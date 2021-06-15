import os
import gc
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from typing import List, Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import id_to_string, get_network, print_gpu_status, print_ram_status
from flags import Flags
from postprocessing import DecodingManager
from checkpoint import load_checkpoint
from augmentations import get_test_transform
from dataset import LoadEvalDataset, collate_eval_batch


def compose_test_dataloader(
    test_data, batch_size, token_to_id, id_to_token, height, width, num_workers
):
    transforms = get_test_transform(height=height, width=width)
    dataset = LoadEvalDataset(
        test_data, token_to_id, id_to_token, crop=False, transform=transforms, rgb=3
    )
    test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval_batch,
    )
    return test_dataloader

def make_encoder_values(models, d, device, model_order: dict) -> List[torch.Tensor]:
    encoder_values = []
    for model_name, model in models:
        d_idx = model_order.get(model_name, None)
        assert d_idx is not None, f"There's no model_name '{model_name}'"
        input_images = d[d_idx]["image"].to(device).float()
        encoder_value = model(input_images)
        encoder_values.append(encoder_value)  # NOTE: to save as pickle
    return encoder_values


def make_decoder_values(
    models: List[nn.Module],
    parser,
    enc_dataloader: DataLoader,
    dec_dataloader: DataLoader,
    manager: DecodingManager,
    device,
    weight_list,
) -> List[str]:
    st_id = models[0].decoder.st_id
    # print(f"{'='*20} DECODING {'='*20}")
    results_de = []
    num_steps = parser.max_sequence + 1
    with torch.no_grad():
        for step, (paths, predicteds) in tqdm(
            enumerate(dec_dataloader), desc="[Decoding]"
        ):
            out = []

            # align <SOS>
            batch_size = len(predicteds[0])
            target = torch.LongTensor(batch_size).fill_(st_id).to(device)

            # initialize decoding manager
            if manager is not None:
                manager.reset(sequence_length=num_steps)

            for _ in range(num_steps):
                one_step_out = None
                for m, model in enumerate(models):
                    input = predicteds[m].to(device)
                    _out = model.step_forward(input, target)

                    if _out.ndim > 2:
                        _out = _out.squeeze()
                    assert _out.ndim == 2  # [B, VOCAB_SIZE]

                    if one_step_out == None:
                        one_step_out = F.softmax(_out, dim=-1) * weight_list[m]
                    else:
                        one_step_out += F.softmax(_out, dim=-1) * weight_list[m]

#                 one_step_out = one_step_out / len(models)
                one_step_out = one_step_out / sum(weight_list)

                if manager is not None:
                    target, one_step_out = manager.sift(one_step_out)
                else:
                    target = torch.argmax(one_step_out, dim=-1)

                out.append(one_step_out)
            decoded_values = torch.stack(out, dim=1).to(
                device
            )  # [B, MAX_LENGTH, VOCAB_SIZE]

            _, sequences = torch.topk(decoded_values, k=1, dim=-1)
            sequences = sequences.squeeze()
            sequences_str = id_to_string(sequences, enc_dataloader, do_eval=1)

            for path, predicted in zip(paths, sequences_str):
                results_de.append((path, predicted))

            for model in models:
                model.reset_status()

    return results_de


def remap_model_idx(model_order, data_loaders: list):
    if all(data_loaders) is not True:
        idx2name = {i: name for name, i in model_order.items()}
        remapped_order = dict()
        new_order = 0
        for idx, d in enumerate(data_loaders):
            if d is not None:
                model_name = idx2name[idx]
                remapped_order[model_name] = new_order
                new_order += 1
    else:
        remapped_order = model_order
    print(f"[+] MODEL ID: {remapped_order}\n")
    return remapped_order


def remap_test_dataloaders(test_data_loaders) -> List[DataLoader]:
    test_data_loaders = [l for l in test_data_loaders if l is not None]
    return test_data_loaders


def truncate_aligned_models(models: List[nn.Module], verbose: bool) -> None:
    for _ in range(len(models)):
        del models[0]
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print_gpu_status()


def load_encoder_models(
    checkpoints: List[str], dataset: Dataset, is_cuda, device, verbose: bool = False
) -> List[nn.Module]:
    enc_models = []
    enc_total_params = 0
    for c in checkpoints:
        ckpt = load_checkpoint(c, cuda=is_cuda)
        network_type = ckpt['network']
        param_dict = ckpt['model']

        # load model weights
        enc = OrderedDict()
        for (key, value) in param_dict.items():
            if key.startswith("encoder"):
                enc[key] = value

        # compose model
        encoder_name = f"{network_type}_encoder"
        options = Flags(ckpt["configs"]).get()
        enc_model = get_network(encoder_name, options, enc, device, dataset)
        enc_model.eval()
        enc_models.append((options.network, enc_model))

        if verbose:
            enc_params = [p.numel() for p in enc_model.parameters()]
            enc_total_params += sum(enc_params)

    if verbose:
        print(
            "[+] Encoders\n",
            f"The number of models : {len(enc_models)}\n",
            f"The number of total params : {enc_total_params}\n",
        )
    return enc_models


def load_decoder_models(
    checkpoints: dict, dataset: Dataset, is_cuda, device, verbose: bool = False
) -> List[nn.Module]:
    dec_models = []
    dec_total_params = 0
    for c in checkpoints:
        ckpt = load_checkpoint(c, cuda=is_cuda)
        network_type = ckpt["network"]
        param_dict = ckpt['model']

        # compose model checkpoint weights
        dec = OrderedDict()
        for (key, value) in param_dict.items():
            if key.startswith("decoder"):
                dec[key] = value

        # load model
        decoder_name = f"{network_type}_decoder"
        options = Flags(ckpt["configs"]).get()
        dec_model = get_network(decoder_name, options, dec, device, dataset)
        dec_model.eval()
        dec_models.append(dec_model)

        if verbose:
            dec_params = [p.numel() for p in dec_model.parameters()]
            dec_total_params += sum(dec_params)

    if verbose:
        print(
            "[+] Decoders\n",
            f"The number of models : {len(dec_models)}\n",
            f"The number of total params : {dec_total_params}\n",
        )
    return dec_models

def remove_all_files_in_dir(dir):
    for fpath in glob(os.path.join(dir, '*')):
        os.remove(fpath)

class DecoderDataset(Dataset):
    def __init__(self, tmp_dir: str):
        self.paths = sorted(glob(os.path.join(tmp_dir, "*")))

    def __getitem__(self, idx):
        output = torch.load(self.paths[idx], map_location="cpu")
        return output

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def collate_fn(batch):
        paths_aggregated = []
        batch_aggregated = None

        for idx, (paths, batch_each_model) in enumerate(batch):
            if idx == 0:
                num_models = len(batch_each_model)
                batch_aggregated = [[] for _ in range(num_models)]

            paths_aggregated.extend(paths)
            for m, batch in enumerate(batch_each_model):
                batch_aggregated[m].append(batch)

        # 모델별 텐서를 병합
        for i in range(len(batch_aggregated)):
            batch_aggregated[i] = torch.vstack(batch_aggregated[i])

        return paths_aggregated, batch_aggregated
