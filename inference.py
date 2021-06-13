import os
from glob import glob
import gc
from collections import OrderedDict
from copy import deepcopy
import argparse
import random
from tqdm import tqdm
import time
from typing import List
import csv
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from metrics import word_error_rate, sentence_acc, final_metric
from checkpoint import load_checkpoint
from dataset import LoadEvalDataset, collate_eval_batch, START, PAD
from train import get_valid_transforms
from flags import Flags
from utils import id_to_string, get_network, get_optimizer, set_seed
from decoding import decode
from postprocessing import get_decoding_manager
from augmentations import get_test_transform
from utils import print_memory_status

NO_TEACHER_FORCING = 0.0
SATRN_IDX = 0
SWIN_IDX = 1
ASTER_IDX = 2
ORDER = dict(MySATRN=SATRN_IDX, SWIN=SWIN_IDX, ASTER=ASTER_IDX)  # 디폴트 모델ID


def make_encoder_values(models, d, device):
    global ORDER
    encoder_values = []
    for n, (model_name, model) in enumerate(models):
        d_idx = ORDER.get(model_name, None)
        assert d_idx is not None, f"There's no model_name '{model_name}'"
        input_images = d[d_idx]["image"].to(device).float()
        encoder_value = model(input_images)
        encoder_values.append(encoder_value) # NOTE: to save as pickle
    return encoder_values


def remap_model_idx(data_loaders: list):
    global ORDER
    if all(data_loaders) is not True:
        idx2name = {i: name for name, i in ORDER.items()}
        remapped_order = dict()
        new_order = 0
        for idx, d in enumerate(data_loaders):
            if d is not None:
                model_name = idx2name[idx]
                remapped_order[model_name] = new_order
                new_order += 1
        ORDER = remapped_order
    print(f"[+] MODEL ID: {ORDER}\n")


def truncate_aligned_models(models: List[nn.Module]) -> None:
    for _ in range(len(models)):
        del models[0]
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_status()


class DecoderDataset(Dataset):
    def __init__(self, tmp_dir: str):
        self.paths = sorted(glob(os.path.join(tmp_dir, '*')))

    def __getitem__(self, idx):
        output = torch.load(self.paths[idx], map_location='cpu')
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
    

def main(parser):
    set_seed(21)
    start = time.time()
    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    tmp_dir = './tmp_enc_results'

    # NOTE: Load Test Data
    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]

    # df = pd.read_csv("./configs/data_info.txt")
    # test_image_names = set(df[df["fold"] == 4]["image_name"].head(100).values)
    # with open(os.path.join(os.path.dirname(parser.file_path), "gt.txt"), "r") as fd:
    #     data = []
    #     for line in fd:
    #         data.append(line.strip().split("\t"))
    #     dataset_len = round(len(data) * 1.0)
    #     data = data[:dataset_len]
    # test_data = [
    #     [os.path.join(root, x[0]), x[0], dummy_gt]
    #     for x in data
    #     if x[0] in test_image_names
    # ]
    
    # NOTE: Compose DataLoaders & Encoders
    enc_models = [] 
    data_loaders = [None] * 3  # [SATRN, SWIN, ASTER] - 모델 고유 순서를 바탕으로 채워짐
    enc_total_params = 0
    token_to_id_ = None
    id_to_token_ = None

    for parser_checkpoint in parser.checkpoint:
        checkpoint = load_checkpoint(parser_checkpoint, cuda=is_cuda)
        model_name = checkpoint["network"]
        height = checkpoint["configs"]["input_size"]["height"]
        width = checkpoint["configs"]["input_size"]["width"]

        if token_to_id_ is None and id_to_token_ is None:
            token_to_id_ = checkpoint["token_to_id"]
            id_to_token_ = checkpoint["id_to_token"]

        assert token_to_id_ is not None 
        assert id_to_token_ is not None

        loader_idx = ORDER[model_name]
        if data_loaders[loader_idx] is None: # 모델별 데이터로더가 생성되지 않을 때만 생성하도록
            transforms = get_test_transform(height=height, width=width)
            dataset = LoadEvalDataset(
                test_data,
                token_to_id_,
                id_to_token_,
                crop=False,
                transform=transforms,
                rgb=3,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=parser.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_eval_batch,
            )
            data_loaders[loader_idx] = dataloader

        assert data_loaders[loader_idx] is not None  # NOTE: for DEBUG

        # load model weights
        enc = OrderedDict()
        for (key, value) in checkpoint["model"].items():
            if key.startswith("encoder"):
                enc[key] = value

        # compose model
        encoder_name = f"{model_name}_encoder"
        options = Flags(checkpoint["configs"]).get()
        enc_model = get_network(
            encoder_name, options, enc, device, data_loaders[loader_idx].dataset
        )
        enc_model.eval()
        enc_models.append((options.network, enc_model))

        # count number of params
        enc_params = [p.numel() for p in enc_model.parameters()]
        enc_total_params += sum(enc_params)

    print(
        "[+] Encoders\n",
        f"The number of models : {len(enc_models)}\n",
        f"The number of total params : {enc_total_params}\n",
    )
    
    loading_time = (time.time() - start) / 60
    print(
        "[+] Time Check\n",
        f"Data & Encoder Loading Time(min) : {loading_time:.2f}\n",
    )
    start = time.time()

    print_memory_status()
    remap_model_idx(data_loaders) # 모델ID 업데이트
    data_loaders = [l for l in data_loaders if l is not None] # data_loaders에서 None 원소 제외

    print(f"{'='*20} ENCODING {'='*20}")
    os.makedirs(tmp_dir, exist_ok=True)
    with torch.no_grad():
        for step, d in tqdm(enumerate(zip(*data_loaders)), desc="[Encoding]"):
            file_path = d[0]['file_path']
            encoder_values = make_encoder_values(enc_models, d, device)  # list
            batch_result = (file_path, encoder_values)
            torch.save(batch_result, os.path.join(tmp_dir, f'batch{step:0>4d}')) # (Paths, tensor([BATCH_SIZE, X, X]))
            if step % 400 == 0 and step != 0: print_memory_status()
    
    encoding_time = (time.time() - start) / 60
    print(
        "[+] Time Check\n",
        f"Encoding Time(min) : {encoding_time:.2f}\n",
    )
    start = time.time()
    print(f"{'='*20} Done! {'='*20}")

    gc.collect()
    torch.cuda.empty_cache()
    truncate_aligned_models(enc_models)
    
    dec_models = []
    dec_total_params = 0
    for parser_checkpoint in parser.checkpoint:
        checkpoint = load_checkpoint(parser_checkpoint, cuda=is_cuda)
        model_name = checkpoint["network"]

        # compose model checkpoint weights
        dec = OrderedDict()
        for (key, value) in checkpoint["model"].items():
            if key.startswith("decoder"):
                dec[key] = value

        # load model
        decoder_name = f"{model_name}_decoder"
        options = Flags(checkpoint["configs"]).get()
        dec_model = get_network(decoder_name, options, dec, device, data_loaders[0].dataset)
        dec_model.eval()
        dec_models.append(dec_model)
        dec_params = [p.numel() for p in dec_model.parameters()]
        dec_total_params += sum(dec_params)

    print(
        "[+] Decoders\n",
        f"The number of models : {len(dec_models)}\n",
        f"The number of total params : {dec_total_params}\n",
    )

    loading_time = (time.time() - start) / 60
    print(
        "[+] Time Check\n",
        f"Decoder Loading Time(min) : {loading_time:.2f}\n",
    )
    start = time.time()


    # NOTE: compose decoding manager
    manager = (
        get_decoding_manager(
            tokens_path=parser.tokens_path, batch_size=parser.batch_size
        )
        if parser.decoding_manager
        else None
    )

    # NOTE: load decoding dataset
    decoding_dataset = DecoderDataset(tmp_dir)
    decoder_dataloader = DataLoader(
        decoding_dataset, 
        batch_size=4, # NOTE: batch_size = encoder batch size * decoder batch size
        shuffle=False, 
        drop_last=False, 
        collate_fn=decoding_dataset.collate_fn
        )


    st_id = dec_models[0].decoder.st_id
    print(f"{'='*20} DECODING {'='*20}")
    results_de = []
    num_steps = parser.max_sequence + 1
    with torch.no_grad():
        # for step, (paths, predicteds) in tqdm(enumerate(decoding_dataset), desc='[Decoding]'):
        for step, (paths, predicteds) in tqdm(enumerate(decoder_dataloader), desc='[Decoding]'):
            out = []

            # align <SOS>
            batch_size = len(predicteds[0])
            target = (
                torch.LongTensor(batch_size)
                .fill_(st_id)
                .to(device)
            )

            # initialize decoding manager
            if manager is not None:
                manager.reset(sequence_length=num_steps)
            
            for _ in range(num_steps):
                one_step_out = None
                for m, model in enumerate(dec_models):
                    input = predicteds[m].to(device)
                    _out = model.step_forward(input, target)
                    
                    if _out.ndim > 2:
                        _out = _out.squeeze()

                    assert _out.ndim == 2 # [B, VOCAB_SIZE]

                    if one_step_out == None:
                        one_step_out = F.softmax(_out, dim=-1)
                    else:
                        one_step_out += F.softmax(_out, dim=-1)

                one_step_out = one_step_out / len(dec_models)

                if manager is not None:
                    target, one_step_out = manager.sift(one_step_out)
                else:
                    target = torch.argmax(one_step_out, dim=-1)
                    # target = target.squeeze()

                out.append(one_step_out)
            decoded_values = torch.stack(out, dim=1).to(device)  # [B, MAX_LENGTH, VOCAB_SIZE]

            _, sequences = torch.topk(decoded_values, k=1, dim=-1)
            sequences = sequences.squeeze()
            sequences_str = id_to_string(sequences, data_loaders[0], do_eval=1)
            for path, predicted in zip(paths, sequences_str):
                results_de.append((path, predicted))

            for model in dec_models:
                model.reset_status()

            if step % 100 == 0 and step != 0: print_memory_status() # verbose

    spend_time = (time.time() - start) / 60
    print(f"{'='*20} Done! WALL TIME: {spend_time:.4f} {'='*20}")
    
    export_dir = f"{parser.output_dir}"
    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, "output.csv"), "w") as w:
        for path, predicted in results_de:
            w.write(path + "\t" + predicted + "\n")

    decoding_time = (time.time() - start) / 60
    print(
        "[+] Time Check\n",
        f"Decoding Time(min) : {decoding_time:.2f}\n",
    )
    print(f"{'='*20} Done! {'='*20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=[
            "/opt/ml/code/models/aster-fold-0-0.7878.pth",
            "/opt/ml/code/models/aster-fold-1-0.7869.pth",
            "/opt/ml/code/models/aster-fold-2-0.7900.pth",
            "/opt/ml/code/models/aster-fold-3-0.7861.pth",
            "/opt/ml/code/models/aster-fold-4-0.7846.pth",
            # "/opt/ml/code/models/satrn/satrn-fold-2-0.8171.pth",
            # "",
            # "",
            # "",
            # "",
            # "/opt/ml/code/models/swin-fold-2-0.8322.pth",
            # "/opt/ml/code/models/swin-fold-3-0.8293.pth",
            # "/opt/ml/code/models/swin-fold-4-0.8259.pth",
        ],
        nargs="*",
        help="Path of checkpoint file",
    )
    parser.add_argument(
        "--max_sequence",
        dest="max_sequence",
        default=230,
        type=int,
        help="maximun sequence when doing inference",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=16,
        type=int,
        help="batch size when doing inference",
    )
    parser.add_argument(
        "--decode_type",
        dest="decode_type",
        default="greedy",
        type=str,
        help="디코딩 방식 설정. 'greedy', 'beam'",
    )
    parser.add_argument(
        "--beam_width",
        dest="beam_width",
        default=3,
        type=int,
        help="빔서치 사용 시 스텝별 후보 수 설정",
    )
    parser.add_argument(
        "--decoding_manager", default=False, help="DecodingManager 사용 여부 결정"
    )
    parser.add_argument(
        "--tokens_path",
        default="/opt/ml/input/data/train_dataset/tokens.txt",
        help="DecodingManager 사용시 활용할 토큰 파일 경로",
    )

    eval_dir = os.environ.get("SM_CHANNEL_EVAL", "../input/data/")
    file_path = os.path.join(eval_dir, 'eval_dataset/input.txt')
    # file_path = os.path.join('./eval_dataset/input.txt') # NOTE: for DEBUG
    # file_path = "/content/data/train_dataset/gt.txt"  # NOTE: for COLAB
    parser.add_argument(
        "--file_path",
        dest="file_path",
        default=file_path,
        type=str,
        help="file path when doing inference",
    )

    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "submit")
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="output directory",
    )
    parser = parser.parse_args()
    main(parser)
