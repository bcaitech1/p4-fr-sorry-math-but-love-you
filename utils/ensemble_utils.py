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
import sys

sys.path.append("../")
from .utils import id_to_string, get_network, print_gpu_status, print_ram_status
from .flags import Flags
from .checkpoint import load_checkpoint

from postprocessing.postprocessing import DecodingManager


def make_encoder_values(
    models: List[nn.Module], d: List[torch.Tensor], device, model_order: dict
) -> List[torch.Tensor]:
    """인코더 모델의 추론 결과를 리턴하는 함수

    Args:
        models (List[nn.Module]): 앙상블에 활용할 인코더 모델 리스트
        d (List[torch.Tensor]): 미니 배치 데이터
        device ([type]): 연산에 활용할 디바이스
        model_order (dict): 모델별 ID. 앙상블 수행 시 할당됨

    Returns:
        List[torch.Tensor]: 모델별 추론 결과 리스트
    """
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
) -> List[str]:
    """디코더 모델의 추론 결과를 리턴하는 함수

    Args:
        models (List[nn.Module]): 앙상블에 활용할 인코더 모델 리스트
        parser ([type]): 앙상블 구동 시 입력한 parser
        enc_dataloader (DataLoader): 인코더 데이터로더
        dec_dataloader (DataLoader): 디코더 데이터로더
        manager (DecodingManager): 생성 과정 중 후처리에 활용할 DecodingManager
        device ([type]): 연산에 활용할 디바이스

    Returns:
        List[str]: 디코딩 결과를 문자열 형태의 수식으로 복원한 추론 결과 리스트
    """
    st_id = models[0].decoder.st_id
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
                        one_step_out = F.softmax(_out, dim=-1)
                    else:
                        one_step_out += F.softmax(_out, dim=-1)

                one_step_out = one_step_out / len(models)

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
    """모델 ID를 리맵핑. 모델 아키텍처 수에 관계 없이 앙상블이 가능하도록 하기 위한 함수임"""
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
    """데이터 로더를 리맵핑. 모델 아키텍처 수에 관계 없이 앙상블이 가능하도록 하기 위한 함수임"""
    test_data_loaders = [l for l in test_data_loaders if l is not None]
    return test_data_loaders


def truncate_aligned_models(models: List[nn.Module], verbose: bool) -> None:
    """입력된 모델을 kill하는 함수. 한정된 GPU 자원의 과부하 방지를 위해 사용"""
    for _ in range(len(models)):
        del models[0]
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print_gpu_status()


def load_encoder_models(
    checkpoints: List[str], dataset: Dataset, is_cuda, device, verbose: bool = False
) -> List[nn.Module]:
    """인코더 모델을 불러오는 함수"""
    enc_models = []
    enc_total_params = 0
    for c in checkpoints:
        ckpt = load_checkpoint(c, cuda=is_cuda)
        network_type = ckpt["network"]
        param_dict = ckpt["model"]

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
    """디코더 모델을 불러오는 함수"""
    dec_models = []
    dec_total_params = 0
    for c in checkpoints:
        ckpt = load_checkpoint(c, cuda=is_cuda)
        network_type = ckpt["network"]
        param_dict = ckpt["model"]

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
    """앙상블 과정 중 활용되는 임시폴더 내 파일을 모두 제거"""
    for fpath in glob(os.path.join(dir, "*")):
        os.remove(fpath)
