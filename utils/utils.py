import os
import sys
import random
import psutil
from typing import List
from psutil import virtual_memory
from datetime import datetime
import warnings
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset

from .flags import Flags
from networks import (
    EfficientSATRN,
    EfficientSATRN_encoder,
    EfficientSATRN_decoder,
    LiteSATRN,
    SWIN, 
    SWIN_encoder, 
    SWIN_decoder,
    ASTER, 
    ASTER_encoder, 
    ASTER_decoder
)


def get_network(
    model_type: str,
    FLAGS: Flags,
    model_checkpoint: str,
    device,
    dataset: Dataset,
    decoding_manager=None,
) -> nn.Module:
    """모델을 불러오는 함수

    Args:
        model_type (str): 불러올 모델 아키텍쳐
        FLAGS (Flags): 모델 configuration 정보
        model_checkpoint (str): 사전 학습한 weigt를 불러올 경우 해당 경로 입력
        device:
        dataset (Dataset): 데이터셋
        decoding_manager (optional): 후처리 클래스 DecodingManager 사용시 입력
    """
    model = None
    if model_type == "EfficientSATRN" or model_type == "MySATRN":
        model = EfficientSATRN(
            FLAGS, dataset, model_checkpoint, decoding_manager
        ).to(device)
    elif model_type == "LiteSATRN":
        model = LiteSATRN(
            FLAGS, dataset, model_checkpoint, decoding_manager
        ).to(device)
    elif model_type == "EfficientSATRN_encoder" or model_type == "MySATRN_encoder":
        model = EfficientSATRN_encoder(FLAGS, dataset, model_checkpoint).to(
            device
        )
    elif model_type == "EfficientSATRN_decoder" or model_type == "MySATRN_decoder":
        model = EfficientSATRN_decoder(FLAGS, dataset, model_checkpoint).to(
            device
        )
    elif model_type == "SWIN":
        model = SWIN(FLAGS, dataset, model_checkpoint).to(device)
    elif model_type == "SWIN_encoder":
        model = SWIN_encoder(FLAGS, dataset, model_checkpoint).to(device)
    elif model_type == "SWIN_decoder":
        model = SWIN_decoder(FLAGS, dataset, model_checkpoint).to(device)
    elif model_type == "EfficientASTER" or model_type == "ASTER":
        model = ASTER(FLAGS, dataset, model_checkpoint, decoding_manager).to(
            device
        )
    elif model_type == "ASTER_encoder":
        model = ASTER_encoder(FLAGS, model_checkpoint).to(device)
    elif model_type == "ASTER_decoder":
        model = ASTER_decoder(FLAGS, dataset, model_checkpoint).to(device)
    else:
        raise NotImplementedError
    return model


def get_optimizer(
    optimizer: str, params: List[torch.Tensor], lr: float, weight_decay: float = None
):
    """옵티마이저를 리턴하는 함수"""
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def print_gpu_status() -> None:
    """GPU 이용 상태를 출력"""
    total_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 3)
    reserved = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 3)
    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 3)
    free = round(reserved - allocated, 3)
    print(
        "[+] GPU Status\n",
        f"Total: {total_mem} GB\n",
        f"Reserved: {reserved} GB\n",
        f"Allocated: {allocated} GB\n",
        f"Residue: {free} GB\n",
    )


def print_system_envs():
    """시스템 환경을 출력"""
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    print(
        "[+] System environments\n",
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )


def print_ram_status():
    """램 이용 상태를 출력"""
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20  # Bytes to MB
    print(f"[+] Memory Status\n", f"Usage: {rss: 10.5f} MB\n")


# Fixed version of id_to_string
def id_to_string(tokens, data_loader, do_eval=0):
    """디코더를 통해 얻은 추론 결과를 문자열로 구성된 수식으로 복원하는 함수"""
    result = []
    if do_eval:
        eos_id = data_loader.dataset.token_to_id["<EOS>"]
        special_ids = set(
            [
                data_loader.dataset.token_to_id["<PAD>"],
                data_loader.dataset.token_to_id["<SOS>"],
                eos_id,
            ]
        )

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
                elif token == eos_id:
                    break
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result


def set_seed(seed: int = 21):
    """시드값을 고정하는 함수. 실험 재현을 위해 사용"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timestamp():
    return datetime.now().strftime(format="%m%d-%H%M%S")
