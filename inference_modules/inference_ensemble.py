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

from utils.checkpoint import load_checkpoint
from utils.flags import Flags
from utils.utils import id_to_string, get_network, get_optimizer, set_seed
from postprocessing.postprocessing import get_decoding_manager
from data.augmentations import get_test_transforms
from data.loader import compose_test_dataloader
from data.dataset import DecoderDataset
from utils.utils import print_gpu_status, print_ram_status
from utils.ensemble_utils import (
    load_encoder_models,
    load_decoder_models,
    make_encoder_values,
    make_decoder_values,
    remap_model_idx,
    remap_test_dataloaders,
    truncate_aligned_models,
    remove_all_files_in_dir,
)

NO_TEACHER_FORCING = 0.0
SATRN_IDX = 0
SWIN_IDX = 1
ASTER_IDX = 2
ORDER = dict(MySATRN=SATRN_IDX, SWIN=SWIN_IDX, ASTER=ASTER_IDX)  # 디폴트 모델별 ID
VERBOSE_DEC_INFO = True
VERBOSE_ENC_INFO = True


def main(parser):
    global ORDER, VERBOSE_DEC_INFO, VERBOSE_ENC_INFO
    set_seed(21)
    start = time.time()
    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    tmp_dir = "./tmp_enc_results"
    tmp_export_dir = "./tmp_outputs"
    export_dir = parser.output_dir
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(tmp_export_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    remove_all_files_in_dir(tmp_dir)  # NOTE: 실행시 임시 추론 폴더 내 모든 파일이 삭제됨
    remove_all_files_in_dir(tmp_export_dir)  # NOTE: 실행시 임시 추론 폴더 내 모든 파일이 삭제됨

    # NOTE: Load Test Data
    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence
    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]

    print(f'{"="*20} Load Data {"="*20}')
    # NOTE: Compose DataLoaders
    enc_dataloaders = [None] * 3  # [SATRN, SWIN, ASTER] - 모델 고유 순서를 바탕으로 채워짐
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
        if enc_dataloaders[loader_idx] is None:  # 모델별 데이터로더가 생성되지 않을 때만 생성하도록
            dataloader = compose_test_dataloader(
                test_data=test_data,
                batch_size=parser.batch_size,
                token_to_id=token_to_id_,
                id_to_token=id_to_token_,
                # num_workers=parser.num_workers
                num_workers=0,
                transforms=get_test_transforms(height, width),
            )
            enc_dataloaders[loader_idx] = dataloader

        assert enc_dataloaders[loader_idx] is not None  # NOTE: FOR DEBUG

    print_gpu_status()  # NOTE: GPU 메모리 확인
    print_ram_status()  # NOTE: 램 사용량 확인
    ORDER = remap_model_idx(ORDER, enc_dataloaders)  # 모델ID 업데이트
    enc_dataloaders = [
        l for l in enc_dataloaders if l is not None
    ]  # data_loaders에서 None 원소 제외

    loading_time = (time.time() - start) / 60
    print(
        "[+] Time Check\n",
        f"Data Loading Time(min) : {loading_time:.2f}\n",
    )
    time_ckpt = time.time()

    print(f"{'='*20} Inference {'='*20}")
    with torch.no_grad():
        current_num_batches = 0
        need_enc_loading = True
        output_id = 0
        total_steps = len(enc_dataloaders[0])

        for step, d in tqdm(enumerate(zip(*enc_dataloaders)), desc="[Inference]"):
            file_path = d[0]["file_path"]

            if need_enc_loading:
                enc_models = load_encoder_models(
                    checkpoints=parser.checkpoint,
                    dataset=enc_dataloaders[0].dataset,
                    is_cuda=is_cuda,
                    device=device,
                    verbose=VERBOSE_ENC_INFO,
                )
                need_enc_loading = False

                if VERBOSE_ENC_INFO:
                    print(f"*** Encoder Loading Status ***")
                    print_gpu_status()
                    print_ram_status()

            encoder_values = make_encoder_values(
                models=enc_models, d=d, device=device, model_order=ORDER
            )  # list

            batch_result = (file_path, encoder_values)
            torch.save(
                batch_result, os.path.join(tmp_dir, f"batch{current_num_batches:0>4d}")
            )  # (Paths, tensor([BATCH_SIZE, X, X]))
            current_num_batches += 1
            if VERBOSE_ENC_INFO:
                VERBOSE_ENC_INFO = False

            if (
                parser.max_cache <= current_num_batches or step == total_steps - 1
            ):  # 디코딩으로 넘어감
                gc.collect()
                torch.cuda.empty_cache()
                if VERBOSE_DEC_INFO:
                    print("*** Encoder Truncation Status ***")
                    print_ram_status()
                truncate_aligned_models(enc_models, VERBOSE_DEC_INFO)
                need_enc_loading = True

                dec_models = load_decoder_models(
                    checkpoints=parser.checkpoint,
                    dataset=enc_dataloaders[0].dataset,
                    is_cuda=is_cuda,
                    device=device,
                    verbose=VERBOSE_DEC_INFO,
                )
                if VERBOSE_DEC_INFO:
                    print(f"*** Decoder Loading Status ***")
                    print_gpu_status()
                    print_ram_status()

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
                dec_dataloader = DataLoader(
                    decoding_dataset,
                    batch_size=4,  # NOTE: batch_size = encoder batch size * decoder batch size
                    shuffle=False,
                    drop_last=False,
                    collate_fn=decoding_dataset.collate_fn,
                )

                results_de = make_decoder_values(
                    models=dec_models,
                    parser=parser,
                    enc_dataloader=enc_dataloaders[0],
                    dec_dataloader=dec_dataloader,
                    manager=manager,
                    device=device,
                )
                with open(
                    os.path.join(tmp_export_dir, f"output{output_id:0>5d}.csv"), "w"
                ) as w:
                    for path, predicted in results_de:
                        w.write(path + "\t" + predicted + "\n")

                remove_all_files_in_dir(tmp_dir)
                gc.collect()
                torch.cuda.empty_cache()
                if VERBOSE_DEC_INFO:
                    print("*** Decoder Truncation Status ***")
                    print_ram_status()
                truncate_aligned_models(dec_models, verbose=VERBOSE_DEC_INFO)
                current_num_batches = 0  # 배치 넘버링 초기화
                output_id += 1  # 추론 결과 넘버링 갱신

                if VERBOSE_DEC_INFO:
                    loading_time = (time.time() - time_ckpt) / 60
                    print(
                        "[+] Time Check\n",
                        f"Data Loading Time(min) : {loading_time:.2f}\n",
                    )
                    VERBOSE_DEC_INFO = False

    # NOTE: Export final output
    output_list = sorted(glob(os.path.join(tmp_export_dir, "*")))
    output = []
    for o in output_list:
        temp_output = pd.read_csv(o, sep="\t", header=None)
        temp_output.columns = ["path", "predicted"]
        output.append(temp_output)

    output = pd.concat(output, axis=0, ignore_index=True)
    with open(os.path.join(export_dir, "output.csv"), "w") as w:
        for _, row in output.iterrows():
            path = row.path
            predicted = row.predicted
            w.write(path + "\t" + predicted + "\n")

    loading_time = (time.time() - start) / 60
    print(
        "[+] Time Check\n",
        f"Total Inference Time(min) : {loading_time:.2f}\n",
    )