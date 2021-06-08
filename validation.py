import os
import argparse
import random
from tqdm import tqdm
import csv
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from metrics import word_error_rate, sentence_acc, final_metric
from checkpoint import load_checkpoint
from dataset import LoadEvalDataset, collate_eval_batch, START, PAD
from train import get_valid_transforms
from flags import Flags
from utils import id_to_string, get_network, get_optimizer, set_seed
from decoding import decode


def validate(parser):
    """학습한 모델의 성능 검증을 위한 함수. NOTE: 제출용 함수가 아님!

    Args:
        parser ([type]): [description]
    """
    import time
    from dataset import collate_batch, LoadDataset, split_gt

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    checkpoint = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    set_seed(options.seed)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )

    # Load data
    valid_transform = get_valid_transforms(
        height=options.input_size.height, width=options.input_size.width
    )

    valid_data = []
    for i, path in enumerate(options.data.train):
        prop = 1.0
        if len(options.data.dataset_proportions) > i:
            prop = options.data.dataset_proportions[i]
        _, valid = split_gt(path, prop, options.data.test_proportions)
        valid_data += valid

    valid_dataset = LoadDataset(
        valid_data,
        options.data.token_paths,
        crop=options.data.crop,
        transform=valid_transform,
        rgb=options.data.rgb,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )

    print(
        "[+] Data\n",
        "The number of test samples : {}".format(len(valid_dataset)),
    )

    # Load model
    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        valid_dataset,
    )
    model.eval()

    correct_symbols = 0
    total_symbols = 0
    wer = 0
    num_wer = 0
    sent_acc = 0
    num_sent_acc = 0

    # Infernce
    print("[+] Decoding Type:", parser.decode_type)
    start = time.time()
    with torch.no_grad():
        with tqdm(
            desc=f"Validation",
            total=len(valid_data_loader.dataset),
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for d in valid_data_loader:
                input = d["image"].float().to(device)

                curr_batch_size = len(input)
                expected = d["truth"]["encoded"].to(device)
                expected[expected == -1] = valid_data_loader.dataset.token_to_id[PAD]

                sequence = decode(
                    model=model,
                    input=input,
                    data_loader=valid_data_loader,
                    expected=expected,
                    method=parser.decode_type,
                    beam_width=parser.beam_width,
                )

                expected[expected == valid_data_loader.dataset.token_to_id[PAD]] = -1
                expected_str = id_to_string(expected, valid_data_loader, do_eval=1)
                sequence_str = id_to_string(sequence, valid_data_loader, do_eval=1)
                wer += word_error_rate(sequence_str, expected_str)
                num_wer += 1
                sent_acc += sentence_acc(sequence_str, expected_str)
                num_sent_acc += 1
                correct_symbols += torch.sum(
                    sequence.to(device) == expected[:, 1:], dim=(0, 1)
                ).item()
                total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

                pbar.update(curr_batch_size)

    # Validation
    inference_time = (time.time() - start) / 60  # minutes
    valid_sentence_accuracy = sent_acc / num_sent_acc
    valid_wer = wer / num_wer
    valid_score = final_metric(
        sentence_acc=valid_sentence_accuracy, word_error_rate=valid_wer
    )
    print(f"INFERENCE TIME: {inference_time}")
    print(
        f"SCORE: {valid_score} SENTENCE ACC: {valid_sentence_accuracy} WER: {valid_wer}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default="./log/MySATRN_best_model.pth",
        type=str,
        help="Path of checkpoint file",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=128,
        type=int,
        help="batch size when doing inference",
    )
    parser.add_argument(
        "--decode_type",
        dest="decode_type",
        default="greedy",  # 'greedy'로 설정하면 기존과 동일하게 inference
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

    parser = parser.parse_args()
    validate(parser)  # 성능 검증 시 활용 NOTE. 제출 전용이 아님!
