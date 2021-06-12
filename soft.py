import os
import argparse
from collections import OrderedDict
import random
from tqdm import tqdm
import csv
import torch
from torch.utils.data import DataLoader

from metrics import word_error_rate, sentence_acc, final_metric
from checkpoint import load_checkpoint
from dataset import LoadEvalDataset, collate_eval_batch, START, PAD
from train import get_valid_transforms
from flags import Flags
from utils import id_to_string, get_network, get_optimizer, set_seed
from decoding import decode
from augmentations import get_test_transform

import pickle
import torch.nn.functional as F
import pandas as pd


def make_encoder_values(models: list, input_images, expected):
    encoder_values = [[] for _ in range(len(models))]
    for n, model in enumerate(models):
        encoder_value = model(input_images, expected, False, 0.0)
        encoder_values[n].append(encoder_value)
    return encoder_values


def main(parser):
    set_seed(21)
    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    transformed = get_test_transform(256, 512)

    token_to_id_ = load_checkpoint(parser.checkpoint[0], cuda=is_cuda)["token_to_id"]
    id_to_token_ = load_checkpoint(parser.checkpoint[0], cuda=is_cuda)["id_to_token"]

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    # with open(parser.file_path, "r") as fd:
    #     reader = csv.reader(fd, delimiter="\t")
    #     data = list(reader)
    # test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    # df = pd.read_csv(os.path.join(os.path.dirname(parser.file_path), 'data_info.txt'))
    df = pd.read_csv("./configs/data_info.txt")
    test_image_names = set(df[df["fold"] == 4]["image_name"].values)
    with open(os.path.join(os.path.dirname(parser.file_path), "gt.txt"), "r") as fd:
        data = []
        for line in fd:
            data.append(line.strip().split("\t"))
        dataset_len = round(len(data) * 1.0)
        data = data[:dataset_len]
    test_data = [
        [os.path.join(root, x[0]), x[0], dummy_gt]
        for x in data
        if x[0] in test_image_names
    ]

    test_dataset = LoadEvalDataset(
        test_data, token_to_id_, id_to_token_, crop=False, transform=transformed, rgb=3
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_eval_batch,
    )

    SATRN_en_models = []
    SATRN_de_models = []

    for parser_checkpoint in parser.checkpoint:
        checkpoint = load_checkpoint(parser_checkpoint, cuda=is_cuda)
        enc = OrderedDict()
        dec = OrderedDict()
        for (key, value) in checkpoint["model"].items():
            if key.startswith("encoder"):
                enc[key] = value
            else:
                dec[key] = value
        options = Flags(checkpoint["configs"]).get()
        model_en = get_network("MySATRN_en", options, enc, device, test_dataset)
        model_de = get_network("MySATRN_de", options, dec, device, test_dataset)
        model_en.eval()
        model_de.eval()
        SATRN_en_models.append(model_en)
        SATRN_de_models.append(model_de)

    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    print("Start Encoding")
    results_en = []  # img, (predict0, predict1, ... , expected)
    with torch.no_grad():
        for d in tqdm(test_data_loader, desc="[Encoding]"):
            input = d["image"].to(device).float()  # 4, 3, 256, 512
            expected = d["truth"]["encoded"].to(device)  # 4, 232

            encoder_values = make_encoder_values(
                SATRN_en_models, input, expected
            )  # list

            results_en.append((d["file_path"], encoder_values))

    print("Start Decoding")
    results_de = []
    with torch.no_grad():
        for result_en in tqdm(results_en, desc="[Decoding]"):
            path = result_en[0]
            predicteds = result_en[1]

            out = []
            num_steps = parser.max_sequence + 1
            features_list = [
                [None] * model_de.decoder.layer_num for _ in range(len(SATRN_de_models))
            ]
            target = (
                torch.LongTensor((predicteds[0][0].size(0)))
                .fill_(model_de.decoder.st_id)
                .to(device)
            )
            for t in range(num_steps):
                one_step_out = None
                for m, model_de in enumerate(SATRN_de_models):
                    input = predicteds[m][0].to(device)
                    _out, features_list[m] = model_de(
                        input, expected, t, target, features_list[m], False, 0.0
                    )
                    if one_step_out == None:
                        one_step_out = F.softmax(_out, dim=-1)
                    else:
                        one_step_out += F.softmax(_out, dim=-1)
                one_step_out = one_step_out / len(SATRN_de_models)

                target = torch.argmax(one_step_out[:, -1:, :], dim=-1)
                target = target.squeeze()
                out.append(one_step_out)

            out = torch.stack(out, dim=1).to(device)  # [b, max length, 1, class length]
            decoded_values = out.squeeze(2)
            decoded_values = decoded_values.transpose(1, 2)

            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)
            sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
            for path, predicted in zip(path, sequence_str):
                results_de.append((path, predicted))

        os.makedirs(parser.output_dir + "_de", exist_ok=True)
        with open(os.path.join(parser.output_dir + "_de", "output.csv"), "w") as w:
            for path, predicted in results_de:
                w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=["./log/0.7907 F0 dual opt MySATRN_best_model.pth"],
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
        default=128,
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

    eval_dir = os.environ.get("SM_CHANNEL_EVAL", "../input/data/")
    # file_path = os.path.join(eval_dir, 'eval_dataset/input.txt')
    file_path = os.path.join(eval_dir, "train_dataset/input.txt")
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
