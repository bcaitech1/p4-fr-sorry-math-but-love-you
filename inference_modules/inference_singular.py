import os
import argparse
import random
from tqdm import tqdm
import csv
import torch
from torch.utils.data import DataLoader

from utils import (
    word_error_rate,
    sentence_acc,
    final_metric,
    load_checkpoint,
    Flags,
    id_to_string,
    get_network,
    get_optimizer,
    set_seed
)
from data import (
    LoadEvalDataset,
    collate_eval_batch,
    START,
    PAD,
    get_valid_transforms
)
from postprocessing import decode, get_decoding_manager



def main(parser):
    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    set_seed(options.seed)

    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )

    transformed = get_valid_transforms(
        height=options.input_size.height, width=options.input_size.width
    )
    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)

    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data,
        checkpoint["token_to_id"],
        checkpoint["id_to_token"],
        crop=False,
        transform=transformed,
        rgb=options.data.rgb,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_eval_batch,
    )

    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset)),
    )
    manager = (
        get_decoding_manager(
            tokens_path="./configs/tokens.txt", batch_size=parser.batch_size
        )
        if parser.decoding_manager
        else None
    )

    model = get_network(
        model_type=options.network,
        FLAGS=options,
        model_checkpoint=model_checkpoint,
        device=device,
        dataset=test_dataset,
        decoding_manager=manager,
    )
    model.eval()

    print("[+] Decoding Type:", parser.decode_type)
    results = []
    with torch.no_grad():
        for d in tqdm(test_data_loader):
            input = d["image"].float().to(device)
            expected = d["truth"]["encoded"].to(device)
            sequence = decode(
                model=model,
                input=input,
                data_loader=test_data_loader,
                expected=expected,
                method=parser.decode_type,
                beam_width=parser.beam_width,
            )
            sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)

            for path, predicted in zip(d["file_path"], sequence_str):
                results.append((path, predicted))

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted in results:
            w.write(path + "\t" + predicted + "\n")