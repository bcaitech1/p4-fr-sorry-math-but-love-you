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


def validate(parser):
    import time
    from dataset import collate_batch, LoadDataset, split_gt

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
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
    print(options.input_size.height)

    valid_transform = get_valid_transforms(height=options.input_size.height, width=options.input_size.width)

    valid_data = []
    for i, path in enumerate(options.data.train):
        prop = 1.0
        if len(options.data.dataset_proportions) > i:
            prop = options.data.dataset_proportions[i]
        _, valid = split_gt(path, prop, options.data.test_proportions)
        valid_data += valid

    valid_dataset = LoadDataset(
        # valid_data, options.data.token_paths, crop=options.data.crop, transform=transformed, rgb=options.data.rgb
        valid_data, options.data.token_paths, crop=options.data.crop, transform=valid_transform, rgb=options.data.rgb
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
        "The number of test samples : {}\n".format(len(valid_dataset)),
    )

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

                if parser.decode_type == 'greedy':
                    output = model(
                        input=input, 
                        expected=expected, 
                        is_train=False,
                        teacher_forcing_ratio=0.0
                        )
                    decoded_values = output.transpose(1, 2) # [B, VOCAB_SIZE, MAX_LEN]
                    _, sequence = torch.topk(decoded_values, 1, dim=1) # sequence: [B, 1, MAX_LEN]
                    sequence = sequence.squeeze(1) # [B, MAX_LEN], 각 샘플에 대해 시퀀스가 생성 상태

                elif parser.decode_type == 'beam':
                    sequence = model.beam_search(
                        input=input,
                        data_loader=valid_data_loader,
                        beam_width=parser.beam_width,
                        max_sequence=expected.size(-1)-1 # expected에는 이미 시작 토큰 개수까지 포함
                    )
                elif parser.decode_type == 'greedy-ensemble':
                    raise NotImplementedError
                
                elif parser.decode_type == 'beam-ensemble':
                    raise NotImplementedError
                
                else:
                    raise NotImplementedError

                expected[expected == valid_data_loader.dataset.token_to_id[PAD]] = -1
                expected_str = id_to_string(expected, valid_data_loader, do_eval=1)
                sequence_str = id_to_string(sequence, valid_data_loader, do_eval=1)
                wer += word_error_rate(sequence_str, expected_str)
                num_wer += 1
                sent_acc += sentence_acc(sequence_str, expected_str)
                num_sent_acc += 1
                correct_symbols += torch.sum(sequence.to(device) == expected[:, 1:], dim=(0, 1)).item()
                total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

                pbar.update(curr_batch_size)
    inference_time = (time.time() - start) / 60 # minutes
    valid_sentence_accuracy = sent_acc / num_sent_acc
    valid_wer = wer / num_wer
    valid_score = final_metric(sentence_acc=valid_sentence_accuracy, word_error_rate=valid_wer)
    print(f'INFERENCE TIME: {inference_time}')
    print(f'SCORE: {valid_score} SENTENCE ACC: {valid_sentence_accuracy} WER: {valid_wer}')


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
    print(options.input_size.height)

    # transformed = get_valid_transforms(height=options.input_size.height, width=options.input_size.width)
    transformed = A.Compose([A.Resize(256, 512, p=1.), ToTensorV2()])

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=transformed,
        rgb=options.data.rgb
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

    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        test_dataset,
    )
    model.eval()
    results = []
    

    with torch.no_grad():
        for d in tqdm(test_data_loader):
            input = d["image"].float().to(device)
            expected = d["truth"]["encoded"].to(device)

            if parser.decode_type == 'greedy':
                output = model(input, expected, False, 0.0)
                decoded_values = output.transpose(1, 2)
                _, sequence = torch.topk(decoded_values, 1, dim=1)
                sequence = sequence.squeeze(1)

            elif parser.decode_type == 'beam':
                sequence = model.beam_search(
                    input=input, 
                    data_loader=test_data_loader,
                    beam_width=parser.beam_width,
                    max_sequence=parser.max_sequence,
                    )
            elif parser.decode_type == 'greedy-ensemble':
                    raise NotImplementedError
                
            elif parser.decode_type == 'beam-ensemble':
                raise NotImplementedError
            
            else:
                raise NotImplementedError

            sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
            
            for path, predicted in zip(d["file_path"], sequence_str):
                results.append((path, predicted))

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted in results:
            w.write(path + "\t" + predicted + "\n")


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
    #-----------------------
    parser.add_argument(
        "--decode_type",
        dest="decode_type",
        default='greedy',
        type=str,
        help="디코딩 방식 설정. 'greedy', 'beam', 'greedy-ensemble', 'beam-ensemble'",
    )

    parser.add_argument(
        "--beam_width",
        dest="beam_width",
        default=5,
        type=int,
        help="빔서치 사용 시 스텝별 후보 수 설정",
    )
    #-----------------------

    eval_dir = os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/')
    file_path = os.path.join(eval_dir, 'eval_dataset/input.txt')
    parser.add_argument(
        "--file_path",
        dest="file_path",
        default=file_path,
        type=str,
        help="file path when doing inference",
    )

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'submit')
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="output directory",
    )

    parser = parser.parse_args()
    main(parser)
    # validate(parser)
