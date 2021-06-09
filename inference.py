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


def get_test_transform(height, width):
    return A.Compose([
        A.Resize(height, width, p=1.),
        A.Normalize(),
        ToTensorV2(),
    ])

def ensemble(models, input_images, expected):
    decoded_values = None
    for model in models:
        output = model(input_images, expected, False, 0.0)
        if decoded_values is None:
            decoded_values = output.transpose(1, 2)
        else:
            decoded_values += output.transpose(1, 2)
    decoded_values /= len(models)

    return decoded_values

def main(parser):
    torch.manual_seed(21)
    random.seed(21)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence
    
    transformed = get_test_transform(256, 512)
    
    token_to_id_ = load_checkpoint(parser.checkpoint[0], cuda=is_cuda)['token_to_id']
    id_to_token_ = load_checkpoint(parser.checkpoint[0], cuda=is_cuda)['id_to_token']

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, token_to_id_, id_to_token_, crop=False, transform=transformed,
        rgb=3
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_eval_batch,
    )

    SATRN_models = []

    for parser_checkpoint in parser.checkpoint:
        checkpoint = load_checkpoint(parser_checkpoint, cuda=is_cuda)
        options = Flags(checkpoint["configs"]).get()
        model_checkpoint = checkpoint["model"]
        model = get_network(options.network, options, model_checkpoint, device, test_dataset)
        model.eval()
        SATRN_models.append(model)
    
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    results = []
    with torch.no_grad():
        for d in tqdm(test_data_loader):
            input = d["image"].to(device).float()
            expected = d["truth"]["encoded"].to(device)

            decoded_values = ensemble(SATRN_models, input, expected)
            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)
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
        default=["/opt/ml/sorry_math_but_love_you/log/my_satrn/MySATRN_fold2.pth",
                "/opt/ml/sorry_math_but_love_you/log/my_satrn/MySATRN_fold3_7945.pth",
                "/opt/ml/sorry_math_but_love_you/log/my_satrn/MySATRN_fold4.pth"],
        nargs='*',
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
        default=4,
        type=int,
        help="batch size when doing inference",
    )
    parser.add_argument(
        "--decode_type",
        dest="decode_type",
        default='greedy', # 'greedy'로 설정하면 기존과 동일하게 inference
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
