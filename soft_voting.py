import os
import argparse
import random
from tqdm import tqdm
import csv
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from metrics_ import word_error_rate, sentence_acc, final_metric
from checkpoint import load_checkpoint
from dataset import LoadEvalDataset, collate_eval_batch, START, PAD
from train import get_valid_transforms
from flags import Flags
from utils import id_to_string, get_network, get_optimizer, set_seed
from decoding import decode

from collections import OrderedDict
import pickle
import torch.nn.functional as F
import pandas as pd
import gc

def get_test_transform(height, width):
    return A.Compose([
        A.Resize(height, width, p=1.),
        A.Normalize(),
        ToTensorV2(),
    ])


def make_encoder_values(models, d, device):
    encoder_values = [[] for _ in range(len(models))]
    for n, model_info in enumerate(models):
        model_name = model_info[0]
        model = model_info[1]
        if model_name == 'MySATRN':
            input_images = d[0]["image"].to(device).float()
            expected = d[0]["truth"]["encoded"].to(device)
        elif model_name == 'SWIN':
            input_images = d[1]["image"].to(device).float()
            expected = d[1]["truth"]["encoded"].to(device)    
        elif model_name == 'ASTER':
            input_images = d[2]["image"].to(device).float()
            expected = d[2]["truth"]["encoded"].to(device) 
        else:
            raise NotImplementedError        
        encoder_value = model(input_images, expected, False, 0.0)
        encoder_values[n].append(encoder_value)
    return encoder_values


def main(parser):
    torch.manual_seed(21)
    random.seed(21)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence
        
    token_to_id_ = load_checkpoint(parser.checkpoint[0], cuda=is_cuda)['token_to_id']
    id_to_token_ = load_checkpoint(parser.checkpoint[0], cuda=is_cuda)['id_to_token']

    root = os.path.join(os.path.dirname(parser.file_path), "images")

    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    # df = pd.read_csv(os.path.join(os.path.dirname(parser.file_path), 'data_info.txt'))
    # test_image_names = set(df[df['fold']==4]['image_name'].values)
    # with open(os.path.join(os.path.dirname(parser.file_path), 'gt.txt'), "r") as fd:
    #     data=[]
    #     for line in fd:
    #         data.append(line.strip().split("\t"))
    #     dataset_len = round(len(data) * 1.)
    #     data = data[:dataset_len]
    # test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data if x[0] in test_image_names]
    
    img_size_list = [(256,512)] # ,(384,384),(256,256) satrn, swin, aster
    test_datasets = []
    for i in range(len(img_size_list)):
        test_datasets.append(LoadEvalDataset(
            test_data, token_to_id_, id_to_token_, crop=False, transform=get_test_transform(*img_size_list[i]),
            rgb=3
    ))
        
    data_loaders = []
    for j in range(len(img_size_list)):
        data_loaders.append(DataLoader(
            test_datasets[j],
            batch_size=parser.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_eval_batch,
    ))

    SATRN_en_models = []
    
    for parser_checkpoint in parser.checkpoint:
        checkpoint = load_checkpoint(parser_checkpoint, cuda=is_cuda)
        enc = OrderedDict()
        for (key, value) in checkpoint['model'].items():
            if key.startswith('encoder'):
                enc[key] = value
        options = Flags(checkpoint["configs"]).get()
        model_en = get_network('MySATRN_encoder', options, enc, device, test_datasets[0])
        model_en.eval()
        SATRN_en_models.append((options.network, model_en))

    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))


    print('Start Encoding')
    results_en = [] # img, (predict0, predict1, ... , expected)
    with torch.no_grad():
        for d in tqdm(zip(*data_loaders)):
            # input = d["image"].to(device).float() # b, 3, w, h
            expected = d[0]["truth"]["encoded"].to(device) # b, 232
            encoder_values = make_encoder_values(SATRN_en_models, d, device) # list

            results_en.append((d[0]['file_path'], encoder_values))

    gc.collect()
    torch.cuda.empty_cache()

    SATRN_de_models = []

    for parser_checkpoint in parser.checkpoint:
        checkpoint = load_checkpoint(parser_checkpoint, cuda=is_cuda)
        dec = OrderedDict()
        for (key, value) in checkpoint['model'].items():
            if key.startswith('decoder'):
                dec[key] = value
        options = Flags(checkpoint["configs"]).get()
        model_de = get_network('MySATRN_decoder', options, dec, device, test_datasets[0])
        model_de.eval()
        SATRN_de_models.append(model_de)


    print('Start Decoding')
    results_de = []
    with torch.no_grad():
        for result_en in tqdm(results_en):
            path = result_en[0]
            predicteds = result_en[1]

            out = []
            num_steps = parser.max_sequence + 1
            target = torch.LongTensor((predicteds[0][0].size(0))).fill_(model_de.decoder.st_id).to(device)
            for t in range(num_steps):
                one_step_out = None
                for m, model_de in enumerate(SATRN_de_models):
                    input = predicteds[m][0].to(device)
                    _out = model_de.step_forward(
                        input, expected, target
                        )
                    if one_step_out == None:
                        one_step_out = F.softmax(_out, dim=-1)
                    else:
                        one_step_out += F.softmax(_out, dim=-1)
                one_step_out = one_step_out/len(SATRN_de_models)

                target = torch.argmax(one_step_out[:, -1:, :], dim=-1)
                target = target.squeeze()
                out.append(one_step_out)
                       
            out = torch.stack(out, dim=1).to(device)    # [b, max length, 1, class length]
            decoded_values = out.squeeze(2)
            decoded_values = decoded_values.transpose(1, 2)

            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)
            sequence_str = id_to_string(sequence, data_loaders[0], do_eval=1)
            for path, predicted in zip(path, sequence_str):
                results_de.append((path, predicted))

        os.makedirs(parser.output_dir +'_de', exist_ok=True)
        with open(os.path.join(parser.output_dir+'_de', "output.csv"), "w") as w:
            for path, predicted in results_de:
                w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=[
            # "/content/drive/MyDrive/Colab Notebooks/OCR/p4-fr-sorry-math-but-love-you/pth/Copy of SATRN_fold2.pth",
            "/content/drive/MyDrive/Colab Notebooks/OCR/p4-fr-sorry-math-but-love-you/pth/Copy of MySATRN_fold3_7945.pth"
            # "/content/drive/MyDrive/Colab Notebooks/OCR/p4-fr-sorry-math-but-love-you/pth/Copy of aster-fold-0-0.7878.pth"
            ],
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
        default=32,
        type=int,
        help="batch size when doing inference",
    )
    parser.add_argument(
        "--decode_type",
        dest="decode_type",
        default='greedy', 
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

    eval_dir = os.environ.get('SM_CHANNEL_EVAL', '../input/data/')
    file_path = os.path.join(eval_dir, 'eval_dataset/input.txt')
    # file_path = os.path.join(eval_dir, 'train_dataset/input.txt')
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
