import os
import csv
import random
import numpy as np
import pandas as pd
from typing import *
import pandas as pd
from PIL import Image, ImageOps
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, DataLoader


START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):

    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if '' in truth_tokens: truth_tokens.remove('')
    return truth_tokens


def load_vocab(tokens_paths: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Generation 과정에서 활용할 토큰을 불러와 vocab에 추가하는 함수
    Args:
        tokens_paths (str): 토큰 정보가 담긴 파일 경로(tokens.txt)
    Returns:
        token_to_id: {토큰명:ID} 꼴 딕셔너리
        id_to_token: {ID:토큰명} 꼴 딕셔너리
    """
    tokens = []
    tokens.extend(SPECIAL_TOKENS)
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token


def split_gt(groundtruth: str, proportion: float=1.0, test_percent=None) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Ground Truth 이미지 디렉토리로부터 일부만을 불러온 뒤, split하는 함수

    Args:
        groundtruth (str): GT 디렉토리 경로
        proportion (float, optional): 디렉토리로부터 불러올 데이터 비율. Defaults to 1.0.
        test_percent ([type], optional):
            - 불러온 데이터를 학습/검증 데이터로 split할 비율
            - 0.3으로 설정 시 30%를 테스트 데이터, 70%를 학습 데이터로 사용
            - Defaults to None.

    Returns:
        (1) split할 경우(test_percent != None): (학습용 이미지 경로, GT) 리스트, (검증용 이미지 경로, GT) 리스트
        (2) split하지 않을 경우(test_percent == None): (학습용 이미지 경로, GT) 리스트
    """
    # root = os.path.join(os.path.dirname(groundtruth), "images")
    # with open(groundtruth, "r") as fd:
    #     data=[]
    #     for line in fd:
    #         data.append(line.strip().split("\t"))
    #     random.shuffle(data)
    #     dataset_len = round(len(data) * proportion)
    #     data = data[:dataset_len]
    #     data = [[os.path.join(root, x[0]), x[1]] for x in data]
    
    # if test_percent:
    #     test_len = round(len(data) * test_percent)
    #     return data[test_len:], data[:test_len]
    # else:
    #     return data

    # Author: Junchul Choi
    root = os.path.join(os.path.dirname(groundtruth), "images")
    print(root)
    print(os.path.dirname(groundtruth))
    df = pd.read_csv(os.path.join(os.path.dirname(groundtruth), 'data_info_2.txt'))
    val_image_names = set(df[df['fold']==2]['image_name'].values)
    train_image_names = set(df[df['fold']!=2]['image_name'].values)
    ####----------------------
    with open(groundtruth, "r") as fd:
        data=[]
        for line in fd:
            data.append(line.strip().split("\t"))
        random.shuffle(data)
        dataset_len = round(len(data) * proportion)
        data = data[:dataset_len]
        train_data = [[os.path.join(root, x[0]), x[1]] for x in data if x[0] in train_image_names]
        val_data = [[os.path.join(root, x[0]), x[1]] for x in data if x[0] in val_image_names]
    return train_data, val_data



def collate_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

def collate_eval_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "file_path":[d["file_path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
    }

class LoadDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        crop=False,
        preprocessing=True,
        transform=None,
        rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadDataset, self).__init__()
        self.crop = crop
        self.preprocessing = preprocessing
        self.transform = transform
        self.rgb = rgb
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.data = [
            {
                "path": p,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
            # image = cv2.cvtColor(cv2.imread(item["path"]), cv2.COLOR_BGR2RGB)
        elif self.rgb == 1:
            image = image.convert("L")
            # image = cv2.imread(item["path"], 2)
        else:
            raise NotImplementedError

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            w, h = image.size
            if h / w > 2:
                image = image.rotate(90, expand=True)
            image = np.array(image)
            image = self.transform(image=image)['image']

        return {"path": item["path"], "truth": item["truth"], "image": image}

class LoadEvalDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        token_to_id,
        id_to_token,
        crop=False,
        preprocessing=True,
        transform=None,
        rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadEvalDataset, self).__init__()
        self.crop = crop
        self.rgb = rgb
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.preprocessing = preprocessing
        self.transform = transform
        self.data = [
            {
                "path": p,
                "file_path":p1,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, p1,truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
            # image = cv2.cvtColor(cv2.imread(item["path"]), cv2.COLOR_BGR2RGB)
        elif self.rgb == 1:
            image = image.convert("L")
            # image = cv2.imread(item["path"], 2)
        else:
            raise NotImplementedError

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)
                
        if self.transform:
            w, h = image.size
            if h / w > 2:
                image = image.rotate(90, expand=True)
            image = np.array(image)
            image = self.transform(image=image)['image']

        return {"path": item["path"], "file_path":item["file_path"],"truth": item["truth"], "image": image}

# def dataset_loader(options, transformed):
def dataset_loader(options, train_transform, valid_transform):

    # Read data
    train_data, valid_data = [], [] 
    if options.data.random_split:
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train, valid = split_gt(path, prop, options.data.test_proportions)
            train_data += train
            valid_data += valid
    else:
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train_data += split_gt(path, prop)
        for i, path in enumerate(options.data.test):
            valid = split_gt(path)
            valid_data += valid

    # Load data
    train_dataset = LoadDataset(
        # train_data, options.data.token_paths, crop=options.data.crop, transform=transformed, rgb=options.data.rgb
        train_data, options.data.token_paths, crop=options.data.crop, transform=train_transform, rgb=options.data.rgb
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True
    )

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
        pin_memory=True,
        drop_last=True
    )

    return train_data_loader, valid_data_loader, train_dataset, valid_dataset
