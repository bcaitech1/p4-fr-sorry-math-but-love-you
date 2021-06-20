from torch.utils.data import Dataset
from typing import Tuple, Dict, List
from PIL import Image, ImageOps
from glob import glob
import torch
import os
import numpy as np
import sys

from utils import encode_truth, load_vocab

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


class LoadDataset(Dataset):
    def __init__(
        self,
        groundtruth: str,
        tokens_file: str,
        crop: bool = False,
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
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            w, h = image.size
            if h / w > 2:
                image = image.rotate(90, expand=True)
            image = np.array(image)
            image = self.transform(image=image)["image"]

        return {"path": item["path"], "truth": item["truth"], "image": image}


class LoadEvalDataset(Dataset):
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
                "file_path": p1,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, p1, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        if self.rgb == 3:
            image = image.convert("RGB")
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            w, h = image.size
            if h / w > 2:
                image = image.rotate(90, expand=True)
            image = np.array(image)
            image = self.transform(image=image)["image"]

        return {
            "path": item["path"],
            "file_path": item["file_path"],
            "truth": item["truth"],
            "image": image,
        }

class DistillationDataset(Dataset):
    def __init__(
        self,
        groundtruth: str,
        tokens_file: str,
        crop: bool = False,
        preprocessing=True,
        student_transform=None,
        teacher_transform=None,
        rgb=3,
    ):
        super(DistillationDataset, self).__init__()
        self.crop = crop
        self.preprocessing = preprocessing
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
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
        elif self.rgb == 1:
            image = image.convert("L")
        else:
            raise NotImplementedError

        if self.crop:
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)
        
        w, h = image.size
        if h / w > 2:
            image = image.rotate(90, expand=True)
        image = np.array(image)
        
        student_image = self.student_transform(image=image)['image']
        teacher_image = self.teacher_transform(image=image)['image']

        return {"path": item["path"], "truth": item["truth"], "student_image": student_image, 'teacher_image': teacher_image}


class DecoderDataset(Dataset):
    """앙상블 과정 중 디코딩에 활용되는 디코더 데이터셋
    인코더를 거쳐 임시폴더에 저장된 텐서를 불러옴
    """

    def __init__(self, tmp_dir: str):
        self.paths = sorted(glob(os.path.join(tmp_dir, "*")))

    def __getitem__(self, idx):
        output = torch.load(self.paths[idx], map_location="cpu")
        return output

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def collate_fn(batch):
        """디코더에 입력할 데이터를 iterating하는 과정에서 사용될 collate function."""
        paths_aggregated = []
        batch_aggregated = None

        for idx, (paths, batch_each_model) in enumerate(batch):
            if idx == 0:
                num_models = len(batch_each_model)
                batch_aggregated = [[] for _ in range(num_models)]

            paths_aggregated.extend(paths)
            for m, batch in enumerate(batch_each_model):
                batch_aggregated[m].append(batch)

        # 모델별 텐서를 병합
        for i in range(len(batch_aggregated)):
            batch_aggregated[i] = torch.vstack(batch_aggregated[i])

        return paths_aggregated, batch_aggregated
