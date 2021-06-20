import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import sys

from utils import split_gt
from .dataset import LoadDataset, LoadEvalDataset, DistillationDataset


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
            "encoded": torch.tensor(padded_encoded),
        },
    }

def collate_distillation_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]

    return {
        "path": [d["path"] for d in data],
        "student_image": torch.stack([d["student_image"] for d in data], dim=0),
        "teacher_image": torch.stack([d["teacher_image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded),
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
        "file_path": [d["file_path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded),
        },
    }


def dataset_loader(
    options, train_transform, valid_transform, fold
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:

    # Read data
    train_data, valid_data = [], []

    for i, path in enumerate(options.data.train):
        train, valid = split_gt(path, fold)
        train_data += train
        valid_data += valid

    # Load data
    train_dataset = LoadDataset(
        train_data,
        options.data.token_paths,
        crop=options.data.crop,
        transform=train_transform,
        rgb=options.data.rgb,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
        drop_last=True,
        pin_memory=True,
    )
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
        drop_last=True,
        pin_memory=True,
    )

    return train_data_loader, valid_data_loader, train_dataset, valid_dataset


def compose_test_dataloader(
    test_data, batch_size, token_to_id, id_to_token, num_workers, transforms
):
    dataset = LoadEvalDataset(
        test_data, token_to_id, id_to_token, crop=False, transform=transforms, rgb=3
    )
    test_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_eval_batch,
    )
    return test_dataloader


def get_distillation_dataloaders(
    student_options,
    teacher_options,
    student_transform,
    teacher_transform,
    valid_transform,
    fold: int,
) -> Tuple[DataLoader, DataLoader]:

    # Read data
    train_data, valid_data = [], []

    for path in student_options.data.train:
        train, valid = split_gt(path, fold)
        train_data += train
        valid_data += valid

    # Load data
    distillation_dataset = DistillationDataset(
        train_data,
        student_options.data.token_paths,
        crop=student_options.data.crop,
        student_transform=student_transform,  # NOTE
        teacher_transform=teacher_transform,
        rgb=student_options.data.rgb,
    )
    valid_dataset = LoadDataset(
        valid_data,
        student_options.data.token_paths,
        crop=student_options.data.crop,
        transform=valid_transform,
        rgb=student_options.data.rgb,
    )
    distillation_dataloader = DataLoader(
        distillation_dataset,
        batch_size=student_options.batch_size,
        shuffle=True,
        num_workers=student_options.num_workers,
        collate_fn=collate_distillation_batch,
        drop_last=True,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=student_options.num_workers,
        collate_fn=collate_batch,
        drop_last=True,
        pin_memory=True,
    )

    return distillation_dataloader, valid_dataloader
