import os
import pandas as pd
import random
from typing import Tuple, Dict, List

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


def encode_truth(truth: str, token_to_id: dict):
    """입력한 수식 문자열을 인코딩하는 함수"""
    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if "" in truth_tokens:
        truth_tokens.remove("")
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


def split_gt(
    groundtruth: str, fold: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
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

    # Author: Junchul Choi
    root = os.path.join(os.path.dirname(groundtruth), "images")
    df = pd.read_csv(os.path.join(os.path.dirname(groundtruth), "data_info.txt"))
    val_image_names = set(df[df["fold"] == fold]["image_name"].values)
    train_image_names = set(df[df["fold"] != fold]["image_name"].values)
    with open(groundtruth, "r") as fd:
        data = []
        for line in fd:
            data.append(line.strip().split("\t"))
        random.shuffle(data)
        dataset_len = len(data)
        data = data[:dataset_len]
        train_data = [
            [os.path.join(root, x[0]), x[1]] for x in data if x[0] in train_image_names
        ]
        val_data = [
            [os.path.join(root, x[0]), x[1]] for x in data if x[0] in val_image_names
        ]
    return train_data, val_data
