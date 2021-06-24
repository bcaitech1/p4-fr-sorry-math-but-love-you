import argparse
import warnings
from importlib import import_module

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference_type",
        default='single',
        help="추론 방식을 설정. 'single(단일 모델 추론)', 'ensemble(다중 모델 추론)'",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=["/opt/ml/ensemble/log/satrn_fold-0-0.8148.pth"],
        nargs="*",
        help="추론에 활용할 학습 모델 파일 경로",
    )
    parser.add_argument(
        "--max_sequence",
        dest="max_sequence",
        default=230,
        type=int,
        help="수식 문장 최대 생성 길이",
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
        default="greedy",
        type=str,
        help="디코딩 방식 설정. 'greedy(그리디 디코딩)', 'beam(빔서치)'. NOTE: 빔서치는 단일 모델 추론(singular)에서만 작동, SwinTRN 사용불가",
    )
    parser.add_argument(
        "--beam_width",
        dest="beam_width",
        default=3,
        type=int,
        help="빔서치 사용 시 Beam Size 설정",
    )
    parser.add_argument(
        "--decoding_manager", default=True, help="DecodingManager 사용 여부 결정"
    )
    parser.add_argument(
        "--tokens_path",
        default="./configs/tokens.txt",
        help="DecodingManager 사용시 활용할 토큰 파일 경로",
    )
    parser.add_argument(
        "--max_cache", type=int, default=50, help="최대 몇 개의 피클 파일을 저장할 지 결정. NOTE: 앙상블 추론(ensemble)에서만 사용됨"
    )
    parser.add_argument(
        "--file_path",
        dest="file_path",
        default='../input/data/eval_dataset/input.txt',
        type=str,
        help="추론 시 활용할 데이터 경로",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default='./result/',
        type=str,
        help="추론 결과를 저장할 디렉토리 경로",
    )

    parser = parser.parse_args()

    # Singular model inference
    if len(parser.checkpoint) == 1:
        del parser.max_cache
        parser.checkpoint = parser.checkpoint[0]

    # Ensemble models inference
    elif len(parser.checkpoint) > 1 and parser.inference_type == 'ensemble':
        if parser.decode_type != 'greedy':
            parser.decode_type = 'greedy'
            warnings.warn("'ensemble' inference just support 'greedy'. Changed decode_type: 'beam' -> 'greedy'")
    
    elif len(parser.checkpoint) > 1 and parser.inference_type == 'single':
        raise ValueError("Cannot run with 'single' inference type since the number of checkpoints is greater than 1")

    else:
        raise NotImplementedError

    # run inference
    print('='*100)
    print(parser)
    print('='*100)
    inference_module = getattr(import_module(f"inference_modules.inference_{parser.inference_type}"), 'main')
    inference_module(parser)
