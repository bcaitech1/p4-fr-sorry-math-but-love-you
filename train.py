import argparse
from importlib import import_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_type',
        default="single_opt",
        help="""
        인코더/디코더 optimizer 부여 방식 설정
        'single_opt' - 모델 전체에 단일 optimizer를 적용하여 학습 진행
        'dual_opt' - 모델의 인코더와 디코더에 optimzer를 개별 적용하여 학습 진행
        각 optimzer의 learning rate는 모델 configuration에 따라 결정됨
        """
    )
    parser.add_argument(
        '--project_name', default="[프로젝트명]", help="Weight & Bias에 표시될 프로젝트명"
    )
    parser.add_argument(
        "--exp_name",
        default="[실험명]",
        help="Weight & Bias에 표시될 실험명",
    )
    parser.add_argument(
        "--config_file",
        default="./configs/EfficientSATRN.yaml",
        type=str,
        help="모델 configuration 파일 경로",
    )
    parser = parser.parse_args()

    # run train
    print('='*100)
    print(parser)
    print('='*100)
    train_module = getattr(import_module(f"train_modules.train_{parser.train_type}"), 'main')
    train_module(parser)
