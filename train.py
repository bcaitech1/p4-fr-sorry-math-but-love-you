import argparse
from utils import set_seed
from importlib import import_module
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        default='SATRN',
        help="W&B에 표시될 프로젝트명. 모델명으로 통일!"
    )
    parser.add_argument(
        "--exp_name",
        default="TF(8>3&(-5,5)) & SSR(.3)>GD(.3)>Norm & Dual Opt & Dec(256_1024) & Fold0",
        help="실험명(SATRN-베이스라인, SARTN-Loss변경 등)",
    )
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        default="./configs/My_SATRN.yaml",
        type=str,
        help="Path of configuration file",
    )
    parser.add_argument(
        "-c",
        "--train_type",
        default="default",
        type=str,
        help="학습 종류 설정. 'dual_opt(인코더/디코더 개별 lr 부여)', 'single_opt'(인코더/디코더 통합 lr 부여)"
    ) # NOTE. single_opt 설정시 yaml 파일 내 기록된 learning rate를 모델에 적용
    parser.add_argument(
        '--enc_lr',
        default=5e-4,
        type=float,
        help="인코더에 부여할 lr. dual_opt 학습 시 작동"
    )
    parser.add_argument(
        '--dec_lr',
        default=5e-4,
        type=float,
        help="인코더에 부여할 lr. dual_opt 학습 시 작동"
    )
    args = parser.parse_args()

    # wandb init
    wandb.init(project=args.project_name)
    wandb.run.name = f"{args.exp_name}"
    wandb.config.update(args)

    # random seed 고정
    set_seed(args.seed)
    
    train_module = getattr(
        import_module(f"train_{args.dst}"), "train"
    )

    print('='*100)
    print(args)
    print('='*100)

    train_module(args)