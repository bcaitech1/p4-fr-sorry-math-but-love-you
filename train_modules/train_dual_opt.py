import os
import argparse
import random
import time
from tqdm import tqdm
import yaml
import shutil
import numpy as np
import torch
from torch import nn, optim
from transformers import get_constant_schedule_with_warmup
import wandb
from utils import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    write_wandb,
    set_seed,
    print_system_envs,
    get_optimizer,
    get_network,
    id_to_string,
    Flags,
    load_vocab
)
from data import get_train_transforms, get_valid_transforms, dataset_loader, START, PAD  # NOTE
from schedulers import (
    CircularLRBeta,
    CustomCosineAnnealingWarmUpRestarts,
    TeacherForcingScheduler,
)
from utils.metrics import word_error_rate, sentence_acc, final_metric

os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "all"


def _train_one_epoch(
    data_loader,
    model,
    epoch_text,
    criterion,
    enc_optimizer,
    dec_optimizer,
    enc_lr_scheduler,
    dec_lr_scheduler,
    max_grad_norm,
    device,
    tf_scheduler,  # Teacher Forcing Scheduler
    is_logging: bool
):
    torch.set_grad_enabled(True)
    model.train()

    losses = []
    grad_norms = []
    correct_symbols = 0
    total_symbols = 0
    wer = 0
    num_wer = 0
    sent_acc = 0
    num_sent_acc = 0

    with tqdm(
        desc=f"{epoch_text} Train",
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for d in data_loader:
            input = d["image"].to(device).float()
            tf_ratio = tf_scheduler.step()

            curr_batch_size = len(input)
            expected = d["truth"]["encoded"].to(device)
            expected[expected == -1] = data_loader.dataset.token_to_id[PAD]

            output = model(input, expected, True, tf_ratio)  # [B, MAX_LEN, VOCAB_SIZE]

            decoded_values = output.transpose(1, 2)  # [B, VOCAB_SIZE, MAX_LEN]
            _, sequence = torch.topk(decoded_values, k=1, dim=1)  # [B, 1, MAX_LEN]
            sequence = sequence.squeeze(1)  # [B, MAX_LEN], Metric 측정을 위해

            loss = criterion(decoded_values, expected[:, 1:])  # [SOS] 이후부터

            enc_optim_params = [
                p
                for param_group in enc_optimizer.param_groups
                for p in param_group["params"]
            ]
            dec_optim_params = [
                p
                for param_group in dec_optimizer.param_groups
                for p in param_group["params"]
            ]

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            loss.backward()

            enc_grad_norm = nn.utils.clip_grad_norm_(
                enc_optim_params, max_norm=max_grad_norm
            )
            dec_grad_norm = nn.utils.clip_grad_norm_(
                dec_optim_params, max_norm=max_grad_norm
            )

            grad_norms.append(enc_grad_norm)
            grad_norms.append(dec_grad_norm)

            enc_optimizer.step()
            dec_optimizer.step()

            losses.append(loss.item())

            expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
            expected_str = id_to_string(expected, data_loader, do_eval=1)
            sequence_str = id_to_string(sequence, data_loader, do_eval=1)
            wer += word_error_rate(sequence_str, expected_str)
            num_wer += 1
            sent_acc += sentence_acc(sequence_str, expected_str)
            num_sent_acc += 1
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)
            enc_lr_scheduler.step()
            dec_lr_scheduler.step()

            # Weight & Bias logging
            if is_logging:
                wandb.log(
                    {
                        "enc_lr": enc_lr_scheduler.get_lr()[0],
                        "dec_lr": dec_lr_scheduler.get_last_lr()[0],
                        "tf_ratio": tf_ratio,
                    }
                )

    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
        "wer": wer,
        "num_wer": num_wer,
        "sent_acc": sent_acc,
        "num_sent_acc": num_sent_acc,
    }

    try:
        result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
    except:
        result["grad_norm"] = np.mean(grad_norms)

    return result


def _valid_one_epoch(data_loader, model, epoch_text, criterion, device):
    model.eval()

    losses = []
    correct_symbols = 0
    total_symbols = 0
    wer = 0
    num_wer = 0
    sent_acc = 0
    num_sent_acc = 0

    with torch.no_grad():
        with tqdm(
            desc=f"{epoch_text} Validation",
            total=len(data_loader.dataset),
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for d in data_loader:
                input = d["image"].to(device).float()

                curr_batch_size = len(input)
                expected = d["truth"]["encoded"].to(device)

                expected[expected == -1] = data_loader.dataset.token_to_id[PAD]
                output = model(input, expected, False, 0.0)

                decoded_values = output.transpose(1, 2)  # [B, VOCAB_SIZE, MAX_LEN]
                _, sequence = torch.topk(
                    decoded_values, 1, dim=1
                )  # sequence: [B, 1, MAX_LEN]
                sequence = sequence.squeeze(1)  # [B, MAX_LEN], 각 샘플에 대해 시퀀스가 생성 상태

                loss = criterion(decoded_values, expected[:, 1:])

                losses.append(loss.item())

                expected[expected == data_loader.dataset.token_to_id[PAD]] = -1
                expected_str = id_to_string(expected, data_loader, do_eval=1)
                sequence_str = id_to_string(sequence, data_loader, do_eval=1)
                wer += word_error_rate(sequence_str, expected_str)
                num_wer += 1
                sent_acc += sentence_acc(sequence_str, expected_str)
                num_sent_acc += 1
                correct_symbols += torch.sum(
                    sequence == expected[:, 1:], dim=(0, 1)
                ).item()
                total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()
                pbar.update(curr_batch_size)

    expected = id_to_string(expected, data_loader)
    sequence = id_to_string(sequence, data_loader)

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
        "wer": wer,
        "num_wer": num_wer,
        "sent_acc": sent_acc,
        "num_sent_acc": num_sent_acc,
    }
    return result


def main(parser):
    config_file = parser.config_file
    options = Flags(config_file).get()
    is_logging = True if parser.project_name is not None else False

    # set random seed
    set_seed(seed=options.seed)

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))
    print_system_envs()  # Print system environments

    # Load checkpoint and print result
    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=is_cuda)
        if options.checkpoint != ""
        else default_checkpoint
    )

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
            "Train Symbol Accuracy : {:.5f}\n".format(
                checkpoint["train_symbol_accuracy"][-1]
            ),
            "Train Sentence Accuracy : {:.5f}\n".format(
                checkpoint["train_sentence_accuracy"][-1]
            ),
            "Train WER : {:.5f}\n".format(checkpoint["train_wer"][-1]),
            "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
            "Validation Symbol Accuracy : {:.5f}\n".format(
                checkpoint["validation_symbol_accuracy"][-1]
            ),
            "Validation Sentence Accuracy : {:.5f}\n".format(
                checkpoint["validation_sentence_accuracy"][-1]
            ),
            "Validation WER : {:.5f}\n".format(checkpoint["validation_wer"][-1]),
            "Validation Loss : {:.5f}\n".format(checkpoint["validation_losses"][-1]),
        )

    (
        train_data_loader,
        validation_data_loader,
        train_dataset,
        valid_dataset,
    ) = dataset_loader(
        options,
        train_transform=get_train_transforms(
            height=options.input_size.height, width=options.input_size.width
        ),
        valid_transform=get_valid_transforms(
            height=options.input_size.height, width=options.input_size.width
        ),
        fold=options.data.fold,
    )

    print(
        "[+] Data\n",
        "The number of train samples : {}\n".format(len(train_dataset)),
        "The number of validation samples : {}\n".format(len(valid_dataset)),
        "The number of classes : {}\n".format(len(train_dataset.token_to_id)),
    )

    # TODO. DecodingManager
    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        train_dataset,
    )
    model.train()

    # define loss
    criterion = model.criterion.to(device)

    # define optimizer

    enc_params_to_optimise = [
        param for param in model.encoder.parameters() if param.requires_grad
    ]
    dec_params_to_optimise = [
        param for param in model.decoder.parameters() if param.requires_grad
    ]
    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Encoder parameters: {}\n".format(
            sum(p.numel() for p in enc_params_to_optimise),
        ),
        "Decoder parameters: {} \n".format(
            sum(p.numel() for p in dec_params_to_optimise),
        ),
    )
    total_steps = len(train_data_loader) * options.num_epochs  # 전체 스텝 수
    t_0 = total_steps // options.scheduler.cycle  # 주기를 3으로 설정
    enc_t_up = int(t_0 * options.scheduler.warmup_ratio)  # 한 주기에서 10%의 스텝을 warm-up으로 사용
    dec_t_up = int(t_0 * options.scheduler.warmup_ratio)

    enc_optimizer = optim.Adam(enc_params_to_optimise, lr=0)
    dec_optimizer = optim.Adam(dec_params_to_optimise, lr=5e-4)

    enc_optimizer_state = checkpoint.get("enc_optimizer", None)
    if enc_optimizer_state is not None:
        enc_optimizer.load_state_dict(enc_optimizer_state)

    dec_optimizer_state = checkpoint.get("dec_optimizer", None)
    if dec_optimizer_state is not None:
        dec_optimizer.load_state_dict(dec_optimizer_state)

    # Custom Cosine Annealing 파라미터 명세 볼 만한 곳: https://bit.ly/2SGDhxO
    # T_0: 한 주기에 대한 스텝 수
    # T_mult: 주기 반복마다 주기 길이를 T_mult배로 바꿈
    # eta_max: warm-up을 통해 도달할 최대 LR
    # T_up: 한 주기 내에서 warm-up을 할 스텝 수
    # gamma: 주기 반복마다 주기 진폭을 gamma배로 바꿈

    enc_lr_scheduler = CustomCosineAnnealingWarmUpRestarts(
        enc_optimizer,
        T_0=t_0,
        T_mult=1,
        eta_max=options.optimizer.enc_lr,
        T_up=enc_t_up,
        gamma=0.8,
    )

    dec_lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=dec_optimizer, num_warmup_steps=dec_t_up
    )

    enc_lr_scheduler_state = checkpoint.get("enc_scheduler", None)
    if enc_lr_scheduler_state is not None:
        enc_lr_scheduler.load_state_dict(enc_lr_scheduler_state)

    dec_lr_scheduler_state = checkpoint.get("dec_scheduler", None)
    if dec_lr_scheduler_state is not None:
        dec_lr_scheduler.load_state_dict(dec_lr_scheduler_state)

    # Define Teacher Forcing Scheduler
    tf_scheduler = TeacherForcingScheduler(
        num_steps=total_steps,
        tf_max=options.teacher_forcing_ratio.tf_max,
        tf_min=options.teacher_forcing_ratio.tf_min,
    )
    print(
        "[+] Teacher Forcing\n",
        "Type: Arctan\n",
        f"Steps: {total_steps}\n" f"TF-MAX: {options.teacher_forcing_ratio.tf_max}\n",
        f"TF-MIN: {options.teacher_forcing_ratio.tf_min}\n",
    )

    # Log for W&B
    if is_logging:
        wandb.config.update(dict(options._asdict()))  # logging to W&B

    # Log for tensorboard
    if not os.path.exists(options.prefix):
        os.makedirs(options.prefix)
    log_file = open(os.path.join(options.prefix, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(options.prefix, "train_config.yaml"))
    if options.print_epochs is None:
        options.print_epochs = options.num_epochs
    start_epoch = checkpoint["epoch"]
    train_symbol_accuracy = checkpoint["train_symbol_accuracy"]
    train_sentence_accuracy = checkpoint["train_sentence_accuracy"]
    train_wer = checkpoint["train_wer"]
    train_losses = checkpoint["train_losses"]
    validation_symbol_accuracy = checkpoint["validation_symbol_accuracy"]
    validation_sentence_accuracy = checkpoint["validation_sentence_accuracy"]
    validation_wer = checkpoint["validation_wer"]
    validation_losses = checkpoint["validation_losses"]
    learning_rates = checkpoint["lr"]
    grad_norms = checkpoint["grad_norm"]

    best_score = 0.0

    # Train
    for epoch in range(options.num_epochs):
        start_time = time.time()

        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=options.num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(options.num_epochs)),
        )

        train_result = _train_one_epoch(
            data_loader=train_data_loader,
            model=model,
            epoch_text=epoch_text,
            criterion=criterion,
            enc_optimizer=enc_optimizer,
            dec_optimizer=dec_optimizer,
            enc_lr_scheduler=enc_lr_scheduler,
            dec_lr_scheduler=dec_lr_scheduler,
            max_grad_norm=options.max_grad_norm,
            device=device,
            tf_scheduler=tf_scheduler,
            is_logging=is_logging
        )

        train_losses.append(train_result["loss"])
        grad_norms.append(train_result["grad_norm"])
        train_epoch_symbol_accuracy = (
            train_result["correct_symbols"] / train_result["total_symbols"]
        )
        train_symbol_accuracy.append(train_epoch_symbol_accuracy)
        train_epoch_sentence_accuracy = (
            train_result["sent_acc"] / train_result["num_sent_acc"]
        )

        train_sentence_accuracy.append(train_epoch_sentence_accuracy)
        train_epoch_wer = train_result["wer"] / train_result["num_wer"]
        train_wer.append(train_epoch_wer)
        train_epoch_score = final_metric(
            sentence_acc=train_epoch_sentence_accuracy, word_error_rate=train_epoch_wer
        )
        epoch_lr = enc_lr_scheduler.get_lr()[0]  # cycle

        validation_result = _valid_one_epoch(
            validation_data_loader,
            model,
            epoch_text,
            criterion,
            device,
        )

        validation_losses.append(validation_result["loss"])
        validation_epoch_symbol_accuracy = (
            validation_result["correct_symbols"] / validation_result["total_symbols"]
        )
        validation_symbol_accuracy.append(validation_epoch_symbol_accuracy)

        validation_epoch_sentence_accuracy = (
            validation_result["sent_acc"] / validation_result["num_sent_acc"]
        )
        validation_sentence_accuracy.append(validation_epoch_sentence_accuracy)
        validation_epoch_wer = validation_result["wer"] / validation_result["num_wer"]
        validation_wer.append(validation_epoch_wer)
        validation_epoch_score = final_metric(
            sentence_acc=validation_epoch_sentence_accuracy,
            word_error_rate=validation_epoch_wer,
        )

        # Save checkpoint
        # make config
        with open(config_file, "r") as f:
            option_dict = yaml.safe_load(f)

        if best_score < 0.9 * validation_epoch_sentence_accuracy + 0.1 * (
            1 - validation_epoch_wer
        ):
            save_checkpoint(
                {
                    "epoch": start_epoch + epoch + 1,
                    "train_losses": train_losses,
                    "train_symbol_accuracy": train_symbol_accuracy,
                    "train_sentence_accuracy": train_sentence_accuracy,
                    "train_wer": train_wer,
                    "validation_losses": validation_losses,
                    "validation_symbol_accuracy": validation_symbol_accuracy,
                    "validation_sentence_accuracy": validation_sentence_accuracy,
                    "validation_wer": validation_wer,
                    "lr": epoch_lr,
                    "grad_norm": grad_norms,
                    "model": model.state_dict(),
                    "enc_optimizer": enc_optimizer.state_dict(),
                    "dec_optimizer": dec_optimizer.state_dict(),
                    "configs": option_dict,
                    "token_to_id": train_data_loader.dataset.token_to_id,
                    "id_to_token": train_data_loader.dataset.id_to_token,
                    "network": options.network,
                    "enc_scheduler": enc_lr_scheduler.state_dict(),
                    "dec_scheduler": dec_lr_scheduler.state_dict(),
                },
                prefix=options.prefix,
            )
            best_score = 0.9 * validation_epoch_sentence_accuracy + 0.1 * (
                1 - validation_epoch_wer
            )
            print(f"best score: {best_score}")
            print("model is saved")

        # Summary
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % options.print_epochs == 0 or epoch == options.num_epochs - 1:
            output_string = (
                "{epoch_text}: "
                "Train Symbol Accuracy = {train_symbol_accuracy:.5f}, "
                "Train Sentence Accuracy = {train_sentence_accuracy:.5f}, "
                "Train WER = {train_wer:.5f}, "
                "Train Loss = {train_loss:.5f}, "
                "Validation Symbol Accuracy = {validation_symbol_accuracy:.5f}, "
                "Validation Sentence Accuracy = {validation_sentence_accuracy:.5f}, "
                "Validation WER = {validation_wer:.5f}, "
                "Validation Loss = {validation_loss:.5f}, "
                "lr = {lr} "
                "(time elapsed {time})"
            ).format(
                epoch_text=epoch_text,
                train_symbol_accuracy=train_epoch_symbol_accuracy,
                train_sentence_accuracy=train_epoch_sentence_accuracy,
                train_wer=train_epoch_wer,
                train_loss=train_result["loss"],
                validation_symbol_accuracy=validation_epoch_symbol_accuracy,
                validation_sentence_accuracy=validation_epoch_sentence_accuracy,
                validation_wer=validation_epoch_wer,
                validation_loss=validation_result["loss"],
                lr=epoch_lr,
                time=elapsed_time,
            )
            print(output_string)
            log_file.write(output_string + "\n")

            if is_logging:
                write_wandb(
                    epoch=start_epoch + epoch + 1,
                    grad_norm=train_result["grad_norm"],
                    train_loss=train_result["loss"],
                    train_symbol_accuracy=train_epoch_symbol_accuracy,
                    train_sentence_accuracy=train_epoch_sentence_accuracy,
                    train_wer=train_epoch_wer,
                    train_score=train_epoch_score,
                    validation_loss=validation_result["loss"],
                    validation_symbol_accuracy=validation_epoch_symbol_accuracy,
                    validation_sentence_accuracy=validation_epoch_sentence_accuracy,
                    validation_wer=validation_epoch_wer,
                    validation_score=validation_epoch_score,
                )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--project_name", default="REFACTORING-TEST", help="W&B에 표시될 프로젝트명. 모델명으로 통일!"
#     )
#     parser.add_argument(
#         "--exp_name",
#         default="train_dual_opt.py(SATRN)",
#         help="실험명(SATRN-베이스라인, SARTN-Loss변경 등)",
#     )
#     parser.add_argument(
#         "-c",
#         "--config_file",
#         dest="config_file",
#         default="./configs/EfficientSATRN.yaml",
#         type=str,
#         help="Path of configuration file",
#     )
#     parser = parser.parse_args()

#     # initilaize W&B
#     run = wandb.init(project=parser.project_name, name=parser.exp_name)

#     # train
#     main(parser.config_file)

#     # fishe W&B
#     run.finish()
