import os
import argparse
import random
import time
from tqdm import tqdm
import yaml
import shutil
from psutil import virtual_memory
import multiprocessing
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import wandb
from utils import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    write_wandb,
    Flags,
    set_seed,
    print_system_envs,
    get_optimizer,
    get_network,
    id_to_string,
    get_timestamp,
    load_vocab,
)
from data import (
    get_train_transforms,
    get_valid_transforms,
    dataset_loader,
    START,
    PAD,
    get_distillation_dataloaders,
)
from schedulers import (
    CircularLRBeta,
    CustomCosineAnnealingWarmUpRestarts,
    TeacherForcingScheduler,
)
from utils.metrics import word_error_rate, sentence_acc, final_metric

os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "all"


def loss_fn_kd(
    outputs, labels, teacher_outputs, T=10, alpha=0.1
):  # NOTE: popular choice - 0.9*CE + 0.1*KD
    KD_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)
    ) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1.0 - alpha)
    return KD_loss


def _train_one_epoch(
    distillation_loader,
    student_model,
    teacher_model,
    epoch_text,
    optimizer,
    lr_scheduler,
    max_grad_norm,
    device,
    tf_scheduler,
    is_logging: bool,
):
    torch.set_grad_enabled(True)
    student_model.train()

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
        total=len(distillation_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for d in distillation_loader:
            student_input = d["student_image"].to(device).float()
            expected = d["truth"]["encoded"].to(device)
            expected[expected == -1] = distillation_loader.dataset.token_to_id[PAD]
            tf_ratio = tf_scheduler.step()  # Teacher Forcing Scheduler
            curr_batch_size = len(student_input)

            student_output = student_model(
                student_input, expected, True, tf_ratio
            )  # [B, MAX_LEN, VOCAB_SIZE]
            student_decoded_values = student_output.transpose(
                1, 2
            )  # [B, VOCAB_SIZE, MAX_LEN]
            _, sequence = torch.topk(
                student_decoded_values, k=1, dim=1
            )  # [B, 1, MAX_LEN]
            sequence = sequence.squeeze(1)  # [B, MAX_LEN]

            # get teacher output
            with torch.no_grad():
                teacher_input = d["teacher_image"].to(device).float()
                teacher_output = teacher_model(
                    teacher_input, expected, False, 0.0
                )  # [B, MAX_LEN, VOCAB_SIZE]
                teacher_decoded_values = teacher_output.transpose(1, 2)

            loss = loss_fn_kd(
                outputs=student_decoded_values,
                labels=expected[:, 1:],
                teacher_outputs=teacher_decoded_values,
            )
            optim_params = [
                p
                for param_group in optimizer.param_groups
                for p in param_group["params"]
            ]
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(optim_params, max_norm=max_grad_norm)
            grad_norms.append(grad_norm)

            optimizer.step()
            losses.append(loss.item())

            expected[expected == distillation_loader.dataset.token_to_id[PAD]] = -1
            expected_str = id_to_string(expected, distillation_loader, do_eval=1)
            sequence_str = id_to_string(sequence, distillation_loader, do_eval=1)
            wer += word_error_rate(sequence_str, expected_str)
            num_wer += 1
            sent_acc += sentence_acc(sequence_str, expected_str)
            num_sent_acc += 1
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)
            lr_scheduler.step()

            # Weight & Bias
            if is_logging:
                if isinstance(lr_scheduler.get_lr(), float) or isinstance(
                    lr_scheduler.get_lr(), int
                ):
                    wandb.log(
                        {"learning_rate": lr_scheduler.get_lr(), "tf_ratio": tf_ratio}
                    )
                else:
                    wandb.log(
                        {
                            "learning_rate": lr_scheduler.get_lr()[0],
                            "tf_ratio": tf_ratio,
                        }
                    )

    expected = id_to_string(expected, distillation_loader)
    sequence = id_to_string(sequence, distillation_loader)

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
    teacher_ckpt = load_checkpoint(parser.teacher_ckpt)

    student_options = Flags(config_file).get()
    teacher_options = Flags(teacher_ckpt["configs"]).get()

    is_logging = True if parser.project_name is not None else False

    # set random seed
    set_seed(seed=student_options.seed)

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(student_options.network, device))

    # Print system environments
    print_system_envs()

    # Load checkpoint and print result
    student_ckpt = (
        load_checkpoint(student_options.checkpoint, cuda=is_cuda)
        if student_options.checkpoint != ""
        else default_checkpoint
    )

    model_checkpoint = student_ckpt["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(student_ckpt["epoch"]),
            "Train Symbol Accuracy : {:.5f}\n".format(
                student_ckpt["train_symbol_accuracy"][-1]
            ),
            "Train Sentence Accuracy : {:.5f}\n".format(
                student_ckpt["train_sentence_accuracy"][-1]
            ),
            "Train WER : {:.5f}\n".format(student_ckpt["train_wer"][-1]),
            "Train Loss : {:.5f}\n".format(student_ckpt["train_losses"][-1]),
            "Validation Symbol Accuracy : {:.5f}\n".format(
                student_ckpt["validation_symbol_accuracy"][-1]
            ),
            "Validation Sentence Accuracy : {:.5f}\n".format(
                student_ckpt["validation_sentence_accuracy"][-1]
            ),
            "Validation WER : {:.5f}\n".format(student_ckpt["validation_wer"][-1]),
            "Validation Loss : {:.5f}\n".format(student_ckpt["validation_losses"][-1]),
        )

    student_transform = get_train_transforms(
        student_options.input_size.height,
        student_options.input_size.width,
    )
    teacher_transform = get_train_transforms(
        teacher_options.input_size.height,
        teacher_options.input_size.width,
    )
    valid_transform = get_valid_transforms(
        student_options.input_size.height,
        student_options.input_size.width,
    )

    distillation_loader, valid_loader = get_distillation_dataloaders(
        student_options=student_options,
        teacher_options=teacher_options,
        student_transform=student_transform,
        teacher_transform=teacher_transform,
        valid_transform=valid_transform,
        fold=student_options.data.fold,
    )
    print(
        "[+] Data\n",
        "The number of train samples : {}\n".format(len(distillation_loader.dataset)),
        "The number of validation samples : {}\n".format(len(valid_loader.dataset)),
        "The number of classes : {}\n".format(len(distillation_loader.dataset.token_to_id)),
    )

    # define model
    student_model = get_network(
        model_type=student_options.network,
        FLAGS=student_options,
        model_checkpoint=model_checkpoint,
        device=device,
        dataset=distillation_loader.dataset,
    )
    student_model.train()
    criterion = student_model.criterion

    teacher_model = get_network(
        model_type=teacher_options.network,
        FLAGS=teacher_options,
        model_checkpoint=teacher_ckpt["model"],
        device=device,
        dataset=distillation_loader.dataset,
    )
    teacher_model.eval()

    # define optimizer
    enc_params_to_optimise = [
        param for param in student_model.encoder.parameters() if param.requires_grad
    ]
    dec_params_to_optimise = [
        param for param in student_model.decoder.parameters() if param.requires_grad
    ]
    params_to_optimise = [*enc_params_to_optimise, *dec_params_to_optimise]
    print(
        "[+] Network\n",
        "Type: {}\n".format(student_options.network),
        "Encoder parameters: {}\n".format(
            sum(p.numel() for p in enc_params_to_optimise),
        ),
        "Decoder parameters: {} \n".format(
            sum(p.numel() for p in dec_params_to_optimise),
        ),
    )

    # Get optimizer and optimizer
    if student_options.scheduler.scheduler == "CustomCosine":
        optimizer = get_optimizer(
            student_options.optimizer.optimizer,
            params_to_optimise,
            lr=0,
            weight_decay=student_options.optimizer.weight_decay,
        )
        optimizer_state = student_ckpt.get("optimizer")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        # Custom Cosine Annealing 파라미터 명세 볼 만한 곳: https://bit.ly/2SGDhxO
        # T_0: 한 주기에 대한 스텝 수
        # T_mult: 주기 반복마다 주기 길이를 T_mult배로 바꿈
        # eta_max: warm-up을 통해 도달할 최대 LR
        # T_up: 한 주기 내에서 warm-up을 할 스텝 수
        # gamma: 주기 반복마다 주기 진폭을 gamma배로 바꿈

        total_steps = len(distillation_loader) * student_options.num_epochs  # 전체 스텝 수
        t_0 = total_steps // 1  # 주기를 1로 설정
        t_up = int(
            t_0 * student_options.scheduler.warmup_ratio
        )  # 한 주기에서 10%의 스텝을 warm-up으로 사용

        lr_scheduler = CustomCosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=t_0,
            T_mult=1,
            eta_max=student_options.optimizer.lr,
            T_up=t_up,
            gamma=0.8,
        )

        tf_scheduler = TeacherForcingScheduler(
            num_steps=total_steps,
            tf_max=student_options.teacher_forcing_ratio.tf_max,
            tf_min=student_options.teacher_forcing_ratio.tf_min,
        )
        print(
            "[+] Teacher Forcing\n",
            "Type: Arctan\n",
            f"Steps: {total_steps}\n",
            f"TF-MAX: {student_options.teacher_forcing_ratio.tf_max}\n",
            f"TF-MIN: {student_options.teacher_forcing_ratio.tf_min}\n",
        )

    else:
        optimizer = get_optimizer(
            student_options.optimizer.optimizer,
            params_to_optimise,
            lr=student_options.optimizer.lr,
            weight_decay=student_options.optimizer.weight_decay,
        )
        optimizer_state = student_ckpt.get("optimizer")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        if student_options.scheduler.scheduler == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=student_options.schduler.patience
            )
        elif student_options.scheduler.scheduler == "StepLR":
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=student_options.optimizer.lr_epochs,
                gamma=student_options.optimizer.lr_factor,
            )
        elif student_options.scheduler.scheduler == "Cycle":
            for param_group in optimizer.param_groups:
                param_group["initial_lr"] = student_options.optimizer.lr
            cycle = len(distillation_loader) * student_options.num_epochs
            lr_scheduler = CircularLRBeta(
                optimizer, student_options.optimizer.lr, 10, 10, cycle, [0.95, 0.85]
            )
    if student_ckpt["scheduler"]:
        lr_scheduler.load_state_dict(student_ckpt["scheduler"])

    # Log for W&B
    if is_logging:
        wandb.config.update(dict(student_options._asdict()))  # logging to W&B

    if not os.path.exists(student_options.prefix):
        os.makedirs(student_options.prefix)
    log_file = open(os.path.join(student_options.prefix, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(student_options.prefix, "train_config.yaml"))
    if student_options.print_epochs is None:
        student_options.print_epochs = student_options.num_epochs
    start_epoch = student_ckpt["epoch"]
    train_symbol_accuracy = student_ckpt["train_symbol_accuracy"]
    train_sentence_accuracy = student_ckpt["train_sentence_accuracy"]
    train_wer = student_ckpt["train_wer"]
    train_losses = student_ckpt["train_losses"]
    validation_symbol_accuracy = student_ckpt["validation_symbol_accuracy"]
    validation_sentence_accuracy = student_ckpt["validation_sentence_accuracy"]
    validation_wer = student_ckpt["validation_wer"]
    validation_losses = student_ckpt["validation_losses"]
    learning_rates = student_ckpt["lr"]
    grad_norms = student_ckpt["grad_norm"]

    best_score = 0.0

    # Train
    for epoch in range(student_options.num_epochs):
        start_time = time.time()

        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=student_options.num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(student_options.num_epochs)),
        )

        train_result = _train_one_epoch(
            distillation_loader=distillation_loader,
            student_model=student_model,
            teacher_model=teacher_model,
            epoch_text=epoch_text,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_grad_norm=student_options.max_grad_norm,
            device=device,
            tf_scheduler=tf_scheduler,
            is_logging=is_logging,
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
        epoch_lr = lr_scheduler.get_lr()  # cycle
        validation_result = _valid_one_epoch(
            data_loader=valid_loader,
            model=student_model,
            epoch_text=epoch_text,
            criterion=criterion,
            device=device,
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
                    "model": student_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "configs": option_dict,
                    "token_to_id": distillation_loader.dataset.token_to_id,
                    "id_to_token": distillation_loader.dataset.id_to_token,
                    "network": student_options.network,
                    "scheduler": lr_scheduler.state_dict(),
                },
                prefix=student_options.prefix,
            )
            best_score = final_metric(
                sentence_acc=validation_epoch_sentence_accuracy,
                word_error_rate=validation_epoch_wer,
            )
            print(f"best score: {best_score}")
            print("model is saved")

        # Summary
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if (
            epoch % student_options.print_epochs == 0
            or epoch == student_options.num_epochs - 1
        ):
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