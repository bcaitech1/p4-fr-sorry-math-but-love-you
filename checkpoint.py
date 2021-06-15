import os
import torch
import wandb

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,
    "train_losses": [],
    "train_symbol_accuracy": [],
    "train_sentence_accuracy": [],
    "train_wer": [],
    "validation_losses": [],
    "validation_symbol_accuracy": [],
    "validation_sentence_accuracy": [],
    "validation_wer": [],
    "lr": [],
    "grad_norm": [],
    "model": {},
    "configs": {},
    "token_to_id": {},
    "id_to_token": {},
    "scheduler": {},
    "scheduler_name": "",
}


def save_checkpoint(checkpoint, dir="./checkpoints", prefix=""):
    filename = f"{checkpoint['network']}_best_model.pth"
    if not os.path.exists(os.path.join(prefix, dir)):
        os.makedirs(os.path.join(prefix, dir))
    torch.save(checkpoint, os.path.join(prefix, dir, filename))


def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)


def write_wandb(
    epoch,
    grad_norm,
    train_loss,
    train_symbol_accuracy,
    train_sentence_accuracy,
    train_wer,
    train_score,
    validation_loss,
    validation_symbol_accuracy,
    validation_sentence_accuracy,
    validation_wer,
    validation_score,
):
    wandb.log(
        dict(
            epoch=epoch,
            train_loss=train_loss,
            train_symbol_accuracy=train_symbol_accuracy,
            train_sentence_accuracy=train_sentence_accuracy,
            train_wer=train_wer,
            train_score=train_score,
            validation_loss=validation_loss,
            validation_symbol_accuracy=validation_symbol_accuracy,
            validation_sentence_accuracy=validation_sentence_accuracy,
            validation_wer=validation_wer,
            validation_score=validation_score,
            grad_norm=grad_norm,
        )
    )
