import os
import random
from psutil import virtual_memory
import numpy as np
import torch
import torch.optim as optim
from networks.Attention import Attention
from networks.SATRN import SATRN
from networks.My_SATRN import MySATRN
from networks.SWIN import SWIN

def get_network(
    model_type,
    FLAGS,
    model_checkpoint,
    device,
    train_dataset,
):
    model = None
    if model_type == "Attention":
        model = Attention(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == 'SATRN':
        model = SATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == 'MySATRN':
        model = MySATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == 'SWIN':
        model = SWIN(FLAGS, train_dataset, model_checkpoint).to(device)
        checkpoint = torch.load('/opt/ml/p4-fr-sorry-math-but-love-you_sub/pth/swin_tiny_patch4_window7_224.pth', map_location='cuda')
        model.encoder.load_state_dict(checkpoint['model'], strict=False)
    else:
        raise NotImplementedError

    return model

def get_optimizer(optimizer, params, lr, weight_decay=None):
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)

    elif optimizer == "Adadelta":
        optimizer = optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def print_system_envs():
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    print(
        "[+] System environments\n",
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

def id_to_string(tokens, data_loader, do_eval=0):
    result = []
    if do_eval:
        special_ids = [
            data_loader.dataset.token_to_id["<PAD>"],
            data_loader.dataset.token_to_id["<SOS>"],
            data_loader.dataset.token_to_id["<EOS>"]
            ]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "
        result.append(string)
    return result
    
def set_seed(seed: int=21):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False