import os
import random
from psutil import virtual_memory
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from networks.Attention import Attention
from networks.SATRN import SATRN
from networks.SWIN import SWIN
from networks.My_SATRN import MySATRN

import warnings

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
    elif model_type == 'SWIN':
        model = SWIN(FLAGS, train_dataset, model_checkpoint).to(device)
        checkpoint = torch.load('/opt/ml/p4-fr-sorry-math-but-love-you_sub/pth/swin_tiny_patch4_window7_224.pth', map_location='cuda')
        model.encoder.load_state_dict(checkpoint['model'], strict=False)
    elif model_type == "MySATRN":
        model = MySATRN(FLAGS, train_dataset, model_checkpoint).to(device)
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
        eos_id = data_loader.dataset.token_to_id['<EOS>']
        special_ids = set([
            data_loader.dataset.token_to_id['<PAD>'],
            data_loader.dataset.token_to_id["<SOS>"],
            eos_id
            ])
        # special_ids = [
        #     data_loader.dataset.token_to_id["<PAD>"],
        #     data_loader.dataset.token_to_id["<SOS>"],
        #     data_loader.dataset.token_to_id["<EOS>"]
        #     ]
    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
                elif token == eos_id:
                    break
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

def get_timestamp():
    return datetime.now().strftime(format='%m%d-%H%M%S')


class TeacherForcingScheduler:
    """Teacher Forcing 스케줄러 클래스. Train에 활용
    Example:
        # Define TF Scheduler
        total_steps = len(train_data_loader)*options.num_epochs
        teacher_forcing_ratio = 0.6
        tf_scheduler = TeacherForcingScheduler(
            num_steps=total_steps,
            tf_max=teacher_forcing_ratio
            )
        
        # Train phase
        tf_ratio = tf_scheduler.step()
        output = model(input, expected, False, tf_ratio)

    Args:
        num_steps (int): 총 스텝 수
        tf_max (float): 최대 teacher forcing ratio. tf_max에서 시작해서 코사인 함수를 그리며 0으로 마무리 됨
    """
    def __init__(self, num_steps: int, tf_max: float):
        linspace = self._get_linspace(num_steps, tf_max)
        self.__scheduler = iter(linspace)
        
    def step(self):
        try:
            return next(self.__scheduler)
        except:
            warnings.warn('Teacher forcing scheduler has been done. Return just 0 for now.')
            return 0.0

    @staticmethod
    def _get_linspace(num_steps, tf_max):
       from copy import deepcopy
       factor = tf_max / 2
       x = np.linspace(0, np.pi, num_steps)
       x = np.cos(x)
       x *= factor
       x += factor
    #    return deepcopy(x.tolist())
       return x