import math
import numpy as np
import warnings
from torch.optim.lr_scheduler import _LRScheduler


class CircularLRBeta:

    def __init__(
        self, optimizer, lr_max, lr_divider, cut_point, step_size, momentum=None
    ):
        self.lr_max = lr_max
        self.lr_divider = lr_divider
        self.cut_point = step_size // cut_point
        self.step_size = step_size
        self.iteration = 0
        self.cycle_step = int(step_size * (1 - cut_point / 100) / 2)
        self.momentum = momentum
        self.optimizer = optimizer

    def get_lr(self):
        if self.iteration > 2 * self.cycle_step:
            cut = (self.iteration - 2 * self.cycle_step) / (
                self.step_size - 2 * self.cycle_step
            )
            lr = self.lr_max * (1 + (cut * (1 - 100) / 100)) / self.lr_divider

        elif self.iteration > self.cycle_step:
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        else:
            cut = self.iteration / self.cycle_step
            lr = self.lr_max * (1 + cut * (self.lr_divider - 1)) / self.lr_divider

        return lr

    def get_momentum(self):
        if self.iteration > 2 * self.cycle_step:
            momentum = self.momentum[0]

        elif self.iteration > self.cycle_step:
            cut = 1 - (self.iteration - self.cycle_step) / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        else:
            cut = self.iteration / self.cycle_step
            momentum = self.momentum[0] + cut * (self.momentum[1] - self.momentum[0])

        return momentum

    def step(self):
        lr = self.get_lr()

        if self.momentum is not None:
            momentum = self.get_momentum()

        self.iteration += 1

        if self.iteration == self.step_size:
            self.iteration = 0

        for group in self.optimizer.param_groups:
            group['lr'] = lr

            if self.momentum is not None:
                group['betas'] = (momentum, group['betas'][1])

        return lr
    
    def get_state_dict(self):
        return None

class CustomCosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CustomCosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
        
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class TeacherForcingScheduler:
    """Teacher Forcing 스케줄러 클래스. Train에 활용
    Example:
        # Define TF Scheduler
        total_steps = len(train_data_loader)*options.num_epochs
        teacher_forcing_ratio = 0.6
        tf_scheduler = TeacherForcingScheduler(
            num_steps=total_steps,
            tf_max=teacher_forcing_ratio
            tf_min=0.4
            )
        
        # Train phase
        tf_ratio = tf_scheduler.step()
        output = model(input, expected, False, tf_ratio)

    Args:
        num_steps (int): 총 스텝 수
        tf_max (float): 최대 teacher forcing ratio. tf_max에서 시작해서 코사인 함수를 그리며 0으로 마무리 됨
        tf_min (float, optional): 최소 teacher forcing ratio. Defaults to 0.4
    """
    def __init__(self, num_steps: int, tf_max: float=1.0, tf_min: float=0.4):
        linspace = self._get_arctan(num_steps, tf_max, tf_min)
        self.__scheduler = iter(linspace)
        self.tf_max = tf_max
        self.tf_min = tf_min
        
        
    def step(self):
        try:
            return next(self.__scheduler)
        except:
            warnings.warn(f'Teacher forcing scheduler has been done. Return just tf_min({self.tf_min}) for now.')
            return self.tf_min

    @staticmethod
    def _get_arctan(num_steps, tf_max, tf_min):
        
        # Old Arctan
        # https://wandb.ai/smbly/Augmentations/runs/2ujnba9s?workspace=user-smbly
        # x = np.linspace(-5, 5, num_steps)
        # x = -np.arctan(x)
        # x -= x[-1]
        # x *= (tf_max/x[0])
        
        # New Arctan
        diff = tf_max - tf_min
        inflection = int(num_steps * 0.1)
        
        x = np.linspace(-5, 4, num_steps) # NOTE. for transformer
        x = -np.arctan(x)
        x -= x[-1]
        x *= (diff/x[0])
        x += tf_min
        x = x[inflection:]

        return x
    
    @staticmethod
    def _get_cosine(num_steps, tf_max): # NOTE. tf_min 적용 안 돼서 무조건 0으로 수렴함
        factor = tf_max / 2
        x = np.linspace(0, np.pi, num_steps)
        x = np.cos(x)
        x *= factor
        x += factor
        return x