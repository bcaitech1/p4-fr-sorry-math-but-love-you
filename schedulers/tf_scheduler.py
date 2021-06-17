import math
import numpy as np
import warnings


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

    def __init__(self, num_steps: int, tf_max: float = 1.0, tf_min: float = 0.4):
        linspace = self._get_arctan(num_steps, tf_max, tf_min)
        self.__scheduler = iter(linspace)
        self.tf_max = tf_max
        self.tf_min = tf_min

    def step(self):
        try:
            return next(self.__scheduler)
        except:
            # 스케줄링이 끝났는데 학습은 종료되지 않은 경우 tf_min을 리턴
            warnings.warn(
                f"Teacher forcing scheduler has been done. Return just tf_min({self.tf_min}) for now."
            )
            return self.tf_min

    @staticmethod
    def _get_arctan(num_steps: int, tf_max: float, tf_min: float):
        diff = tf_max - tf_min
        inflection = int(num_steps * 0.1)
        x = np.linspace(-5, 5, num_steps)  # NOTE. for transformer
        x = -np.arctan(x)
        x -= x[-1]
        x *= diff / x[0]
        x += tf_min
        x = x[inflection:]
        return x

    @staticmethod
    def _get_cosine(num_steps: int, tf_max: float):  # NOTE. 아직 tf_min 미적용. 무조건 0으로 하강함
        factor = tf_max / 2
        x = np.linspace(0, np.pi, num_steps)
        x = np.cos(x)
        x *= factor
        x += factor
        return x
