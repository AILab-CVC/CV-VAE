import numpy as np
import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers.trainer_utils import SchedulerType
from transformers.utils import logging

from transformers.optimization import (
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    # get_inverse_sqrt_schedule,
    # get_reduce_on_plateau_schedule,
)

logger = logging.get_logger(__name__)


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(
        self,
        warm_up_steps,
        lr_min,
        lr_max,
        lr_start,
        max_decay_steps,
        verbosity_interval=0,
    ):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (
                self.lr_max - self.lr_start
            ) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (
                self.lr_max_decay_steps - self.lr_warm_up_steps
            )
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + np.cos(t * np.pi)
            )
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(
        self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0
    ):
        assert (
            len(warm_up_steps)
            == len(f_min)
            == len(f_max)
            == len(f_start)
            == len(cycle_lengths)
        )
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[
                cycle
            ] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (
                self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle]
            )
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                1 + np.cos(t * np.pi)
            )
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[
                cycle
            ] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_lengths[cycle] - n
            ) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_ratio: float = 0.0,
):

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    # return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return max(
        0.0,
        0.5
        * (
            (1.0 + min_lr_ratio)
            + (1.0 - min_lr_ratio)
            * math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        ),
    )


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    # SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    # SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    min_lr_ratio: Optional[float] = 0.0,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    # if name == SchedulerType.CONSTANT or name == SchedulerType.REDUCE_ON_PLATEAU:
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(
            f"{name} requires `num_warmup_steps`, please provide that argument."
        )

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # if name == SchedulerType.INVERSE_SQRT:
    #     return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(
            f"{name} requires `num_training_steps`, please provide that argument."
        )

    logger.info(f"Initialize lr scheduler with min_lr_ratio: {min_lr_ratio}")
    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
    )
