import math

from omegaconf import DictConfig

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, min_value=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_value = min_value
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        value = self.min_value + (1.0 - self.min_value) * (1.0 - progress)
        return max(self.min_value, value)


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1, min_value=0.0):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_value = min_value
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        full_range_value = 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        value = self.min_value + full_range_value * (1. - self.min_value)
        return max(self.min_value, value)


class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1, min_value=0.0):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.min_value = min_value
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return self.min_value

        full_range_value = 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0)))
        value = self.min_value + full_range_value * (1. - self.min_value)
        return max(self.min_value, value)


def instantiate_scheduler(optimizer: Optimizer, cfg: DictConfig) -> LambdaLR:
    if cfg.scheduler.name == 'constant':
        return ConstantLRSchedule(optimizer=optimizer)

    elif cfg.scheduler.name == 'constant-with-warmup':
        return WarmupConstantSchedule(
            optimizer=optimizer,
            warmup_steps=cfg.scheduler.warmup_steps,
        )

    elif cfg.scheduler.name == 'linear-with-warmup':
        t_total = cfg.scheduler.t_total if cfg.scheduler.t_total is not None else cfg.task.max_steps  # FIXME how do we infer max_steps if it is not set?
        if t_total <= 0:
            raise ValueError(f'Invalid t_total: {t_total}')

        return WarmupLinearSchedule(
            optimizer=optimizer,
            warmup_steps=cfg.scheduler.warmup_steps,
            t_total=t_total,
            min_value=cfg.scheduler.min_learning_rate / cfg.optimizer.learning_rate,
        )

    elif cfg.scheduler.name == 'cosine-with-warmup':
        t_total = cfg.scheduler.t_total if cfg.scheduler.t_total is not None else cfg.task.max_steps  # FIXME same
        if t_total <= 0:
            raise ValueError(f'Invalid t_total: {t_total}')

        return WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=cfg.scheduler.warmup_steps,
            t_total=t_total,
            cycles=cfg.scheduler.cycles,
            min_value=cfg.scheduler.min_learning_rate / cfg.optimizer.learning_rate,
        )

    else:
        raise ValueError(f'Unknown scheduler: {cfg.scheduler.name}')
