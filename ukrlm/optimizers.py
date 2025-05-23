import torch
import torch.nn as nn
import optimi

from omegaconf import DictConfig


# from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
# Copyright 2019 Ross Wightman, Apache-2.0 License
def param_groups_weight_decay(model: nn.Module, weight_decay=1e-5, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def instantiate_optimizer(model: nn.Module, cfg: DictConfig):
    if getattr(cfg.optimizer, 'filter_bias_norm_wd', False):
        parameters = param_groups_weight_decay(model, cfg.optimizer.weight_decay)
    else:
        parameters = model.parameters()

    if cfg.optimizer.name == 'torch.optim.AdamW':
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            eps=cfg.optimizer.epsilon,
        )
    elif cfg.optimizer.name == 'optimi.StableAdamW':
        optimizer = optimi.StableAdamW(
            parameters,
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            eps=cfg.optimizer.epsilon,
            decouple_lr=cfg.optimizer.decouple_lr,
        )
    else:
        raise ValueError(f'Unknown optimizer: {cfg.optimizer.name}')

    return optimizer
