import torch
from torch import nn
from torchtyping import TensorType

from torchvision import models

from omegaconf import DictConfig


class BERTBase(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        raise NotImplementedError()

    def forward(self, X: ...) -> ...:
        raise NotImplementedError()
