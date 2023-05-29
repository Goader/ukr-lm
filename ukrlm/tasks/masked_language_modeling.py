from typing import Any

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig

import torch
from torch import nn
from torch.nn import functional as F
from torchtyping import TensorType

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info


class MaskedLanguageModelingTask(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.save_hyperparameters(cfg)

        raise NotImplementedError()
        # for objective in self.cfg.task.objectives: ...

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        # TODO
        raise NotImplementedError()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.task.learning_rate,
            weight_decay=self.hparams.task.weight_decay
        )
        return optimizer
