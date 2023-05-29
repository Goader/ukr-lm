from omegaconf import DictConfig

import torch
import pytorch_lightning as pl


class C4DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError()
