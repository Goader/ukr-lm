from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset


class LegalCorpusDataset(Dataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


class LegalCorpusDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError()
