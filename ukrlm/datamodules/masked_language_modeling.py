from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from ukrlm.datasets import MultiSourceDataset
from ukrlm.utils import instantiate_datasets


class MaskedLanguageModelingDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_datasets = dict()
        self.val_datasets = dict()

        self.tokenizer = None
        self.collator = None

    def prepare_data(self) -> None:
        AutoTokenizer.from_pretrained(self.cfg.model.name)

    def setup(self, stage: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.cfg.task.mlm_probability,
            pad_to_multiple_of=self.cfg.task.pad_to_multiple_of
        )

        self.train_datasets = instantiate_datasets(self.cfg.datamodule.datasets.train, self.cfg)
        self.val_datasets = instantiate_datasets(self.cfg.datamodule.datasets.val, self.cfg)

    def train_dataloader(self):
        if not self.tokenizer or not self.collator:
            raise RuntimeError('Tokenizer and collator must be initialized before training dataloader')

        # FIXME temporary weights since we have only one dataset for now
        dataset = MultiSourceDataset(datasets=self.train_datasets,
                                     weights={dataset_name: 1.0 for dataset_name in self.train_datasets.keys()},)
        return DataLoader(dataset=dataset,
                          batch_size=self.cfg.datamodule.batch_size,
                          num_workers=self.cfg.datamodule.num_workers,
                          collate_fn=self.collator)

    def val_dataloader(self):
        if not self.tokenizer or not self.collator:
            raise RuntimeError('Tokenizer and collator must be initialized before validation dataloader')

        dataloaders = {
            name: DataLoader(dataset=dataset,
                             batch_size=self.cfg.datamodule.batch_size,
                             num_workers=self.cfg.datamodule.num_workers,
                             collate_fn=self.collator)
            for name, dataset in self.val_datasets.items()
        }
        return dataloaders
