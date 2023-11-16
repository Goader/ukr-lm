from omegaconf import DictConfig

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from datasets import IterableDataset, IterableDatasetDict
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DataCollator,
    DataCollatorForLanguageModeling,
)

from ukrlm.datasets import MultiSourceDataset, instantiate_datasets
from ukrlm.tokenizers import LibertaTokenizer


class MaskedLanguageModelingDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_datasets: dict[str, IterableDataset | IterableDatasetDict] = dict()
        self.val_datasets: dict[str, IterableDataset | IterableDatasetDict] = dict()

        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.collator: DataCollator | None = None

    def prepare_data(self) -> None:
        # AutoTokenizer.from_pretrained(self.cfg.model.name)  # FIXME tokenizer is not yet on huggingface hub
        pass

    def setup(self, stage: str | None = None) -> None:
        # FIXME tokenizer is not yet on huggingface hub
        # self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        self.tokenizer = LibertaTokenizer.from_pretrained(self.cfg.datamodule.tokenizer_path)
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.cfg.task.mlm_probability,
            pad_to_multiple_of=self.cfg.task.pad_to_multiple_of
        )

        self.train_datasets = instantiate_datasets(self.cfg.datamodule.datasets.train, self.cfg)
        self.val_datasets = instantiate_datasets(self.cfg.datamodule.datasets.val, self.cfg)

        # FIXME update the function and should we split long sentences?
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                max_length=512,
                truncation=True,
                padding='longest',
                return_special_tokens_mask=True,
            )

        # FIXME what with distributed training? how should we handle it?

        # FIXME should we do it after interleaving?
        for name, dataset in list(self.train_datasets.items()):
            self.train_datasets[name] = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=dataset.column_names,
            )

        for name, dataset in list(self.val_datasets.items()):
            self.val_datasets[name] = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=dataset.column_names,
            )

    def train_dataloader(self):
        if not self.tokenizer or not self.collator:
            raise RuntimeError('Tokenizer and collator must be initialized before training dataloader')

        # FIXME temporary weights since we have only one dataset for now
        dataset = MultiSourceDataset(datasets=self.train_datasets,
                                     weights=dict.fromkeys(self.train_datasets.keys(), 1.0))
        return DataLoader(dataset=dataset,
                          batch_size=self.cfg.datamodule.batch_size,
                          num_workers=self.cfg.datamodule.num_workers,
                          pin_memory=self.cfg.datamodule.pin_memory,
                          collate_fn=self.collator)

    def val_dataloader(self):
        if not self.tokenizer or not self.collator:
            raise RuntimeError('Tokenizer and collator must be initialized before validation dataloader')

        dataloaders = {
            name: DataLoader(dataset=dataset,
                             batch_size=self.cfg.datamodule.batch_size,
                             num_workers=self.cfg.datamodule.num_workers,
                             pin_memory=self.cfg.datamodule.pin_memory,
                             collate_fn=self.collator)
            for name, dataset in self.val_datasets.items()
        }
        return list(dataloaders.values())[0]  # FIXME temporary solution
