import os

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

        args = (self.cfg, self.trainer.global_rank, self.trainer.world_size)
        self.train_datasets = instantiate_datasets(self.cfg.datamodule.datasets.train, *args)
        self.val_datasets = instantiate_datasets(self.cfg.datamodule.datasets.val, *args)

        def tokenize_function(examples):
            output = self.tokenizer(
                examples['text'],
                return_special_tokens_mask=True,
            )

            outputs = {
                'id': [],
                'input_ids': [],
                'attention_mask': [],
                'special_tokens_mask': [],
            }

            # splitting into multiple examples if the input is too long
            for doc_idx in range(len(examples['text'])):
                doc_id = examples['id'][doc_idx]
                input_ids = output['input_ids'][doc_idx]
                attention_mask = output['attention_mask'][doc_idx]
                special_tokens_mask = output['special_tokens_mask'][doc_idx]

                max_length = self.cfg.model.max_position_embeddings
                for i in range(0, len(input_ids), max_length):
                    outputs['id'].append(doc_id)
                    outputs['input_ids'].append(input_ids[i:i + max_length])
                    outputs['attention_mask'].append(attention_mask[i:i + max_length])
                    outputs['special_tokens_mask'].append(special_tokens_mask[i:i + max_length])

            return outputs

        # FIXME should we do it after interleaving?
        for name, dataset in list(self.train_datasets.items()):
            self.train_datasets[name] = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=256,
                remove_columns=['text'],
            )

        # FIXME is this properly distributed?
        for name, dataset in list(self.val_datasets.items()):
            self.val_datasets[name] = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=256,
                remove_columns=dataset.column_names,
            ).remove_columns(['id'])  # FIXME we could replace removal with proper integer IDs

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
