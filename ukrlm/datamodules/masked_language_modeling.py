import os
from typing import Dict, Any

from omegaconf import DictConfig

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import DataLoader, Dataset, IterableDataset, DistributedSampler

import datasets
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DataCollator,
    DataCollatorForLanguageModeling,
)

from ukrlm.datasets import (
    MultiSourceDataset,
    SkipExamplesDataset,
    ExamplesPassedCounterDataset,
    instantiate_datasets,
)
from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.collators import DataCollatorForNgramMasking
from ukrlm.samplers import DistributedResumeSampler


def tokenize_function(examples: list[dict], tokenizer: PreTrainedTokenizerBase, max_length: int):
    output = tokenizer(
        examples['text'],
        return_special_tokens_mask=True,
    )

    outputs = {
        'id': [],
        'input_ids': [],
        'attention_mask': [],
        'special_tokens_mask': [],
    }

    max_length = max_length - 2  # for CLS and SEP

    # splitting into multiple examples if the input is too long
    for doc_idx in range(len(examples['text'])):
        doc_id = examples['id'][doc_idx]
        input_ids = output['input_ids'][doc_idx][1:-1]  # removing CLS and SEP
        attention_mask = output['attention_mask'][doc_idx][1:-1]  # same
        special_tokens_mask = output['special_tokens_mask'][doc_idx][1:-1]  # same

        for i in range(0, len(input_ids), max_length):
            outputs['id'].append(doc_id)
            outputs['input_ids'].append([
                tokenizer.cls_token_id,
                *input_ids[i:i + max_length],
                tokenizer.sep_token_id
            ])
            outputs['attention_mask'].append([1, *attention_mask[i:i + max_length], 1])
            outputs['special_tokens_mask'].append([1, *special_tokens_mask[i:i + max_length], 1])

    return outputs


class MaskedLanguageModelingDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.train_datasets: dict[str, Dataset | IterableDataset] = dict()
        self.val_datasets: dict[str, Dataset | IterableDataset] = dict()

        self.joined_train_dataset: Dataset | IterableDataset | None = None

        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.collator: DataCollator | None = None

        self.distributed_sampler: DistributedResumeSampler | None = None

        self.examples_passed_counter_per_dataset: dict[str, ExamplesPassedCounterDataset] = dict()
        self.skip_train_examples_per_dataset: dict[str, int] = dict()

    def prepare_data(self) -> None:
        # AutoTokenizer.from_pretrained(self.cfg.model.name)  # FIXME tokenizer is not yet on huggingface hub
        pass

    def instantiate_collator(self):
        if self.cfg.task.collator == 'DataCollatorForLanguageModeling':
            collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm_probability=self.cfg.task.mlm_probability,
                pad_to_multiple_of=self.cfg.task.pad_to_multiple_of
            )
        elif self.cfg.task.collator == 'DataCollatorForNgramMasking':
            collator = DataCollatorForNgramMasking(
                tokenizer=self.tokenizer,
                mlm_probability=self.cfg.task.mlm_probability,
                max_ngram_size=1,
                pad_to_multiple_of=self.cfg.task.pad_to_multiple_of
            )
        else:
            raise ValueError(f'unknown collator: {self.cfg.task.collator}')

        return collator

    def setup(self, stage: str | None = None) -> None:
        # FIXME tokenizer is not yet on huggingface hub
        # self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        self.tokenizer = LibertaTokenizer.from_pretrained(self.cfg.datamodule.tokenizer_path)
        self.collator = self.instantiate_collator()

        args = (self.cfg, self.trainer.global_rank, self.trainer.world_size)
        self.train_datasets = instantiate_datasets(self.cfg.datamodule.datasets.train, *args)
        self.val_datasets = instantiate_datasets(self.cfg.datamodule.datasets.val, *args)

        for name, dataset in list(self.train_datasets.items()):
            dataset_config = self.cfg.datasets.get(name, dict())

            if dataset_config.get('tokenized', False):
                continue

            mapped_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=256,
                remove_columns=['text'],
                fn_kwargs=dict(
                    tokenizer=self.tokenizer,
                    max_length=self.cfg.datamodule.max_length
                )
            )

            examples_passed_counter_dataset = ExamplesPassedCounterDataset(mapped_dataset)
            self.examples_passed_counter_per_dataset[name] = examples_passed_counter_dataset

            self.train_datasets[name] = examples_passed_counter_dataset

        # FIXME is this properly distributed?
        for name, dataset in list(self.val_datasets.items()):
            dataset_config = self.cfg.datasets.get(name, dict())

            if dataset_config.get('tokenized', False):
                continue

            self.val_datasets[name] = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=256,
                remove_columns=dataset.column_names,
                fn_kwargs=dict(
                    tokenizer=self.tokenizer,
                    max_length=self.cfg.datamodule.max_length,
                )
            ).remove_columns(['id'])  # FIXME we could replace removal with proper integer IDs

        # joined train dataset
        # FIXME temporary weights since we have only one dataset for now
        # train_dataset = MultiSourceDataset(datasets=self.train_datasets,
        #                              weights=dict.fromkeys(self.train_datasets.keys(), 1.0))
        if len(self.train_datasets) != 1:
            raise NotImplementedError('only one dataset is supported for now')
        self.joined_train_dataset = list(self.train_datasets.values())[0]

        self.distributed_sampler = DistributedResumeSampler(
            dataset=self.joined_train_dataset,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
            shuffle=False,
            seed=self.cfg.seed,
        ) if self.cfg.task.strategy == 'ddp' else None

    def train_dataloader(self):
        print('train dataset', self.joined_train_dataset)
        if not self.tokenizer or not self.collator:
            raise RuntimeError('Tokenizer and collator must be initialized before training dataloader')

        return DataLoader(dataset=self.joined_train_dataset,
                          batch_size=self.cfg.datamodule.batch_size,
                          num_workers=self.cfg.datamodule.num_workers,
                          sampler=self.distributed_sampler,
                          pin_memory=self.cfg.datamodule.pin_memory,
                          prefetch_factor=self.cfg.datamodule.prefetch_factor,
                          multiprocessing_context=self.cfg.datamodule.multiprocessing_context,
                          collate_fn=self.collator)

    def val_dataloader(self):
        if not self.tokenizer or not self.collator:
            raise RuntimeError('Tokenizer and collator must be initialized before validation dataloader')

        dataloaders = {
            name: DataLoader(dataset=dataset,
                             batch_size=self.cfg.datamodule.batch_size,
                             num_workers=self.cfg.datamodule.num_workers,
                             pin_memory=self.cfg.datamodule.pin_memory,
                             prefetch_factor=self.cfg.datamodule.prefetch_factor,
                             multiprocessing_context=self.cfg.datamodule.multiprocessing_context,
                             collate_fn=self.collator)
            for name, dataset in self.val_datasets.items()
        }
        return list(dataloaders.values())[0]  # FIXME temporary solution

    def state_dict(self) -> Dict[str, Any]:
        batch_size = self.cfg.datamodule.batch_size
        gradient_accumulation_steps = self.cfg.task.gradient_accumulation_steps
        world_size = self.trainer.world_size

        passed_batches = self.trainer.fit_loop.epoch_loop.batch_idx
        passed_examples = passed_batches * batch_size * world_size

        return {
            'epoch': self.trainer.current_epoch,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'world_size': world_size,
            'passed_batches': passed_batches,
            'passed_examples': passed_examples,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        last_epoch = state_dict['epoch']
        skip_samples = state_dict['passed_examples']
        self.distributed_sampler.resume(
            epoch=last_epoch,
            skip_samples=skip_samples,
        )
        self.distributed_sampler.set_epoch(self.trainer.current_epoch)

