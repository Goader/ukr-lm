import datasets
from datasets import load_dataset, load_from_disk, Value, Features
from datasets.distributed import split_dataset_by_node
from omegaconf import DictConfig

from .examples_passed_counter_dataset import ExamplesPassedCounterDataset
from .multisource_dataset import MultiSourceDataset
from .skip_examples_dataset import SkipExamplesDataset


def dataset_by_name(
        name: str,
        cfg: DictConfig,
        rank: int,
        world_size: int
) -> datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset | datasets.IterableDatasetDict:
    if name == 'c4':
        raise NotImplementedError('C4 dataset is not yet implemented')
    # TODO add possibility for tokenized
    elif name == 'cc100':
        dataset = load_dataset(
            path='cc100',
            lang='uk',
            split='train',
            streaming=cfg.datasets.cc100.streaming,
            keep_in_memory=cfg.datasets.cc100.keep_in_memory,
            cache_dir=cfg.huggingface_cache_dir,
            num_proc=cfg.datamodule.num_workers if not cfg.datasets.cc100.streaming else None,
        )

        if not cfg.datasets.cc100.streaming:
            # TODO what is an optimal number of shards?
            dataset = dataset.to_iterable_dataset(num_shards=cfg.datasets.cc100.num_shards)

        dataset = split_dataset_by_node(dataset, rank, world_size)
        dataset = dataset.cast_column('id', Value('int64'))

        return dataset
    elif name == 'culturax':
        print('culturax tokenized', cfg.datasets.culturax.tokenized)
        if not cfg.datasets.culturax.tokenized:
            print('loading culturax from the huggingface')
            dataset = load_dataset(
                path='uonlp/CulturaX',
                name='uk',
                split='train',
                streaming=cfg.datasets.culturax.streaming,
                keep_in_memory=cfg.datasets.culturax.keep_in_memory,
                cache_dir=cfg.huggingface_cache_dir,
                num_proc=cfg.datamodule.num_workers if not cfg.datasets.culturax.streaming else None,
                use_auth_token=True,
            )

            if not cfg.datasets.culturax.streaming:
                dataset = dataset.to_iterable_dataset(num_shards=cfg.datasets.culturax.num_shards)

            dataset = split_dataset_by_node(dataset, rank, world_size)

            def zip_with_index(examples: list[dict], indices: list[int], rank: int = 0, world_size: int = 1):
                examples['id'] = [i * world_size + rank for i in indices]
                return examples

            dataset = dataset.map(
                zip_with_index,
                with_indices=True,
                batched=True,
                batch_size=256,
                remove_columns=['timestamp', 'url', 'source'],
                features=Features({'id': Value('int64'), 'text': Value('string')}),
                fn_kwargs={'rank': rank, 'world_size': world_size}
            )
        else:
            print('loading culturax from the disk')
            dataset = load_from_disk(
                dataset_path=cfg.datasets.culturax.tokenized_dataset_path,
                keep_in_memory=cfg.datasets.culturax.keep_in_memory,
            )

        return dataset
    elif name == 'ukrcorpus':
        print('loading ukr-corpus from the disk')
        dataset = load_from_disk(
            dataset_path=cfg.datasets.ukrcorpus.tokenized_dataset_path,
            keep_in_memory=cfg.datasets.ukrcorpus.keep_in_memory,
        )
        return dataset
    elif name == 'treebank':
        dataset = load_dataset(
            path='Goader/ukrainian-treebank-lm',
            split='train',
            streaming=cfg.datasets.treebank.streaming,
            keep_in_memory=cfg.datasets.treebank.keep_in_memory,
            cache_dir=cfg.huggingface_cache_dir,
            num_proc=cfg.datamodule.num_workers if not cfg.datasets.treebank.streaming else None,
        )
        dataset = dataset.rename_column('document_id', 'id')
        return dataset
    else:
        raise ValueError(f'Unknown dataset name: {name}')


def instantiate_datasets(
        datasets: list[str],
        cfg: DictConfig,
        rank: int,
        word_size: int
) -> dict[str, datasets.Dataset | datasets.DatasetDict | datasets.IterableDataset | datasets.IterableDatasetDict]:
    return {
        name: dataset_by_name(name, cfg, rank, word_size)
        for name in datasets
    }
