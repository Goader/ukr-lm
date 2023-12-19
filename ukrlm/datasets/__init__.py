from datasets import IterableDataset, IterableDatasetDict, load_dataset, Value
from datasets.distributed import split_dataset_by_node
from omegaconf import DictConfig

from .multisource_dataset import MultiSourceDataset


def dataset_by_name(name: str, cfg: DictConfig, rank: int, word_size: int) -> IterableDataset | IterableDatasetDict:
    if name == 'c4':
        raise NotImplementedError('C4 dataset is not yet implemented')
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

        dataset = split_dataset_by_node(dataset, rank, word_size)
        dataset = dataset.cast_column('id', Value('int64'))

        return dataset
    elif name == 'treebank':
        return load_dataset(
            path='Goader/ukrainian-treebank-lm',
            split='train',
            streaming=cfg.datasets.treebank.streaming,
            keep_in_memory=cfg.datasets.treebank.keep_in_memory,
            cache_dir=cfg.huggingface_cache_dir,
            num_proc=cfg.datamodule.num_workers if not cfg.datasets.treebank.streaming else None,
        )
    else:
        raise ValueError(f'Unknown dataset name: {name}')


def instantiate_datasets(
        datasets: list[str],
        cfg: DictConfig,
        rank: int,
        word_size: int
) -> dict[str, IterableDataset | IterableDatasetDict]:
    return {
        name: dataset_by_name(name, cfg, rank, word_size)
        for name in datasets
    }
