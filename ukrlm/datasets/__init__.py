from datasets import IterableDataset, IterableDatasetDict, load_dataset
from omegaconf import DictConfig

from .multisource_dataset import MultiSourceDataset


def dataset_by_name(name: str, cfg: DictConfig) -> IterableDataset | IterableDatasetDict:
    if name == 'c4':
        raise NotImplementedError('C4 dataset is not yet implemented')
    elif name == 'cc100':
        return load_dataset(
            path='cc100',
            lang='uk',
            split='train',
            streaming=cfg.datasets.cc100.streaming,
            cache_dir=cfg.huggingface_cache_dir,
            # TODO add more options: num_proc?
        )
    else:
        raise ValueError(f'Unknown dataset name: {name}')


def instantiate_datasets(datasets: list[str], cfg: DictConfig) -> dict[str, IterableDataset | IterableDatasetDict]:
    return {
        name: dataset_by_name(name, cfg)
        for name in datasets
    }
