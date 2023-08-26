from torch.utils.data import Dataset
from omegaconf import DictConfig

from ukrlm.datasets import C4, CC100


def dataset_by_name(name: str) -> type[Dataset]:
    if name == 'c4':
        return C4
    elif name == 'cc100':
        return CC100
    else:
        raise ValueError(f'Unknown dataset name: {name}')


def instantiate_datasets(datasets: list[str], cfg: DictConfig) -> dict[str, Dataset]:
    return {
        name: dataset_by_name(name)(cfg)
        for name in datasets
    }
