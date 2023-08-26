from omegaconf import DictConfig

from torch.utils.data import IterableDataset
from datasets import load_dataset


class CC100(IterableDataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset = load_dataset(
            path='cc100',
            name='uk',
            split='train',
            streaming=self.cfg.datasets.cc100.streaming,
            cache_dir=self.cfg.huggingface_cache_dir,
        )

    def __iter__(self):
        return iter(self.dataset)
