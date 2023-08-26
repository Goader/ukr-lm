from omegaconf import DictConfig

from torch.utils.data import IterableDataset


class C4(IterableDataset):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError()
