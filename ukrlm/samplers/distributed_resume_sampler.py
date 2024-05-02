import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Dataset, DistributedSampler


T_co = TypeVar('T_co', covariant=True)


class DistributedResumeSampler(DistributedSampler[T_co]):
    r"""
    Sampler that allows you to resume training from a checkpoint, basically
    skipping the first `n` samples for the first epoch, where `n` is the
    number of samples that were already processed before the checkpoint was
    saved.

    It follows the same API as :class:`~torch.utils.data.DistributedSampler`
    with an additional method :meth:`resume` that allows you to set the epoch
    and the number of samples that were already processed.
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.resume_epoch: int = 0
        self.skip_samples: int = 0

        self.resume_num_samples: int = self.num_samples
        self.resume_total_size: int = self.total_size

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # skip samples
        if self.resume_epoch == self.epoch:
            indices = indices[self.skip_samples:]
            num_samples = self.resume_num_samples
            total_size = self.resume_total_size
        else:
            num_samples = self.num_samples
            total_size = self.total_size

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_size]
        assert len(indices) == total_size

        # subsample
        indices = indices[self.rank:total_size:self.num_replicas]
        assert len(indices) == num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def resume(self, epoch: int, skip_samples: int) -> None:
        self.resume_epoch = epoch
        self.skip_samples = skip_samples

        resume_samples = len(self.dataset) - skip_samples
        if self.drop_last and resume_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.resume_num_samples = math.ceil(
                (resume_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.resume_num_samples = math.ceil(resume_samples / self.num_replicas)  # type: ignore[arg-type]
        self.resume_total_size = self.resume_num_samples * self.num_replicas
