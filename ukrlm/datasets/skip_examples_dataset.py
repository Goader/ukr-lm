import time

from torch.utils.data import IterableDataset

from lightning.pytorch.utilities import rank_zero_info


class SkipExamplesDataset(IterableDataset):
    def __init__(
            self,
            dataset: IterableDataset,
            skip_n: int,
    ):
        """
        Skips `skip_n` examples from the dataset and then starts yielding examples.

        :param dataset: PyTorch dataset
        :param skip_n: number of examples to skip
        """
        super().__init__()

        self.dataset = dataset
        self.skip_n = skip_n

    def __iter__(self):
        if self.skip_n == 0:
            yield from self.dataset
            return

        rank_zero_info(f'Skipping {self.skip_n} examples')
        t0 = time.time()

        iterator = iter(self.dataset)
        for _ in range(self.skip_n):
            try:
                next(iterator)
            except StopIteration:
                break

        rank_zero_info(f'Skipped {self.skip_n} examples in {time.time() - t0:.2f} seconds')

        yield from iterator
