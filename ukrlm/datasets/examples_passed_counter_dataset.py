from torch.utils.data import IterableDataset
import torch.multiprocessing as mp


class ExamplesPassedCounterDataset(IterableDataset):
    def __init__(
            self,
            dataset: IterableDataset,
    ):
        """
        Counts the number of examples passed through the dataset.

        :param dataset: PyTorch dataset
        """
        super().__init__()

        self.dataset = dataset

        # does not require lock, since the last step in each process should produce the same value
        self.global_examples_passed = mp.Value('i', 0, lock=False)
        self.local_examples_passed = 0

    @property
    def examples_passed(self) -> int:
        return self.global_examples_passed.value

    def __iter__(self):
        iterator = iter(self.dataset)
        for example in iterator:
            self.local_examples_passed += 1
            self.global_examples_passed.value = self.local_examples_passed
            yield example
