from torch.utils.data import IterableDataset


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
        self.examples_passed = 0

    def __iter__(self):
        iterator = iter(self.dataset)
        for example in iterator:
            self.examples_passed += 1
            yield example
