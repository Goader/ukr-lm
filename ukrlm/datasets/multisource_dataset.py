from copy import deepcopy
import random

from torch.utils.data import IterableDataset


class MultiSourceDataset(IterableDataset):
    def __init__(
            self,
            datasets: dict[str, IterableDataset],
            weights: dict[str, float],
            stopping_strategy: str = 'all_exhausted',
            overload: bool = False
    ):
        """
        Merges multiple datasets into one, so that multiple corpora can be used for training.
        This allows to treat all corpora as one big dataset, which is useful for training on multiple corpora.
        Instances from each dataset are sampled with probability proportional to the corresponding weight.

        :param datasets: dictionary of PyTorch datasets
        :param weights: dictionary of weights for each dataset (must be the same length as datasets,
            dataset sizes may be used)
        :param stopping_strategy: strategy for stopping the iteration, possible values:
            - 'all_exhausted' (default) - stop when all datasets are exhausted
            - 'first_exhausted' - stop when any dataset is exhausted
        :param overload: if `stopping_strategy` is not `all_exhausted`, then this parameter is ignored, otherwise
            if True, then the datasets will be overloaded instantly after exhaustion, until all
            datasets will get exhausted at least once, otherwise exhausted datasets will be omitted
        """
        super().__init__()

        if datasets.keys() != weights.keys():
            raise ValueError('The number of datasets and weights must be equal')

        if not all(w > 0 for w in weights.values()):
            raise ValueError('All weights must be positive')

        self.datasets = datasets
        self.weights = weights
        self.stopping_strategy = stopping_strategy
        self.overload = overload

    def __iter__(self):
        dataset_names = list(self.datasets.keys())
        weights = [self.weights[name] for name in dataset_names]
        name2index = {name: i for i, name in enumerate(dataset_names)}

        iterators = {name: iter(dataset) for name, dataset in self.datasets.items()}
        exhausted = {name: False for name in dataset_names}

        while True:
            dataset_name = random.choices(dataset_names, weights=weights, k=1)[0]

            try:
                yield next(iterators[dataset_name])
            except StopIteration:
                exhausted[dataset_name] = True
                if self.stopping_strategy == 'first_exhausted':
                    break
                elif self.stopping_strategy == 'all_exhausted':
                    if all(exhausted.values()):
                        break
                    elif not self.overload:
                        index = name2index[dataset_name]
                        weights[index] = 0
                else:
                    raise ValueError(f'Unknown stopping strategy: {self.stopping_strategy}')
