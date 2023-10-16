from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
        dataset: pd.DataFrame,
        val_size: float,
        test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_dataset, val_test_dataset = train_test_split(
        dataset,
        test_size=val_size + test_size,
        random_state=42,
        stratify=dataset['source'],
    )

    val_dataset, test_dataset = train_test_split(
        val_test_dataset,
        test_size=test_size / (val_size + test_size),
        random_state=42,
        stratify=val_test_dataset['source'],
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='path to the CSV dataset')
    parser.add_argument('--val-size', type=float, default=0.1, help='validation size')
    parser.add_argument('--test-size', type=float, default=0.1, help='test size')
    parser.add_argument('--output-dir', type=str, default='.', help='output directory')
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)

    train_dataset, val_dataset, test_dataset = \
        split_dataset(dataset, args.val_size, args.test_size)

    output_dir = Path(args.output_dir)
    train_dataset.to_csv(output_dir / 'train.csv', index=False)
    val_dataset.to_csv(output_dir / 'val.csv', index=False)
    test_dataset.to_csv(output_dir / 'test.csv', index=False)
