from argparse import ArgumentParser

from datasets import load_dataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='path to the data directory')
    parser.add_argument('--ids_file', type=str, default=None, help='path to the ids file')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the output directory')
    parser.add_argument('--num_shards', type=int, default=240, help='number of shards')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_proc', type=int, default=None)
    args = parser.parse_args()

    print('Loading the ids...')
    with open(args.ids_file, 'r') as f:
        ids = set([line.strip() for line in f.readlines()])

    print('Loading the dataset...')
    dataset = load_dataset(
        path='parquet',
        data_dir=args.data_dir,
        split='train',
        num_proc=args.num_proc,
    )

    print('Loaded dataset:')
    print(dataset)

    def filter_function(examples: list[dict]):
        return [
            id_ in ids
            for id_ in examples['id']
        ]

    print('Filtering the dataset...')
    dataset = dataset.filter(
        filter_function,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
    )

    print('Saving the filtered dataset...')
    dataset.save_to_disk(
        args.output_dir,
        num_proc=args.num_proc,
        num_shards=args.num_shards,
    )
