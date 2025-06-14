from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from ukrlm.datamodules.masked_language_modeling import tokenize_function


# DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent \
#                          / 'research' / 'tokenizer' / 'experiment-6-liberta-v2' / 'spm.model'
DEFAULT_TOKENIZER_PATH = 'Goader/liberta-large-v2'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='name of the dataset')
    parser.add_argument('--data_dir', type=str, default=None, help='path to the data directory')
    parser.add_argument('--load_from_disk', action='store_true', help='load the dataset from disk')
    parser.add_argument('--tokenizer_path', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the output directory')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the dataset')
    parser.add_argument('--max_length', type=int, default=512, help='max length of the input sequence')
    parser.add_argument('--num_shards', type=int, default=240, help='number of shards')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_proc', type=int, default=None)
    parser.add_argument('--cache_dir', type=str, default=None, help='path to the cache directory')
    args = parser.parse_args()

    print('Loading the tokenizer...')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print('Loading the dataset...')

    if args.name == 'culturax':
        dataset = load_dataset(
            path='uonlp/CulturaX',
            name='uk',
            split='train',
            streaming=False,
            keep_in_memory=False,
            cache_dir=args.cache_dir,
            num_proc=args.num_proc,
            use_auth_token=True,
        )
    else:  # filepath
        assert args.data_dir is not None, 'data_dir must be provided'
        if args.load_from_disk:
            dataset = load_from_disk(args.data_dir)
        else:
            dataset = load_dataset(
                path='parquet',
                data_dir=args.data_dir,
                split='train',
                num_proc=args.num_proc,
            )

    print('Loaded dataset:')
    print(dataset)

    print('Tokenizing the dataset...')
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
        fn_kwargs=dict(tokenizer=tokenizer, max_length=args.max_length),
        num_proc=args.num_proc,
    )

    dataset = dataset.select_columns(['input_ids'])

    if args.shuffle:
        print('Shuffling the dataset...')
        dataset = dataset.shuffle(seed=42)

    print('Adding the id column...')
    dataset = dataset.map(
        lambda examples, indices: {'id': indices},
        with_indices=True,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
    )

    print('Saving the tokenized dataset...')
    dataset.save_to_disk(
        args.output_dir,
        num_proc=args.num_proc,
        num_shards=args.num_shards,
    )
