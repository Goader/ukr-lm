from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.datamodules.masked_language_modeling import tokenize_function


DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent \
                         / 'research' / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='name of the dataset')
    parser.add_argument('--tokenizer_path', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the output directory')
    parser.add_argument('--max_length', type=int, default=512, help='max length of the input sequence')
    parser.add_argument('--num_proc', type=int, default=None)
    parser.add_argument('--cache_dir', type=str, default=None, help='path to the cache directory')
    args = parser.parse_args()

    print('Loading the tokenizer...')

    tokenizer = LibertaTokenizer.from_pretrained(args.tokenizer_path)

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
        # FIXME
        dataset = load_dataset(
            path='arrow',
            data_files=[args.name],
            split='train',
            num_proc=args.num_proc,
        )

    print('Loaded dataset:')
    print(dataset)

    print('Adding the id column...')
    dataset = dataset.map(
        lambda examples, indices: {'id': indices},
        with_indices=True,
        batched=True,
        batch_size=256,
    )

    print('Tokenizing the dataset...')
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=256,
        remove_columns=dataset.column_names,
        fn_kwargs=dict(tokenizer=tokenizer, max_length=args.max_length),
    )

    dataset = dataset.select_columns(['id', 'input_ids'])

    print('Saving the tokenized dataset...')
    dataset.save_to_disk(
        args.output_dir,
        num_proc=args.num_proc,
    )
