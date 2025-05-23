from argparse import ArgumentParser

from datasets import load_dataset
from transformers import AutoTokenizer


# DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent \
#                          / 'research' / 'tokenizer' / 'experiment-6-liberta-v2' / 'spm.model'
DEFAULT_TOKENIZER_PATH = 'Goader/liberta-large-v2'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, help='path to the data directory')
    parser.add_argument('--tokenizer_path', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the output directory')
    parser.add_argument('--num_shards', type=int, default=240, help='number of shards')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_proc', type=int, default=None)
    parser.add_argument('--cache_dir', type=str, default=None, help='path to the cache directory')
    args = parser.parse_args()

    print('Loading the tokenizer...')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print('Loading the dataset...')

    dataset = load_dataset(
        path='parquet',
        data_dir=args.data_dir,
        split='train',
        num_proc=args.num_proc,
    )

    print('Loaded dataset:')
    print(dataset)

    def tokenize_function(examples: list[dict]):
        output = tokenizer(
            examples['text'],
            return_attention_mask=False,
            return_special_tokens_mask=False,
            return_length=True,
        )

        return {
            'id': examples['id'],
            'source': examples['source'],
            'url': examples['url'],
            'subsource': examples['subsource'],
            'tokens': output['length'],
        }

    print('Tokenizing the dataset...')
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
        num_proc=args.num_proc,
    )

    print('Saving the tokenized dataset...')
    dataset.save_to_disk(
        args.output_dir,
        num_proc=args.num_proc,
        num_shards=args.num_shards,
    )
