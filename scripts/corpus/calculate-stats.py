from argparse import ArgumentParser

from datasets import load_from_disk


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--num-proc', type=int, default=1)
    args = parser.parse_args()

    dataset = load_from_disk(args.path)
    print(dataset)

    print('Number of documents:', f'{len(dataset):,}')

    total_tokens = 0
    def count_tokens(examples):
        global total_tokens
        total_tokens += sum([len(input_ids) - 2 for input_ids in examples['input_ids']])
        return examples

    dataset = dataset.map(count_tokens, batched=True, batch_size=1000, num_proc=1)
    print('Total tokens:', f'{total_tokens:,}')

    print('Average tokens per document:', f'{total_tokens / len(dataset):,}')
