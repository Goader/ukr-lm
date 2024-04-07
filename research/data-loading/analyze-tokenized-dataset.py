from argparse import ArgumentParser

from datasets import load_dataset, load_from_disk


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='path to the dataset')
    parser.add_argument('--show_n_examples', type=int, default=5, help='number of examples to show')
    args = parser.parse_args()

    # FIXME
    dataset = load_from_disk(
        args.name,
    )

    # printing out a few examples
    print(dataset)

    for idx in range(args.show_n_examples):
        example = dataset[idx]
        print(example)
        print()

    # calculating the number of documents, examples, tokens and an average number of tokens per document
    num_documents = -1
    num_examples = len(dataset)
    num_tokens = 0
    full_context = 0

    for idx in range(len(dataset)):
        example = dataset[idx]
        num_documents = max(num_documents, example['id'])

        tokens = len(example['input_ids'])
        num_tokens += tokens - 2  # removing CLS and SEP
        full_context += int(tokens == 512)

    print(f'Number of documents: {num_documents + 1:,}')
    print(f'Number of examples: {num_examples:,}')
    print(f'Number of tokens: {num_tokens:,}')
    print()
    print(f'Average number of examples per document: {num_examples / (num_documents + 1):.2f}')
    print(f'Average number of tokens per document: {num_tokens / (num_documents + 1):.2f}')
    print(f'Average number of tokens per example: {num_tokens / num_examples:.2f}')
    print()
    print(f'Ratio of examples with full context: {full_context / num_examples:.2%}')


