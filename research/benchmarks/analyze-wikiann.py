from datasets import load_dataset, Split


if __name__ == '__main__':
    dataset = load_dataset('wikiann', 'uk', split=Split.ALL)

    # each item contains a dictionary with the following keys:
    # - 'tokens' - list of tokens
    # - 'ner_tags' - list of NER tags
    # - 'spans' - list of spans

    # print out the total number of tokens
    total_tokens = sum(len(item['tokens']) for item in dataset)
    print('Total tokens:', total_tokens)

    # print out the total number of spans
    total_spans = sum(len(item['spans']) for item in dataset)
    print('Total NER spans:', total_spans)

    # print out the average number of tokens and spans per document
    print('Average tokens per document:', total_tokens / len(dataset))
    print('Average spans per document:', total_spans / len(dataset))
