from argparse import ArgumentParser
from typing import Iterator
from pathlib import Path
import jsonlines
import tqdm

from spacy.lang.uk import Ukrainian


nlp = Ukrainian()
tokenizer = nlp.tokenizer


def spacy_tokenize(text: str) -> list[str]:
    doc = tokenizer(text)

    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        tokens.append(token.text.lower())
    return tokens


def read_batch(source: Path, batch_size: int = 20000) -> Iterator[list[str]]:
    docs = []
    with jsonlines.open(source, 'r') as reader:
        for doc in reader.iter(skip_empty=True, skip_invalid=True):
            docs.append(doc['text'])

            if len(docs) >= batch_size:
                yield docs
                docs = []

        # the last unfinished batch
        if docs:
            yield docs


def write_batch(output: Path, tokenized_docs: list[list[str]]):
    with open(output, 'a+', encoding='utf-8') as f:
        for doc in tokenized_docs:
            f.write(' '.join(doc) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('sources', nargs='+', help='source JSONL files')
    parser.add_argument('--batch_size', type=int, default=20000, help='loadable batch size')
    parser.add_argument('--overwrite', action='store_true', help='overwrite already tokenized files')
    args = parser.parse_args()

    for source in (pbar := tqdm.tqdm(sorted(args.sources))):
        source = Path(source).resolve()
        output = source.parent / (source.stem + '-tokenized.txt')

        pbar.set_postfix_str(source.name)

        if args.overwrite:
            output.open('w').close()  # erases all the contents
        elif output.exists():
            continue

        for batch in read_batch(source, args.batch_size):
            tokenized_batch = [spacy_tokenize(doc) for doc in batch]
            write_batch(output, tokenized_batch)
