from argparse import ArgumentParser
from pathlib import Path
import time

from datasets import Dataset, load_dataset
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex,
)


def prepare_dataset(input_path: str) -> Dataset:
    input_path = Path(input_path)
    directory = input_path.parent
    filename = input_path.name

    dataset = load_dataset(str(directory), data_files=filename, split='train')
    return dataset


def batch_iterator(dataset: Dataset, batch_size: int):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]['text']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='text file to train tokenizer')
    parser.add_argument('--output', type=str, required=True, help='output filepath')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--pre-tokenizer', type=str, default='byte-level',
                        choices=['byte-level', 'bert'], help='pre-tokenizer')
    parser.add_argument('--limit-alphabet', type=int, default=500, help='limit alphabet, if not byte-level')
    parser.add_argument('--vocab-size', type=int, default=32000, help='vocab size')
    parser.add_argument('--min-frequency', type=int, default=2, help='min frequency')
    parser.add_argument('--max-token-length', type=int, default=100, help='max token length')
    # parser.add_argument('--model-type', type=str, default='bpe', choices=['bpe', 'unigram'], help='model type')
    args = parser.parse_args()

    tokenizer = Tokenizer(model=models.BPE(dropout=0.1))

    special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']
    if args.pre_tokenizer == 'byte-level':
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            show_progress=True,
            special_tokens=special_tokens,
            end_of_word_suffix='</w>',
            max_token_length=args.max_token_length,
        )
    elif args.pre_tokenizer == 'bert':
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.BPEDecoder(suffix='</w>')
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            show_progress=True,
            special_tokens=special_tokens,
            limit_alphabet=args.limit_alphabet,
            end_of_word_suffix='</w>',
            max_token_length=args.max_token_length,
        )
    else:
        raise ValueError(f"Invalid pre-tokenizer: {args.pre_tokenizer}")

    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', 1), ('[SEP]', 2)],
    )

    t0 = time.time()
    dataset = prepare_dataset(args.input)
    tokenizer.train_from_iterator(batch_iterator(dataset, args.batch_size), trainer=trainer, length=len(dataset))
    tokenizer.save(args.output)
    print(f"Training time: {time.time() - t0:.2f} sec")

    # Test tokenizer
    tokenizer = Tokenizer.from_file(args.output)
