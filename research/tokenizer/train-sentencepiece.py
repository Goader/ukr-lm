from argparse import ArgumentParser
import time

import sentencepiece as spm


def load_symbols(filepath: str) -> list[str]:
    with open(filepath, 'r') as f:
        symbols = f.read().splitlines()
    return [s for s in symbols if s]


if __name__ == '__main__':
    parser = ArgumentParser(
        description='TODO. See https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto'
    )
    parser.add_argument('--input', type=str, required=True, help='Input filepath to the raw text file.')
    parser.add_argument('--model-prefix', type=str, required=True,
                        help='Output filepath prefix for the trained SentencePiece model. May include a whole path as '
                             'a prefix. For example, "path/to/model" will create files "path/to/model.model" and '
                             '"path/to/model.vocab".')
    parser.add_argument('--model-type', type=str, default='bpe', choices=['bpe', 'unigram'],
                        help='Tokenizer model type')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Output vocabulary size.')
    parser.add_argument('--self-test-sample-size', type=int, default=0,
                        help='Enables self-testing mode, where a small sample of data is used '
                             'to verify the correctness of the trained model.')
    parser.add_argument('--character-coverage', type=float, default=0.9995,
                        help='Specifies the desired character coverage for training. For example, '
                             'setting it to 0.9995 ensures that at least 99.95%% of the input text '
                             'characters (all, not unique) are covered by the model\'s vocabulary.')
    parser.add_argument('--input-sentence-size', type=int, default=10_000_000,
                        help='Specifies the maximum number of sentences to use for training. '
                             'Useful for large corpora to limit memory usage during training.')
    parser.add_argument('--shuffle-input-sentence', type=bool, default=True,
                        help='Randomly sample input sentences in advance, '
                             'this option is effective only if input_sentence_size > 0.')
    parser.add_argument('--seed-sentencepiece-size', type=int, default=1000000,
                        help='The seed SentencePiece model is a pre-trained model used as the initial vocabulary '
                             'for the unsupervised training process.')
    parser.add_argument('--shrinking-factor', type=float, default=0.75,
                        help='Controls the shrinking factor applied during vocabulary reduction.'
                             'Smaller values preserve more tokens in the vocabulary.')
    parser.add_argument('--num-threads', type=int, default=16,
                        help='Specifies the number of threads to use during BPE merge '
                             'operations for BPE model training.')
    parser.add_argument('--num-sub-iterations', type=int, default=2,
                        help='Specifies the number of EM sub-iterations to use during '
                             'BPE merge operations for BPE model training.')
    parser.add_argument('--max-sentencepiece-length', type=int, default=16,
                        help='Specifies the maximum length of sentence pieces. Measured in Unicode characters.')
    parser.add_argument('--max-sentence-length', type=int, default=10000000,
                        help='Specifies the maximum length of sentences to use for training. Measured in bytes.')

    # pre-tokenization options
    parser.add_argument('--split-by-unicode-script', type=bool, default=True,
                        help='This option splits input text into tokens based on Unicode script boundaries. '
                             'For example, the text "HelloСвіт" would be pre-tokenized as ["Hello", "Світ"] '
                             'because the Cyrillic script is different from the Latin script.')
    parser.add_argument('--split-by-whitespace', type=bool, default=True,
                        help='This option splits input text into tokens based on whitespace characters '
                             '(e.g., spaces, tabs, newlines). It treats each word as a token and is a common choice '
                             'for languages where words are typically separated by spaces.')
    parser.add_argument('--split-by-number', type=bool, default=True,
                        help='When this option is used, SentencePiece will split the input text into tokens at the '
                             'boundaries between numbers and non-numbers. For example, in the text "abc123def", '
                             'SentencePiece would pre-tokenize it as ["abc", "123", "def"].')
    parser.add_argument('--split-digits', type=bool, default=False,
                        help='This option splits input text into tokens at digit boundaries. For example, the text '
                             '"abc123def" would be pre-tokenized as ["abc", "1", "2", "3", "def"].')
    parser.add_argument('--treat-whitespace-as-suffix', type=bool, default=False,
                        help='This option, when used with split_by_whitespace, treats whitespace characters as '
                             'suffixes rather than prefixes. For example, the text "Hello World" would be '
                             'pre-tokenized as ["Hello ", "World"].')
    parser.add_argument('--allow-whitespace-only-pieces', type=bool, default=False,
                        help='This option, when used with split_by_whitespace, allows whitespace characters to '
                             'be treated as tokens. For example, the text "Hello World" would be pre-tokenized as '
                             '["Hello", " ", "World"].')

    # special tokens and rules
    parser.add_argument('--control-symbols', type=str, default='<cls>,<sep>,<mask>',
                        help='Special tokens that are used for control purposes, like BERT\'s [CLS] and [SEP]. '
                             'We only reserve ids for these tokens. Even if these tokens appear in the input text, '
                             'they are not handled as one token. User needs to insert ids explicitly after encoding.')
    parser.add_argument('--control-symbols-file', type=str, default=None,
                        help='Load control symbols from file, https://github.com/google/sentencepiece/issues/536')
    parser.add_argument('--user-defined-symbols', type=str, default=None,
                        help='Defines some custom pieces, which will get replaced during the training with special '
                             'token boundary, so that this pieces will never get split. When "@" is specified as user '
                             'defined symbol, "@" is always treated as one piece, meaning that no piece containing '
                             '"@" inside will not be extracted. Probably the name of "symbol" is misleading. The '
                             'intention is user-defined-piece.')
    parser.add_argument('--user-defined-symbols-file', type=str, default=None,
                        help='Load user defined symbols from file, https://github.com/google/sentencepiece/issues/536')
    parser.add_argument('--required-chars', type=str, default=None,
                        help='This option allows you to specify a list of characters that must be included in the '
                             'vocabulary. If a character is not included in the vocabulary, it will be treated as '
                             'an unknown character. This option is useful for languages that use a large number of '
                             'characters, such as Chinese, Japanese, and Korean.')
    parser.add_argument('--required-chars-file', type=str, default=None,
                        help='Load required chars from file, https://github.com/google/sentencepiece/issues/536')

    # normalization options
    parser.add_argument('--byte-fallback', type=bool, default=False,
                        help='This option allows you to specify whether SentencePiece should fall back to byte-level '
                             'tokenization when the input text contains characters that are not included in the '
                             'vocabulary. If this option is set to true, SentencePiece will fall back to byte-level '
                             'tokenization for unknown characters. If this option is set to false, SentencePiece '
                             'will treat unknown characters as unknown tokens.')
    parser.add_argument('--vocabulary-output-piece-score', type=bool, default=True,
                        help='This option allows you to specify whether SentencePiece should output the score of each '
                             'token in the vocabulary. The score is a floating point number between 0 and 1 that '
                             'indicates the probability of the token appearing in the input text.')
    parser.add_argument('--normalization-rule-name', type=str, default='identity',
                        help='This option allows you to specify a normalization rule name. SentencePiece provides '
                             'several normalization rules, such as nfkc, nmt_nfkc_cf, and identity. '
                             'See what each normalization includes.')
    parser.add_argument('--normalization-rule-tsv', type=str, default=None, help='Load normalization rule tsv')
    parser.add_argument('--denormalization-rule-tsv', type=str, default=None, help='Load denormalization rule tsv')
    parser.add_argument('--add-dummy-prefix', type=bool, default=True,
                        help='This option allows you to specify whether SentencePiece should add a dummy prefix '
                             'at the beginning of the input text.')
    parser.add_argument('--remove-extra-whitespaces', type=bool, default=True,
                        help='Extra whitespaces are removed from the output text. For specific purposes, '
                             'such as code tokenization, you may want to disable this option.')
    parser.add_argument('--hard-vocab-limit', type=int, default=False,
                        help='Enables hard vocabulary limit. When the vocabulary size reaches the limit, '
                             'the least frequent tokens are dropped to satisfy the size constraint. '
                             'If you prioritize capturing important subword units and are willing to allow a slightly '
                             'larger vocabulary size to accommodate them, you can use set it to False.')

    # special tokens representation
    parser.add_argument('--unk-piece', type=str, default='<unk>',
                        help='UNK piece. This is used to specify the unknown token in the text.')
    parser.add_argument('--bos-piece', type=str, default='<s>',
                        help='BOS piece. This is used for decoding to specify the beginning of the output text.')
    parser.add_argument('--eos-piece', type=str, default='</s>',
                        help='EOS piece. This is used for decoding to specify the end of the output text.')
    parser.add_argument('--pad-piece', type=str, default='<pad>',
                        help='PAD piece. This is used to pad sequences to the common length for faster processing.')
    parser.add_argument('--unk-id', type=int, default=1,
                        help='The id of the unknown token. If -1, it is disabled.')
    parser.add_argument('--bos-id', type=int, default=-1,
                        help='The id of the beginning of sentence token. If -1, it is disabled.')
    parser.add_argument('--eos-id', type=int, default=-1,
                        help='The id of the end of sentence token. If -1, it is disabled.')
    parser.add_argument('--pad-id', type=int, default=0,
                        help='The id of the padding token. If -1, it is disabled.')
    parser.add_argument('--unk-surface', type=str, default='<unk>',
                        help='The surface form of the unknown token - used for decoding to specify the '
                             'unknown token in the output text.')

    # TODO add Unigram parameters

    # training process
    parser.add_argument('--train-extremely-large-corpus', type=bool, default=True,
                        help='Enables a mode for training on extremely large corpora '
                             'by adjusting internal buffer sizes.')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed, which ensures the reproducibility of training.')
    parser.add_argument('--minloglevel', type=int, default=0,
                        help='Specifies the minimum level of logging messages to display. '
                             'Values are 0 (display INFO messages), 1 (display WARNING messages), '
                             '2 (display ERROR messages).')
    args = parser.parse_args()

    # prepare everything for training
    spm.set_random_generator_seed(args.random_seed)

    EXCLUDED_ARGS = ['random_seed', 'control_symbols_file', 'user_defined_symbols_file', 'required_chars_file']
    kwargs = {
        k: v for k, v in vars(args).items()
        if k not in EXCLUDED_ARGS and v is not None
    }

    if args.control_symbols_file is not None:
        if args.control_symbols is not None:
            print('WARNING: control_symbols_file and control_symbols are both specified. '
                  'control_symbols_file will be used.')
        kwargs['control_symbols'] = load_symbols(args.control_symbols_file)

    if args.user_defined_symbols_file is not None:
        if args.user_defined_symbols is not None:
            print('WARNING: user_defined_symbols_file and user_defined_symbols are both specified. '
                  'user_defined_symbols_file will be used.')
        kwargs['user_defined_symbols'] = load_symbols(args.user_defined_symbols_file)

    if args.required_chars_file is not None:
        if args.required_chars is not None:
            print('WARNING: required_chars_file and required_chars are both specified. '
                  'required_chars_file will be used.')
        kwargs['required_chars'] = load_symbols(args.required_chars_file)

    # training process
    t1 = time.time()
    spm.SentencePieceTrainer.Train(**kwargs)
    print(f"Training time: {time.time() - t1:.2f} sec")

    # evaluate the model
    # TODO
