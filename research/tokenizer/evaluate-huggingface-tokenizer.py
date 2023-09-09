from argparse import ArgumentParser
from colorama import Fore, Style
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed
from spacy.lang.uk import Ukrainian


PURPLE = '\x1b[1;95m'
ORANGE = '\x1b[1;33m'
YELLOW = '\x1b[1;33m'
GREEN = '\x1b[1;32m'
BOLD = '\x1b[1m'

RESET = Style.RESET_ALL


TEST_SENTENCES = [
    'Чортків',
    'чорт',
    'сатана',
    'ангел',
    'скальпель',
    'русло',
    'Гальтюк',
    'Ігорович',
    'Теодозій',
    'Серет',
    'пательня',
    'калабаня',
    'санки',
    'порча',
    'вроки',
    'поршень',
    'знічев\'я',
    'п\'ятниця',
    'об\'єкт',
    'сім\'я',
    'пів\'яблука',
    'карго-культ',
    'стоїцизм',
    'антропологія',
    'макіавелізм',
    'азимут',
    'рейв',
    'чувак',
    'тачка',
    'дедлайн',
    'блекаут',
    'лате',
    'стафілокок',
    'блять',
    'сука',
    'підар',
    'ельф',
    'орк',
    'чугайстер',
    'компілятор',
    'блокчейн',
    'джаз',
    'сиквел',
    'підскажіть',
    'презедент',
    'PowerPoint',
    'Diablo',
    'Kraków',
    '— Видно…'
]


def format_pieces(pieces):
    return '  '.join(pieces)


def spacy_tokenize(text):
    return [
        token.text
        for token in spacy_tokenizer(text)
        if not token.is_space and not token.is_punct and token.text
    ]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the model file.')
    parser.add_argument('--validation-data', nargs='+', type=str, help='Path to the validation data.')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed, which ensures the reproducibility of evaluation.')
    args = parser.parse_args()

    # prepare everything for training
    set_seed(args.random_seed)

    # evaluate the model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model)

    spacy_tokenizer = Ukrainian().tokenizer

    # test sentences
    print(PURPLE + 'Test sentences:' + RESET)
    max_length = max(map(len, TEST_SENTENCES))
    pieces_counts = []
    for s in TEST_SENTENCES:
        pieces = tokenizer.tokenize(s)
        pieces_counts.append(len(pieces))
        print(f'{s:>{max_length}}    {format_pieces(pieces)}')

    print()
    print(GREEN + 'Statistics:' + RESET)
    print(f'  {ORANGE}Average number of pieces:{RESET} {sum(pieces_counts) / len(pieces_counts):.3f}')
    print(f'  {ORANGE}Vocabulary size:{RESET} {tokenizer.vocab_size}')
    print()

    # metrics
    for filepath in args.validation_data:
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            validation_data = [line.strip() for line in f if line.strip()]

        print(f'{GREEN}Validation File:{RESET} {filepath.name}')

        raw_tokenized = []
        spacy_tokenized = []
        for line in validation_data:
            raw_tokenized.append(tokenizer.tokenize(line))
            spacy_tokenized.append(spacy_tokenize(line))

        number_of_sentences = len(validation_data)
        number_of_chars = sum(map(len, validation_data))
        number_of_raw_pieces = sum(map(len, raw_tokenized))

        print(f'  {ORANGE}Number of sentences:{RESET} {number_of_sentences}')
        print(f'  {ORANGE}Number of characters:{RESET} {number_of_chars}')
        print(f'  {ORANGE}Number of raw pieces:{RESET} {number_of_raw_pieces}')
        print()

        print(f'  {ORANGE}Average number of raw pieces per sentence:{RESET} {number_of_raw_pieces / number_of_sentences:.3f}')
        print(f'  {ORANGE}Average number of characters per raw piece:{RESET} {number_of_chars / number_of_raw_pieces:.3f}')
        print()

        pieces = []

        vocabulary_hits = 0
        byte_fallbacks = 0
        unknowns = 0
        for tokenized in spacy_tokenized:
            for token in tokenized:
                tokenized_token = tokenizer.tokenize(token)
                pieces.append(tokenized_token)

                if len(tokenized_token) == 1 and tokenizer.unk_token != tokenized_token[0]:
                    vocabulary_hits += 1
                if False:  # there is no byte-level fallback in the HuggingFace tokenizers
                    byte_fallbacks += 1
                if any(piece == tokenizer.unk_token for piece in tokenized_token):
                    unknowns += 1

        number_of_tokens = sum(map(len, spacy_tokenized))
        number_of_pieces = sum(map(len, pieces))

        print(f'  {ORANGE}Number of tokens:{RESET} {number_of_tokens}')
        print(f'  {ORANGE}Number of pieces:{RESET} {number_of_pieces}')
        print(f'  {ORANGE}Average number of pieces per token:{RESET} {number_of_pieces / number_of_tokens:.3f}')
        print()

        print(f'  {ORANGE}Direct vocabulary hits:{RESET} {vocabulary_hits} - {vocabulary_hits / number_of_tokens:.2%}')
        print(f'  {ORANGE}Byte fallbacks:{RESET} {byte_fallbacks} - {byte_fallbacks / number_of_tokens:.2%}')
        print(f'  {ORANGE}Unknowns:{RESET} {unknowns} - {unknowns / number_of_tokens:.2%}')
        print()
