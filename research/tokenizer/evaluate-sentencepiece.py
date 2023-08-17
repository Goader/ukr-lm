from argparse import ArgumentParser
import colorama
from colorama import Fore, Style

import sentencepiece as spm


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
]


def format_pieces(pieces):
    return '  '.join(pieces)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the model file.')
    parser.add_argument('--validation-data', type=str, help='Path to the validation data.')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed, which ensures the reproducibility of evaluation.')
    args = parser.parse_args()

    # prepare everything for training
    spm.set_random_generator_seed(args.random_seed)

    # evaluate the model
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    # test sentences
    print(Fore.GREEN + 'Test sentences:' + Style.RESET_ALL)
    max_length = max(map(len, TEST_SENTENCES))
    pieces_counts = []
    for s in TEST_SENTENCES:
        pieces = sp.EncodeAsPieces(s)
        pieces_counts.append(len(pieces))
        print(f'{s:>{max_length}}    {format_pieces(pieces)}')

    print()
    print(Fore.GREEN + 'Statistics:' + Style.RESET_ALL)
    print(f'Average number of pieces: {sum(pieces_counts) / len(pieces_counts):.3f}')
    print()

    # metrics
    with open(args.validation_data, 'r') as f:
        validation_data = f.readlines()

    tokenized = []
    for line in validation_data:
        tokenized.append(sp.EncodeAsPieces(line))

    print(Fore.GREEN + 'Metrics:' + Style.RESET_ALL)
    print(f'Vocabulary size: {sp.GetPieceSize()}')
    print(f'Number of sentences: {len(validation_data)}')
    print(f'Number of tokens: {sum(map(len, validation_data))}')
    print(f'Number of pieces: {sum(map(len, tokenized))}')
    print(f'Average number of pieces per token: {sum(map(len, tokenized)) / sum(map(len, validation_data)):.3f}')



