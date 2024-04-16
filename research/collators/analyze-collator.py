from argparse import ArgumentParser
from pathlib import Path
from time import time

import torch
from transformers import DataCollatorForWholeWordMask

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.collators import DataCollatorForNgramMasking
from ukrlm.datamodules.masked_language_modeling import tokenize_function
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'


def print_sequence(sequence, tokenizer):
    for idx, token_id in enumerate(sequence):
        token = tokenizer.decode(token_id)
        print(token, end=' // ')
    print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--collator', type=str, required=True, help='name of the collator')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--measure-time', action='store_true', help='measure the time of the collation')
    args = parser.parse_args()

    print('Loading the tokenizer')
    tokenizer = LibertaTokenizer.from_pretrained(args.tokenizer)

    if args.collator == 'ngram':
        collator = DataCollatorForNgramMasking(tokenizer)
    elif args.collator == 'whole-word-hf':
        collator = DataCollatorForWholeWordMask(tokenizer)

    queries = [
        'Прикладове навчання — це підхід до машинного навчання, який дозволяє вирішувати практичні задачі, не будуючи модель з нуля.',
        'Машинне навчання — це галузь штучного інтелекту, яка досліджує методи побудови систем, які можуть навчатися на основі емпіричних даних.',
        'Автономне навчання — це підхід до машинного навчання, який дозволяє моделі навчатися без участі людини.',
        'Асталафармоподібне страндоблексофобічне ефеметрічне віджетофілійне офорломобуванання.',
    ]

    inputs = []
    for query in queries:
        inputs.append(tokenizer(query, return_special_tokens_mask=True))

    print()
    print('Input sequences:')
    for sequence in inputs:
        print_sequence(sequence['input_ids'], tokenizer)

    if args.measure_time:
        print('Measuring the time of the collation (100 iterations)')
        start_time = time()
        for _ in range(100):
            collator_output = collator(inputs)
        end_time = time()
        print(f'Time: {end_time - start_time:.3f} s')
    else:
        collator_output = collator(inputs)

    print()
    print('Collator output:')
    for sequence in collator_output['input_ids']:
        print_sequence(sequence, tokenizer)
