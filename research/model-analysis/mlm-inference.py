from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from prettytable import PrettyTable

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the CKPT file')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    args = parser.parse_args()

    print('Loading the model and tokenizer')

    model = load_ckpt(args.checkpoint)
    tokenizer = LibertaTokenizer.from_pretrained(args.tokenizer)

    print('Query the model with sequences. Type "exit" to quit.')

    input_sequence = input('\nEnter a sequence: ')
    while input_sequence != 'exit':
        inputs = tokenizer(input_sequence, return_tensors='pt')
        outputs = model(**inputs)

        # decode the output
        logits = outputs.logits
        predicted_token_ids = torch.topk(logits, 5, dim=-1).indices[0]
        print(predicted_token_ids.size())

        # print the whole sequence with the masked tokens replaced with the top 5 predictions
        for idx, input_id in enumerate(inputs['input_ids'][0]):
            if input_id == tokenizer.mask_token_id:
                print('MASK:', end=' ')
                for token_id in predicted_token_ids[idx]:
                    token = tokenizer.decode(token_id)
                    print(token, end=' ')
                print()
            else:
                token = tokenizer.decode(input_id)
                print(token)

        input_sequence = input('\nEnter a sequence: ')
