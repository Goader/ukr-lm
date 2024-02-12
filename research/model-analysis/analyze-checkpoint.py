from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from prettytable import PrettyTable

from ukrlm.tokenizers import LibertaTokenizer


DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the CKPT file')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--dont-analyze-weights', action='store_true', help='do not analyze weights')
    parser.add_argument('--dont-analyze-output', action='store_true', help='do not analyze model output')
    parser.add_argument('--analyze-with-masks', action='store_true', help='analyze with masks')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = {key.removeprefix('model.'): value for key, value in ckpt['state_dict'].items()}

    config = ckpt['config']

    model = AutoModelForMaskedLM.from_config(config)
    model.load_state_dict(state_dict)
    model.eval()

    # analyzing weights
    if not args.dont_analyze_weights:
        print('Analyzing weights')
        table = PrettyTable(['Max Abs', 'Mean Abs', 'Frobenius Norm', 'Size', 'Name'])
        for name, params in state_dict.items():
            max_abs = torch.max(torch.abs(params))
            mean_abs = torch.mean(torch.abs(params))
            frobenius_norm = torch.norm(params)
            size_str = str(tuple(params.size()))

            table.add_row([f'{max_abs:.4f}', f'{mean_abs:.4f}', f'{frobenius_norm:.4f}', size_str, name])

        print(table)
        print('\n')

        print('Classification Layer Weights')
        with torch.no_grad():
            print('Top 5 Max Norm', torch.topk(torch.norm(model.cls.predictions.decoder.weight, dim=-1), k=5), sep='\n')
        print('\n')

    # analyzing model output
    if not args.dont_analyze_output:
        print('Analyzing model output')
        # TODO substitue with AutoTokenizer
        tokenizer = LibertaTokenizer.from_pretrained(args.tokenizer)
        input_strings = [
            'Невже в тебе ніколи не виникало жагучого бажання знати, що ж відбувається з твоїм тілом, коли ти спиш?',
        ]

        for input_string in input_strings:
            print(input_string)
            model_input = tokenizer(input_string, return_tensors='pt', max_length=512, truncation=True)

            with torch.no_grad():
                model_output = model(**model_input)

            table = PrettyTable(['Token', 'Token ID', 'Predicted Token', 'Predicted Token ID',
                                 'Max Abs', 'Mean Abs', 'Frobenius Norm'])

            for token, token_id, predicted_token_id, logits in zip(
                model_input['input_ids'][0], model_input['input_ids'][0],
                torch.argmax(model_output.logits[0], dim=-1), model_output.logits[0]
            ):
                max_abs = torch.max(torch.abs(logits))
                mean_abs = torch.mean(torch.abs(logits))
                frobenius_norm = torch.norm(logits)

                table.add_row([tokenizer.decode(token), int(token_id), tokenizer.decode(predicted_token_id),
                               int(predicted_token_id), f'{max_abs:.4f}', f'{mean_abs:.4f}', f'{frobenius_norm:.4f}'])

            print(table)
