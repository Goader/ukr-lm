from argparse import ArgumentParser
from pathlib import Path
import os

from transformers import AutoTokenizer

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent \
                         / 'research' / 'tokenizer' / 'experiment-6-liberta-v2' / 'spm.model'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the CKPT checkpoint')
    parser.add_argument('--tokenizer-path', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--repo-id', type=str, required=True, help='name of the repository')
    parser.add_argument('--commit-message', type=str, default='Pushing model to the hub', help='commit message')
    args = parser.parse_args()

    print('Loading the tokenizer...')

    tokenizer = LibertaTokenizer.from_pretrained(args.tokenizer_path)

    LibertaTokenizer.register_for_auto_class("AutoTokenizer")

    print('Loading the model...')

    model = load_ckpt(args.checkpoint_path)

    print('Pushing the model to the hub...')

    print(model)

    tokenizer.push_to_hub(
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        private=True,
        token=os.getenv('HF_TOKEN'),
    )

    model.push_to_hub(
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        private=True,
        token=os.getenv('HF_TOKEN'),
        safe_serialization=False,
    )
