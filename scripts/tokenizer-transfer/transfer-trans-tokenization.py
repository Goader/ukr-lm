from argparse import ArgumentParser
from pathlib import Path
import json
import os

from transtokenizers import create_aligned_corpus, align, map_tokens, smooth_mapping, remap_model
from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch
import torch.nn as nn


# taken from trans-tokenizers and slightly modified
def remap_embeddings(source_tokenizer: str, target_tokenizer: str, mapping: list[tuple[str, list[tuple[str, float]]]], embeddings: torch.Tensor) -> torch.Tensor:
    # load tokenizers for the two models
    old_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer, trust_remote_code=True)
    new_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer, trust_remote_code=True)

    # add an unk token if none exist
    if old_tokenizer.unk_token_id is None:
        if old_tokenizer.pad_token_id is not None:
            print("WARNING: The old tokenizer had no UNK token, so we used the PAD token instead")
            old_tokenizer.unk_token_id = old_tokenizer.pad_token_id
            old_tokenizer.unk_token = old_tokenizer.pad_token
        elif old_tokenizer.bos_token_id is not None:
            print("WARNING: The old tokenizer had no UNK and PAD token, so we used the BOS token instead")
            old_tokenizer.unk_token_id = old_tokenizer.bos_token_id
            old_tokenizer.unk_token = old_tokenizer.bos_token
        else:
            print("WARNING: The old tokenizer had neither UNK, PAD or BOS special tokens")
            old_tokenizer.unk_token_id = 0

    # remap the embeddings
    print("Remapping the model...")
    with torch.no_grad():
        old_embeddings = embeddings
        new_embeddings = torch.empty(len(new_tokenizer), embeddings.shape[1])

        # for each token in the new tokenizer, find the corresponding tokens in the old tokenizer, and average their embeddings
        from tqdm import tqdm
        from functools import reduce

        for new_id in tqdm(range(len(new_tokenizer))):

            old_tokens = mapping[new_id][1]  # list of (ids,weight) in the old tokenizer

            # sort the list such that the smallest weights come first (for numerical stability)
            old_tokens = sorted(old_tokens, key=lambda x: x[1])

            # map tokens to their ids
            old_ids = [(old_tokenizer.convert_tokens_to_ids(old_token), weight) for old_token, weight in old_tokens]
            old_ids = [(old_id if old_id is not None else old_tokenizer.unk_token_id, weight) for old_id, weight in old_ids]

            # we use a weighted average, where the first token in the list has 0.4 weight, the second 0.2, and the remaining 0.4 are equally distributed among all tokens (including the first two)
            if len(old_ids) > 1:
                new_embeddings[new_id] = reduce(lambda a, b: a.add_(b), [old_embeddings[old_id]*weight for old_id, weight in old_ids])
            elif len(old_ids) == 1:
                new_embeddings[new_id] = old_embeddings[old_ids[0][0]]
            # use the unknown token embedding if the token is not in the old tokenizer
            else:
                new_embeddings[new_id] = old_embeddings[old_tokenizer.unk_token_id]
    
    # check if all tokens have been initialized
    uninitialized = torch.isnan(new_embeddings).any(dim=1)
    if uninitialized.any():
        print(f"WARNING: {uninitialized.sum().item()} tokens were not initialized and will use UNK embedding")
        new_embeddings[uninitialized] = old_embeddings[old_tokenizer.unk_token_id]

    return new_embeddings


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--source_model_or_path', type=str, required=True, help='path or name of the source model')
    parser.add_argument('--source_tokenizer', type=str, required=True, help='path or name of the source tokenizer')
    parser.add_argument('--target_tokenizer', type=str, required=True, help='path or name of the target tokenizer')
    parser.add_argument('--export_dir', type=str, required=True, help='path to the export directory')
    args = parser.parse_args()

    if 'TT_HOME' not in os.environ:
        print('WARNING: TT_HOME is not set, using current directory')

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    corpus = create_aligned_corpus(
        source_language='en',
        target_language='uk',
        source_tokenizer=args.source_tokenizer,
        target_tokenizer=args.target_tokenizer,
    )

    mapped_tokens_file = align(corpus, fast_align_path='fast_align')

    tokenized_possible_translations, untokenized_possible_translations = map_tokens(
        mapped_tokens_file=mapped_tokens_file,
        source_tokenizer=args.source_tokenizer,
        target_tokenizer=args.target_tokenizer
    )

    smoothed_mapping = smooth_mapping(args.target_tokenizer, tokenized_possible_translations)

    with open(export_dir / 'smoothed_mapping.json', 'w') as f:
        json.dump(smoothed_mapping, f)

    if not Path(args.source_model_or_path).exists():
        model_defined = True
        model = AutoModelForMaskedLM.from_pretrained(args.source_model_or_path, trust_remote_code=True)
        source_embeddings = model.get_input_embeddings().weight.data
    else:
        model_defined = False
        source_embeddings = torch.load(args.source_model_or_path)

    new_input_embeddings = remap_embeddings(
        source_tokenizer=args.source_tokenizer,
        target_tokenizer=args.target_tokenizer,
        mapping=smoothed_mapping,
        embeddings=source_embeddings,
    )

    # Save input and output embeddings using checkpoint format
    checkpoint = {
        'input_embeddings': new_input_embeddings,
    }

    if model_defined and model.get_output_embeddings() is not None:
        new_output_embeddings = remap_embeddings(
            source_tokenizer=args.source_tokenizer,
            target_tokenizer=args.target_tokenizer,
            mapping=smoothed_mapping,
            embeddings=model.get_output_embeddings().weight.data
        )
        checkpoint['output_embeddings'] = new_output_embeddings
    
    torch.save(checkpoint, export_dir / 'embeddings.ckpt')
