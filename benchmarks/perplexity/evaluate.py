from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Iterator
import tqdm
import os

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    set_seed,
)
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from torch.utils.data import DataLoader
from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.collators import DataCollatorForNgramMasking
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = \
    Path(__file__).parent.parent.parent / 'research' / 'tokenizer' / 'experiment-6-liberta-v2' / 'spm.model'


def load_huggingface_dataset(dataset_name: str, cache_dir: Optional[str] = None) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    if dataset_name == 'treebank':
        dataset = load_dataset('Goader/ukrainian-treebank-lm', 'document', split='all', cache_dir=cache_dir)
    elif dataset_name == 'spivavtor':
        dataset = load_dataset('grammarly/spivavtor', split='validation', cache_dir=cache_dir)
        dataset = dataset.rename_column('tgt', 'text')
    elif dataset_name == 'wikipedia':
        dataset = load_dataset('wikimedia/wikipedia', '20233101.uk', split='train', cache_dir=cache_dir)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return dataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the CKPT file or HuggingFace checkpoint')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--dataset', type=str, required=True, help='name of the dataset to train on')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training and evaluation')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    args = parser.parse_args()

    print(f'Evaluating on {args.dataset}...')

    dataset = load_huggingface_dataset(args.dataset)

    if args.checkpoint.endswith('.ckpt'):
        model = load_ckpt(args.checkpoint)
        tokenizer = LibertaTokenizer(args.tokenizer)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            args.checkpoint,
            token=os.getenv('HF_TOKEN', None),
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            add_prefix_space=True,
            trust_remote_code=True,
        )
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.seed is not None:
        set_seed(args.seed)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)

    # Create masked dataset using batched processing
    masked_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Creating masked examples"
    )

    collator = DataCollatorForNgramMasking(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        keep_original_probability=0.0,
        substitute_probability=0.0,
        max_ngram_size=1,
    )

    dataloader = DataLoader(
        masked_dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
    )

    total_loss = 0
    total_correct = 0
    total_predictions = 0

    for batch in tqdm.tqdm(dataloader, desc='Evaluating'):
        # Stack the tensors into batches
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # Compute metrics only on masked tokens
        masked_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        predictions = outputs.logits.argmax(dim=-1)
        
        # Update metrics
        total_loss += outputs.loss.item()
        total_correct += (predictions[masked_positions[0], masked_positions[1]] == labels[masked_positions[0], masked_positions[1]]).sum().item()
        total_predictions += len(masked_positions[0])

    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    mlm_accuracy = total_correct / total_predictions

    print(f"Perplexity: {perplexity:.2f}")
    print(f"MLM Accuracy: {mlm_accuracy:.2%}")
