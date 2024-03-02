from argparse import ArgumentParser
from pathlib import Path
import tqdm
import csv

import torch
import numpy as np
from transformers import set_seed, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertForMaskedLM
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = \
    Path(__file__).parent.parent.parent / 'research' / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the model dir')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--data', type=str, required=True, help='path to the inference data (CSV file)')
    parser.add_argument('--output', type=str, required=True, help='path to the output CSV file')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    if Path(args.tokenizer).exists():
        tokenizer = LibertaTokenizer(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = load_dataset(
        'csv',
        data_files=args.data,
        split='train'
    )

    def concatenate_title_and_text(example):
        return {
            'id': example['Id'],
            'text': example['title'] + '. ' + example['text'],
        }

    dataset = dataset.map(concatenate_title_and_text)

    def tokenize(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

    dataset = dataset.map(tokenize, batched=True, batch_size=256)
    dataset = dataset.remove_columns(['Id', 'title', 'text'])
    collator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=512)

    model.eval()
    with open(args.output, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'Predicted'])
        for input in tqdm.tqdm(dataset.iter(
                batch_size=args.batch_size,
                drop_last_batch=False
        ), total=len(dataset) // args.batch_size):

            document_ids = input['id']
            input = collator({
                'input_ids': input['input_ids'],
                'attention_mask': input['attention_mask']
            })

            input_ids = input['input_ids'].to(device)
            attention_mask = input['attention_mask'].to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            predicted_class_ids = logits.argmax(dim=-1).detach().cpu().numpy()

            for document_id, predicted_class_id in zip(document_ids, predicted_class_ids):
                writer.writerow({'Id': document_id, 'Predicted': predicted_class_id})

