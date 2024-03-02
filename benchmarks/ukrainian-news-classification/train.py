from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
from transformers import (
    set_seed,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertForMaskedLM
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = \
    Path(__file__).parent.parent.parent / 'research' / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'

NUM_LABELS = 7


def convert_model(model: BertForMaskedLM) -> BertForSequenceClassification:
    model.config.num_labels = NUM_LABELS
    clf = AutoModelForSequenceClassification.from_config(model.config)
    clf.bert.embeddings.load_state_dict(model.bert.embeddings.state_dict())
    clf.bert.encoder.load_state_dict(model.bert.encoder.state_dict())
    return clf


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the CKPT file or HuggingFace checkpoint')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--train-data', type=str, required=True, help='path to the train data')
    parser.add_argument('--val-data', type=str, required=True, help='path to the validation data')
    parser.add_argument('--test-data', type=str, required=True, help='path to the test data')
    parser.add_argument('--output-best-model', type=str, default='models/best', help='path to the best model')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if args.checkpoint.endswith('.ckpt'):
        model = load_ckpt(args.checkpoint)
        model = convert_model(model)
        tokenizer = LibertaTokenizer(args.tokenizer)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint,
            num_labels=NUM_LABELS,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = load_dataset(
        'csv',
        data_files={
            'train': args.train_data,
            'validation': args.val_data,
            'test': args.test_data
        },
    )

    def concatenate_title_and_text(example):
        return {
            'id': example['Id'],
            'text': example['title'] + '. ' + example['text'],
            'label': example['source'],
        }

    dataset = dataset.map(concatenate_title_and_text)

    def tokenize(batch):
        return tokenizer(
            batch['text'],
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ) | {'label': batch['label']}

    dataset = dataset.map(tokenize, batched=True, batch_size=256)
    collator = DataCollatorWithPadding(tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            'accuracy': accuracy_score(labels, predictions),
            'macro_f1': f1_score(labels, predictions, average='macro'),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='models',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='macro_f1',
        ),
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    trainer.train()

    print('Evaluating on test dataset')
    trainer.evaluate(dataset['test'])
    trainer.save_model(args.output_best_model)
