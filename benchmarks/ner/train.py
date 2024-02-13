from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers.models.bert.modeling_bert import BertForTokenClassification, BertForMaskedLM
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from sklearn.metrics import f1_score, accuracy_score

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = \
    Path(__file__).parent.parent.parent / 'research' / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'


def load_huggingface_dataset(dataset_name: str, cache_dir: Optional[str] = None) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    if dataset_name == 'wikiann':
        dataset = load_dataset('wikiann', 'uk', cache_dir=cache_dir)
    elif dataset_name == 'ner-uk':
        dataset = load_dataset('benjamin/ner-uk', cache_dir=cache_dir)
    else:
        raise ValueError(f'unknown dataset for this script - {dataset_name}')
    return dataset


def convert_model(model: BertForMaskedLM, num_labels: int) -> BertForTokenClassification:
    model.config.num_labels = num_labels
    clf = AutoModelForTokenClassification.from_config(model.config)
    clf.bert.embeddings.load_state_dict(model.bert.embeddings.state_dict())
    clf.bert.encoder.load_state_dict(model.bert.encoder.state_dict())
    return clf


def collect_word_ids(seq: list[str], tokenizer) -> list[int]:
    word_ids = [None]  # CLS
    tokenized = tokenizer(seq)
    for i, word_tokenization in enumerate(tokenized['input_ids']):
        subtokens = len(word_tokenization) - 2  # minus CLS and SEP
        word_ids.extend([i] * subtokens)
    word_ids.append(None)  # SEP
    return word_ids


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the CKPT file')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--dataset', type=str, choices=['wikiann', 'ner-uk'],
                        required=True, help='name of the dataset to train on')
    args = parser.parse_args()

    dataset = load_huggingface_dataset(args.dataset)
    label_names = dataset['train'].features['ner_tags'].feature.names

    model = load_ckpt(args.checkpoint)
    model = convert_model(model, num_labels=dataset['train'].features['ner_tags'].feature.num_classes)

    tokenizer = LibertaTokenizer(args.tokenizer)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding='longest',
            max_length=512,
            return_tensors='pt'
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, (seq, labels) in enumerate(zip(examples['tokens'], all_labels)):
            word_ids = collect_word_ids(seq, tokenizer)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    dataset = dataset.map(tokenize_and_align_labels, batched=True, batch_size=256)
    collator = DataCollatorForTokenClassification(tokenizer, padding='longest')
    metric = evaluate.load('seqeval')

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return all_metrics

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
            metric_for_best_model='overall_f1',
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
