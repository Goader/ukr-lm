from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import os

import torch
import evaluate
import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    set_seed,
)
from transformers.models.bert.modeling_bert import BertForTokenClassification, BertForMaskedLM
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from sklearn.metrics import f1_score, accuracy_score

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = \
    Path(__file__).parent.parent.parent / 'research' / 'tokenizer' / 'experiment-6-liberta-v2' / 'spm.model'


def load_huggingface_dataset(dataset_name: str, cache_dir: Optional[str] = None) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    if dataset_name == 'wikiann':
        dataset = load_dataset('wikiann', 'uk', cache_dir=cache_dir)
    elif dataset_name == 'ner-uk':
        dataset = load_dataset('benjamin/ner-uk', cache_dir=cache_dir)
    elif dataset_name == 'ner-uk-2.0':
        dataset = load_dataset('Goader/ner-uk-2.0', cache_dir=cache_dir)
    elif dataset_name == 'universal-dependencies':
        dataset = load_dataset('universal_dependencies', 'uk_iu', cache_dir=cache_dir)
        dataset = dataset.rename_column('upos', 'pos_tags')
    else:
        raise ValueError(f'unknown dataset for this script - {dataset_name}')
    return dataset


def convert_model(model: BertForMaskedLM, num_labels: int, finetuning_task: str) -> BertForTokenClassification:
    model.config.num_labels = num_labels
    model.config.finetuning_task = finetuning_task
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


def align_labels_with_tokens(labels, word_ids, task='ner'):
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
            if task == 'ner':
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
            else:  # task == 'pos'
                new_labels.append(-100)

    return new_labels


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the CKPT file or HuggingFace checkpoint')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--dataset', type=str, choices=['wikiann', 'ner-uk', 'universal-dependencies', 'ner-uk-2.0'],
                        required=True, help='name of the dataset to train on')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training and evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='batch size for evaluation only')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate for the optimizer')
    parser.add_argument('--scheduler_type', type=str, default='linear', help='type of the learning rate scheduler')
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help='warmup ratio for the learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for the optimizer')
    parser.add_argument('--load_best_model', action='store_true', help='load the best model at the end of training')
    parser.add_argument('--repeat_reproducibility', action='store_true', help='repeat the weights initialization '
                                                                              'for reproducibility')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    dataset = load_huggingface_dataset(args.dataset)
    label_column_name = 'ner_tags' if args.dataset in ['wikiann', 'ner-uk', 'ner-uk-2.0'] else 'pos_tags'
    label_names = dataset['train'].features[label_column_name].feature.names
    finetuning_task = 'pos' if args.dataset in ['universal-dependencies'] else 'ner'

    if args.checkpoint.endswith('.ckpt'):
        model = load_ckpt(args.checkpoint)
        model = convert_model(
            model,
            num_labels=len(label_names),
            finetuning_task=finetuning_task
        )
        tokenizer = LibertaTokenizer(args.tokenizer)
    elif args.repeat_reproducibility:
        # this should not cause the random state to change
        real_model = AutoModelForMaskedLM.from_pretrained(
            args.checkpoint,
            token=os.getenv('HF_TOKEN', None),
            trust_remote_code=True,
        )
        config = AutoConfig.from_pretrained(args.checkpoint)

        # setting the random generator state to be the same, as loading from checkpoint
        AutoModelForMaskedLM.from_config(config)
        model = convert_model(
            real_model,
            num_labels=len(label_names),
            finetuning_task=finetuning_task,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            args.checkpoint,
            num_labels=len(label_names),
            token=os.getenv('HF_TOKEN', None),
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            add_prefix_space=True,
            trust_remote_code=True,
        )

    print(label_names)
    print(dataset['train'].features[label_column_name].feature.num_classes)
    print(model)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding='longest',
            max_length=512,
            return_tensors='pt'
        )
        all_labels = examples[label_column_name]
        new_labels = []
        for i, (seq, labels) in enumerate(zip(examples['tokens'], all_labels)):
            word_ids = collect_word_ids(seq, tokenizer)
            new_labels.append(align_labels_with_tokens(labels, word_ids, task=finetuning_task))

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

    main_metric = 'overall_accuracy' if args.dataset in ['universal-dependencies'] else 'overall_f1'

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f'models/{args.dataset}-seed{args.seed}',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            load_best_model_at_end=args.load_best_model,
            metric_for_best_model=main_metric,
            lr_scheduler_type=args.scheduler_type,
            warmup_ratio=args.warmup_ratio,
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
