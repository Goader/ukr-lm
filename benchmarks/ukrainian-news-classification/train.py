from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertForMaskedLM
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.utils import load_ckpt


DEFAULT_TOKENIZER_PATH = \
    Path(__file__).parent.parent.parent / 'research' / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'


def convert_model(model: BertForMaskedLM) -> BertForSequenceClassification:
    clf = AutoModelForSequenceClassification.from_config(model.config)
    clf.bert = model.bert
    return clf


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the CKPT file')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--train-data', type=str, required=True, help='path to the train data')
    parser.add_argument('--val-data', type=str, required=True, help='path to the validation data')
    parser.add_argument('--test-data', type=str, required=True, help='path to the test data')
    args = parser.parse_args()

    model = load_ckpt(args.checkpoint)
    model = convert_model(model)

    tokenizer = LibertaTokenizer(args.tokenizer)

    dataset = load_dataset(
        'csv',
        data_files={'train': args.train_data, 'validation': args.val_data, 'test': args.test_data},
    )

    def concatenate_title_and_text(example):
        return {
            'id': example['id'],
            'text': example['title'] + '. ' + example['text'],
            'label': example['source'],
        }

    dataset.map(concatenate_title_and_text)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.argmax(predictions, dim=-1)

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        return {
            'accuracy': accuracy_score(labels, predictions),
            'macro_f1': f1_score(labels, predictions, average='macro'),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='models',
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='macro_f1',
        ),
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(dataset['test'])
    trainer.save_model('model')
