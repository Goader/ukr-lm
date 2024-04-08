from argparse import ArgumentParser
from pathlib import Path
import time
import tqdm

from datasets import load_from_disk, load_dataset
from transformers import DataCollatorForLanguageModeling

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ukrlm.tokenizers import LibertaTokenizer
from ukrlm.datamodules.masked_language_modeling import tokenize_function


DEFAULT_TOKENIZER_PATH = Path(__file__).parent.parent / 'tokenizer' / 'experiment-5-overall-v2' / 'spm.model'


class MockModel(nn.Module):
    def __init__(self, vocabulary_size: int = 32000, heaviness: int = 200):
        super().__init__()
        self.embeddings = nn.Embedding(vocabulary_size, 1000)
        self.linear = nn.Linear(1000, 1000)
        self.relu = nn.ReLU()
        self.cls = nn.Linear(1000, 4)
        self.heaviness = heaviness

    def forward(self, x):
        x = self.embeddings(x)

        # simulating heavy model
        for _ in range(self.heaviness):
            x = self.linear(x)
            x = self.relu(x)
        x = self.cls(x)
        return x


class Trainer:
    def __init__(self, model):
        self.model = model.to(torch.device('cuda'))
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, dataloader):
        self.model.train()
        for data in tqdm.tqdm(dataloader):
            input_ids = data['input_ids']
            input_ids = input_ids.to(torch.device('cuda')).view(-1)
            target = torch.randint(0, 4, (input_ids.size(0),)).to(torch.device('cuda'))

            self.optimizer.zero_grad()
            output = self.model(input_ids)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataloader_strategy', type=str, help='name of the strategy')
    parser.add_argument('--tokenizer_path', type=str, default=DEFAULT_TOKENIZER_PATH, help='path to the tokenizer')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--heaviness', type=int, default=10)
    parser.add_argument('--vocabulary_size', type=int, default=32000)
    args = parser.parse_args()

    tokenizer = LibertaTokenizer.from_pretrained(args.tokenizer_path)
    model = MockModel(vocabulary_size=args.vocabulary_size, heaviness=args.heaviness)
    trainer = Trainer(model)

    # heaviness = 1 // ~6.1 it/s // 240 shards // 64 batch size // 8 workers // 456.16 seconds
    if args.dataloader_strategy == 'tokenized-arrow-input-ids':
        """
        Already tokenized version of the dataset with `input_ids` and `id` columns only.
        The number of shards is 240.
        """
        # FIXME path
        dataset = load_from_disk('/media/goader/masters/data-loading/output')
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

        t0 = time.perf_counter()
        trainer.train(dataloader)
        t1 = time.perf_counter()
    # heaviness = 1 // ~6.2 it/s // 240 shards // 64 batch size // 8 workers // 447.83 seconds
    elif args.dataloader_strategy == 'tokenized-arrow-input-ids-prefetching10':
        """
        ...
        """
        # FIXME path
        dataset = load_from_disk('/media/goader/masters/data-loading/output')
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collator,
            prefetch_factor=10,
        )

        t0 = time.perf_counter()
        trainer.train(dataloader)
        t1 = time.perf_counter()
    # heaviness = 1 // jumping from 3.3 to 5.5 it/s // 1 shards // 64 batch size // 8 workers // 692.48 seconds
    # heaviness = 1 // 6.3 it/s with lags to 4.5 it/s // 240 shards // 64 batch size // 8 workers // 521.20 seconds
    elif args.dataloader_strategy == 'raw-iterable':
        """
        ...
        """
        dataset = load_dataset(
            'arrow',
            data_files=['/media/goader/masters/data-loading/cultura_x-train-00000-of-00501.arrow'],
            split='train'
        ).to_iterable_dataset(num_shards=240)
        dataset = dataset.map(
            lambda examples, indices: {'id': indices},
            with_indices=True,
            batched=True,
            batch_size=256,
        ).map(
            tokenize_function,
            batched=True,
            batch_size=256,
            remove_columns=dataset.column_names,
            fn_kwargs=dict(
                tokenizer=tokenizer,
                max_length=512
            )
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, pad_to_multiple_of=8)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

        t0 = time.perf_counter()
        trainer.train(dataloader)
        t1 = time.perf_counter()



    print(f'Training took {t1 - t0:.2f} seconds')
