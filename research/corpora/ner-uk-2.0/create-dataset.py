from argparse import ArgumentParser
from pathlib import Path
import random
import re
import os

from datasets import (
    DatasetDict,
    Dataset,
    ClassLabel,
    Features,
    Value,
    Sequence,
)


def parse_txt(text: str) -> list[tuple[str, int]]:
    sentences = re.split('(\n+)', text.strip())

    filtered_sentences = []
    offsets = [0]
    for sentence in sentences[:-1]:
        # if the sentence is empty (it is a separator), add its length to the previously obtained offset
        if sentence.strip() == '':
            offsets[-1] += len(sentence)
        # otherwise, add the length of the sentence as the offset for the next sentence
        else:
            filtered_sentences.append(sentence)
            offsets.append(offsets[-1] + len(sentence))

    return list(zip(filtered_sentences, offsets))


def parse_bsf(ann: str) -> list[tuple[int, int, str]]:
    entities = []

    for line in ann.split('\n'):
        if line.startswith('T'):
            parts = line.split('\t')

            entity_type = parts[1]
            start = int(parts[2])
            end = int(parts[3])

            entities.append((start, end, entity_type))

    return entities


def remove_overlapping_entities(entities: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    entities = sorted(entities, key=lambda x: x[0])

    new_entities = []
    for i, entity in enumerate(entities):
        if i == 0:
            new_entities.append(entity)
        else:
            prev_start, prev_end, _ = new_entities[-1]
            start, end, _ = entity

            if start >= prev_end:
                new_entities.append(entity)

    return new_entities


def align_entities(sentence: str, entities: list[tuple[int, int, str]]) -> tuple[list[str], list[str]]:
    tokens = sentence.split()
    ner_tags = ['O'] * len(tokens)

    token_offsets = [0]
    for token in tokens[:-1]:
        token_offsets.append(token_offsets[-1] + len(token) + 1)  # +1 for space

    for start, end, entity_type in entities:
        for i, offset in enumerate(token_offsets):
            if start <= offset and end >= offset + len(tokens[i]):
                if ner_tags[i] != 'O':
                    raise ValueError(f'Overlapping entities: {sentence} and {entities}')

                if start == offset:
                    ner_tags[i] = f'B-{entity_type}'
                else:
                    ner_tags[i] = f'I-{entity_type}'

    return tokens, ner_tags


def process_file(text_path: Path, ann_path: Path) -> list[tuple[list[str], list[str]]]:
    with open(text_path, 'r') as f:
        text = f.read().strip()

    with open(ann_path, 'r') as f:
        ann = f.read().strip()

    sentences = parse_txt(text)
    entities = parse_bsf(ann)

    entities = remove_overlapping_entities(entities)

    documents = []
    for sentence, offset in sentences:
        entities_in_sentence = [
            (start - offset, end - offset, entity_type)
            for start, end, entity_type in entities
            if start >= offset and end <= offset + len(sentence)
        ]
        tokens, ner_tags = align_entities(sentence, entities_in_sentence)

        documents.append((tokens, ner_tags))

    return documents


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True, help='path to the dataset')
    parser.add_argument('--repository', type=str, required=False, default=None, help='name of the repository')
    parser.add_argument('--output_dir', type=str, required=False, default=None, help='output directory')
    parser.add_argument('--validation-split', type=float, default=0.1, help='fraction of the dataset to be used for '
                                                                     'the validation split')
    parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
    parser.add_argument('--commit-message', type=str, default='Pushing dataset to the hub', help='commit message')
    args = parser.parse_args()

    random.seed(args.seed)

    # reading in the dev-test split
    print('Reading in the dev-test split...')

    train_ids = set()
    validation_ids = set()
    test_ids = set()

    path = Path(args.dataset_path)

    with open(path / 'dev-test-split.txt', 'r') as f:
        content = f.read().strip()
        parts = content.split('\n\n')
        if parts[0].lower().startswith('dev'):
            train_ids = set(parts[0].split('\n')[1:])
        if parts[1].lower().startswith('test'):
            test_ids = set(parts[1].split('\n')[1:])

    print('Count of train IDs:', len(train_ids))
    print('Count of test IDs:', len(test_ids))
    print()

    # creating the validation split
    print('Creating the validation split...')

    train_validation = list(train_ids)
    random.shuffle(train_validation)

    validation_count = int(args.validation_split * len(train_validation))
    train_ids = set(train_validation[validation_count:])
    validation_ids = set(train_validation[:validation_count])

    print('Count of train IDs:', len(train_ids))
    print('Count of validation IDs:', len(validation_ids))
    print('Count of test IDs:', len(test_ids))
    print()

    # mapping onto filepaths
    print('Mapping onto filepaths...')

    train_paths = dict()
    validation_paths = dict()
    test_paths = dict()

    for folder in path.iterdir():
        if not folder.is_dir():
            continue

        train_paths[folder.name] = dict()
        validation_paths[folder.name] = dict()
        test_paths[folder.name] = dict()

        for file in folder.iterdir():
            file_id = file.stem

            if file_id in train_ids:
                aggregator = train_paths[folder.name]
            elif file_id in validation_ids:
                aggregator = validation_paths[folder.name]
            elif file_id in test_ids:
                aggregator = test_paths[folder.name]
            else:
                print(f'File {file_id} is not in the dev-test split!')
                continue

            aggregator[file_id] = {
                'text': folder / f'{file_id}.txt',
                'ann': folder / f'{file_id}.ann'
            }

    print('Count of train files:', sum(len(files) for files in train_paths.values()))
    print('Count of validation files:', sum(len(files) for files in validation_paths.values()))
    print('Count of test files:', sum(len(files) for files in test_paths.values()))
    print()

    # creating the dataset
    dataset = DatasetDict()

    for split, paths in [('train', train_paths), ('validation', validation_paths), ('test', test_paths)]:
        documents = []

        for source, files in paths.items():
            for file_id, file_paths in files.items():
                sentences = process_file(file_paths['text'], file_paths['ann'])
                for tokens, ner_tags in sentences:
                    documents.append({
                        'document_id': f'{file_id}',
                        'tokens': tokens,
                        'ner_tags': ner_tags,
                        'source': source
                    })

        dataset[split] = Dataset.from_dict({
            'document_id': [doc['document_id'] for doc in documents],
            'tokens': [doc['tokens'] for doc in documents],
            'ner_tags': [doc['ner_tags'] for doc in documents],
            'source': [doc['source'] for doc in documents]
        })

    # mapping ner_tags onto the ClassLabel
    print('Mapping ner_tags onto the ClassLabel...')

    unique_ner_tags = set()
    for split in ['train', 'validation', 'test']:
        for ner_tags in dataset[split]['ner_tags']:
            unique_ner_tags.update(ner_tags)

    defined_ner_tags = [
        'ORG', 'PERS', 'LOC', 'MON',
        'PCT', 'DATE', 'TIME', 'PERIOD',
        'JOB', 'DOC', 'QUANT', 'ART', 'MISC',
    ]

    all_tags = ['O'] + [
        prefix + tag
        for tag in defined_ner_tags
        for prefix in ['B-', 'I-']
    ]

    assert unique_ner_tags == set(all_tags), (f'Unique NER tags do not match the defined NER tags! '
                                              f'{unique_ner_tags} != {all_tags}')

    ner_tags = ClassLabel(names=all_tags)

    dataset = dataset.map(
        lambda x: {'ner_tags': [ner_tags.str2int(tag) for tag in x['ner_tags']]},
        batched=True,
        features=Features({
            'document_id': Value('string'),
            'tokens': Sequence(Value('string')),
            'ner_tags': Sequence(ner_tags),
            'source': Value('string'),
        })
    )

    if not args.output_dir and not args.repository:
        print('No output directory or repository specified')

    # saving the dataset
    if args.output_dir:
        print('Saving the dataset...')

        dataset.save_to_disk(args.output_dir)

    # pushing the dataset to the hub
    if args.repository:
        print('Pushing the dataset to the hub...')

        dataset.push_to_hub(
            repo_id=args.repository,
            commit_message=args.commit_message,
            token=os.getenv('HF_TOKEN', None),
        )
