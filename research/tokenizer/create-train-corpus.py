from argparse import ArgumentParser
import tqdm

from datasets import (
    load_dataset,
    Split,
    Dataset,
)


def load_ukrainian_cc100_part(cc100_filepath: str | None):
    if cc100_filepath is None:
        print('WARNING: CC100 file is not provided, loading from the Hugging Face datasets')
        return load_dataset('cc100', split=Split.TRAIN, num_proc=8)
    return load_dataset('text', data_files=cc100_filepath, split=Split.TRAIN, num_proc=8)


def load_english_multi_news():
    multi_news = load_dataset('multi_news', split=Split.ALL) \
        .rename_column('document', 'text') \
        .remove_columns(['summary'])
    return multi_news


class CorpusReader:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.idx = 0

    def is_empty(self) -> bool:
        return self.idx >= len(self.dataset)

    def process(self, text: str) -> str:
        return text

    def __next__(self) -> str:
        if self.is_empty():
            raise StopIteration

        text = self.dataset[self.idx]['text']
        self.idx += 1
        text = self.process(text)
        return text

    def __iter__(self):
        return self


class EnglishCorpusReader(CorpusReader):
    def process(self, text: str) -> str:
        text = text.replace('|||||', '\n')
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(lines)
        return text


class CorpusWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.ukrainian_size = 0
        self.english_size = 0

    @property
    def size(self) -> int:
        return self.ukrainian_size + self.english_size

    @property
    def english_portion(self) -> float:
        if self.size == 0:
            return 0
        return self.english_size / self.size

    def write(self, text: str, language: str):
        if language == 'uk':
            self.ukrainian_size += len(text)
        elif language == 'en':
            self.english_size += len(text)
        else:
            raise ValueError(f'Unknown language: {language}')

        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(text)
            f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, required=True, help='path to the output file')
    parser.add_argument('--size', type=float, default=5_000, help='millions of characters')
    parser.add_argument('--english-portion', type=float, default=0.015, help='portion of English text')
    parser.add_argument('--cc100-filepath', type=str, default=None, help='path to the CC100 file')
    args = parser.parse_args()

    writer = CorpusWriter(args.output)

    ukrainian_reader = CorpusReader(load_ukrainian_cc100_part(args.cc100_filepath))
    # ukrainian_reader = CorpusReader([])
    english_reader = EnglishCorpusReader(load_english_multi_news())

    ukr_warning = False
    eng_warning = False

    progress_bar = tqdm.tqdm(total=args.size, unit='M')
    while writer.size < args.size * 1_000_000:
        if (ukr_empty := ukrainian_reader.is_empty()) and not ukr_warning:
            print(f'WARNING: Ukrainian dataset is empty')
            ukr_warning = True
        if (eng_empty := english_reader.is_empty()) and not eng_warning:
            print(f'WARNING: English dataset is empty')
            eng_warning = True

        if not eng_empty and (writer.english_portion < args.english_portion or ukr_empty):
            reader = english_reader
            language = 'en'
        elif not ukr_empty:
            reader = ukrainian_reader
            language = 'uk'
        else:
            print('WARNING: desired size is not reached, but both datasets are empty')
            break

        try:
            text = next(reader)
            writer.write(text, language)
        except StopIteration:
            pass

        progress_bar.update(writer.size // 1_000_000 - progress_bar.n)
        progress_bar.refresh()

    progress_bar.close()
