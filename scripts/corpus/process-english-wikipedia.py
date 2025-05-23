from pathlib import Path
import os
import re

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.pipeline.stats.token_stats import TokenStats


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'
ENGLISH_WIKIPEDIA_DIR = MASTERS_SCRATCH_DIR / 'english-wikipedia'

# Create temp directories if they don't exist
os.makedirs(ENGLISH_WIKIPEDIA_DIR, exist_ok=True)


class EnglishWikipediaFormatter(BaseFormatter):
    def __init__(self):
        super().__init__()

        self.clean_parentheses_regex = re.compile(r'(?<=\()[^\w()]*?(\w+?[^()]*?\w*?)[^\w()]*?(?=\))', re.IGNORECASE)
        self.remove_empty_parentheses_regex = re.compile(r' \((\W*?|or)\)', re.IGNORECASE)

    def format(self, text: str) -> str:
        text = self.clean_parentheses_regex.sub(r'\g<1>', text)
        text = self.remove_empty_parentheses_regex.sub('', text)
        text = text.strip()
        return text


if __name__ == '__main__':
    print("Processing English Wikipedia corpus...")
    english_wikipedia_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder='hf://datasets/wikimedia/wikipedia/20231101.en',
                glob_pattern='*.parquet',
                file_progress=True,
                doc_progress=True,
                default_metadata={
                    'source': 'english-wikipedia',
                },
                adapter=lambda self, data, path, id_in_file: {
                    'text': data['text'],
                    'id': f'english-wikipedia/{data["id"]}',
                    'metadata': {
                        'url': data['url'],
                        'timestamp': '2023-11-01T00:00:00Z',
                        'subsource': data['title'],
                    }
                }
            ),
            EnglishWikipediaFormatter(),
            # TokenStats(
            #     output_folder=str(ENGLISH_WIKIPEDIA_DIR / 'token-stats'),
            #     tokenizer_name_or_path='gpt2',
            # ),
            ParquetWriter(
                output_folder=str(ENGLISH_WIKIPEDIA_DIR),
                output_filename='${source}_${rank}.parquet',
                expand_metadata=True,
            ),
        ],
        logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'english-wikipedia'),
        skip_completed=False,
        workers=1,
    )
    english_wikipedia_executor.run()
