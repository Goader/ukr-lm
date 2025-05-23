from pathlib import Path
import os
import re

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.pipeline.stats.token_stats import TokenStats


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'
DEDUPLICATED_DIR = MASTERS_SCRATCH_DIR / 'minhash'
POSTPROCESSED_DIR = MASTERS_SCRATCH_DIR / 'postprocessed'

# Create temp directories if they don't exist
os.makedirs(POSTPROCESSED_DIR, exist_ok=True)


if __name__ == '__main__':
    print("Postprocessing minhash corpus...")
    postprocessed_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=str(DEDUPLICATED_DIR),
                glob_pattern='*.parquet',
                file_progress=True,
                doc_progress=True,
                adapter=lambda self, data, path, id_in_file: {
                    'id': data['id'],
                    'text': data['text'],
                    'metadata': {
                        'url': data['url'],
                        'timestamp': data['timestamp'],
                        'subsource': data['subsource'],
                        'source': data['source'],
                    }
                }
            ),           
            ParquetWriter(
                output_folder=str(POSTPROCESSED_DIR),
                output_filename='ukr_corpus_${rank}.parquet',
                expand_metadata=True,
            ),
        ],
        logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'postprocessed'),
        skip_completed=False,
        workers=10,
    )
    postprocessed_executor.run()
