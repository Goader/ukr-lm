from pathlib import Path
import os

from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor.local import LocalPipelineExecutor


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'


malyuk_reader = ParquetReader(
    data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
    file_progress=True,
    doc_progress=True,
    glob_pattern='*malyuk*.parquet',
    adapter=lambda self, data, path, id_in_file: {
        'text': data['text'],
        'id': data['id'],
        'metadata': {
            'source': data['source'],
            'url': data['url'],
            'timestamp': '1970-01-01T00:00:00Z',
            'subsource': data['subsource'],
        }
    }
)


writer = ParquetWriter(
    output_folder=str(MASTERS_SCRATCH_DIR / 'fixed'),
    output_filename='${source}_${rank}.parquet',
    expand_metadata=True,
)


executor = LocalPipelineExecutor(
    pipeline=[malyuk_reader, writer],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    skip_completed=False,
    tasks=16,
    workers=16,
)


if __name__ == '__main__':
    executor.run()
