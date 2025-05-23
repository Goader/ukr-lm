from pathlib import Path
import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'


class SourceInfoExtractor(PipelineStep):
    def __init__(self, output_file: str):
        self.output_file = output_file

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with open(f'{rank:04d}_{self.output_file}', 'w') as f:
            for doc in data:
                f.write(f'{doc.metadata["url"]}\t{doc.metadata["timestamp"]}\t{doc.text[:50]}\n')
                yield doc


hplt_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
            glob_pattern='*hplt*.parquet',
            file_progress=True,
            doc_progress=True,
        ),
        SourceInfoExtractor(output_file='hplt_source_info.txt'),
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'base_processing'),
    skip_completed=False,
)


fineweb2_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
            glob_pattern='*fineweb*.parquet',
        ),
        SourceInfoExtractor(output_file='fineweb2_source_info.txt'),
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'base_processing'),
    skip_completed=False,
)


culturax_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
            glob_pattern='*cultura-x*.parquet',
        ),
        SourceInfoExtractor(output_file='culturax_source_info.txt'),
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'base_processing'),
    skip_completed=False,
)


if __name__ == '__main__':
    print('Extracting source info for HPLT...')
    hplt_executor.run()
    print('Extracting source info for FineWeb2...')
    fineweb2_executor.run()
    print('Extracting source info for Cultura-X...')
    culturax_executor.run()
