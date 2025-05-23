from pathlib import Path
import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.utils.typeshelper import Languages


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'


main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
            glob_pattern='*.parquet',
            file_progress=True,
            doc_progress=True,
        ),
        GopherRepetitionFilter(
            language=Languages.ukrainian__cyrl,
            exclusion_writer=JsonlWriter(str(MASTERS_SCRATCH_DIR / 'removed' / 'gopher_rep'))
        ),
        GopherQualityFilter(
            min_doc_words=20,  # lower threshold, because of longer and more sensebearing word intensity
            max_avg_word_length=12,
            max_symbol_word_ratio=0.2,
            min_stop_words=None,  # skipping stop words filter for Ukrainian
            language=Languages.ukrainian__cyrl,
            exclusion_writer=JsonlWriter(str(MASTERS_SCRATCH_DIR / 'removed' / 'gopher_qual'))
        ),
        FineWebQualityFilter(
            char_duplicates_ratio=0.05,
            language=Languages.ukrainian__cyrl,
            exclusion_writer=JsonlWriter(str(MASTERS_SCRATCH_DIR / 'removed' / 'fineweb_qual'))
        ),
        ParquetWriter(
            output_folder=str(MASTERS_SCRATCH_DIR / 'filtered'),
            output_filename='${source}_${rank}.parquet',
            expand_metadata=True,
        ),
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'base_processing'),
    skip_completed=False,
    tasks=32,
    workers=16,
)


if __name__ == '__main__':
    main_processing_executor.run()
