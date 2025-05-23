#!/usr/bin/env python3
from pathlib import Path
import os
import time

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'
FINEWEB2_DUPLICATES_FILE = MASTERS_SCRATCH_DIR / 'corpora-sources' / 'fineweb-duplicates.txt'
CULTURAX_DUPLICATES_FILE = MASTERS_SCRATCH_DIR / 'corpora-sources' / 'culturax-duplicates.txt'
FILTERED_DIR = MASTERS_SCRATCH_DIR / 'metadata-deduplicated'

# Create temp directories if they don't exist
os.makedirs(FILTERED_DIR, exist_ok=True)


class DuplicatesLoader:
    """Load duplicates from a file with format: url\ttimestamp"""

    def __init__(self, duplicates_file: str):
        self.duplicates_file = duplicates_file
        self.duplicates: set[tuple[str, str]] = set()
        self._load_duplicates()

    def _load_duplicates(self):
        with open(self.duplicates_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    url, timestamp = parts
                    self.duplicates.add((url, timestamp))
        print(f"Loaded {len(self.duplicates):,} duplicates from {self.duplicates_file}")

    def is_duplicate(self, url: str, timestamp: str) -> bool:
        return (url, timestamp) in self.duplicates


class FilterCorpusStep(PipelineStep):
    """Filter corpus to only include documents that are in the duplicates list"""

    def __init__(self, duplicates_loader: DuplicatesLoader):
        self.duplicates_loader = duplicates_loader
        self.filtered_count = 0
        self.total_count = 0

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.total_count += 1

            url = doc.metadata.get("url", "")
            timestamp = doc.metadata.get("timestamp", "")

            # if no url or timestamp, we cannot identify it as a duplicate, so we preserve it
            if not url or not timestamp:
                yield doc
                continue

            # if it is a duplicate, we do not yield it
            if self.duplicates_loader.is_duplicate(url, timestamp):
                self.filtered_count += 1
                continue

            # otherwise, we yield it
            yield doc

        print(f"[Rank {rank}] Filtered {self.filtered_count:,} documents out of {self.total_count:,}")


class FilterOscarStep(PipelineStep):
    """Filter out documents from OSCAR subsources"""

    def __init__(self):
        self.filtered_count = 0
        self.total_count = 0

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.total_count += 1

            subsource = doc.metadata.get("subsource", "")

            # if it is not an OSCAR corpus, we yield it
            if not subsource.lower().startswith("oscar"):
                yield doc
            else:
                self.filtered_count += 1

        print(f"[Rank {rank}] Filtered {self.filtered_count:,} OSCAR documents out of {self.total_count:,}")


if __name__ == '__main__':
    start_time = time.time()

    # Load duplicates
    fineweb2_duplicates_loader = DuplicatesLoader(FINEWEB2_DUPLICATES_FILE)
    culturax_duplicates_loader = DuplicatesLoader(CULTURAX_DUPLICATES_FILE)

    # Step 1: Filter the FineWeb2 corpus
    print("Filtering FineWeb2 corpus...")
    fineweb2_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
                glob_pattern='*fineweb*.parquet',
                file_progress=True,
                doc_progress=True,
            ),
            FilterCorpusStep(fineweb2_duplicates_loader),
            ParquetWriter(
                output_folder=str(FILTERED_DIR),
                output_filename='${source}_${rank}.parquet',
                expand_metadata=True,
            ),
        ],
        logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'duplicates_analysis'),
        skip_completed=False,
    )
    fineweb2_executor.run()

    # Step 2: Filter the CulturaX corpus
    print("\nFiltering CulturaX corpus...")
    culturax_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
                glob_pattern='*cultura-x*.parquet',
                file_progress=True,
                doc_progress=True,
            ),
            FilterCorpusStep(culturax_duplicates_loader),
            # FilterOscarStep(),
            ParquetWriter(
                output_folder=str(FILTERED_DIR),
                output_filename='${source}_${rank}.parquet',
                expand_metadata=True,
            ),
        ],
        logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'duplicates_analysis'),
        skip_completed=False,
    )
    culturax_executor.run()

    # Step 3: Copy the HPLT corpus
    print("\nCopying HPLT corpus...")
    os.system(f"cp {MASTERS_SCRATCH_DIR}/merged/*hplt*.parquet {FILTERED_DIR}/")

    # # Step 4: Copy the Malyuk corpus
    # print("\nCopying Malyuk corpus...")
    # os.system(f"cp {MASTERS_SCRATCH_DIR}/merged/*malyuk*.parquet {FILTERED_DIR}/")
