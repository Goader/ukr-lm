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
FILTERED_DIR = MASTERS_SCRATCH_DIR / 'metadata-deduplicated'

# Create temp directories if they don't exist
os.makedirs(FILTERED_DIR, exist_ok=True)



class FilterUberTextStep(PipelineStep):
    """Filter corpus to only include documents that are in the duplicates list"""
    
    def __init__(self):
        self.filtered_count = 0
        self.total_count = 0
        
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.total_count += 1
            
            compound_id = doc.metadata.get("url", "")

            if not compound_id.startswith("ubertext"):
                continue
                
            doc.metadata['subsource'] = compound_id.split('.')[1]
            doc.metadata['source'] = 'ubertext2.0'
            doc.metadata['timestamp'] = '1970-01-01T00:00:00Z'

            doc.id = f'ubertext2.0/{compound_id}'

            yield doc

        print(f"[Rank {rank}] Filtered {self.filtered_count:,} documents out of {self.total_count:,}")


if __name__ == '__main__':
    print("Filtering UberText corpus...")
    ubertext_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
                glob_pattern='*malyuk*.parquet',
                file_progress=True,
                doc_progress=True,
            ),
            FilterUberTextStep(),
            ParquetWriter(
                output_folder=str(FILTERED_DIR),
                output_filename='${source}_${rank}.parquet',
                expand_metadata=True,
            ),
        ],
        logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'duplicates_analysis'),
        skip_completed=False,
    )
    ubertext_executor.run()
