from pathlib import Path
import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'

# you can also change ngrams or the number of buckets and their size here
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
)  # better precision -> fewer false positives (collisions)


TOTAL_TASKS = 32
WORKERS = 16


# stage 1 computes minhash signatures for each task (each task gets a set of files)
stage1 = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
            glob_pattern='*.parquet',
            file_progress=True,
            doc_progress=True,
        ),
        MinhashDedupSignature(
            output_folder=str(MASTERS_SCRATCH_DIR / 'minhash' / 'signatures'),
            config=minhash_config,
            language=Languages.ukrainian,
        ),
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    tasks=TOTAL_TASKS,
    workers=WORKERS,
)

# stage 2 finds matches between signatures in each bucket
stage2 = LocalPipelineExecutor(
    pipeline=[
        MinhashDedupBuckets(
            input_folder=str(MASTERS_SCRATCH_DIR / 'minhash' / 'signatures'),
            output_folder=str(MASTERS_SCRATCH_DIR / 'minhash' / 'buckets'),
            config=minhash_config,
        ),
    ],
    tasks=minhash_config.num_buckets,
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    depends=stage1,
    workers=WORKERS,
)


# stage 3 creates clusters of duplicates using the results from all buckets
stage3 = LocalPipelineExecutor(
    pipeline=[
        MinhashDedupCluster(
            input_folder=str(MASTERS_SCRATCH_DIR / 'minhash' / 'buckets'),
            output_folder=str(MASTERS_SCRATCH_DIR / 'minhash' / 'remove_ids'),
            config=minhash_config,
            save_cluster_id=True,
            save_cluster_size=True,
        ),
    ],
    tasks=1,
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    depends=stage2,
    workers=WORKERS,
)

# stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
# the data must match exactly stage 1, so number of tasks and the input source must be the same
stage4 = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
            glob_pattern='*.parquet',
            file_progress=True,
            doc_progress=True,
        ),
        # TokensCounter(tokenizer_name_or_path='Goader/liberta-large-v2'),  # nice way to see how many tokens we had before and after deduplication
        MinhashDedupFilter(
            input_folder=str(MASTERS_SCRATCH_DIR / 'minhash' / 'remove_ids'),
            load_cluster_ids=True,
            load_cluster_sizes=True,
        ),
        ParquetWriter(output_folder=str(MASTERS_SCRATCH_DIR / 'minhash' / 'deduplicated_output')),
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    depends=stage3,
    tasks=TOTAL_TASKS,
    workers=WORKERS,
)


if __name__ == '__main__':
    stage4.run()
