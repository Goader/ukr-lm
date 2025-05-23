#!/usr/bin/env python3
from pathlib import Path
import os
import pylcs
import statistics
import json
import gc
import pickle
import time
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Iterator, Optional

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline, Document


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'
DUPLICATES_FILE = MASTERS_SCRATCH_DIR / 'corpora-sources' / 'hplt-culturax-duplicates.txt'
SIMILARITIES_OUTPUT = MASTERS_SCRATCH_DIR / 'corpora-sources' / 'hplt-culturax-similarities.txt'
TEMP_DIR = MASTERS_SCRATCH_DIR / 'corpora-sources' / 'temp'
FILTERED_HPLT_DIR = TEMP_DIR / 'filtered_hplt'
FILTERED_CULTURAX_DIR = TEMP_DIR / 'filtered_culturax'
BATCH_SIZE = 20_000  # Process 20 thousand documents at a time

# Create temp directories if they don't exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FILTERED_HPLT_DIR, exist_ok=True)
os.makedirs(FILTERED_CULTURAX_DIR, exist_ok=True)


class DuplicatesLoader:
    """Load duplicates from a file with format: url\ttimestamp"""
    
    def __init__(self, duplicates_file: str):
        self.duplicates_file = duplicates_file
        self.duplicates: Set[Tuple[str, str]] = set()
        self._load_duplicates()
        
    def _load_duplicates(self):
        with open(self.duplicates_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    url = parts[0].strip()
                    timestamp = parts[1].strip()
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
            
            if self.duplicates_loader.is_duplicate(url, timestamp):
                self.filtered_count += 1
                yield doc
        
        print(f"[Rank {rank}] Filtered {self.filtered_count:,} documents out of {self.total_count:,}")


class TextCollector(PipelineStep):
    """Collect texts in memory for a batch of documents"""
    
    def __init__(self, corpus_name: str, batch_id: int):
        self.corpus_name = corpus_name
        self.batch_id = batch_id
        self.collected_texts: dict[tuple[str, str], str] = {}
        self.total_collected = 0
        self.skipped_count = 0  # Track documents we've skipped
        
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        documents_to_skip = self.batch_id * BATCH_SIZE
        
        for doc in data:
            url = doc.metadata.get("url", "").strip()
            timestamp = doc.metadata.get("timestamp", "").strip()

            if self.skipped_count < documents_to_skip:
                # Skip documents from previous batches
                self.skipped_count += 1
            else:
                self.collected_texts[f'{url}\t{timestamp}'] = doc.text
                self.total_collected += 1
                
                if self.total_collected >= BATCH_SIZE:
                    print(f"[Rank {rank}] Reached batch limit of {BATCH_SIZE:,} documents")
                    break
            
            yield doc
        
        print(f"[Rank {rank}] Batch {self.batch_id}: Skipped {self.skipped_count:,} documents, "
              f"Collected {len(self.collected_texts):,} texts from {self.corpus_name}")

        # saving the dictionary
        with open(f'{self.corpus_name}_texts_{self.batch_id}.pkl', 'wb') as f:
            pickle.dump(self.collected_texts, f)


def load_texts_batch(corpus_name: str, corpus_dir: Path, batch_id: int) -> dict[tuple[str, str], str]:
    """Load a batch of texts from the given corpus using TextCollector"""
    
    print(f"Loading batch {batch_id} from {corpus_name}...")
    collector = TextCollector(corpus_name, batch_id)
    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=str(corpus_dir),
                glob_pattern="*.parquet",
                file_progress=True,
                doc_progress=True,
            ),
            collector,
        ],
        logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'duplicates_analysis'),
        skip_completed=False,
    )
    executor.run()

    # loading the dictionary
    with open(f'{corpus_name}_texts_{batch_id}.pkl', 'rb') as f:
        collected_texts = pickle.load(f)

    print(f"Number of items in collected_texts: {len(collected_texts)}")

    return collected_texts


class SimilaritiesExtractor(PipelineStep):
    def __init__(self, key2text: dict[str, str]):
        self.key2text = key2text
        self.exact_matches = 0
        self.similarities = []
        self.total_processed = 0
        self.hits = 0

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.total_processed += 1
            
            url = doc.metadata.get("url", "").strip()
            timestamp = doc.metadata.get("timestamp", "").strip()
            text = doc.text
            key = f'{url}\t{timestamp}'
            
            if self.total_processed < 10:
                print(f"Total processed: {self.total_processed}")
                print(f"Hits: {self.hits}")
                print(f"URL: {repr(url)}")
                print(f"Timestamp: {repr(timestamp)}")
                print(f"Key: {repr(key)}")
                print(f"Key in key2text: {key in self.key2text}")
                print()

            if key in self.key2text:
                if 0 < self.hits < 10:
                    print(f"Hits: {self.hits}")

                self.hits += 1
                other_text = self.key2text[key]
                
                # Check for exact matches
                if text == other_text:
                    self.exact_matches += 1
                
                text = text[:10_000]
                other_text = other_text[:10_000]
                
                # Compute LCS similarity
                similarity = pylcs.lcs(text, other_text) / min(len(text), len(other_text))
                self.similarities.append(similarity)

            yield doc
        
        print(f"[Rank {rank}] Hits: {self.hits:,}")

        with open(TEMP_DIR / f'similarities_data.json', 'w') as f:
            json.dump({
                'exact_matches': self.exact_matches,
                'total_processed': self.total_processed,
            }, f)

        with open(TEMP_DIR / f'similarities.pkl', 'wb') as f:
            pickle.dump(self.similarities, f)


def print_stats(similarities: List[float], exact_matches: int, total_processed: int):
    """Print statistics about the similarities"""
    if not similarities:
        print("No similarities computed")
        return
    
    print(f"Number of documents analyzed: {total_processed:,}")
    print(f"Number of exact matches: {exact_matches:,}")
    print(f"Ratio of exact matches: {exact_matches / total_processed:.2%}")
    
    print("\nSimilarity statistics:")
    print(f"  Min: {min(similarities):.4f}")
    print(f"  Max: {max(similarities):.4f}")
    print(f"  Mean: {statistics.mean(similarities):.4f}")
    print(f"  Median: {statistics.median(similarities):.4f}")
    
    # Distribution in ranges
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.0)]
    print("\nDistribution of similarities:")
    for start, end in ranges:
        count = sum(1 for s in similarities if start <= s < end)
        print(f"  {start:.1f} - {end:.2f}: {count:,} ({count / total_processed:.2%})")
    
    # Exact 1.0 matches
    exact_1 = sum(1 for s in similarities if s == 1.0)
    print(f"  Exactly 1.0: {exact_1:,} ({exact_1 / total_processed:.2%})")


def save_similarities(similarities: List[float]):
    """Save similarities to a file for later analysis"""
    with open(SIMILARITIES_OUTPUT, 'w') as f:
        for similarity in similarities:
            f.write(f"{similarity:.6f}\n")
    print(f"Saved {len(similarities):,} similarity values to {SIMILARITIES_OUTPUT}")


def cleanup_temp_files():
    """Clean up temporary files"""
    import shutil
    if os.path.exists(FILTERED_HPLT_DIR):
        shutil.rmtree(FILTERED_HPLT_DIR)
    if os.path.exists(FILTERED_CULTURAX_DIR):
        shutil.rmtree(FILTERED_CULTURAX_DIR)
    print(f"Cleaned up temporary files in {TEMP_DIR}")


if __name__ == '__main__':
    start_time = time.time()
    
    try:
        # Make sure output file is empty
        with open(SIMILARITIES_OUTPUT, 'w') as f:
            pass
        
        # Load duplicates
        duplicates_loader = DuplicatesLoader(DUPLICATES_FILE)
        
        # # Step 1: Filter the HPLT corpus
        # print("Filtering HPLT corpus...")
        # hplt_executor = LocalPipelineExecutor(
        #     pipeline=[
        #         ParquetReader(
        #             data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
        #             glob_pattern='*hplt*.parquet',
        #             file_progress=True,
        #             doc_progress=True,
        #         ),
        #         FilterCorpusStep(duplicates_loader),
        #         ParquetWriter(output_folder=str(FILTERED_HPLT_DIR), expand_metadata=True),
        #     ],
        #     logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'duplicates_analysis'),
        #     skip_completed=False,
        # )
        # hplt_executor.run()
        
        # # Step 2: Filter the CulturaX corpus
        # print("\nFiltering CulturaX corpus...")
        # culturax_executor = LocalPipelineExecutor(
        #     pipeline=[
        #         ParquetReader(
        #             data_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
        #             glob_pattern='*cultura-x*.parquet',
        #             file_progress=True,
        #             doc_progress=True,
        #         ),
        #         FilterCorpusStep(duplicates_loader),
        #         ParquetWriter(output_folder=str(FILTERED_CULTURAX_DIR), expand_metadata=True),
        #     ],
        #     logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'duplicates_analysis'),
        #     skip_completed=False,
        # )
        # culturax_executor.run()
        
        # Step 3: Process similarities between filtered corpora
        print("\nProcessing similarities between filtered corpora...")
        batch_id = 0
        last_batch = False
        similarities = []
        exact_matches = 0
        total_processed = 0

        # only one batch (sampling effectively)
        while not last_batch and batch_id < 1:
            key2text = load_texts_batch('hplt', FILTERED_HPLT_DIR, batch_id)
            if len(key2text) < BATCH_SIZE:
                last_batch = True
            else:
                batch_id += 1
            
            xkey = ('http://www.slideboom.com/presentations/498259/razdel3', '2013-05-19T12:25:37Z')
            print(f"Key: {xkey}")
            print(f"Key in duplicates: {xkey in duplicates_loader.duplicates}")

            print(f"Number of items in key2text: {len(key2text)}")
            print(f"Type of key2text: {type(key2text)}")
            print()

            items = key2text.keys()
            xi = 0
            for item in items:
                print(f'Item #{xi}: {item}')
                xi += 1
                if xi >= 5:
                    break
            
            print(f"Processing batch {batch_id}...")
            extractor = SimilaritiesExtractor(key2text)
            executor = LocalPipelineExecutor(
                pipeline=[
                    ParquetReader(
                        data_folder=str(FILTERED_CULTURAX_DIR),
                        glob_pattern='*.parquet',
                        file_progress=True,
                        doc_progress=True,
                    ),
                    extractor,
                ],
                logging_dir=str(MASTERS_SCRATCH_DIR / 'logs' / 'duplicates_analysis'),
                skip_completed=False,
            )
            executor.run()

            with open(TEMP_DIR / f'similarities_data.json', 'rb') as f:
                data = json.load(f)
                exact_matches += data['exact_matches']
                total_processed += data['total_processed']

            with open(TEMP_DIR / f'similarities.pkl', 'rb') as f:
                similarities.extend(pickle.load(f))

        # Print final statistics
        print("\nFinal statistics:")
        print_stats(similarities, exact_matches, total_processed)
        
        # Save similarities to file
        save_similarities(similarities)
        
        # Print execution time
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        
    finally:
        # Clean up temporary files
        # cleanup_temp_files() 
        pass
