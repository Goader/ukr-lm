from pathlib import Path
import os

from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.extractors.trafilatura import Trafilatura
from datatrove.executor.local import LocalPipelineExecutor


MASTERS_SCRATCH_DIR = Path(os.environ['SCRATCH']) / 'masters'


"""
HPLT v2.0

Location: JSONL files in `hplt-2.0/`

Format:

JSONL with the following fields:
- f: Common Crawl dump name
- u: url
- c: content type (e.g. text/plain)
- ts: crawl timestamp (e.g. 2022-03-01T08:14:43Z)
- collection: collection name
- text: text
- id: uuid
"""

hplt2_reader = JsonlReader(
    data_folder=str(MASTERS_SCRATCH_DIR / 'hplt-2.0'),
    file_progress=True,
    doc_progress=True,
    glob_pattern='*.jsonl',
    default_metadata={
        'source': 'hplt-2.0',
    },
    adapter=lambda self, data, path, id_in_file: {
        'text': data['text'],
        'id': f'hplt-2.0/{data["id"]}',
        'metadata': {
            'url': data['u'],
            'timestamp': data['ts'],
            'subsource': data['f'],
        }
    }
)


"""
FineWeb-2

Location: HF Dataset (cache in `fineweb-2/`)

Format:

Parquet files with the following fields:
- text: text
- id: uuid
- dump: Common Crawl dump name
- url: url
- date: crawl date (e.g. 2016-09-29T22:12:41Z)
"""

fineweb2_reader = ParquetReader(
    data_folder='hf://datasets/HuggingFaceFW/fineweb-2/data/ukr_Cyrl/train',
    file_progress=True,
    doc_progress=True,
    default_metadata={
        'source': 'fineweb-2',
    },
    adapter=lambda self, data, path, id_in_file: {
        'text': data['text'],
        'id': f'fineweb-2/{data["id"].removeprefix("<urn:uuid:").removesuffix(">")}',
        'metadata': {
            'url': data['url'],
            'timestamp': data['date'],
            'subsource': data['dump'],
        }
    }
)

"""
CulturaX

Location: HF Dataset (cache in ???)

Format:

Parquet files with the following fields:
- text: text
- timestamp: crawl timestamp (e.g. 2018/11/19 07:24:51)
- url: url
- source: source name (mC4 | OSCAR-xxxx)
"""

culturax_reader = ParquetReader(
    data_folder='hf://datasets/uonlp/CulturaX/uk',
    file_progress=True,
    doc_progress=True,
    glob_pattern='*.parquet',
    default_metadata={
        'source': 'cultura-x',
    },
    adapter=lambda self, data, path, id_in_file: {
        'text': data['text'],
        'id': f'cultura-x/{path}/{id_in_file}',
        'metadata': {
            'url': data['url'],
            'timestamp': data['timestamp'].replace('/', '-').replace(' ', 'T') + 'Z',
            'subsource': data['source'],
        }
    }
)

"""
Malyuk

Location: HF Dataset (cache in `huggingface_cache/`)

Format:

Parquet files with the following fields:
- id: uuid
- compound_id: compound id (e.g. ubertext.news.filter_rus_gcld+short.orig.jsonl.xz.6da5383273173f92238f70df3722b391595de376)
- text: text
"""

malyuk_reader = JsonlReader(
    data_folder='hf://datasets/lang-uk/malyuk',
    file_progress=True,
    doc_progress=True,
    glob_pattern='*.jsonlines',
    default_metadata={
        'source': 'malyuk',
    },
    adapter=lambda self, data, path, id_in_file: {
        'text': data['text'],
        'id': f'malyuk/{data["id"]}',
        'metadata': {
            'url': data['compound_id'],
            'timestamp': None,
            'subsource': data['compound_id'].split('.')[0],
        }
    }
)

"""
Output

Location: `merged/`

Format:

Parquet files with the following fields:
- id: uuid
- text: text
- source: source name (hplt-2.0 | fineweb-2 | cultura-x | malyuk)
- url: url
- timestamp: crawl timestamp
- subsource: dump name for FineWeb-2 and HPLT v2.0, source for CulturaX, first segment of compound id for Malyuk
"""

writer = ParquetWriter(
    output_folder=str(MASTERS_SCRATCH_DIR / 'merged'),
    output_filename='${source}_${rank}.parquet',
    expand_metadata=True,
)


print('Processing HPLT v2.0...')
hplt2_executor = LocalPipelineExecutor(
    pipeline=[
        hplt2_reader,
        writer,
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    skip_completed=False,
)

print('Processing FineWeb-2...')
fineweb2_executor = LocalPipelineExecutor(
    pipeline=[
        fineweb2_reader,
        writer,
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    skip_completed=False,
)

print('Processing CulturaX...')
cultura_executor = LocalPipelineExecutor(
    pipeline=[
        culturax_reader,
        writer,
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    skip_completed=False,
)

print('Processing Malyuk...')
malyuk_executor = LocalPipelineExecutor(
    pipeline=[
        malyuk_reader,
        writer,
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    skip_completed=False,
)



"""
Ukrainian News

Location: HF Dataset (https://huggingface.co/datasets/zeusfsx/ukrainian-news)

Format:

JSONL files with the following fields:
- id: incrementing integer
- url: url
- title: title
- text: text
- owner: news agency name
- datetime: timestamp (e.g. 2022-12-12T12:19:00+00:00)
"""

ukrainian_news_reader = JsonlReader(
    data_folder='hf://datasets/zeusfsx/ukrainian-news',
    file_progress=True,
    doc_progress=True,
    glob_pattern='*.jsonl',
    default_metadata={
        'source': 'ukrainian-news',
    },
    adapter=lambda self, data, path, id_in_file: {
        'text': data['text'],
        'id': f'ukrainian-news/{data["id"]}',
        'metadata': {
            'url': data['url'],
            'timestamp': data['datetime'][:-6] + 'Z',
            'subsource': data['owner'],
        }
    }
)


ukrainian_news_executor = LocalPipelineExecutor(
    pipeline=[
        ukrainian_news_reader,
        Trafilatura(timeout=1),
        ParquetWriter(
            output_folder=str(MASTERS_SCRATCH_DIR / 'ukrainian-news'),
            output_filename='${source}_${rank}.parquet',
            expand_metadata=True,
        ),
    ],
    logging_dir=str(MASTERS_SCRATCH_DIR / 'datatrove-logs'),
    skip_completed=False,
)


if __name__ == '__main__':
    # hplt2_executor.run()
    # fineweb2_executor.run()
    # cultura_executor.run()
    # malyuk_executor.run()
    ukrainian_news_executor.run()
