# Available Ukrainian Language Corpora

## All outgoing links from Wikipedia

We would rather need a dump to run such an experiment. After the conversation with Krzysztof I learned, that there might be not that many links, they might lead to too specific places, and they might not be of any use.

**Pros:**

* similar to WebText approach

**Cons:**

* may not be that many links -> not that much data
* linked pages may not be diverse enough
* needs a lot of preprocessing

**Status: frozen** 

## [UberText](https://lang.org.ua/uk/corpora/#anchor4)

Corpus of scraped news articles from the several newspapers. They claim to have 6GB of data, zipped version has 1.6GB. The biggest problem is license. All the sentences are shuffled, so that the original text cannot be restored. Usage of the corpus is allowed only on the terms of **Fair Use** ([whatever that means](https://cedem.org.ua/analytics/fair-use/)). Will it be a problem if a model is later used in commercial purposes?

**Pros:**

* quite a lot of data
* data should be pretty high quality (news articles)
* tokenized (tokens and sentences)
* could be used for BERT's NSP (next sentence prediction) as negative examples (would it be biased?)

**Cons:**

* shuffled sentences!
* corpus contains wikipedia and literature too! (6GB contains them or not?)
* license

## [Legal Corpus](https://lang.org.ua/uk/corpora/#anchor4)

Over 9GB of legal documents (laws and other things)

**Pros:**

* 9GB (quite a lot)
* specialized data
* quality data (should be at least)
* tokenized
* no license (i guess :))

**Cons:**

* specialized data :)

## [CC-100](https://data.statmt.org/cc-100/)

Corpus which is created with the aim to replicate the training dataset for XLM-R. Ukrainian part contains 14GB of data. According to the `awesome-ukrainian-nlp` if we deduplicate the data we get around 10GB of it.

**Pros:**

* rather processed corpus
* diverse data (should be at least)
* quality data (same :))
* great license (as far as I understand)

**Cons:**

* not that much data for such a big corpus
* a lot of duplicates (?)

## [OSCAR](https://oscar-project.github.io/documentation/versions/oscar-2301/)

Version `23.01` contains 52GB data in Ukrainian.

**Pros:**

* a lot of data
* diverse data (should be at least)

**Cons:**

* probably a solid part of 52GB data is metadata and JSONL schema
* not very clear how do download it (just needs time to try)

## [mC4](https://github.com/allenai/allennlp/discussions/5056)

A big data corpus used for T5 training (i guess). According to `awesome-nlp-ukrainian` contains 196GB of data. Needs to be verified!

**Pros:**

* a lot of data (if everything is true, then it should be the main corpus)
* diverse data (should be at least)
* ok license (as far as I have understood)

**Cons:**

* not sure about the data quality
* duplicates (?)

## TODO other corpora
