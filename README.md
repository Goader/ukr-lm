# Ukrainian Language Models - Master's Thesis - LiBERTa Family

<!-- Provide a quick summary of what the model is/does. -->
LiBERTa is a family of BERT-like models pre-trained from scratch exclusively for Ukrainian. It was presented during the [UNLP](https://unlp.org.ua/) @ [LREC-COLING 2024](https://lrec-coling-2024.org/). Further details are in the [LiBERTa: Advancing Ukrainian Language Modeling through Pre-training from Scratch](https://aclanthology.org/2024.unlp-1.14/) paper.

The models are available on the HuggingFace Hub ([LiBERTa Collection](https://huggingface.co/collections/Goader/liberta-667d80dabef9acac5039c2e8)):

* [LiBERTa](https://huggingface.co/Goader/liberta-large)
* [LiBERTa-V2](https://huggingface.co/Goader/liberta-large-v2)

This repository contains all the code, that was used to pre-train these models. The pre-training framework is based on PyTorch Lightning and HuggingFace's Transformers.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

Read the [paper](https://aclanthology.org/2024.unlp-1.14/) for more detailed task descriptions.

|                                                                                                                         | NER-UK (Micro F1)   | WikiANN (Micro F1) | UD POS (Accuracy)              | News (Macro F1) |
|:------------------------------------------------------------------------------------------------------------------------|:------------------------:|:------------------:|:------------------------------:|:----------------------------------------:|
| <tr><td colspan="5" style="text-align: center;"><strong>Base Models</strong></td></tr>
| [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)                                                  | 90.86 (0.81)             | 92.27 (0.09)       | 98.45 (0.07)                   | -                                        |
| [roberta-base-wechsel-ukrainian](https://huggingface.co/benjamin/roberta-base-wechsel-ukrainian)                        | 90.81 (1.51)             | 92.98 (0.12)       | 98.57 (0.03)                   | -                                        |
| [electra-base-ukrainian-cased-discriminator](https://huggingface.co/lang-uk/electra-base-ukrainian-cased-discriminator) | 90.43 (1.29)             | 92.99 (0.11)       | 98.59 (0.06)                   | -                                        |
| <tr><td colspan="5" style="text-align: center;"><strong>Large Models</strong></td></tr>
| [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large)                                                | 90.16 (2.98)             | 92.92 (0.19)       | 98.71 (0.04)                   | 95.13 (0.49)                             |
| [roberta-large-wechsel-ukrainian](https://huggingface.co/benjamin/roberta-large-wechsel-ukrainian)                      | 91.24 (1.16)             | __93.22 (0.17)__   | 98.74 (0.06)                   | __96.48 (0.09)__                         |
| [liberta-large](https://huggingface.co/Goader/liberta-large)                                                            | 91.27 (1.22)             | 92.50 (0.07)       | 98.62 (0.08)                   | 95.44 (0.04)                             |
| [liberta-large-v2](https://huggingface.co/Goader/liberta-large-v2)                                                      | __91.73 (1.81)__         | __93.22 (0.14)__   | __98.79 (0.06)__               | 95.67 (0.12)                             |


## Fine-Tuning Hyperparameters

| Hyperparameter | Value |
|:---------------|:-----:|
| Peak Learning Rate  | 3e-5   |
| Warm-up Ratio       | 0.05   |
| Learning Rate Decay | Linear |
| Batch Size          | 16     |
| Epochs              | 10     |
| Weight Decay        | 0.05   |


## How to Get Started with the Model

Use the code below to get started with the model. Note, that the repository contains custom code for tokenization:

Pipeline usage:

```python
>>> from transformers import pipeline

>>> fill_mask = pipeline("fill-mask", "Goader/liberta-large-v2", trust_remote_code=True)
>>> fill_mask("Тарас Шевченко - один з найвизначніших <mask> України.")

[{'score': 0.37743982672691345,
  'token': 23179,
  'token_str': 'поетів',
  'sequence': 'Тарас Шевченко - один з найвизначніших поетів України.'},
 {'score': 0.3221002519130707,
  'token': 12095,
  'token_str': 'письменників',
  'sequence': 'Тарас Шевченко - один з найвизначніших письменників України.'},
 {'score': 0.05367676541209221,
  'token': 17491,
  'token_str': 'художників',
  'sequence': 'Тарас Шевченко - один з найвизначніших художників України.'},
 {'score': 0.04778451472520828,
  'token': 17124,
  'token_str': 'синів',
  'sequence': 'Тарас Шевченко - один з найвизначніших синів України.'},
 {'score': 0.04660917446017265,
  'token': 1354,
  'token_str': 'людей',
  'sequence': 'Тарас Шевченко - один з найвизначніших людей України.'}]
```

Extracting embeddings:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Goader/liberta-large-v2", trust_remote_code=True)
model = AutoModel.from_pretrained("Goader/liberta-large-v2")

encoded = tokenizer('Тарас Шевченко - один з найвизначніших поетів України.', return_tensors='pt')

output = model(**encoded)
```

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

```
@inproceedings{haltiuk-smywinski-pohl-2024-liberta,
    title = "{L}i{BERT}a: Advancing {U}krainian Language Modeling through Pre-training from Scratch",
    author = "Haltiuk, Mykola  and
      Smywi{\'n}ski-Pohl, Aleksander",
    editor = "Romanyshyn, Mariana  and
      Romanyshyn, Nataliia  and
      Hlybovets, Andrii  and
      Ignatenko, Oleksii",
    booktitle = "Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.unlp-1.14",
    pages = "120--128",
    abstract = "Recent advancements in Natural Language Processing (NLP) have spurred remarkable progress in language modeling, predominantly benefiting English. While Ukrainian NLP has long grappled with significant challenges due to limited data and computational resources, recent years have seen a shift with the emergence of new corpora, marking a pivotal moment in addressing these obstacles. This paper introduces LiBERTa Large, the inaugural BERT Large model pre-trained entirely from scratch only on Ukrainian texts. Leveraging extensive multilingual text corpora, including a substantial Ukrainian subset, LiBERTa Large establishes a foundational resource for Ukrainian NLU tasks. Our model outperforms existing multilingual and monolingual models pre-trained from scratch for Ukrainian, demonstrating competitive performance against those relying on cross-lingual transfer from English. This achievement underscores our ability to achieve superior performance through pre-training from scratch with additional enhancements, obviating the need to rely on decisions made for English models to efficiently transfer weights. We establish LiBERTa Large as a robust baseline, paving the way for future advancements in Ukrainian language modeling.",
}
```

## Licence

CC-BY 4.0

## Authors

The model was trained by Mykola Haltiuk as a part of his Master's Thesis under the supervision of Aleksander Smywiński-Pohl, PhD, AGH University of Krakow.

