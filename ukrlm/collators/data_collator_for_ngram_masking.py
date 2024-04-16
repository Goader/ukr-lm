from typing import Optional, List, Dict, Union, Any, Tuple, Mapping
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    pad_without_fast_tokenizer_warning,
    _torch_collate_batch,
)

from ukrlm.tokenizers.tokenization_liberta import SPIECE_UNDERLINE
from ukrlm.collators.utils import (
    is_whitespace,
    is_control,
    is_punctuation,
)


@dataclass
class DataCollatorForNgramMasking(DataCollatorMixin):
    """
    Data collator used for masked language modeling using Ngram Masking. Inputs are dynamically padded to the maximum
    length of a batch if they are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input.
        keep_original_probability (`float`, *optional*, defaults to 0.1):
            The probability with which to keep the original token when replacing tokens with ngrams.
        substitute_probability (`float`, *optional*, defaults to 0.1):
            The probability with which to substitute a token with a random token.
        max_ngram_size (`int`, *optional*, defaults to 1):
            The maximum size of the ngrams to mask.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    keep_original_probability: float = 0.1
    substitute_probability: float = 0.1
    max_ngram_size: int = 1
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = 'pt'

    # Mask over the vocabulary that marks subtokens that are the start of a word
    unigram_start_vocabulary_mask: torch.Tensor = field(init=False)

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                'This tokenizer does not have a mask token which is necessary for masked language modeling.'
            )

        if self.keep_original_probability + self.substitute_probability > 1:
            raise ValueError(
                'The sum of `keep_original_probability` and `substitute_probability` should be less than or equal to 1.'
            )

        if self.max_ngram_size < 1:
            raise ValueError('`max_ngram_size` should be greater than or equal to 1.')

        if self.max_ngram_size > 1:
            raise NotImplementedError('Ngram masking with n > 1 is not yet supported.')

        self.unigram_start_vocabulary_mask = self._build_ngram_mask()

    def _build_ngram_mask(self) -> torch.Tensor:
        unigram_start_vocabulary_mask = torch.zeros(len(self.tokenizer), dtype=torch.bool)
        for token, token_id in self.tokenizer.get_vocab().items():
            # special tokens are not part of words
            if token in self.tokenizer.all_special_tokens:
                continue

            # whitespace, control, and punctuation tokens are marking the start of a word
            if len(token) == 1 and (is_whitespace(token) or is_control(token) or is_punctuation(token)):
                unigram_start_vocabulary_mask[token_id] = True
                continue

            # if the token starts with an underscore, it means it is a first subtoken of a word
            if token.startswith(SPIECE_UNDERLINE):
                unigram_start_vocabulary_mask[token_id] = True
                continue

        return unigram_start_vocabulary_mask

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors='pt', pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                'input_ids': _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, get it from the dict.
        special_tokens_mask = batch.get('special_tokens_mask', None)
        batch['input_ids'], batch['labels'] = self.torch_mask_tokens(
            batch['input_ids'], special_tokens_mask=special_tokens_mask
        )
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
            - `keep_original_probability` original,
            - `substitute_probability` random,
            - otherwise MASK.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, 0.0)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # Assume we have matrix like this:
        # [[False, True, False, True, True, True],
        #  [False, True, False, True, False, True],
        #  [False, True, True, False, True, False],
        unigram_start_mask = self.unigram_start_vocabulary_mask[labels]

        # We have mask of all first subtokens of words, from which we randomly select `mlm_probability` to mask.
        # Since each flag is a start of a single unigram, and we mask the next `max_ngram_size` unigrams, without
        # lowering the probability, we would yield `max_ngram_size` times more masked tokens on average. Thus, we
        # need to divide the probability by `max_ngram_size` to keep the same average number of masked tokens.
        mlm_probability = self.mlm_probability / self.max_ngram_size

        # Probabilities will then look like this:
        # [[0.0, 0.15, 0.0, 0.15, 0.15, 0.15],
        #  [0.0, 0.15, 0.0, 0.15, 0.0, 0.15],
        #  [0.0, 0.15, 0.15, 0.0, 0.15, 0.0],
        probability_matrix.masked_fill_(unigram_start_mask, value=mlm_probability)

        # Let's say we will get this matrix:
        # [[False, True, False, False, True, False],
        #  [False, True, False, True, False, False],
        #  [False, False, False, False, True, False],
        masked_start_indices = torch.bernoulli(probability_matrix).bool()

        # A matrix like this:
        # [[0, 1, 2, 3, 4, 5],
        #  [0, 1, 2, 3, 4, 5],
        #  [0, 1, 2, 3, 4, 5],
        sequence_indices = torch.arange(labels.size(1), device=labels.device) \
            .repeat(labels.size(0), 1)

        # After applying the mask, we will get (according to the starts of the words):
        # [[0, 1, 0, 3, 4, 5],
        #  [0, 1, 0, 3, 0, 5],
        #  [0, 1, 2, 0, 4, 0],
        unigram_start_sequence_indices = sequence_indices.masked_fill(~unigram_start_mask, 0)

        # After applying the mask, we will get (according to the masked starts):
        # [[0, 1, 0, 0, 4, 0],
        #  [0, 1, 0, 3, 0, 0],
        #  [0, 0, 0, 0, 4, 0],
        masked_start_sequence_indices = sequence_indices.masked_fill(~masked_start_indices, 0)

        # Then we simply extend the mask to the whole ngram
        # [[0, 1, 1, 3, 4, 5],
        #  [0, 1, 1, 3, 3, 5],
        #  [0, 1, 2, 2, 4, 4],
        cumulated_unigram_start_sequence_indices = unigram_start_sequence_indices.cummax(dim=1).values

        # Doing same here, but with masked starts:
        # [[0, 1, 1, 1, 4, 4],
        #  [0, 1, 1, 3, 3, 3],
        #  [0, 0, 0, 0, 4, 4],
        cumulated_masked_start_sequence_indices = masked_start_sequence_indices.cummax(dim=1).values

        # We then compare the two set of indices to get the mask of the masked ngrams
        # [[True, True, True, False, True, False],
        #  [True, True, True, True, True, False],
        #  [True, False, False, False, True, True],
        masked_unigrams_mask = torch.eq(
            cumulated_unigram_start_sequence_indices,
            cumulated_masked_start_sequence_indices,
        )

        # We then get clean the mask from the special tokens and the sequence start
        # [[False, True, True, False, True, False],
        #  [False, True, True, True, True, False],
        #  [False, False, False, False, True, True],
        masked_indices = torch.where(
            masked_unigrams_mask & ~special_tokens_mask,
            cumulated_unigram_start_sequence_indices > 0,
            False,
        )

        # then we proceed to modify labels accordingly
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # we replace masked input tokens with tokenizer.mask_token ([MASK])
        mask_probability = 1 - self.keep_original_probability - self.substitute_probability
        indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_probability)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def tf_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError('TensorFlow is not yet supported for this collator.')

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError('NumPy is not yet supported for this collator.')
