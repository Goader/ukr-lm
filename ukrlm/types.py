from typing import TypedDict

from torchtyping import TensorType


class TransformerInput(TypedDict):
    """Input to the transformer model."""

    # Shape: (batch_size, sequence_length, hidden_size)
    input_ids: TensorType['batch', 'seq_len', 'hidden_size']

    # Shape: (batch_size, sequence_length, hidden_size)
    attention_mask: TensorType['batch', 'seq_len', 'hidden_size']

    # Shape: (batch_size, sequence_length, hidden_size)
    token_type_ids: TensorType['batch', 'seq_len', 'hidden_size']
