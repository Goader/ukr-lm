import torch

from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertForMaskedLM


def load_ckpt(path: str) -> BertForMaskedLM:
    """
    Load a checkpoint from a CKPT file.
    Model is loaded on CPU and does not use flash attention.

    :param path: path to the CKPT file
    :return: BertForMaskedLM model
    """

    ckpt = torch.load(path, map_location='cpu')

    # TODO read config somehow from the CKPT file
    config = AutoConfig.from_pretrained(
        'bert-base-uncased',
        max_position_embeddings=512,
        vocab_size=32000,
        pad_token_id=0,
        unk_token_id=1,
        cls_token_id=2,
        sep_token_id=3,
        mask_token_id=4,
    )
    model: BertForMaskedLM = AutoModelForMaskedLM.from_config(config)
    model.load_state_dict(ckpt['state_dict'])
    return model
