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

    config = ckpt['config']
    model: BertForMaskedLM = AutoModelForMaskedLM.from_config(config)
    model.load_state_dict(ckpt['state_dict'])
    return model
