import torch

from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import BertConfig, BertForMaskedLM
from transformers.models.modernbert.modeling_modernbert import ModernBertConfig, ModernBertForMaskedLM


def load_ckpt(path: str) -> BertForMaskedLM | ModernBertForMaskedLM:
    """
    Load a checkpoint from a CKPT file.
    Model is loaded on CPU and does not use flash attention.

    :param path: path to the CKPT file
    :return: BertForMaskedLM model
    """

    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    config = ckpt['huggingface_config']

    if isinstance(config, BertConfig):
        model: BertForMaskedLM = AutoModelForMaskedLM.from_config(config)
        model = model.to_bettertransformer()
    elif isinstance(config, ModernBertConfig):
        model: ModernBertForMaskedLM = AutoModelForMaskedLM.from_config(config)
    else:
        raise ValueError(f'unknown model config type - {type(config)}')


    model_state_dict = {
        k.removeprefix('model.'): p
        for k, p in ckpt['state_dict'].items()
        if k.startswith('model.')
    }

    model.load_state_dict(model_state_dict)

    if isinstance(model, BertForMaskedLM):
        model = model.reverse_bettertransformer()

    return model
