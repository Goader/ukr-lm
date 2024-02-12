from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torchtyping import TensorType

from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModel,
    DebertaV2Config,
)
from transformers.modeling_outputs import ModelOutput, MaskedLMOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PredictionHeadTransform,
)

from omegaconf import DictConfig


@dataclass
class RTDOutput(ModelOutput):
    """
    Base class for replaced token detection outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Replaced token detection (RTD) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Prediction scores of the RTD head (a score for each sequence token before Sigmoid).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# @dataclass
# class PretrainingRTDOutput(ModelOutput):
#     """
#     Base class for outputs of pretraining RTD models.
#
#     Args:
#         mlm_output (`MaskedLMOutput`):
#             The output of the Masked Language Model head.
#         rtd_output (`RTDOutput`):
#             The output of the Replaced Token Detection head.
#         rtd_labels (`torch.LongTensor`):
#             Labels produced by the Masked Language Model head (MLM) for the Replaced Token Detection model (RTD).
#     """
#
#     mlm_output: MaskedLMOutput
#     rtd_output: RTDOutput
#     rtd_labels: torch.LongTensor


# Inspired by transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2LMPredictionHead
# and https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/models/replaced_token_detection_model.py
class DebertaV3RTDHead(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)

        self.transform = DebertaV2PredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(self.embedding_size, 1, bias=False)

        self.bias = nn.Parameter(torch.zeros(1))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DebertaV3ForRTD(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        # FIXME is this correct?
        match cfg.model.name:
            case 'deberta-v3-base':
                model_name = 'microsoft/deberta-v3-base'
            case 'deberta-v3-large':
                model_name = 'microsoft/deberta-v3-large'
            case _:
                raise ValueError(f'unknown model name: {cfg.model.name}')

        config = AutoConfig.from_pretrained(
            model_name,
            max_position_embeddings=self.cfg.model.max_position_embeddings,
            vocab_size=self.cfg.model.vocab_size,
            pad_token_id=self.cfg.model.pad_token_id,
            unk_token_id=self.cfg.model.unk_token_id,
            cls_token_id=self.cfg.model.cls_token_id,
            sep_token_id=self.cfg.model.sep_token_id,
            mask_token_id=self.cfg.model.mask_token_id,
        )

        self.deberta = AutoModel.from_config(config)
        self.cls = DebertaV3RTDHead(self.generator.config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, RTDOutput]:

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state
        logits = self.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return RTDOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
