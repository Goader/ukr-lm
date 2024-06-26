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
from transformers.activations import ACT2FN
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PredictionHeadTransform,
    DebertaV2LMPredictionHead,
    DebertaV2Encoder,
)

from omegaconf import DictConfig


__all__ = [
    'RTDOutput',
    'DebertaV3RTDHead',
    'DebertaV3ForRTD',
    'DebertaV3EnhancedMaskDecoder',
    'DebertaV3ForMLM',
]


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


# Inspired by transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2LMPredictionHead
# and https://github.com/microsoft/DeBERTa/blob/master/DeBERTa/apps/models/replaced_token_detection_model.py
class DebertaV3RTDHead(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps)

        self.classifier = nn.Linear(self.embedding_size, 1)

    def forward(self, hidden_states):
        # selecting all embeddings for the [CLS] token
        ctx_states = hidden_states[:, 0, :].unsqueeze(1)

        # summing the embeddings of the [CLS] token and the hidden states and passing through the layers
        seq_states = self.LayerNorm(ctx_states + hidden_states)
        seq_states = self.dense(seq_states)
        seq_states = self.transform_act_fn(seq_states)

        # Replaced Token Detection (RTD) classification
        logits = self.classifier(seq_states).squeeze(-1)

        return logits


class DebertaV3ForRTD(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()

        self.config = config

        self.deberta = AutoModel.from_config(config)
        self.cls = DebertaV3RTDHead(config)

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

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
            loss = loss_fct(logits.view(-1), labels.view(-1).float())

        return RTDOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaV3EnhancedMaskDecoder(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        # TODO substitute with custom prediction head utilizing word embeddings as decoder?
        self.prediction_head = DebertaV2LMPredictionHead(config)

    def forward(
        self,
        encoder_layers: tuple[torch.Tensor],
        encoder: DebertaV2Encoder,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = encoder_layers[-2]

        # preparing query states and other inputs
        position_ids = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        rel_embeddings = encoder.get_rel_embedding()
        attention_mask = encoder.get_attention_mask(attention_mask)

        query_states = hidden_states + position_embeddings

        # defining the decoder layers (sharing weights)
        decoder_layers = [
            encoder.layer[-1],
            encoder.layer[-1],
        ]

        # passing the query states through the decoder layers
        for i, layer in enumerate(decoder_layers):
            query_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                query_states=query_states,
                rel_embeddings=rel_embeddings,
            )

        logits = self.prediction_head(query_states)
        return logits


# TODO possible optimizations:
#  - skip the last layer in deberta
#  - skip calculating logits for the unmasked tokens
class DebertaV3ForMLM(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()

        self.config = config

        self.deberta = AutoModel.from_config(config)
        self.decoder = DebertaV3EnhancedMaskDecoder(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.deberta.get_input_embeddings()

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
    ) -> Union[Tuple, MaskedLMOutput]:

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        encoder_layers = outputs.hidden_states
        logits = self.decoder(
            encoder_layers=encoder_layers,
            encoder=self.deberta.encoder,
            attention_mask=attention_mask
        )

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
