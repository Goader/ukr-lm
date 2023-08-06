import torch
from torch import nn
from torchtyping import TensorType

from omegaconf import DictConfig

from ..types import TransformerInput


class BERT(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.model = self._construct_model()

    def _construct_model(self) -> nn.Module:
        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.model.hidden_size,
            nhead=self.cfg.model.attn_heads,
            dim_feedforward=self.cfg.model.feedforward_size,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation,
            layer_norm_eps=self.cfg.model.layer_norm_eps,
        )

        model = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=self.cfg.model.n_layers,
            norm=nn.LayerNorm(self.cfg.model.hidden_size),
        )

        return model

    def forward(self, X: TransformerInput) -> TensorType['batch', 'seq_len', 'hidden_size']:
        return self.model(X)
