import torch
from torch import nn
from torchtyping import TensorType

from torchvision import models

from omegaconf import DictConfig


class BERT(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        self.n_layers = cfg.model.n_layers
        self.attn_heads = cfg.model.attn_heads
        self.hidden_size = cfg.model.hidden_size
        self.feedforward_size = cfg.model.feedforward_size

        self.model = self._construct_model()

    def _construct_model(self) -> nn.Module:
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.attn_heads,
            dim_feedforward=self.feedforward_size,
            dropout=self.cfg.model.dropout,
            activation=self.cfg.model.activation,
        )

        model = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.hidden_size)
        )

        return model

    def forward(self, X) -> TensorType['batch', 'seq_len', 'hidden_size']:
        return self.model(X)
