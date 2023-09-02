from omegaconf import DictConfig

import torch
from torch import nn

import pytorch_lightning as pl

from ukrlm.schedulers import instantiate_scheduler


class MaskedLanguageModelingTask(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = model

        self.loss = nn.CrossEntropyLoss()

        # TODO how do we handle multiple objectives?
        # for objective in self.cfg.task.objectives: ...

    def training_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = self.loss(model_output.logits.permute(0, 2, 1), batch['labels'])
        self.log('train_loss', loss, on_step=True, logger=True)
        self.log('train_perplexity', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = self.loss(model_output.logits.permute(0, 2, 1), batch['labels'])
        self.log('val_loss', loss, on_step=True, logger=True)
        self.log('val_perplexity', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = self.loss(model_output.logits.permute(0, 2, 1), batch['labels'])
        self.log('test_loss', loss, on_step=True, logger=True)
        self.log('test_perplexity', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.task.learning_rate,
            weight_decay=self.hparams.task.weight_decay
        )
        scheduler = instantiate_scheduler(optimizer, self.cfg)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
