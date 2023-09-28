from omegaconf import DictConfig

import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
from torchmetrics.classification import MulticlassAccuracy

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
        self.mlm_accuracy = MulticlassAccuracy(
            num_classes=self.cfg.model.vocab_size,
            average='micro',
            ignore_index=-100,
            validate_args=True,
        )

        # TODO how do we handle multiple objectives?
        # for objective in self.cfg.task.objectives: ...

    def training_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        flattened_shape = (-1, logits.size()[-1])
        mlm_accuracy = self.mlm_accuracy(logits.view(*flattened_shape), batch['labels'].view(-1))

        self.log('train_loss', loss, on_step=True, logger=True)
        self.log('train_perplexity', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
        self.log('train_mlm_accuracy', mlm_accuracy, on_step=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        flattened_shape = (-1, logits.size()[-1])
        mlm_accuracy = self.mlm_accuracy(logits.view(*flattened_shape), batch['labels'].view(-1))

        self.log('val_loss', loss, on_step=True, logger=True)
        self.log('val_perplexity', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
        self.log('val_mlm_accuracy', mlm_accuracy, on_step=True, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        flattened_shape = (-1, logits.size()[-1])
        mlm_accuracy = self.mlm_accuracy(logits.view(*flattened_shape), batch['labels'].view(-1))

        self.log('test_loss', loss, on_step=True, logger=True)
        self.log('test_perplexity', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
        self.log('test_mlm_accuracy', mlm_accuracy, on_step=True, prog_bar=False, logger=True)
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
