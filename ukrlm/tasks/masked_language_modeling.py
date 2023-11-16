from typing import Dict, Any

from omegaconf import DictConfig

import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
from torchmetrics.classification import MulticlassAccuracy
from transformers import AutoModelForMaskedLM

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

    def _mlm_accuracy(self, logits, labels) -> float:
        with torch.no_grad():
            flattened_shape = (-1, logits.size()[-1])
            mlm_accuracy = self.mlm_accuracy(logits.view(*flattened_shape), labels.view(-1))
            return mlm_accuracy

    def training_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        # optimizes by not computing accuracy on every step, but only on log_every_n_steps
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            mlm_accuracy = self._mlm_accuracy(logits, batch['labels'])
            self.log('train_loss', loss, on_step=True, logger=True)
            self.log('train_perplexity', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
            self.log('train_mlm_accuracy', mlm_accuracy, on_step=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        mlm_accuracy = self._mlm_accuracy(logits, batch['labels'])

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_perplexity', torch.exp(loss), on_step=True, on_epoch=True, logger=True)
        self.log('val_mlm_accuracy', mlm_accuracy, on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        mlm_accuracy = self._mlm_accuracy(logits, batch['labels'])

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_perplexity', torch.exp(loss), on_step=True, on_epoch=True, logger=True)
        self.log('val_mlm_accuracy', mlm_accuracy, on_step=True, on_epoch=True, logger=True)
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

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['config'] = self.model.config
        if self.cfg.task.use_flash_attention:
            reversed = self.model.reverse_bettertransformer()
            checkpoint['state_dict'] = reversed.state_dict()


    # TODO does this work? implement this
    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     config = checkpoint['config']
    #     model = AutoModelForMaskedLM.from_config(config)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     model = model.to_bettertransformer()
    #     # FIXME this should be a class method
