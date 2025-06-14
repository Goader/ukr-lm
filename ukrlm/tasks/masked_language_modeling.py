from typing import Dict, Any

from omegaconf import DictConfig

import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
from torchmetrics.classification import MulticlassAccuracy

from ukrlm.optimizers import instantiate_optimizer
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

        self.local_step = 0

    def _mlm_accuracy(self, logits, labels) -> float:
        with torch.no_grad():
            flattened_shape = (-1, logits.size()[-1])
            mlm_accuracy = self.mlm_accuracy(logits.view(*flattened_shape), labels.view(-1))
            return mlm_accuracy

    def training_step(self, batch, batch_idx):
        batch_ids = batch['id']
        if min(self.trainer.global_step, self.local_step) < self.trainer.log_every_n_steps + 1:
            print(f'global_rank: {self.global_rank}, global_step: {self.global_step}, batch_ids: {batch_ids}')

            if min(self.trainer.global_step, self.local_step) < 10:
                print(f'global_rank: {self.global_rank}, input_ids: {batch["input_ids"][:5, :10]}')

            self.local_step += 1  # we do not need to increment it any further after initial logging
        del batch['id']

        # temporary fix, because the model does not accept this parameter
        special_tokens_mask = batch.pop('special_tokens_mask', None)

        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        # optimizes by not computing accuracy in every step, but only in log_every_n_steps
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            mlm_accuracy = self._mlm_accuracy(logits, batch['labels'])
            self.log('train_loss', loss, on_step=True, logger=True)
            self.log('train_perplexity', torch.exp(loss), on_step=True, prog_bar=True, logger=True)
            self.log('train_mlm_accuracy', mlm_accuracy, on_step=True, prog_bar=False, logger=True)

            # wandb logs only for the main process, we need grouping to log for all processes
            self.log(f'batch_ids[0]-global_rank-{self.global_rank}', batch_ids[0],
                     on_step=True, logger=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        batch.pop('special_tokens_mask', None)
        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        mlm_accuracy = self._mlm_accuracy(logits, batch['labels'])

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_perplexity', torch.exp(loss), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mlm_accuracy', mlm_accuracy, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch.pop('special_tokens_mask', None)
        model_output = self.model(**batch)
        loss = model_output.loss
        logits = model_output.logits

        mlm_accuracy = self._mlm_accuracy(logits, batch['labels'])

        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_perplexity', torch.exp(loss), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mlm_accuracy', mlm_accuracy, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # FIXME move to bf16 training fully
        # FIXME how do I keep rope and normalization layers in fp32? https://optimi.benjaminwarner.dev/kahan_summation/
        optimizer = instantiate_optimizer(self, self.cfg)
        scheduler = instantiate_scheduler(optimizer, self.cfg)

        # loading optimizer states, but not hyperparameters
        if self.cfg.model.context_extension_phase:
            assert self.cfg.model.checkpoint_path is not None, 'checkpoint_path must be passed for context extension phase'
            param_groups = optimizer.state_dict()['param_groups']
            optimizer_state = torch.load(self.cfg.model.checkpoint_path)['optimizer_states'][0]['state']
            state_dict = {
                'param_groups': param_groups,
                'state': optimizer_state,
            }
            optimizer.load_state_dict(state_dict)
            del state_dict, optimizer_state

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    def on_train_epoch_start(self) -> None:
        # setting the epoch for the distributed sampler
        self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['huggingface_config'] = self.model.config

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.local_step = 0
