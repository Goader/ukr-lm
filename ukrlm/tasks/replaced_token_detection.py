from typing import Dict, Any
from dataclasses import dataclass

from omegaconf import DictConfig

import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_info
from torchmetrics.classification import MulticlassAccuracy, F1Score
from transformers import AutoModelForMaskedLM

from ukrlm.schedulers import instantiate_scheduler


@dataclass
class StepOutput:
    loss: torch.Tensor
    mlm_loss: torch.Tensor
    rtd_loss: torch.Tensor
    mlm_logits: torch.Tensor
    rtd_logits: torch.Tensor


class ReplacedTokenDetectionTask(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.generator = generator
        self.discriminator = discriminator

        self.delta_embeddings = nn.Embedding(
            num_embeddings=cfg.model.vocab_size,
            embedding_dim=cfg.model.hidden_size,
            padding_idx=cfg.model.pad_token_id,
        )

        self.mlm_accuracy = MulticlassAccuracy(
            num_classes=self.cfg.model.vocab_size,
            average='micro',
            ignore_index=-100,
            validate_args=True,
        )

        self.rtd_f1 = F1Score(
            task='binary',
            ignore_index=-100,
            validate_args=True,
        )

    def _mlm_accuracy(self, logits, labels) -> float:
        with torch.no_grad():
            flattened_shape = (-1, logits.size()[-1])
            mlm_accuracy = self.mlm_accuracy(logits.view(*flattened_shape), labels.view(-1))
            return mlm_accuracy

    def _rtd_f1(self, logits, labels) -> float:
        with torch.no_grad():
            flattened_shape = (-1, logits.size()[-1])
            rtd_f1 = self.rtd_f1(logits.view(*flattened_shape), labels.view(-1))
            return rtd_f1

    def common_step(self, batch, batch_idx) -> StepOutput:
        # initial setup
        masked_tokens = batch['labels'] != -100
        mlm_input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # mlm phase
        mlm_output = self.generator(**batch)
        mlm_loss = mlm_output.loss
        mlm_logits = mlm_output.logits

        # prepare inputs for rtd phase
        predicted_tokens = mlm_logits.argmax(dim=-1)  # TODO what if we use topk sampling?

        # FIXME do these predicted_tokens need to be detached?

        rtd_input_ids = mlm_input_ids.clone()
        rtd_input_ids[masked_tokens] = predicted_tokens[masked_tokens]
        labels = torch.where(attention_mask, rtd_input_ids != mlm_input_ids, -100).long()  # FIXME special tokens?

        # rtd phase
        generator_word_embeddings = self.generator.get_input_embeddings()
        # not allowing RTD to update the generator, so it won't make its life easier by worsening the generator
        inputs_embeds = generator_word_embeddings(rtd_input_ids).detach() + self.delta_embeddings(rtd_input_ids)

        # TODO what do we do with this??
        # for now we have generator and discriminator using separate positional embeddings, do we stay like that?
        # they also seem to be absolute positional embeddings, not relative, need to verify this
        rtd_output = self.discriminator(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        rtd_loss = rtd_output.loss
        rtd_logits = rtd_output.logits

        return StepOutput(
            loss=mlm_loss + self.cfg.task.rdt_lambda * rtd_loss,
            mlm_loss=mlm_loss,
            rtd_loss=rtd_loss,
            mlm_logits=mlm_logits,
            rtd_logits=rtd_logits,
        )

    def training_step(self, batch, batch_idx):
        batch_ids = batch['id']
        if self.trainer.global_step < self.trainer.log_every_n_steps + 1:
            print(f'global_rank: {self.global_rank}, global_step: {self.global_step}, batch_ids: {batch_ids}')
        del batch['id']

        output = self.common_step(batch, batch_idx)

        # optimizes by not computing accuracy in every step, but only in log_every_n_steps
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            mlm_accuracy = self._mlm_accuracy(output.mlm_logits, batch['labels'])
            rtd_f1 = self._rtd_f1(output.rtd_logits, batch['labels'])

            self.log('train_mlm_loss', output.mlm_loss, on_step=True, logger=True)
            self.log('train_perplexity', torch.exp(output.mlm_loss), on_step=True, prog_bar=True, logger=True)
            self.log('train_mlm_accuracy', mlm_accuracy, on_step=True, prog_bar=False, logger=True)

            self.log('train_rtd_loss', output.rtd_loss, on_step=True, logger=True)
            self.log('train_rtd_f1', rtd_f1, on_step=True, logger=True)

            self.log('train_loss', output.loss, on_step=True, logger=True)

            # wandb logs only for the main process, we need grouping to log for all processes
            self.log(f'batch_ids[0]-global_rank-{self.global_rank}', batch_ids[0],
                     on_step=True, logger=True, on_epoch=False)

        return output.loss


    def validation_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx)

        mlm_accuracy = self._mlm_accuracy(output.mlm_logits, batch['labels'])
        rtd_f1 = self._rtd_f1(output.rtd_logits, batch['labels'])

        self.log('val_mlm_loss', output.mlm_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_perplexity', torch.exp(output.mlm_loss), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mlm_accuracy', mlm_accuracy, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        self.log('val_rtd_loss', output.rtd_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_rtd_f1', rtd_f1, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        self.log('val_loss', output.loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return output.loss

    def test_step(self, batch, batch_idx):
        output = self.common_step(batch, batch_idx)

        mlm_accuracy = self._mlm_accuracy(output.mlm_logits, batch['labels'])
        rtd_f1 = self._rtd_f1(output.rtd_logits, batch['labels'])

        self.log('test_mlm_loss', output.mlm_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_perplexity', torch.exp(output.mlm_loss), on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_mlm_accuracy', mlm_accuracy, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        self.log('test_rtd_loss', output.rtd_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_rtd_f1', rtd_f1, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        self.log('test_loss', output.loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return output.loss

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
