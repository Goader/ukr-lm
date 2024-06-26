import logging

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import WandbLogger

import torch
from transformers import AutoModelForMaskedLM, AutoConfig, AutoModel, DebertaV2Config
from transformers.models.bert.modeling_bert import BertForMaskedLM

from ukrlm.datamodules import MaskedLanguageModelingDataModule
from ukrlm.tasks.masked_language_modeling import MaskedLanguageModelingTask
from ukrlm.tasks.replaced_token_detection import ReplacedTokenDetectionTask
from ukrlm.models import DebertaV3ForMLM, DebertaV3ForRTD

logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    task: pl.LightningModule
):
    wandb_logger = WandbLogger(project='ukr-lm')
    print(wandb_logger.experiment.id)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.checkpoint_output_dir,
        filename=f'{wandb_logger.experiment.name}_{cfg.model.name}_step{{step}}_perplexity{{train_perplexity:.3f}}',
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=False,
        every_n_train_steps=cfg.task.save_every_n_steps,
    )
    learning_rate_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            learning_rate_monitor,
        ],
        plugins=[SLURMEnvironment(auto_requeue=False)],
        profiler=cfg.profiler,
        # overfit_batches=4,
        # fast_dev_run=True,
        max_epochs=cfg.task.max_epochs,
        max_steps=cfg.task.max_steps,
        val_check_interval=0.1,
        log_every_n_steps=cfg.task.log_every_n_steps,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.task.strategy,
        sync_batchnorm=cfg.task.sync_batchnorm,
        precision=cfg.task.precision,
        gradient_clip_val=cfg.task.gradient_clip_val,
        gradient_clip_algorithm=cfg.task.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.task.accumulate_grad_batches,
        enable_model_summary=True,
    )
    # if no checkpoint_path is passed, then it is None, thus the model will start from the very beginning
    trainer.fit(task, datamodule=datamodule, ckpt_path=cfg.model.checkpoint_path)


def evaluate(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    task: pl.LightningModule
):
    if cfg.model.checkpoint_path is None:
        raise ValueError('no checkpoint path has been passed')

    raise NotImplementedError()


def inference(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    task: pl.LightningModule
):
    if cfg.model.checkpoint_path is None:
        raise ValueError('no checkpoint path has been passed')

    raise NotImplementedError()


@hydra.main(config_path='ukrlm/conf', config_name='config', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    match cfg.datamodule.name:
        case 'masked-language-modeling':
            datamodule = MaskedLanguageModelingDataModule(cfg)
        case _:
            raise ValueError(
                'unknown datamodule, can be either `masked-language-modeling` or ...'
            )

    match cfg.model.name:
        case 'bert-base':
            config = AutoConfig.from_pretrained(
                'bert-base-uncased',
                max_position_embeddings=cfg.model.max_position_embeddings,
                vocab_size=cfg.model.vocab_size,
                pad_token_id=cfg.model.pad_token_id,
                unk_token_id=cfg.model.unk_token_id,
                cls_token_id=cfg.model.cls_token_id,
                sep_token_id=cfg.model.sep_token_id,
                mask_token_id=cfg.model.mask_token_id,
            )
            model: BertForMaskedLM = AutoModelForMaskedLM.from_config(config)
            if cfg.task.use_flash_attention:
                model = model.to_bettertransformer()
            print('Embeddings shape', model.get_input_embeddings().weight.size())
        case 'bert-large':
            config = AutoConfig.from_pretrained(
                'bert-large-uncased',
                max_position_embeddings=cfg.model.max_position_embeddings,
                vocab_size=cfg.model.vocab_size,
                pad_token_id=cfg.model.pad_token_id,
                unk_token_id=cfg.model.unk_token_id,
                cls_token_id=cfg.model.cls_token_id,
                sep_token_id=cfg.model.sep_token_id,
                mask_token_id=cfg.model.mask_token_id,
            )
            model = AutoModelForMaskedLM.from_config(config)
            if cfg.task.use_flash_attention:
                model = model.to_bettertransformer()
            print('Embeddings shape', model.get_input_embeddings().weight.size())
        case 'albert-base-v2':
            raise NotImplementedError()
            # config = AutoConfig.from_pretrained(
            #     'albert-base-v2',
            #     max_position_embeddings=cfg.model.max_position_embeddings,
            #     vocab_size=cfg.model.vocab_size,
            #     pad_token_id=cfg.model.pad_token_id,
            #     unk_token_id=cfg.model.unk_token_id,
            #     cls_token_id=cfg.model.cls_token_id,
            #     sep_token_id=cfg.model.sep_token_id,
            #     mask_token_id=cfg.model.mask_token_id,
            # )
            # model = AutoModelForMaskedLM.from_config(config)
            # print('Embeddings shape', model.get_input_embeddings().weight.size())
        case 'liberta-base':
            raise NotImplementedError()
        case model_name if model_name in ['deberta-v3-base', 'deberta-v3-large']:
            discriminator_config = AutoConfig.from_pretrained(
                'microsoft/' + model_name,
                max_position_embeddings=cfg.model.max_position_embeddings,
                vocab_size=cfg.model.vocab_size,
                pad_token_id=cfg.model.pad_token_id,
                unk_token_id=cfg.model.unk_token_id,
                cls_token_id=cfg.model.cls_token_id,
                sep_token_id=cfg.model.sep_token_id,
                mask_token_id=cfg.model.mask_token_id,
                # deberta specific config
                relative_attention=True,
                position_biased_input=False,
                pos_att_type='p2c|c2p',
                # TODO
            )

            generator_config = DebertaV2Config.from_dict(
                discriminator_config.to_dict(),
                num_hidden_layers=cfg.model.generator_n_layers,
            )

            generator = DebertaV3ForMLM(generator_config)
            discriminator = DebertaV3ForRTD(discriminator_config)

            print('Embeddings shape (generator)', generator.get_input_embeddings().weight.size())
            print('Embeddings shape (discriminator)', discriminator.get_input_embeddings().weight.size())
        case _:
            raise ValueError(
                'unknown model, can be either `bert-base` or `bert-large` or ...'
            )

    match cfg.task.name:
        case 'masked-language-modeling':
            if 'model' not in locals():
                raise ValueError('model must be defined for MLM task. '
                                 'You have probably chosen a wrong model.')
            task = MaskedLanguageModelingTask(cfg, model)
        case 'replaced-token-detection':
            if 'generator' not in locals() or 'discriminator' not in locals():
                raise ValueError('generator and discriminator must be defined for RTD task. '
                                 'You have probably chosen a wrong model.')
            task = ReplacedTokenDetectionTask(cfg, generator, discriminator)
        case _:
            raise ValueError(
                'unknown task, can be either `masked-language-modeling` or ...'
            )

    match cfg.stage:
        case 'train':
            train(cfg, datamodule, task)
        case 'evaluate':
            evaluate(cfg, datamodule, task)
        case 'inference':
            inference(cfg, datamodule, task)
        case _:
            raise ValueError(
                'unknown stage, can be either `train`, `evaluate` or `inference`'
            )


if __name__ == '__main__':
    main()
