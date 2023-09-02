import logging

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from transformers import AutoModelForMaskedLM, AutoConfig

from ukrlm.models.bert import BERT
from ukrlm.datamodules import MaskedLanguageModelingDataModule
from ukrlm.tasks.masked_language_modeling import MaskedLanguageModelingTask

logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    task: pl.LightningModule
):
    datamodule.setup()  # FIXME should it be here?

    # TODO add metrics writing to the checkpoint??
    wandb_logger = WandbLogger(project='ukr-lm')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='output/checkpoints',  # TODO move to config
        filename=f'{cfg.model.name}_step{{step}}_perplexity{{perplexity:.3f}}',  # TODO move to config
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=False,
        every_n_train_steps=50_000,  # TODO move to config
        # train_time_interval=  # TODO should we use it instead of every_n_train_steps?
    )
    # TODO add gradient accumulation parameters
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        overfit_batches=4,  # TODO remove, only for debugging
        # fast_dev_run=True,  # TODO remove, only for debugging
        max_steps=cfg.task.max_steps,
        # max_time=   # TODO this maybe better for Athena with 24h limit
        val_check_interval=1.0,  # TODO move to config
        # val_check_interval=25_000,  # TODO move to config
        log_every_n_steps=200,  # TODO move to config
        # accelerator='
        # strategy=
        # sync_batchnorm=  # TODO what is this?
        precision=16 if cfg.task.fp16 else 32,
        enable_model_summary=True,
        resume_from_checkpoint=cfg.model.checkpoint_path,
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
                pad_token_id=0,
                unk_token_id=1,
                cls_token_id=2,
                sep_token_id=3,
                mask_token_id=4,
            )
            model = AutoModelForMaskedLM.from_config(config)
            print('Embeddings shape', model.get_input_embeddings().weight.size())
        case 'albert-base-v2':
            config = AutoConfig.from_pretrained(
                'albert-base-v2',
                max_position_embeddings=cfg.model.max_position_embeddings,
                vocab_size=cfg.model.vocab_size,
                pad_token_id=0,
                unk_token_id=1,
                cls_token_id=2,
                sep_token_id=3,
                mask_token_id=4,
            )
            model = AutoModelForMaskedLM.from_config(config)
            print('Embeddings shape', model.get_input_embeddings().weight.size())
        case 'liberta-base':
            raise NotImplementedError()
        case _:
            raise ValueError(
                'unknown model, can be either `bert-base` or ...'
            )

    match cfg.task.name:
        case 'masked-language-modeling':
            task = MaskedLanguageModelingTask(cfg, model)
        # case 'text-classification':
        #     task = TextClassificationTask(cfg, model)
        # case 'token-classification':
        #     task = TokenClassificationTask(cfg, model)
        case _:
            raise ValueError(
                'unknown task, can be either `masked-language-modeling`, `text-classification`, `token-classification`'
                ' or ...'
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
