import logging

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch

from ukrlm.models.bert import BERT
from ukrlm.tasks.masked_language_modeling import MaskedLanguageModelingTask

logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    datamodule: pl.LightningDataModule,
    task: pl.LightningModule
):
    raise NotImplementedError()
    datamodule.setup()

    # TODO add metrics writing to the checkpoint??
    # wandb_logger = WandbLogger(project="")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # TODO
    )
    trainer = pl.Trainer(
        # TODO
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
        case 'c4':
            datamodule = C4DataModule(cfg)
        case _:
            raise ValueError(
                'unknown datamodule, can be either `c4` or ...'
            )

    match cfg.model.name:
        case 'bert-base':
            model = BERT(cfg)
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
