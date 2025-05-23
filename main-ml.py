import logging
import os

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

import lightning.pytorch as pl
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
import torch
from transformers import AutoModelForMaskedLM, AutoConfig, AutoModel, DebertaV2Config
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM

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

    if cfg.optimizer.name == 'optimi.StableAdamW' and cfg.task.gradient_clip_val > 0:
        rank_zero_warn(
            'Gradient clipping is not supported for StableAdamW optimizer. '
            'Setting gradient_clip_val to None.'
        )
        cfg.task.gradient_clip_val = None

    # Get number of nodes from SLURM environment variables
    num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    print(f"Number of nodes: {num_nodes}")

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
        # detect_anomaly=True,
        max_epochs=cfg.task.max_epochs,
        max_steps=cfg.task.max_steps,
        val_check_interval=0.1,
        log_every_n_steps=cfg.task.log_every_n_steps,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        num_nodes=num_nodes,
        strategy=cfg.task.strategy,
        sync_batchnorm=cfg.task.sync_batchnorm,
        precision=cfg.task.precision,
        gradient_clip_val=cfg.task.gradient_clip_val,
        gradient_clip_algorithm=cfg.task.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.task.accumulate_grad_batches,
        enable_model_summary=True,
    )
    # if no checkpoint_path is passed, then it is None, thus the model will start from the very beginning
    trainer.fit(
        task,
        datamodule=datamodule,
        ckpt_path=(
            cfg.model.checkpoint_path
            if not getattr(cfg.model, 'context_extension_phase', False)
            else None
        )
    )


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
            model: BertForMaskedLM = AutoModelForMaskedLM.from_config(config)
            if cfg.task.use_flash_attention:
                model = model.to_bettertransformer()
            print('Embeddings shape', model.get_input_embeddings().weight.size())

        case 'albert-base-v2':
            raise NotImplementedError()

        case 'liberta-base':
            raise NotImplementedError()

        case model_name if model_name in ['modernbert-large', 'modernbert-base']:
            config = AutoConfig.from_pretrained(
                'answerdotai/' + model_name,
                vocab_size=cfg.model.vocab_size,
                pad_token_id=cfg.model.pad_token_id,
                unk_token_id=cfg.model.unk_token_id,
                cls_token_id=cfg.model.cls_token_id,
                sep_token_id=cfg.model.sep_token_id,
                mask_token_id=cfg.model.mask_token_id,
                global_rope_theta=cfg.model.global_rope_theta \
                    if cfg.model.context_extension_phase else cfg.model.local_rope_theta,
                local_rope_theta=cfg.model.local_rope_theta,
                reference_compile=cfg.model.reference_compile,
                attn_implementation=cfg.model.attn_implementation,
            )
            model: ModernBertForMaskedLM = AutoModelForMaskedLM.from_config(config)

            if cfg.model.initialize.backbone is not None:
                checkpoint = torch.load(cfg.model.initialize.backbone)

                # tokenizer-related parameters are initialized from scratch
                checkpoint['model.embeddings.tok_embeddings.weight'] = model.model.embeddings.tok_embeddings.weight
                checkpoint['decoder.weight'] = model.decoder.weight
                checkpoint['decoder.bias'] = model.decoder.bias

                model.load_state_dict(checkpoint, strict=True)
                del checkpoint

            if cfg.model.initialize.embeddings is not None:
                checkpoint = torch.load(cfg.model.initialize.embeddings)
                model.get_input_embeddings().weight.data = checkpoint['input_embeddings']

                if 'output_embeddings' in checkpoint:
                    model.get_output_embeddings().weight.data = checkpoint['output_embeddings']
                del checkpoint

            if cfg.model.context_extension_phase:
                assert cfg.model.checkpoint_path is not None, 'checkpoint_path must be passed for context extension phase'
                ckpt = torch.load(cfg.model.checkpoint_path)
                state_dict = {
                    key.removeprefix('model.'): value
                    for key, value in ckpt['state_dict'].items()
                    if 'model.' in key
                }
                model.load_state_dict(state_dict, strict=True)
                del ckpt, state_dict

            if cfg.model.tie_weights:
                model.tie_weights()
                print('Weights tied. Embeddings identity: '
                      f'{model.get_input_embeddings().weight is model.get_output_embeddings().weight}')

            print('Embeddings shape', model.get_input_embeddings().weight.size())
            print('Output embeddings shape', model.get_output_embeddings().weight.size())

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
                # deberta specific config parameters
                relative_attention=True,
                position_biased_input=False,
                pos_att_type='p2c|c2p',
            )

            generator_config = DebertaV2Config.from_dict(
                discriminator_config.to_dict(),
                num_hidden_layers=cfg.model.generator_n_layers,
            )

            generator = DebertaV3ForMLM(generator_config)
            discriminator = DebertaV3ForRTD(discriminator_config)

            if cfg.model.initialize.generator_backbone is not None:
                checkpoint = torch.load(cfg.model.initialize.generator_backbone)

                # tokenizer-related parameters are initialized from scratch
                checkpoint['deberta.embeddings.word_embeddings.weight'] = generator.deberta.embeddings.word_embeddings.weight
                checkpoint['lm_predictions.lm_head.bias'] = generator.lm_predictions.lm_head.bias

                generator.load_state_dict(checkpoint, strict=True)
                del checkpoint

            if cfg.model.initialize.discriminator_backbone is not None:
                checkpoint = torch.load(cfg.model.initialize.discriminator_backbone)

                # tokenizer-related parameters are initialized from scratch
                checkpoint['deberta.embeddings.word_embeddings.weight'] = discriminator.deberta.embeddings.word_embeddings.weight
                checkpoint['mask_predictions.classifier.weight'] = discriminator.mask_predictions.classifier.weight
                checkpoint['mask_predictions.classifier.bias'] = discriminator.mask_predictions.classifier.bias

                discriminator.load_state_dict(checkpoint, strict=True)
                del checkpoint

            if cfg.model.initialize.generator_embeddings is not None:
                checkpoint = torch.load(cfg.model.initialize.generator_embeddings)
                generator.get_input_embeddings().weight.data = checkpoint['input_embeddings']
                del checkpoint

            if cfg.model.initialize.discriminator_embeddings is not None:
                checkpoint = torch.load(cfg.model.initialize.discriminator_embeddings)
                discriminator.get_input_embeddings().weight.data = checkpoint['input_embeddings']
                del checkpoint

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
