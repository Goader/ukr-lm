import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info


class MetricsLoggingCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        precision = trainer.callback_metrics['precision']
        recall = trainer.callback_metrics['recall']
        f1 = trainer.callback_metrics['f1']
        support = int(trainer.callback_metrics['support'])

        rank_zero_info('V')  # fixes the overwriting of the progress bar, maybe there is a better way to do this
        rank_zero_info(f'metrics // precision: {100*precision:.4f}, recall: {100*recall:.4f}, '
                       f'f1: {100*f1:.4f}, support: {support}')
