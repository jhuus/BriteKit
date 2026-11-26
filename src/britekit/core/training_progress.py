from lightning.pytorch.callbacks import TQDMProgressBar


class TrainingProgressBar(TQDMProgressBar):
    """Read the training loss only when refreshing the display."""

    def get_metrics(self, trainer, model):
        metrics = super().get_metrics(trainer, model)
        if trainer.state.fn == "fit":
            loss = trainer.logged_metrics.get("loss")
            if loss is not None:
                metrics["loss"] = loss.item()
        return metrics
