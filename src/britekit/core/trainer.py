#!/usr/bin/env python3

# Defer some imports to improve initialization performance.
import logging
from pathlib import Path
import yaml

from britekit.core.config_loader import get_config
from britekit.models.model_inspector import ModelInspector
from britekit.core.util import cfg_to_pure


class Trainer:
    """
    Run training as specified in configuration.
    """

    def __init__(self):
        import pytorch_lightning as pl
        import torch

        self.cfg = get_config()
        torch.set_float32_matmul_precision("medium")
        if self.cfg.train.seed is not None:
            pl.seed_everything(self.cfg.train.seed, workers=True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.cfg.train.seed)

        if self.cfg.train.deterministic:
            # should also set num_workers = 0 or 1
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # this speeds it up a little
            torch.utils.deterministic.fill_uninitialized_memory = False

    def run(self):
        """
        Run training as specified in configuration.
        """
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
        from pytorch_lightning.loggers import TensorBoardLogger
        import torch

        from britekit.core.data_module import DataModule
        from britekit.models import model_loader

        # load all the data once for performance, then split as needed in each fold
        dm = DataModule()

        val_rocs = []
        for k in range(self.cfg.train.num_folds):
            logger = TensorBoardLogger(
                save_dir="logs",
                name=None if self.cfg.train.num_folds == 1 else f"fold-{k}",
                default_hp_metric=False,
            )
            version = (
                logger.version
                if isinstance(logger.version, int)
                else str(logger.version)
            )

            if self.cfg.train.deterministic:
                deterministic = "warn"
            else:
                deterministic = False

            trainer = pl.Trainer(
                devices=1,
                accelerator="auto",
                callbacks=[
                    ModelCheckpoint(
                        save_top_k=self.cfg.train.save_last_n,
                        mode="max",
                        monitor="epoch",
                        filename=f"v{version}-e{{epoch}}",
                        auto_insert_metric_name=False,
                    ),
                    TQDMProgressBar(refresh_rate=10),
                ],
                deterministic=deterministic,
                max_epochs=self.cfg.train.num_epochs,
                precision="16-mixed" if self.cfg.train.mixed_precision else 32,
                logger=logger,
            )

            dm.prepare_fold(k)

            # create model inside loop so parameters are reset for each fold,
            # and so metrics are tracked correctly
            if self.cfg.train.load_ckpt_path:
                model = model_loader.load_from_checkpoint(
                    self.cfg.train.load_ckpt_path,
                    multi_label=self.cfg.train.multi_label,
                )
                if self.cfg.train.freeze_backbone:
                    model.freeze_backbone()
            else:
                model = model_loader.load_new_model(
                    dm.train_class_names,
                    dm.train_class_codes,
                    dm.train_class_alt_names,
                    dm.train_class_alt_codes,
                    dm.num_train_specs,
                )

            if self.cfg.train.compile:
                model = torch.compile(model)

            # force the log directory to be created, then save model descriptions
            trainer.logger.experiment
            out_path = Path(trainer.logger.log_dir) / "backbone.txt"
            with open(out_path, "w") as out_file:
                out_file.write(f"=== {self.cfg.train.model_type} ===\n\n")
                out_file.writelines([str(model.backbone)])

            inspector = ModelInspector(model.backbone)
            inspector.register_hooks()

            input_shape = (
                1,
                1,
                self.cfg.audio.spec_height,
                self.cfg.audio.spec_width,
            )
            x = torch.randn(*input_shape).to(model.device)
            _ = model.backbone(x)
            inspector.remove()

            out_path = Path(trainer.logger.log_dir) / "backbone-shapes.txt"
            with open(out_path, "w") as out_file:
                out_file.write(f"=== {self.cfg.train.model_type} ===\n\n")
                inspector.print_report(file=out_file)

            out_path = Path(trainer.logger.log_dir) / "head.txt"
            with open(out_path, "w") as out_file:
                if self.cfg.train.head_type is None:
                    out_file.write("=== Default Head ===\n\n")
                else:
                    out_file.write(f"=== {self.cfg.train.head_type} ===\n\n")
                out_file.writelines([str(model.head)])

            # save training parameters in YAML format
            info_str = yaml.dump(cfg_to_pure(model.cfg.train), sort_keys=False)
            info_str = "# Training parameters in YAML format\n" + info_str
            out_path = Path(trainer.logger.log_dir) / "config.yaml"
            with open(out_path, "w") as out_file:
                out_file.write(info_str)

            # run training and optionally test
            trainer.fit(model, dm)

            if self.cfg.train.test_pickle is not None:
                trainer.test(model, dm)

            # save stats from k-fold cross-validation
            if self.cfg.train.num_folds > 1 and "val_roc" in trainer.callback_metrics:
                val_rocs.append(float(trainer.callback_metrics["val_roc"]))

        if val_rocs:
            import math
            import numpy as np

            mean = float(np.mean(val_rocs))
            std = float(np.std(val_rocs, ddof=1)) if len(val_rocs) > 1 else 0.0
            n = len(val_rocs)
            se = std / math.sqrt(n) if n > 1 else 0.0
            ci95 = 1.96 * se  # 95% CI using normal approximation

            logging.info("Using micro-averaged ROC AUC")
            scores_str = ", ".join(f"{v:.4f}" for v in val_rocs)
            logging.info(f"folds: {scores_str}")
            logging.info(f"mean: {mean:.4f}")
            logging.info(f"standard deviation: {std:.4f}")
            logging.info(f"95% confidence interval: {mean-ci95:.4f} to {mean+ci95:.4f}")

    def find_lr(self, num_batches: int = 100):
        """
        Suggest a learning rate and produce a plot.
        """
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import TQDMProgressBar
        from pytorch_lightning.tuner import Tuner

        from britekit.core.data_module import DataModule
        from britekit.models import model_loader

        dm = DataModule()
        dm.prepare_fold(0)

        trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            callbacks=[
                TQDMProgressBar(refresh_rate=10),
            ],
            deterministic=self.cfg.train.deterministic,
            max_epochs=1,
            precision="16-mixed" if self.cfg.train.mixed_precision else 32,
        )

        model = model_loader.load_new_model(
            dm.train_class_names,
            dm.train_class_codes,
            dm.train_class_alt_names,
            dm.train_class_alt_codes,
            dm.num_train_specs,
        )

        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            model, datamodule=dm, min_lr=1e-7, max_lr=10, num_training=num_batches
        )

        assert lr_finder is not None
        return lr_finder.suggestion(), lr_finder.plot(suggest=True)
