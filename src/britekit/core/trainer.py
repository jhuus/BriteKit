# Defer some imports to improve initialization performance.
from pathlib import Path

from britekit.core.config_loader import get_config


class Trainer:
    """
    Run training as specified in configuration.
    """

    def __init__(self):
        import pytorch_lightning as pl
        import torch

        self.cfg, _ = get_config()
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

        for k in range(self.cfg.train.num_folds):
            logger = TensorBoardLogger(
                save_dir="logs", name=f"fold-{k}", default_hp_metric=False
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
                out_file.writelines([str(model.backbone)])

            out_path = Path(trainer.logger.log_dir) / "head.txt"
            with open(out_path, "w") as out_file:
                out_file.writelines([str(model.head)])

            # run training and optionally test
            trainer.fit(model, dm)

            if self.cfg.train.test_pickle is not None:
                trainer.test(model, dm)

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
