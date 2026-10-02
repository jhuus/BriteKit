import copy

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from britekit.core import config_loader
from britekit.core.base_config import BaseConfig
from britekit.core.trainer import Trainer
from britekit.models.base_model import BaseModel


class TinyModel(BaseModel):
    def __init__(self):
        super().__init__(
            "test", None, 4, ["A", "B"], ["A", "B"], ["", ""], ["", ""], 8, True
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(4, 2)
        self.logged_names = []

    def log(self, name, *args, **kwargs):
        # A compiled Lightning training/validation step would enter this method
        # under Dynamo, specializing on names and changing logger/LR state.
        assert not torch.compiler.is_compiling()
        self.logged_names.append(name)
        return super().log(name, *args, **kwargs)


class TinyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_class_names = self.train_class_codes = ["A", "B"]
        self.train_class_alt_names = self.train_class_alt_codes = ["", ""]
        self.num_train_specs = 8
        self.samples = [
            {
                "input": torch.randn(1, 8, 8),
                "segment_labels": torch.tensor([float(i % 2), float(1 - i % 2)]),
            }
            for i in range(8)
        ]
        self.val_data = self.samples

    def prepare_fold(self, fold):
        pass

    def class_weights(self):
        return np.ones(2)

    def train_dataloader(self):
        return DataLoader(self.samples, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.samples, batch_size=4)


def test_compiled_training_keeps_logging_eager_and_checkpoints_loadable(
    monkeypatch, tmp_path
):
    cfg = BaseConfig()
    cfg.audio.spec_height = cfg.audio.spec_width = 8
    cfg.train.compile = True
    cfg.train.num_epochs = 2
    cfg.train.optimizer = "adamw"
    monkeypatch.setattr(config_loader, "_base_config", cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("britekit.core.data_module.DataModule", TinyDataModule)
    model = TinyModel()
    initial_state = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(
        "britekit.models.model_loader.load_new_model", lambda *args: model
    )

    # Exercise Dynamo and compiled autograd on CPU without a GPU/compiler toolchain.
    original_compile = torch.compile
    monkeypatch.setattr(
        torch, "compile", lambda fn: original_compile(fn, backend="aot_eager")
    )
    original_trainer = pl.Trainer

    def cpu_trainer(**kwargs):
        kwargs.update(accelerator="cpu", enable_model_summary=False)
        return original_trainer(**kwargs)

    monkeypatch.setattr(pl, "Trainer", cpu_trainer)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        Trainer().run()
        assert {"lr", "loss", "val_loss", "val_roc"} <= set(model.logged_names)
        assert any(
            not torch.equal(value, initial_state[key])
            for key, value in model.state_dict().items()
        )

        path = sorted(tmp_path.glob("logs/version_*/checkpoints/*e1.ckpt"))[0]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["state_dict"].keys() == initial_state.keys()
        restored = TinyModel().eval()
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        inputs = torch.randn(2, 1, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs)[0], restored(inputs)[0])
    finally:
        torch.set_num_threads(old_threads)
        torch._dynamo.reset()
