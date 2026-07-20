from copy import deepcopy
from types import SimpleNamespace

import yaml

from britekit.core import util
from britekit.core.base_config import BaseConfig
from britekit.core.predictor import Predictor
from britekit.models.base_model import BaseModel


def test_checkpoint_records_completed_epochs():
    model = SimpleNamespace(
        cfg=BaseConfig(), identifier="model-id", training_date="2026-07-14"
    )
    model.cfg.train.num_epochs = 45
    checkpoint = {"epoch": 42}

    BaseModel.on_save_checkpoint(model, checkpoint)

    assert checkpoint["training_cfg"]["train"]["num_epochs"] == 43
    assert model.cfg.train.num_epochs == 45


def test_loading_old_checkpoint_corrects_completed_epochs():
    cfg = BaseConfig()
    training_cfg = util.cfg_to_pure(deepcopy(cfg))
    training_cfg["train"]["num_epochs"] = 45
    checkpoint = {
        "epoch": 42,
        "identifier": "model-id",
        "training_date": "2026-07-14",
        "training_cfg": training_cfg,
    }
    model = SimpleNamespace(cfg=cfg)

    BaseModel.on_load_checkpoint(model, checkpoint)

    assert model.training_cfg["train"]["num_epochs"] == 43


def test_manifest_uses_checkpoint_completed_epochs(tmp_path):
    training_cfg = util.cfg_to_pure(BaseConfig())
    training_cfg["train"]["num_epochs"] = 43
    model = SimpleNamespace(
        identifier="model-id",
        training_date="2026-07-14",
        training_cfg=training_cfg,
    )
    predictor = SimpleNamespace(
        models=[model],
        class_names=["Species"],
        class_codes=["SPEC"],
        cfg=BaseConfig(),
        ov=None,
    )

    Predictor.save_manifest(predictor, str(tmp_path))

    with open(tmp_path / "manifest.yaml") as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    assert manifest["model 1"]["train"]["num_epochs"] == 43
