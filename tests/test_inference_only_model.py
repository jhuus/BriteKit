import subprocess
import sys


def test_inference_only_model_loads_checkpoint_without_training_dependencies(
    tmp_path,
):
    checkpoint_path = tmp_path / "model.ckpt"
    script = f"""
import os
import sys

os.environ["BRITEKIT_INFERENCE_ONLY"] = "1"

import torch
from britekit.models.base_model import BaseModel

parameters = {{
    "model_type": "test",
    "head_type": None,
    "hidden_channels": 1,
    "train_class_names": ["Species"],
    "train_class_codes": ["SPEC"],
    "train_class_alt_names": [""],
    "train_class_alt_codes": [""],
    "num_train_specs": 1,
    "multi_label": True,
}}
model = BaseModel(**parameters)
checkpoint = {{
    "hyper_parameters": parameters,
    "state_dict": model.state_dict(),
    "identifier": "model-id",
    "training_date": "2026-08-30",
    "training_cfg": {{}},
}}
torch.save(checkpoint, {str(checkpoint_path)!r})
loaded = BaseModel.load_from_checkpoint({str(checkpoint_path)!r})

assert loaded.identifier == "model-id"
assert "lightning" not in sys.modules
assert "torchmetrics" not in sys.modules
assert "timm.optim" not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], check=True)
