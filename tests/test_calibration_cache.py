from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from britekit.models.base_model import BaseModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_calibration_cache_tracks_config_and_cuda_stream(monkeypatch):
    model = BaseModel(
        model_type="test",
        head_type=None,
        hidden_channels=1,
        train_class_names=["Species"],
        train_class_codes=["SPEC"],
        train_class_alt_names=[""],
        train_class_alt_codes=[""],
        num_train_specs=1,
        multi_label=True,
    )
    model.set_config(deepcopy(model.cfg))
    cfg = model.cfg.infer
    cfg.scaling_coefficient = 1.0
    cfg.scaling_intercept = 0.0
    logits = torch.zeros((1, 1), device="cuda")

    with torch.inference_mode():
        w, b = model._calibration_tensors(logits)
        w_again, b_again = model._calibration_tensors(logits)
        assert w_again is w and b_again is b

        cfg.scaling_intercept = -0.0
        assert model._calibration_tensors(logits)[1].signbit().item()

        with torch.cuda.stream(torch.cuda.Stream()):
            w_alt, b_alt = model._calibration_tensors(logits)
            assert w_alt is not w and b_alt is not b
            assert model._calibration_tensors(logits)[0] is w_alt
        assert model._calibration_tensors(logits)[0] is not w_alt

        coefficients = [1.0]
        monkeypatch.setattr(
            model,
            "cfg",
            SimpleNamespace(
                infer=SimpleNamespace(
                    scaling_coefficient=coefficients, scaling_intercept=0.0
                )
            ),
        )
        assert model._calibration_tensors(logits)[0].item() == 1.0
        coefficients[0] = 2.0
        assert model._calibration_tensors(logits)[0].item() == 2.0
