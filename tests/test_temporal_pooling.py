import os
import subprocess
import sys
from unittest.mock import patch

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.models import model_loader
from britekit.models.head_factory import make_head
from britekit.models.temporal_pooling import pool_temporal_logits


@pytest.fixture(autouse=True)
def config():
    cfg = BaseConfig()
    cfg.audio.spec_height = 32
    cfg.audio.spec_width = 64
    cfg.audio.spec_duration = 2
    cfg.train.head_type = "temporal_sed"
    cfg.train.hidden_channels = 8
    cfg.train.lse_temp = 0.7
    set_base_config(cfg)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield cfg
    torch.set_num_threads(old_threads)
    set_base_config(BaseConfig())


def new_model():
    with patch("britekit.models.model_loader.get_device", return_value="cpu"):
        return model_loader.load_new_model(
            ["Alpha", "Beta"], ["A", "B"], ["", ""], ["", ""], 2
        )


def save_checkpoint(model, path, legacy=False):
    checkpoint = dict(
        epoch=0,
        state_dict=model.state_dict(),
        hyper_parameters=dict(model.hparams),
        **{"pytorch-lightning_version": pl.__version__},
    )
    model.on_save_checkpoint(checkpoint)
    if legacy:
        del checkpoint["hyper_parameters"]["temporal_pooling"]
        del checkpoint["training_cfg"]["train"]["temporal_pooling"]
    torch.save(checkpoint, path)


def test_linear_softmax_matches_probability_formula_and_gradients():
    logits = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    probabilities = logits.sigmoid()
    expected = (probabilities.square().sum(-1) / probabilities.sum(-1)).logit()
    actual = pool_temporal_logits(logits, "linear_softmax")
    torch.testing.assert_close(actual, expected)
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    actual_gradient = torch.autograd.grad(actual.sum(), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_positive_segment_can_suppress_low_confidence_frames():
    logits = torch.tensor([[[0.05, 0.9, 0.1]]]).logit().requires_grad_()
    clip = pool_temporal_logits(logits, "linear_softmax")
    F.binary_cross_entropy_with_logits(clip, torch.ones_like(clip)).backward()
    # Gradient descent lowers the low-probability frames and raises the peak.
    assert logits.grad[0, 0, 0] > 0
    assert logits.grad[0, 0, 1] < 0
    assert logits.grad[0, 0, 2] > 0
    logits.grad = None
    clip = pool_temporal_logits(logits, "logsumexp")
    F.binary_cross_entropy_with_logits(clip, torch.ones_like(clip)).backward()
    assert (logits.grad < 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_extreme_logits_remain_finite_with_gradients(dtype):
    logits = torch.tensor(
        [[[-1000, -1000, -1000], [1000, 1000, 1000], [-1000, 0, 1000]]],
        dtype=dtype,
        requires_grad=True,
    )
    clip = pool_temporal_logits(logits, "linear_softmax")
    assert clip.dtype == torch.float32
    torch.testing.assert_close(clip[0, :2], torch.tensor([-1000.0, 1000.0]))
    F.binary_cross_entropy_with_logits(clip, torch.tensor([[1.0, 0.0, 1.0]])).backward()
    assert torch.isfinite(clip).all()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[0, 0] < 0).all()
    assert (logits.grad[0, 1] > 0).all()


@pytest.mark.parametrize("head_type", ["temporal_sed", "prototype_sed"])
def test_changing_pooling_preserves_initial_weights_and_frame_logits(head_type):
    torch.manual_seed(99)
    original = make_head(head_type, 4, 8, 2).eval()
    torch.manual_seed(99)
    linear = make_head(head_type, 4, 8, 2, temporal_pooling="linear_softmax").eval()
    assert original.state_dict().keys() == linear.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, linear.state_dict()[key], rtol=0, atol=0)
    inputs = torch.randn(2, 4, 3, 12)
    original_clip, original_frames = original(inputs)
    linear_clip, linear_frames = linear(inputs)
    torch.testing.assert_close(original_frames, linear_frames, rtol=0, atol=0)
    legacy = 0.5 * (torch.logsumexp(original_frames / 0.5, dim=-1) - np.log(12))
    torch.testing.assert_close(original_clip, legacy, rtol=0, atol=0)
    probabilities = linear_frames.sigmoid()
    torch.testing.assert_close(
        linear_clip.sigmoid(), probabilities.square().sum(-1) / probabilities.sum(-1)
    )


@pytest.mark.parametrize("head_type", [None, "basic", "basic_sed", "temporal_sed"])
def test_invalid_pooling_is_rejected(config, head_type):
    config.train.head_type = head_type
    config.train.temporal_pooling = (
        "unknown" if head_type == "temporal_sed" else "linear_softmax"
    )
    with pytest.raises(ValueError, match="temporal_pooling"):
        new_model()


@pytest.mark.parametrize("model_type", ["bknet.3", "timm.convnextv2_femto"])
@pytest.mark.parametrize("pooling", ["logsumexp", "linear_softmax", "legacy"])
def test_checkpoint_restores_pooling_independent_of_runtime_config(
    config, tmp_path, model_type, pooling
):
    config.train.model_type = model_type
    expected_pooling = "logsumexp" if pooling == "legacy" else pooling
    config.train.temporal_pooling = expected_pooling
    model = new_model().eval()
    inputs = torch.randn(2, 1, 32, 64)
    with torch.no_grad():
        expected = model(inputs)
    checkpoint = tmp_path / "model.ckpt"
    # Saving must record the actual head, even if the global config changes.
    config.train.temporal_pooling = (
        "linear_softmax" if pooling != "linear_softmax" else "logsumexp"
    )
    config.train.lse_temp = 9
    save_checkpoint(model, checkpoint, legacy=pooling == "legacy")
    with patch("britekit.models.model_loader.get_device", return_value="cpu"):
        loaded = model_loader.load_from_checkpoint(str(checkpoint)).eval()
    assert loaded.head.temporal_pooling == expected_pooling
    assert config.train.temporal_pooling == expected_pooling
    assert loaded.head.lse_temp == 0.7
    assert config.train.lse_temp == 0.7
    with torch.no_grad():
        for original, restored in zip(expected, loaded(inputs)):
            torch.testing.assert_close(original, restored)


def test_compiled_mixed_precision_forward_and_backward():
    head = make_head("temporal_sed", 4, 8, 2, temporal_pooling="linear_softmax")
    compiled = torch.compile(head, backend="aot_eager", fullgraph=True)
    try:
        inputs = torch.randn(2, 4, 3, 12, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            clip, frames = compiled(inputs)
            loss = F.binary_cross_entropy_with_logits(clip, torch.ones_like(clip))
            loss += 0.2 * F.binary_cross_entropy_with_logits(
                frames, torch.rand_like(frames)
            )
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(inputs.grad).all()
        for parameter in head.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        assert head.frame_head.weight.grad.abs().sum() > 0
    finally:
        torch._dynamo.reset()


def test_inference_only_loads_linear_softmax(config, tmp_path):
    config.train.model_type = "bknet.3"
    config.train.temporal_pooling = "linear_softmax"
    path = tmp_path / "model.ckpt"
    save_checkpoint(new_model(), path)
    script = """
import sys
import torch
from britekit.models.bknet import BKNetModel
torch.set_num_threads(1)
model = BKNetModel.load_from_checkpoint(sys.argv[1]).eval()
assert model.head.temporal_pooling == "linear_softmax"
with torch.no_grad():
    clip, frames = model.head(torch.randn(1, model.backbone.num_features, 2, 12))
    p = frames.sigmoid()
    torch.testing.assert_close(clip.sigmoid(), p.square().sum(-1) / p.sum(-1))
assert "lightning" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={**os.environ, "BRITEKIT_INFERENCE_ONLY": "1"},
        check=True,
    )


def test_onnx_openvino_linear_softmax_parity(tmp_path):
    pytest.importorskip("onnx")
    ov = pytest.importorskip("openvino")
    head = make_head("temporal_sed", 4, 8, 2, temporal_pooling="linear_softmax").eval()
    inputs = torch.randn(2, 4, 3, 12)
    path = tmp_path / "head.onnx"
    torch.onnx.export(head, (inputs,), str(path), opset_version=17, dynamo=False)
    compiled = ov.Core().compile_model(
        str(path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"}
    )
    with torch.no_grad():
        expected = head(inputs)
    outputs = compiled(inputs.numpy())
    for index, value in enumerate(expected):
        np.testing.assert_allclose(
            outputs[compiled.output(index)], value.numpy(), rtol=1e-4, atol=1e-5
        )
