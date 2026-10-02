import copy
import json
import pickle
from unittest.mock import patch

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch import nn

from britekit.commands import prototype_export
from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.core.exceptions import ModelError
from britekit.core.util import compress_spectrogram
from britekit.models import model_loader
from britekit.models.base_model import BaseModel
from britekit.models.head_factory import is_sed, make_head
from britekit.models.prototype_head import PrototypeSEDHead


@pytest.fixture(autouse=True)
def config():
    cfg = BaseConfig()
    cfg.audio.spec_height = 32
    cfg.audio.spec_width = 64
    cfg.audio.spec_duration = 2
    cfg.train.model_type = "effnet.1"
    cfg.train.head_type = "prototype_sed"
    cfg.train.prototypes_per_class = 3
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


def save_checkpoint(model, path):
    checkpoint = dict(
        epoch=0,
        state_dict=model.state_dict(),
        hyper_parameters=dict(model.hparams),
        **{"pytorch-lightning_version": pl.__version__},
    )
    model.on_save_checkpoint(checkpoint)
    torch.save(checkpoint, path)


def test_shapes_pooling_and_gradients():
    head = make_head("prototype_sed", 4, 8, 2, prototypes_per_class=3, lse_temp=0.7)
    assert is_sed("prototype_sed")
    x = torch.randn(2, 4, 3, 7, requires_grad=True)
    clip, frames = head(x)
    assert clip.shape == (2, 2)
    assert frames.shape == (2, 2, 7)
    manual = (head.similarity_maps(x).amax(3) * head.weights[None, :, :, None]).sum(
        2
    ) + head.bias[None, :, None]
    torch.testing.assert_close(frames, manual)
    # LSE is bounded by temporal minimum and maximum.
    assert (clip <= frames.amax(-1) + 1e-6).all()
    assert (clip >= frames.amin(-1) - 1e-6).all()
    (clip.square().mean() + frames.square().mean()).backward()
    for tensor in (x, head.prototypes, head.raw_weights, head.bias):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_vectors_are_finite(dtype):
    head = PrototypeSEDHead(4, 2, 3)
    with torch.no_grad():
        head.prototypes.zero_()
    x = torch.zeros(2, 4, 3, 5, dtype=dtype, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype != torch.float32):
        clip, frames = head(x)
    (clip.sum() + frames.sum()).backward()
    assert torch.isfinite(clip).all() and torch.isfinite(frames).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(head.prototypes.grad).all()


def test_species_isolation_and_positive_readout():
    head = PrototypeSEDHead(4, 2, 3)
    x = torch.randn(2, 4, 3, 5)
    before = head(x)
    with torch.no_grad():
        head.prototypes[0].normal_()
        head.raw_weights[0].fill_(-10)
        head.bias[0].add_(5)
    after = head(x)
    for a, b in zip(before, after):
        torch.testing.assert_close(a[:, 1], b[:, 1])
        assert not torch.allclose(a[:, 0], b[:, 0])
    assert (head.weights >= 0).all()
    head.zero_grad()
    head(x)[1][:, 1].sum().backward()
    assert head.prototypes.grad[0].count_nonzero() == 0
    assert head.raw_weights.grad[0].count_nonzero() == 0


@pytest.mark.parametrize("count,temp", [(0, 0.5), (3, 0), (3, float("nan"))])
def test_invalid_options(count, temp):
    with pytest.raises(ValueError):
        PrototypeSEDHead(4, 2, count, temp)


@pytest.mark.parametrize("pooling", ["logsumexp", "linear_softmax"])
def test_checkpoint_roundtrip_and_existing_loss(config, tmp_path, pooling):
    config.train.temporal_pooling = pooling
    model = new_model().eval()
    x = torch.randn(2, 1, 32, 64)
    clip, frames = model(x)
    assert frames.shape == (2, 2, 8)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = model._calc_loss(clip, frames, labels, labels)
    loss.backward()
    assert model.head.prototypes.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.backbone.parameters()
    )
    path = tmp_path / "model.ckpt"
    save_checkpoint(model, path)
    config.train.prototypes_per_class = 20
    config.train.lse_temp = 9
    config.train.temporal_pooling = "logsumexp"
    with patch("britekit.models.model_loader.get_device", return_value="cpu"):
        loaded = model_loader.load_from_checkpoint(str(path)).eval()
    assert loaded.head.prototypes_per_class == 3
    assert loaded.head.lse_temp == 0.7
    assert loaded.head.temporal_pooling == pooling
    assert config.train.temporal_pooling == pooling
    assert config.train.prototypes_per_class == 3
    for expected, actual in zip((clip, frames), loaded(x)):
        torch.testing.assert_close(expected, actual)


def test_initialize_backbone_preserves_fresh_head(config, tmp_path):
    config.train.head_type = "temporal_sed"
    source = new_model()
    path = tmp_path / "source.ckpt"
    save_checkpoint(source, path)
    config.train.head_type = "prototype_sed"
    target = new_model()
    fresh_head = copy.deepcopy(target.head.state_dict())
    model_loader.initialize_backbone(target, str(path))
    for key, value in source.backbone.state_dict().items():
        torch.testing.assert_close(value, target.backbone.state_dict()[key])
    for key, value in fresh_head.items():
        torch.testing.assert_close(value, target.head.state_dict()[key])
    target.freeze_backbone()
    assert not any(p.requires_grad for p in target.backbone.parameters())
    assert all(p.requires_grad for p in target.head.parameters())
    target.train_class_codes.reverse()
    with pytest.raises(ModelError, match="train_class_codes"):
        model_loader.initialize_backbone(target, str(path))
    target.train_class_codes.reverse()
    config.audio.max_freq = 7000
    with pytest.raises(ModelError, match="audio.max_freq"):
        model_loader.initialize_backbone(target, str(path))


def test_export_matches_and_saved_examples(config, tmp_path):
    config.audio.spec_height = 2
    config.audio.spec_width = 4
    model = BaseModel(
        "test",
        "prototype_sed",
        2,
        ["Alpha", "Beta"],
        ["A", "B"],
        ["", ""],
        ["", ""],
        2,
        True,
    )
    model.backbone = nn.Identity()
    model.head = PrototypeSEDHead(1, 2, 1)
    with torch.no_grad():
        model.head.prototypes.fill_(1)
    data = dict(
        class_names=["Alpha", "Beta"],
        class_codes=["A", "B"],
        spec_values=[
            compress_spectrogram(np.ones((2, 4), dtype=np.float32)),
            compress_spectrogram(np.zeros((2, 4), dtype=np.float32)),
        ],
        spec_class_indexes=[[0], [1]],
        spec_segment_ids=[10, 20],
        spec_recording_ids=[100, 200],
    )
    path = tmp_path / "train.pkl"
    with path.open("wb") as f:
        pickle.dump(data, f)
    dest = tmp_path / "prototypes"
    with patch("britekit.models.model_loader.load_from_checkpoint", return_value=model):
        prototype_export(
            str(path), "model.ckpt", str(dest), top_k=1, batch_size=1, device="cpu"
        )
    manifest = json.loads((dest / "manifest.json").read_text())
    for prototype in manifest["prototypes"]:
        match = prototype["matches"][0]
        assert match["segment_id"] == 10
        assert match["similarity"] == pytest.approx(1)
        assert match["time_cell"] == 0 and match["frequency_cell"] == 0
        assert (dest / match["example_file"]).exists()
    vectors = np.load(dest / "prototypes.npz")
    assert vectors["vectors"].shape == (2, 1, 1)
    assert vectors["class_codes"].tolist() == ["A", "B"]
    with pytest.raises(FileExistsError):
        prototype_export(str(path), "model.ckpt", str(dest))


def test_inference_only_prototype_checkpoint(config, tmp_path):
    import os
    import subprocess
    import sys

    model = new_model().eval()
    path = tmp_path / "prototype.ckpt"
    save_checkpoint(model, path)
    script = """
import sys
import torch
from britekit.models.effnet import EffNetModel
model = EffNetModel.load_from_checkpoint(sys.argv[1]).eval()
model.apply_training_config(model.cfg)
assert model.head.prototypes_per_class == 3
assert model.head.lse_temp == .7
with torch.no_grad():
    clip, frames = model(torch.ones(1, 1, 32, 64))
assert frames.shape == (1, 2, 8)
assert 'lightning' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={**os.environ, "BRITEKIT_INFERENCE_ONLY": "1"},
        check=True,
    )


def test_onnx_openvino_parity(config, tmp_path):
    pytest.importorskip("onnx")
    ov = pytest.importorskip("openvino")
    from britekit.commands import ckpt_onnx

    config.infer.openvino_block_size = 2
    model = new_model().eval()
    checkpoint = tmp_path / "prototype.ckpt"
    save_checkpoint(model, checkpoint)
    with patch("britekit.models.model_loader.get_device", return_value="cpu"):
        ckpt_onnx(input_path=str(checkpoint))
    compiled = ov.Core().compile_model(
        str(checkpoint.with_suffix(".onnx")), "CPU", {"INFERENCE_PRECISION_HINT": "f32"}
    )
    inputs = torch.randn(2, 1, 32, 64)
    with torch.no_grad():
        expected = model(inputs)
    outputs = compiled(inputs.numpy())
    for index, value in enumerate(expected):
        np.testing.assert_allclose(
            outputs[compiled.output(index)], value.numpy(), rtol=1e-4, atol=1e-5
        )


def test_bad_backbone_does_not_partially_load(config, tmp_path):
    source = new_model()
    path = tmp_path / "bad.ckpt"
    save_checkpoint(source, path)
    checkpoint = torch.load(path, weights_only=False)
    key = next(
        k
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("backbone.") and v.ndim > 1
    )
    checkpoint["state_dict"][key] = torch.zeros(1)
    torch.save(checkpoint, path)
    target = new_model()
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ModelError, match="shape_mismatch"):
        model_loader.initialize_backbone(target, str(path))
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_prototype_load_rejects_missing_parameters(config, tmp_path):
    model = new_model()
    path = tmp_path / "bad.ckpt"
    save_checkpoint(model, path)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["state_dict"]["head.prototypes"]
    torch.save(checkpoint, path)
    with patch("britekit.models.model_loader.get_device", return_value="cpu"):
        with pytest.raises(RuntimeError, match="head.prototypes"):
            model_loader.load_from_checkpoint(str(path))
