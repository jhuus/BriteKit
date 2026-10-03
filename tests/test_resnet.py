import lightning.pytorch as pl
import numpy as np
import pytest
import timm
import torch
from torch import nn
from torch.nn import functional as F

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.models import model_loader
from britekit.models.resnet import MODEL_REGISTRY, ResNetModel


@pytest.fixture(autouse=True)
def config(monkeypatch):
    cfg = BaseConfig()
    cfg.audio.spec_height = 192
    cfg.audio.spec_width = 384
    cfg.audio.spec_duration = 3
    cfg.train.sed_fps = 4
    cfg.train.head_type = "prototype_sed"
    cfg.train.prototypes_per_class = 20
    cfg.train.hidden_channels = 8
    set_base_config(cfg)
    monkeypatch.setattr(model_loader, "get_device", lambda: "cpu")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield cfg
    torch.set_num_threads(old_threads)
    set_base_config(BaseConfig())


def new_model(config, model_type="resnet.1"):
    config.train.model_type = model_type
    return model_loader.load_new_model(
        [f"Class {i}" for i in range(30)],
        [str(i) for i in range(30)],
        [""] * 30,
        [""] * 30,
        100,
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


@pytest.mark.parametrize(
    "model_type,backbone_params",
    [
        ("resnet.1", 4_922_056),
        ("resnet.2", 7_083_144),
        ("resnet.3", 2_955_976),
        ("resnet.4", 4_625_544),
        ("resnet.5", 5_606_440),
        ("resnet.6", 6_686_152),
    ],
)
def test_variant_shapes_parameters_and_training(config, model_type, backbone_params):
    model = new_model(config, model_type)
    assert isinstance(model, ResNetModel)
    assert sum(p.numel() for p in model.backbone.parameters()) == backbone_params
    assert sum(p.numel() for p in model.head.parameters()) == 307_830
    inputs = torch.randn(2, 1, 192, 384, requires_grad=True)
    with torch.no_grad():
        assert model.backbone(inputs).shape == (2, 512, 6, 12)

    clip, frames = model(inputs)
    assert clip.shape == (2, 30)
    assert frames.shape == (2, 30, 12)
    loss = F.binary_cross_entropy_with_logits(clip, torch.rand_like(clip))
    loss += 0.2 * F.binary_cross_entropy_with_logits(frames, torch.rand_like(frames))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    assert model.head.prototypes.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("head_type", [None, "prototype_sed"])
def test_baseline_matches_timm_initialization_outputs_and_gradients(config, head_type):
    config.train.head_type = head_type
    torch.manual_seed(99)
    backbone = new_model(config).backbone.eval()
    torch.manual_seed(99)
    reference = timm.create_model(
        "resnet10t",
        pretrained=False,
        in_chans=1,
        num_classes=30 if head_type is None else 0,
        global_pool="avg" if head_type is None else "",
    ).eval()
    assert backbone.state_dict().keys() == reference.state_dict().keys()
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)

    # Activate the residual branches for a useful gradient comparison; timm's
    # zero-initialized last BN otherwise blocks gradients through their convs.
    for model in (backbone, reference):
        for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
            nn.init.ones_(stage[0].bn2.weight)
    inputs = torch.randn(1, 1, 192, 384, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual = backbone(inputs)
    expected = reference(reference_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
    for parameter, reference_parameter in zip(
        backbone.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, reference_parameter.grad, rtol=0, atol=0
        )


@pytest.mark.parametrize("model_type", list(MODEL_REGISTRY))
def test_short_variants_exclude_distant_time_context(config, model_type):
    # Output time cell 6 is centered on input column 192. Columns 260:280 are
    # within the baseline's 199-column context, outside the short one's 103.
    inputs = torch.ones(2, 1, 192, 384)
    inputs[1, :, :, 260:280] += 1
    backbone = new_model(config, model_type).backbone.eval()
    # Positive, normalized weights make every reachable input contribute,
    # independent of the random initialization or inactive ReLU branches.
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.constant_(module.weight, 1 / module.weight[0].numel())
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
    with torch.no_grad():
        features = backbone(inputs)[..., 6]
    if model_type in ("resnet.1", "resnet.2"):
        assert (features[1] - features[0]).min() > 1e-4
    else:
        torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "model_type,head_type",
    [(name, "prototype_sed") for name in MODEL_REGISTRY]
    + [("resnet.4", head) for head in (None, "basic", "temporal_sed")]
    + [("timm.resnet10t", "prototype_sed")],
)
def test_checkpoint_roundtrip_and_head_compatibility(
    config, tmp_path, model_type, head_type
):
    config.train.head_type = head_type
    config.train.drop_rate = 0.1
    config.train.drop_path_rate = 0.2
    if head_type in ("prototype_sed", "temporal_sed"):
        config.train.temporal_pooling = "linear_softmax"
    model = new_model(config, model_type).eval()
    inputs = torch.randn(1, 1, 192, 384)
    with torch.no_grad():
        expected_clip, expected_frames = model(inputs)
    assert expected_clip.shape == (1, 30)
    if model.use_sed:
        assert expected_frames.shape == (1, 30, 12)
    else:
        assert expected_frames is None

    path = tmp_path / "resnet.ckpt"
    save_checkpoint(model, path)
    # Inference must recover the saved architecture and head, rather than use
    # whatever configuration happens to be active when the model is loaded.
    config.train.model_type = "effnet.1"
    config.train.head_type = "basic"
    config.train.prototypes_per_class = 5
    config.train.temporal_pooling = "logsumexp"
    config.train.drop_rate = 0
    config.train.drop_path_rate = 0
    loaded = model_loader.load_from_checkpoint(str(path)).eval()
    assert loaded.model_type == model_type
    assert loaded.head_type == head_type
    assert loaded.backbone.drop_rate == 0.1
    assert loaded.backbone.layer4[0].drop_path.drop_prob == pytest.approx(0.2)
    if head_type == "prototype_sed":
        assert loaded.head.prototypes_per_class == 20
    if model.use_sed:
        assert loaded.head.temporal_pooling == "linear_softmax"
    with torch.no_grad():
        clip, frames = loaded(inputs)
    torch.testing.assert_close(clip, expected_clip, rtol=0, atol=0)
    if expected_frames is None:
        assert frames is None
    else:
        torch.testing.assert_close(frames, expected_frames, rtol=0, atol=0)


def test_unknown_variant_rejected(config):
    with pytest.raises(ValueError, match="Unknown model type: resnet.99"):
        new_model(config, "resnet.99")


@pytest.mark.parametrize("model_type", ["resnet.4", "resnet.5", "resnet.6"])
def test_short_variant_onnx_export(config, tmp_path, model_type):
    onnx = pytest.importorskip("onnx")
    from britekit.commands import ckpt_onnx

    config.infer.openvino_block_size = 1
    model = new_model(config, model_type).eval()
    # Exercise the shortened residual convolutions during export validation.
    for stage in (
        model.backbone.layer1,
        model.backbone.layer2,
        model.backbone.layer3,
        model.backbone.layer4,
    ):
        nn.init.ones_(stage[0].bn2.weight)
    path = tmp_path / "resnet.ckpt"
    save_checkpoint(model, path)
    ckpt_onnx(input_path=str(path))
    onnx.checker.check_model(str(path.with_suffix(".onnx")))

    ov = pytest.importorskip("openvino")
    compiled = ov.Core().compile_model(
        str(path.with_suffix(".onnx")), "CPU", {"INFERENCE_PRECISION_HINT": "f32"}
    )
    inputs = torch.randn(1, 1, 192, 384)
    with torch.no_grad():
        expected = model(inputs)
    outputs = compiled(inputs.numpy())
    for index, value in enumerate(expected):
        np.testing.assert_allclose(
            outputs[compiled.output(index)], value.numpy(), rtol=1e-4, atol=1e-5
        )
