"""Numerical and training integration checks for optional binary focal loss."""

import pytest
import torch
from torch.nn import functional as F

from britekit.core.base_config import BaseConfig
from britekit.core.config_loader import set_base_config
from britekit.models.base_model import BaseModel


@pytest.fixture
def model():
    cfg = BaseConfig()
    cfg.train.loss_type = "focal"
    set_base_config(cfg)
    model = BaseModel(
        "test", None, 1, ["A", "B"], ["a", "b"], ["", ""], ["", ""], 1, True
    )
    yield model
    set_base_config(BaseConfig())


def test_formula_with_smoothed_targets_and_class_weights(model):
    logits = torch.tensor([[1.2, -0.3], [-1.1, 2.0]], dtype=torch.float64)
    targets = torch.tensor([[0.95, 0.05], [0.05, 0.95]], dtype=torch.float64)
    p = logits.sigmoid()
    pt = targets * p + (1 - targets) * (1 - p)
    alpha = 0.25 * targets + 0.75 * (1 - targets)
    expected = (
        alpha * (1 - pt).square() * (-targets * p.log() - (1 - targets) * (1 - p).log())
    )
    model.set_class_weights([2.0, 3.0])
    torch.testing.assert_close(
        model._loss_per_sample(logits, targets),
        (expected * torch.tensor([2.0, 3.0])).mean(1),
    )


def test_easy_examples_are_downweighted(model):
    logits = torch.tensor([[0.0, 0.0], [5.0, -5.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    ratio = model._binary_loss(logits, targets) / F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    assert (ratio[1] < ratio[0]).all()


def test_frame_and_clip_reduction_and_extreme_gradients(model):
    model.use_sed = True
    model.cfg.train.frame_loss_weight = 0.3
    clips = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
    frames = torch.tensor([[[-1000.0, 1000.0], [1000.0, -1000.0]]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    frame_targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    loss = model._calc_loss(clips, frames, targets, targets, frame_targets)
    expected = 0.7 * model._binary_loss(clips, targets).mean()
    expected += 0.3 * model._binary_loss(frames.transpose(1, 2), frame_targets).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(clips.grad).all()
    assert torch.isfinite(frames.grad).all()


def test_teacher_distillation_retains_bce(model):
    logits = torch.tensor([[1.0, -2.0]])
    hard = torch.tensor([[1.0, 0.0]])
    teacher = torch.tensor([[0.7, 0.2]])
    loss = model._calc_loss(logits, None, hard, hard, teacher_labels=teacher)
    expected = 0.5 * model._binary_loss(logits, hard).mean()
    expected += 0.5 * F.binary_cross_entropy_with_logits(logits, teacher)
    torch.testing.assert_close(loss, expected)


def test_bce_default_and_gamma_zero(model):
    logits = torch.randn(2, 2)
    targets = torch.rand(2, 2)
    model.cfg.train.focal_gamma = 0
    model.cfg.train.focal_alpha = 0.5
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    torch.testing.assert_close(model._binary_loss(logits, targets), 0.5 * bce)
    model.cfg.train.loss_type = "bce"
    torch.testing.assert_close(model._binary_loss(logits, targets), bce)


def test_training_step_applies_existing_smoothing_once(model):
    from unittest.mock import Mock

    model.cfg.train.pos_label_smoothing = 0.05
    model.cfg.train.neg_label_smoothing = 0.05
    model.cfg.train.offpeak_weight = 0
    logits = torch.tensor([[0.7, -0.4]], requires_grad=True)
    model.forward = Mock(return_value=(logits, None))
    model.log = Mock()
    model.optimizer = Mock(param_groups=[{"lr": 0.001}])
    targets = torch.tensor([[1.0, 0.0]])
    loss = model.training_step(
        {"input": torch.zeros(1, 1), "segment_labels": targets}, 0
    )
    expected = model._binary_loss(logits, torch.tensor([[0.95, 0.05]])).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("loss_type", "unknown", "loss_type"),
        ("focal_alpha", -0.1, "focal_alpha"),
        ("focal_alpha", 1.1, "focal_alpha"),
        ("focal_gamma", -1.0, "focal_gamma"),
        ("focal_gamma", float("nan"), "focal_gamma"),
    ],
)
def test_invalid_configuration(model, field, value, message):
    setattr(model.cfg.train, field, value)
    with pytest.raises(ValueError, match=message):
        BaseModel("test", None, 1, ["A"], ["a"], [""], [""], 1, True)
