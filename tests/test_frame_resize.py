from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from britekit.models.base_model import BaseModel


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("target", [12, 16])
def test_frame_resize_preserves_values_dtype_and_gradient(
    device, dtype, autocast, target
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    frames = torch.randn(2, 4, 12, device=device, dtype=dtype, requires_grad=True)
    segments = frames.mean(dim=-1)
    model = SimpleNamespace(
        backbone=lambda x: x,
        head=lambda x: (segments, x),
        use_sed=True,
        cfg=SimpleNamespace(
            train=SimpleNamespace(sed_fps=target),
            audio=SimpleNamespace(spec_duration=1),
        ),
    )
    with torch.autocast(device_type=device, enabled=autocast):
        expected = F.interpolate(frames, size=target, mode="linear")
        actual_segments, actual = BaseModel.forward(model, frames)
    assert actual_segments is segments
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    weights = torch.randn_like(expected)
    expected_gradient = torch.autograd.grad(
        expected, frames, weights, retain_graph=True
    )[0]
    actual_gradient = torch.autograd.grad(actual, frames, weights)[0]
    assert torch.equal(actual_gradient, expected_gradient)
