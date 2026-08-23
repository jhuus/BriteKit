#!/usr/bin/env python3

from typing import Any, cast

import torch
from timm.layers import pad_same, ScaledStdConv2d, ScaledStdConv2dSame
from torch import nn
from torch.nn import functional as F


class _SamePaddingConv2d(nn.Conv2d):
    """Ordinary Conv2d retaining timm's dynamic SAME-padding behavior."""

    def forward(self, x):
        x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        return super().forward(x)


def _fold_scaled_std_conv(module: nn.Conv2d) -> nn.Conv2d:
    same_padding = isinstance(module, ScaledStdConv2dSame) and module.same_pad
    conv_type = _SamePaddingConv2d if same_padding else nn.Conv2d
    conv = conv_type(
        module.in_channels,
        module.out_channels,
        cast(Any, module.kernel_size),
        stride=cast(Any, module.stride),
        padding=cast(Any, module.padding),
        dilation=cast(Any, module.dilation),
        groups=module.groups,
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )

    with torch.no_grad():
        weight = F.batch_norm(
            module.weight.reshape(1, module.out_channels, -1),
            None,
            None,
            weight=(module.gain * module.scale).view(-1),
            training=True,
            momentum=0.0,
            eps=module.eps,
        ).reshape_as(module.weight)
        conv.weight.copy_(weight)
        if module.bias is not None:
            assert conv.bias is not None
            conv.bias.copy_(module.bias)
    return conv


def fold_scaled_std_convs(module: nn.Module) -> None:
    """Replace timm weight-standardized convolutions with equivalent Conv2d layers."""
    scaled_conv_types = (ScaledStdConv2d, ScaledStdConv2dSame)
    for name, child in list(module.named_children()):
        if isinstance(child, scaled_conv_types):
            setattr(module, name, _fold_scaled_std_conv(child))
        else:
            fold_scaled_std_convs(child)
