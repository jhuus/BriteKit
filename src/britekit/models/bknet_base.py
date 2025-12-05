"""
BKNet (BriteKit Network) - A configurable CNN backbone for bioacoustic classification.

This module implements a GENet-style architecture with:
- BasicBlocks (two 3x3 convs) for early stages
- Bottleneck blocks (1x1 -> 3x3 -> 1x1) for later stages
- Standard convolutions or depthwise separable convolutions
- Configurable channel widths, depths, and expansion ratios
- Optional stochastic depth (drop_path) for regularization
- Optional Squeeze-and-Excitation (SE) attention

It is essentially a reimplementation of timm's gernet, which is an
implementation of GENet (https://arxiv.org/abs/2006.14090).

This implementation is a bit more configurable, and lets us experiment with variants.
For example, this one has an SE block option, which timm's does not, although the SE
block option does not seem to actually help.

This implementation was created by Claude Opus 4.5, as directed by Jan Huus.
"""

from typing import cast, Dict, List, Optional, Type

import torch
import torch.nn as nn


# -------------------------------------------------
# DropPath (Stochastic Depth)
# -------------------------------------------------
def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """
    Drop paths (stochastic depth) per sample.

    This is the same as the DropConnect impl for EfficientNet, etc.
    but renamed to avoid confusion with dropout which operates on
    individual elements rather than entire paths.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # Work with diff tensor shapes; not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (stochastic depth) per sample.

    When applied to the main path of a residual block, this effectively
    allows the network to dynamically adjust its depth during training.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


# -------------------------------------------------
# Squeeze-and-Excitation (SE) Module
# -------------------------------------------------
class SEModule(nn.Module):
    """
    Squeeze-and-Excitation block.

    Applies channel attention by:
    1. Global average pooling to squeeze spatial dimensions
    2. Two FC layers (reduce -> expand) to learn channel weights
    3. Sigmoid to get attention weights
    4. Scale input channels by attention weights

    Args:
        channels: Number of input/output channels
        rd_ratio: Reduction ratio for the bottleneck (default 0.25 = 1/4)
        rd_channels: Explicit number of reduced channels (overrides rd_ratio if set)
        act_layer: Activation function (default ReLU)
    """

    def __init__(
        self,
        channels: int,
        rd_ratio: float = 0.25,
        rd_channels: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        if rd_channels is None:
            rd_channels = max(1, int(channels * rd_ratio))

        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: global average pooling
        x_se = x.mean(dim=(2, 3), keepdim=True)

        # Excitation: FC -> Act -> FC -> Sigmoid
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        x_se = self.sigmoid(x_se)

        # Scale
        return x * x_se


# -------------------------------------------------
# Helper: Conv + BN + optional activation
# -------------------------------------------------
class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm2d + optional activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act_layer: Optional[Type[nn.Module]] = nn.ReLU,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# -------------------------------------------------
# BasicBlock: two 3x3 convs with residual connection
# Used in early stages of GENet
# -------------------------------------------------
class BasicBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutions.

    Structure:
        x -> 3x3 conv (stride) -> BN -> Act -> 3x3 conv -> BN -> [SE] -> (+shortcut) -> Act

    This matches GENet's stages.0 and stages.1 structure.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        act_layer: Type[nn.Module] = nn.ReLU,
        drop_path_rate: float = 0.0,
        use_se: bool = False,
        se_ratio: float = 0.25,
    ):
        super().__init__()

        # First 3x3 conv (with stride for downsampling)
        self.conv1 = ConvBnAct(
            in_ch, out_ch, kernel_size=3, stride=stride, act_layer=act_layer
        )

        # Second 3x3 conv (no activation - applied after residual add)
        self.conv2 = ConvBnAct(out_ch, out_ch, kernel_size=3, stride=1, act_layer=None)

        # SE attention (applied after conv2, before residual add)
        self.se = (
            SEModule(out_ch, rd_ratio=se_ratio, act_layer=act_layer)
            if use_se
            else nn.Identity()
        )

        # Shortcut connection
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvBnAct(
                in_ch, out_ch, kernel_size=1, stride=stride, act_layer=None
            )
        else:
            self.shortcut = cast(ConvBnAct, nn.Identity())

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)

        out = self.drop_path(out)
        out = out + identity
        out = self.act(out)
        return out


# -------------------------------------------------
# Bottleneck: 1x1 -> 3x3 -> 1x1 with residual connection
# Used in later stages of GENet
# -------------------------------------------------
class Bottleneck(nn.Module):
    """
    Bottleneck block with 1x1 -> 3x3 -> 1x1 structure.

    Structure:
        x -> 1x1 conv -> BN -> Act -> 3x3 conv (stride) -> BN -> Act -> 1x1 conv -> BN -> [SE] -> (+shortcut) -> Act

    The middle 3x3 conv uses `mid_ch` channels (controlled by expansion ratio).
    If `depthwise=True`, the 3x3 conv is depthwise (groups=mid_ch).
    This matches GENet's stages.2, stages.3, stages.4 structure.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: int,
        stride: int = 1,
        act_layer: Type[nn.Module] = nn.ReLU,
        depthwise: bool = False,
        drop_path_rate: float = 0.0,
        use_se: bool = False,
        se_ratio: float = 0.25,
    ):
        super().__init__()

        # 1x1 conv to mid_ch
        self.conv1 = ConvBnAct(
            in_ch, mid_ch, kernel_size=1, stride=1, act_layer=act_layer
        )

        # 3x3 conv (with stride for downsampling)
        # If depthwise=True, use groups=mid_ch for depthwise separable conv
        groups = mid_ch if depthwise else 1
        self.conv2 = ConvBnAct(
            mid_ch,
            mid_ch,
            kernel_size=3,
            stride=stride,
            groups=groups,
            act_layer=act_layer,
        )

        # 1x1 conv to out_ch (no activation - applied after residual add)
        self.conv3 = ConvBnAct(mid_ch, out_ch, kernel_size=1, stride=1, act_layer=None)

        # SE attention (applied after conv3, before residual add)
        self.se = (
            SEModule(out_ch, rd_ratio=se_ratio, act_layer=act_layer)
            if use_se
            else nn.Identity()
        )

        # Shortcut connection
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvBnAct(
                in_ch, out_ch, kernel_size=1, stride=stride, act_layer=None
            )
        else:
            self.shortcut = cast(ConvBnAct, nn.Identity())

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)

        out = self.drop_path(out)
        out = out + identity
        out = self.act(out)
        return out


# -------------------------------------------------
# BKNetBaseModel: GENet-style backbone
# -------------------------------------------------
class BKNetBaseModel(nn.Module):
    """
    BKNet (BriteKit Network) backbone - a GENet-style CNN.

    Config dict `cfg` should contain:
        stem_ch: int                   # output channels of stem conv
        stage_out_chs: List[int]       # output channels per stage
        stage_mid_chs: List[int]       # middle channels for bottleneck stages (0 = BasicBlock)
        num_blocks: List[int]          # number of blocks per stage
        stage_strides: List[int]       # stride for first block of each stage (optional, default [2,2,2,2,1])
        stage_depthwise: List[bool]    # whether to use depthwise 3x3 in bottleneck (optional, default all False)

    When stage_mid_chs[i] == 0, that stage uses BasicBlocks.
    When stage_mid_chs[i] > 0, that stage uses Bottleneck blocks with that mid_ch.
    When stage_depthwise[i] == True, the 3x3 conv in bottleneck is depthwise (groups=mid_ch).

    Additional kwargs:
        drop_path_rate: float          # max drop path rate, linearly scaled per block (default 0.0)
        use_se: bool                   # whether to use SE attention in all blocks (default False)
        se_ratio: float                # SE reduction ratio (default 0.25)

    forward(x) returns features of shape [B, C, H, W].
    `num_features` is set to the final output channels, for use by classifier heads.
    """

    def __init__(
        self,
        cfg: Dict,
        num_classes: int = 0,
        in_chans: int = 1,
        act_layer: Type[nn.Module] = nn.ReLU,
        final_ch: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.act_layer = act_layer

        # Parse config
        stem_ch: int = cfg["stem_ch"]
        stage_out_chs: List[int] = cfg["stage_out_chs"]
        stage_mid_chs: List[int] = cfg["stage_mid_chs"]
        num_blocks: List[int] = cfg["num_blocks"]

        num_stages = len(stage_out_chs)

        # Default strides: downsample in all stages except the last
        default_strides = [2] * (num_stages - 1) + [1] if num_stages > 0 else []
        stage_strides: List[int] = cfg.get("stage_strides", default_strides)

        # Default depthwise: False for all stages
        default_depthwise = [False] * num_stages
        stage_depthwise: List[bool] = cfg.get("stage_depthwise", default_depthwise)

        assert (
            len(stage_mid_chs) == num_stages
        ), "stage_mid_chs must match stage_out_chs length"
        assert (
            len(num_blocks) == num_stages
        ), "num_blocks must match stage_out_chs length"
        assert (
            len(stage_strides) == num_stages
        ), "stage_strides must match stage_out_chs length"
        assert (
            len(stage_depthwise) == num_stages
        ), "stage_depthwise must match stage_out_chs length"

        # Drop path rate (stochastic depth)
        drop_path_rate: float = kwargs.get("drop_path_rate", 0.0)

        # SE attention settings
        use_se: bool = kwargs.get("use_se", False)
        se_ratio: float = kwargs.get("se_ratio", 0.25)

        # Calculate total number of blocks for linear drop path scaling
        total_blocks = sum(num_blocks)

        # -------------------------
        # Stem: single conv with stride 2
        # -------------------------
        self.stem = ConvBnAct(
            in_chans, stem_ch, kernel_size=3, stride=2, act_layer=act_layer
        )

        # -------------------------
        # Stages
        # -------------------------
        stages = []
        in_ch = stem_ch
        block_idx_global = 0  # For linear drop path scaling

        for stage_idx in range(num_stages):
            out_ch = stage_out_chs[stage_idx]
            mid_ch = stage_mid_chs[stage_idx]
            n_blocks = num_blocks[stage_idx]
            first_stride = stage_strides[stage_idx]
            use_depthwise = stage_depthwise[stage_idx]

            stage_blocks = []
            for block_idx in range(n_blocks):
                # First block uses the stage's stride, subsequent blocks use stride=1
                stride = first_stride if block_idx == 0 else 1

                # Linear scaling of drop path rate
                block_drop_path = (
                    drop_path_rate * block_idx_global / max(total_blocks - 1, 1)
                )

                if mid_ch == 0:
                    # Use BasicBlock
                    block = BasicBlock(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        stride=stride,
                        act_layer=act_layer,
                        drop_path_rate=block_drop_path,
                        use_se=use_se,
                        se_ratio=se_ratio,
                    )
                else:
                    # Use Bottleneck
                    block = cast(
                        BasicBlock,
                        Bottleneck(
                            in_ch=in_ch,
                            out_ch=out_ch,
                            mid_ch=mid_ch,
                            stride=stride,
                            act_layer=act_layer,
                            depthwise=use_depthwise,
                            drop_path_rate=block_drop_path,
                            use_se=use_se,
                            se_ratio=se_ratio,
                        ),
                    )

                stage_blocks.append(block)
                in_ch = out_ch
                block_idx_global += 1

            stages.append(nn.Sequential(*stage_blocks))

        self.stages = nn.ModuleList(stages)

        # -------------------------
        # Final conv (expansion to final_ch)
        # -------------------------
        if final_ch is not None and final_ch > 0:
            self.final_conv = ConvBnAct(
                in_ch, final_ch, kernel_size=1, stride=1, act_layer=act_layer
            )
            self.num_features = final_ch
        else:
            self.final_conv = cast(ConvBnAct, nn.Identity())
            self.num_features = in_ch

        self.num_classes = num_classes

        # Default classifier head (matches timm ClassifierHead structure)
        self.drop_rate = kwargs.get("drop_rate", 0.0)
        if num_classes > 0:
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
            )
            self.drop = nn.Dropout(p=self.drop_rate, inplace=False)
            self.fc = nn.Linear(self.num_features, num_classes)
            self.flatten = nn.Identity()  # Already flattened in global_pool
        else:
            self.global_pool = cast(nn.Sequential, nn.Identity())
            self.drop = cast(nn.Dropout, nn.Identity())
            self.fc = cast(nn.Linear, nn.Identity())
            self.flatten = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the classifier head (pool + flatten + dropout + fc)."""
        x = self.global_pool(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.flatten(x)  # Identity when using default head
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.num_classes > 0:
            x = self.forward_head(x)
        return x
