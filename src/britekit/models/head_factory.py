from typing import Optional

import torch
import torch.nn as nn


class ChannelReducer(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        groups: int = 8,
        act: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        Act = nn.ReLU if act == "relu" else nn.GELU
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.groups = groups

        assert (
            in_ch % groups == 0 and out_ch % groups == 0
        ), "groups must divide channels"
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            Act(),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x: [B, C, T]
        return self.block(x)


class BiTemporalSEDHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        self.reduce = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Causal + anti-causal convolutions for bidirectional context
        self.conv_fwd = nn.Conv1d(
            hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False
        )
        self.conv_bwd = nn.Conv1d(
            hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        self.temporal_attention = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.frame_head = nn.Conv1d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.pool(x).squeeze(2)  # [B, C, T]
        x = self.reduce(x)  # [B, H, T]

        # Bidirectional
        fwd = self.conv_fwd(x)
        bwd = self.conv_bwd(x.flip(-1)).flip(-1)
        x = self.bn(torch.cat([fwd, bwd], dim=1))
        x = self.drop(self.act(x))

        frame_logits = self.frame_head(x)  # [B, num_classes, T]
        attn = torch.softmax(self.temporal_attention(x), dim=-1)  # [B, 1, T]
        segment_logits = (attn * frame_logits).sum(dim=-1)

        return segment_logits, frame_logits


class ScalableSEDHead(nn.Module):
    """
    Scalable version of Basic SED head:
      [freq-pool] -> [ChannelReducer] -> Conv1d(k3) -> ReLU -> Dropout -> Conv1d(1x1 to classes)
      + attention pooling for segment logits
    Returns (segment_logits [B,C], frame_logits [B,C,T]).
    """

    def __init__(
        self,
        in_channels: int,  # backbone channels, e.g., 1920 for GerNet, 616 for EffNet
        hidden_channels: int,  # e.g. 256
        num_classes: int,
        conv_dropout: float = 0.0,
        reducer_groups: int = 8,
        attn_temp: float = 0.7,
        use_smoother: bool = False,
        smoother_ks: int = 5,
    ):
        super().__init__()
        # Reduce frequency, keep time
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # [B,C,F,T] -> [B,C,1,T]
        self.squeeze = lambda z: z.squeeze(2)  # -> [B,C,T]

        # Channel reducer (controls size/cost of the head)
        self.reduce = ChannelReducer(
            in_ch=in_channels,
            out_ch=hidden_channels,
            groups=reducer_groups,
            dropout=conv_dropout,
        )

        # Light temporal conv stack
        self.temporal = nn.Sequential(
            nn.Conv1d(
                hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_dropout),
        )

        # Frame classifier and attention
        self.frame_head = nn.Conv1d(hidden_channels, num_classes, kernel_size=1)
        self.attn = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.attn_temp = attn_temp

        # Optional classwise smoother on logits
        self.smoother = None
        if use_smoother:
            self.smoother = nn.Conv1d(
                num_classes,
                num_classes,
                kernel_size=smoother_ks,
                padding=smoother_ks // 2,
                groups=num_classes,
                bias=False,
            )
            with torch.no_grad():
                self.smoother.weight.fill_(1.0 / smoother_ks)

    def forward(self, x):  # x: [B, C, F, T]
        x = self.squeeze(self.freq_pool(x))  # [B, C, T]
        x = self.reduce(x)  # [B, H, T]
        x = self.temporal(x)  # [B, H, T]

        frame_logits = self.frame_head(x)  # [B, num_classes, T]
        if self.smoother is not None:
            frame_logits = self.smoother(frame_logits)

        attn_logits = self.attn(x) / self.attn_temp  # [B, 1, T]
        attn = torch.softmax(attn_logits, dim=-1)  # [B, 1, T]
        segment_logits = (attn * frame_logits).sum(dim=-1)  # [B, num_classes]
        return segment_logits, frame_logits


class BasicSEDHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # [B, C, F, T] → [B, C, 1, T]
        self.squeeze = lambda x: x.squeeze(2)  # → [B, C, T]

        self.temporal_attention = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.frame_classifier = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.pool(x)  # [B, C, 1, T]
        x = self.squeeze(x)  # [B, C, T]

        frame_logits = self.frame_classifier(x)  # [B, C, T]
        attn = torch.softmax(self.temporal_attention(x), dim=-1)  # [B, 1, T]
        segment_logits = torch.sum(attn * frame_logits, dim=-1)  # [B, C]

        return segment_logits, frame_logits


def is_sed(head_type: Optional[str]):
    # Return true for SED heads only
    if not head_type:
        return False
    elif head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {head_type}")
    return HEAD_REGISTRY[head_type][1]


def make_head(
    head_type: str,
    in_channels: int,
    hidden_channels: int,
    num_classes: int,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Create a classifier head by name."""
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {head_type}")
    return HEAD_REGISTRY[head_type][0](
        in_channels, hidden_channels, num_classes, drop_rate
    )


def build_basic_head(
    in_channels: int, hidden_channels: int, num_classes: int, drop_rate: float
) -> nn.Module:
    # Basic: GlobalPool → Dropout → Linear
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Dropout(drop_rate),
        nn.Linear(in_channels, num_classes),
    )


def build_effnet_head(
    in_channels: int, hidden_channels: int, num_classes: int, drop_rate: float
) -> nn.Module:
    # Matches EfficientNet head: Conv2d → BN → SiLU → GlobalPool → Linear
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
        nn.BatchNorm2d(hidden_channels),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(drop_rate),
        nn.Linear(hidden_channels, num_classes),
    )


def build_hgnet_head(
    in_channels: int, hidden_channels: int, num_classes: int, drop_rate: float
) -> nn.Module:
    # Matches HGNet: GlobalPool → Conv2d → ReLU → Dropout → Linear
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dropout(drop_rate),
        nn.Linear(hidden_channels, num_classes),
    )


def build_basic_sed_head(
    in_channels: int, hidden_channels: int, num_classes: int, drop_rate: float
) -> nn.Module:
    return BasicSEDHead(in_channels, hidden_channels, num_classes, drop_rate)


def build_bitemporal_sed_head(
    in_channels: int, hidden_channels: int, num_classes: int, drop_rate: float
) -> nn.Module:
    return BiTemporalSEDHead(in_channels, hidden_channels, num_classes, drop_rate)


def build_scalable_sed_head(
    in_channels: int, hidden_channels: int, num_classes: int, drop_rate: float
) -> nn.Module:
    return ScalableSEDHead(in_channels, hidden_channels, num_classes, drop_rate)


HEAD_REGISTRY = {
    # name: (method, is_sed)
    "basic": (build_basic_head, False),
    "effnet": (build_effnet_head, False),
    "hgnet": (build_hgnet_head, False),
    "basic_sed": (build_basic_sed_head, True),
    "bitemporal_sed": (build_bitemporal_sed_head, True),
    "scalable_sed": (build_scalable_sed_head, True),
}
