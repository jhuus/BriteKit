"""Class-specific prototype features with the existing SED temporal pooling."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeSEDHead(nn.Module):
    """Each class reads only its own prototypes; input is [B, D, F, T].

    Similarities are raw cosine similarities (not probabilities). Frequency
    max-pooling preserves time. Positive readout weights make the class logit
    monotone in each of its prototype activations. No auxiliary loss is used.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        prototypes_per_class: int = 5,
        lse_temp: float = 0.5,
    ):
        super().__init__()
        if min(in_channels, num_classes, prototypes_per_class) < 1:
            raise ValueError("Channel, class and prototype counts must be positive")
        if not math.isfinite(lse_temp) or lse_temp <= 0:
            raise ValueError("lse_temp must be finite and positive")
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.lse_temp = lse_temp
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, prototypes_per_class, in_channels)
        )
        # softplus keeps weights positive without dead gradients at zero.
        self.raw_weights = nn.Parameter(
            torch.full((num_classes, prototypes_per_class), math.log(math.expm1(1.0)))
        )
        self.bias = nn.Parameter(torch.full((num_classes,), -2.0))

    @property
    def weights(self):
        return F.softplus(self.raw_weights)

    def similarity_maps(self, x):
        """Return [B, species, prototypes, F, T] cosine similarities.

        Float32 normalization also handles zero vectors under mixed precision.
        The convolution can still use autocast for its expensive operation.
        """
        features = F.normalize(x.float(), dim=1, eps=1e-6)
        prototypes = F.normalize(self.prototypes.float(), dim=-1, eps=1e-6)
        kernels = prototypes.flatten(0, 1).unsqueeze(-1).unsqueeze(-1)
        similarities = F.conv2d(features, kernels)
        return similarities.reshape(
            x.shape[0],
            self.num_classes,
            self.prototypes_per_class,
            x.shape[2],
            x.shape[3],
        )

    def forward(self, x):
        activations = self.similarity_maps(x).amax(dim=3)
        frame_logits = (activations * self.weights[None, :, :, None]).sum(dim=2)
        frame_logits = frame_logits + self.bias[None, :, None]
        segment_logits = self.lse_temp * (
            torch.logsumexp(frame_logits / self.lse_temp, dim=-1)
            - math.log(frame_logits.shape[-1])
        )
        return segment_logits, frame_logits
