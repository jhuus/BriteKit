"""Pool frame logits into segment logits without changing the frame outputs."""

import math

import torch
from torch.nn import functional as F


def validate_temporal_pooling(method: str) -> None:
    if method not in ("logsumexp", "linear_softmax"):
        raise ValueError(f"Unknown temporal_pooling: {method}")


def pool_temporal_logits(
    frame_logits: torch.Tensor, method: str, lse_temp: float = 0.5
) -> torch.Tensor:
    """Return [B, classes] logits from [B, classes, time] logits.

    Linear-softmax pools probabilities as sum(p**2) / sum(p), with gradients
    through both sums. Its log odds are log(sum(p**2)) - log(sum(p*(1-p))).
    Computing these in log space avoids saturation, division by zero, and
    probability clipping, including under mixed precision.
    """
    if method == "logsumexp":
        return lse_temp * (
            torch.logsumexp(frame_logits / lse_temp, dim=-1)
            - math.log(frame_logits.shape[-1])
        )
    if method == "linear_softmax":
        if frame_logits.dtype in (torch.float16, torch.bfloat16):
            frame_logits = frame_logits.float()
        log_p = F.logsigmoid(frame_logits)
        log_not_p = F.logsigmoid(-frame_logits)
        return torch.logsumexp(2 * log_p, dim=-1) - torch.logsumexp(
            log_p + log_not_p, dim=-1
        )
    raise ValueError(f"Unknown temporal_pooling: {method}")
