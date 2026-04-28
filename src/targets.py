"""Compositional target functions.

The iterated sawtooth f_k(x) = phi^{(k)}(x) with phi(x) = |2x - 1| has 2^k peaks
on [0, 1] and is the canonical witness used by Telgarsky (2016) for the
depth-separation theorem. Composition doubles the number of linear regions per
application, so f_k is provably *not* approximable by any depth-2 ReLU network
with sub-exponential width.
"""
from __future__ import annotations

import torch


def sawtooth_target(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Iterated sawtooth f_k(x) = phi^{(k)}(x), phi(x) = |2x - 1|.

    Args:
        x: Tensor of shape (..., 1) with values in [0, 1].
        k: Number of compositions. Produces 2**k peaks on [0, 1].

    Returns:
        Tensor of shape (..., 1) with f_k(x).
    """
    out = x
    for _ in range(k):
        out = torch.abs(2.0 * out - 1.0)
    return out


def make_dataset(
    n: int = 1200,
    k: int = 4,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform grid over [0, 1] with sawtooth targets.

    Returns x of shape (n, 1) and y = f_k(x) of shape (n, 1).
    """
    x = torch.linspace(0.0, 1.0, n, device=device).view(-1, 1)
    y = sawtooth_target(x, k=k)
    return x, y
