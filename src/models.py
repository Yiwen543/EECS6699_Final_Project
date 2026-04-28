"""ReLU MLP builders and parameter-matched-pair helpers.

A ``ModelBuilder(depth=L, width=W)`` realizes a fully-connected ReLU network with
``L`` *affine layers* (so ``L - 1`` hidden ReLU activations). With
``in_dim = out_dim = 1`` the parameter count is::

    P(L, W) = (1 * W + W) + (L - 2) * (W * W + W) + (W * 1 + 1)
            = (2 * W + 1) + (L - 2) * (W^2 + W)              for L >= 2.

For ``L = 2`` this collapses to ``P(2, W) = 3W + 1``; we use this in
:func:`matched_shallow_width` to construct a depth-2 model whose parameter count
matches a given deep model within +/- a few units.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class ModelBuilder(nn.Module):
    """Fully-connected ReLU MLP, scalar input, scalar output."""

    def __init__(self, depth: int, width: int):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2 (input layer + output layer).")
        layers: list[nn.Module] = [nn.Linear(1, width), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.depth = depth
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_shallow_width(target_params: int) -> int:
    """Smallest width W such that a depth-2 net has >= ``target_params`` params.

    Depth-2 parameter count is ``3W + 1`` (see module docstring).
    """
    return max(1, math.ceil((target_params - 1) / 3))


def make_matched_pair(
    deep_depth: int = 9,
    deep_width: int = 4,
    seed: int | None = None,
) -> Tuple[ModelBuilder, ModelBuilder, dict]:
    """Build a parameter-matched (deep, shallow) pair plus a metadata dict.

    The shallow network is depth-2 with width chosen so that
    ``|P_shallow - P_deep| <= 3``.
    """
    if seed is not None:
        torch.manual_seed(seed)
    deep = ModelBuilder(depth=deep_depth, width=deep_width)
    p_deep = count_parameters(deep)
    s_width = matched_shallow_width(p_deep)
    if seed is not None:
        torch.manual_seed(seed + 1)  # decorrelate shallow init from deep
    shallow = ModelBuilder(depth=2, width=s_width)
    info = {
        "deep_depth": deep_depth,
        "deep_width": deep_width,
        "deep_params": p_deep,
        "shallow_depth": 2,
        "shallow_width": s_width,
        "shallow_params": count_parameters(shallow),
    }
    return deep, shallow, info
