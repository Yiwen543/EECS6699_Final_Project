"""Empirical Lipschitz estimation and linear-region counting for ReLU MLPs.

Two Lipschitz estimators are provided:
  - ``empirical_lipschitz_fd``:  max adjacent finite difference on a fine 1D grid.
    Exact for piecewise-linear functions; no training needed.
  - ``spectral_lipschitz``:      product-of-spectral-norms upper bound.
    Looser but fully analytical; useful as a normalisation-free proxy.

Linear regions are counted via ``activation_pattern_regions``, which enumerates
distinct ReLU activation patterns (binary sign vectors of pre-activations) on
a dense 1D grid — each distinct pattern corresponds to exactly one linear region.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Empirical Lipschitz (finite differences)
# ---------------------------------------------------------------------------

@torch.no_grad()
def empirical_lipschitz_fd(
    model: nn.Module,
    n_points: int = 20_000,
    device: str = "cpu",
) -> float:
    """Estimate empirical Lipschitz constant via max adjacent finite difference.

    For a 1D→1D network on [0, 1]:

        L̂ = max_i  |f(x_{i+1}) - f(x_i)| / h,    h = 1 / (n_points − 1).

    This is **exact** for piecewise-linear functions: the maximum local slope
    is achieved at a region boundary, which the dense grid approximates to
    within one grid spacing.  Increase ``n_points`` for tighter estimates.

    Args:
        model:    Trained ReLU MLP with scalar input and output.
        n_points: Number of uniformly spaced evaluation points on [0, 1].
        device:   Torch device string.

    Returns:
        Scalar float estimate of the empirical Lipschitz constant.
    """
    model.eval()
    x = torch.linspace(0.0, 1.0, n_points, device=device).unsqueeze(1)
    y = model(x).squeeze(1)
    h = 1.0 / (n_points - 1)
    slopes = (y[1:] - y[:-1]).abs() / h
    return float(slopes.max().item())


# ---------------------------------------------------------------------------
# Spectral Lipschitz bound
# ---------------------------------------------------------------------------

def spectral_lipschitz(model: nn.Module) -> float:
    """Upper bound on Lipschitz constant via product of spectral norms.

    For a ReLU MLP with weight matrices W_1, …, W_L:

        L(f) ≤ ∏_l ‖W_l‖_2

    where ‖W‖_2 = σ_max(W) is the largest singular value (operator 2-norm).
    ReLU has Lipschitz constant 1 and does not contribute to the product.
    Bias vectors have no effect on the Lipschitz constant and are ignored.

    Args:
        model: Any ``nn.Module`` containing ``nn.Linear`` sublayers.

    Returns:
        Scalar float upper bound on the network's Lipschitz constant.
    """
    lip = 1.0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            sv_max = float(torch.linalg.svdvals(m.weight).max().item())
            lip *= sv_max
    return lip


# ---------------------------------------------------------------------------
# Linear-region counter (activation patterns)
# ---------------------------------------------------------------------------

@torch.no_grad()
def activation_pattern_regions(
    model: nn.Module,
    n_grid: int = 10_000,
    device: str = "cpu",
) -> int:
    """Count distinct linear regions via activation-pattern enumeration.

    For a ReLU MLP, every input x induces a binary *activation pattern*
    a(x) ∈ {0,1}^N  (N = total hidden neurons; 1 = neuron active, 0 = dead).
    Two inputs with identical patterns lie in the same linear region.

    This function evaluates the model on a uniform grid of ``n_grid`` points
    in [0, 1], collects all activation patterns via forward-hook, and returns
    the count of **distinct** patterns — an empirical lower bound on the true
    number of linear regions.

    Args:
        model:  Trained ReLU MLP with scalar input.
        n_grid: Grid resolution (more points → tighter lower bound).
        device: Torch device string.

    Returns:
        Integer count of distinct activation patterns observed on the grid.
    """
    model.eval()
    x = torch.linspace(0.0, 1.0, n_grid, device=device).unsqueeze(1)

    activation_cols: list[torch.Tensor] = []

    def _make_hook(storage: list[torch.Tensor]):
        def hook(module: nn.Module, inp, out: torch.Tensor) -> None:
            storage.append((out > 0).int().cpu())
        return hook

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(_make_hook(activation_cols)))

    _ = model(x)

    for h in hooks:
        h.remove()

    if not activation_cols:
        return 1

    # (n_grid, total_hidden_neurons) binary matrix
    pattern_matrix = torch.cat(activation_cols, dim=1).numpy()
    unique_count = len(np.unique(pattern_matrix, axis=0))
    return int(unique_count)


# ---------------------------------------------------------------------------
# Per-seed summary helper
# ---------------------------------------------------------------------------

def lipschitz_summary(
    models: list[nn.Module],
    n_points: int = 20_000,
    device: str = "cpu",
) -> dict[str, float]:
    """Compute mean/std of empirical and spectral Lipschitz across a list of models.

    Convenience wrapper for multi-seed experiments.

    Returns:
        Dict with keys ``fd_mean``, ``fd_std``, ``spec_mean``, ``spec_std``.
    """
    fd_vals   = [empirical_lipschitz_fd(m, n_points=n_points, device=device) for m in models]
    spec_vals = [spectral_lipschitz(m) for m in models]
    return {
        "fd_mean":   float(np.mean(fd_vals)),
        "fd_std":    float(np.std(fd_vals)),
        "spec_mean": float(np.mean(spec_vals)),
        "spec_std":  float(np.std(spec_vals)),
        "fd_per_seed":   fd_vals,
        "spec_per_seed": spec_vals,
    }
