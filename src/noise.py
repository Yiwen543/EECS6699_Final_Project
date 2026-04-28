"""Noise injection and adversarial perturbation utilities.

Day 1 implements the stochastic (Phase 2 + 3) perturbations:

* :func:`add_input_noise`  — Gaussian perturbation of inputs, evaluation-time.
* :func:`add_label_noise`  — Gaussian perturbation of training labels.
* :func:`evaluate_with_input_noise` — sweep-friendly wrapper.

FGSM and PGD attacks (Phase 4) are stubbed and will be filled in on Day 2.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Stochastic perturbations
# -----------------------------------------------------------------------------
def add_input_noise(
    x: torch.Tensor,
    sigma: float,
    clip: tuple[float, float] | None = (0.0, 1.0),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``x + N(0, sigma^2)``, optionally clipped to ``clip``."""
    if sigma <= 0.0:
        return x.clone()
    noise = torch.randn(x.shape, generator=generator, device=x.device) * sigma
    out = x + noise
    if clip is not None:
        out = out.clamp(*clip)
    return out


def add_label_noise(
    y: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return ``y + N(0, sigma^2)``."""
    if sigma <= 0.0:
        return y.clone()
    noise = torch.randn(y.shape, generator=generator, device=y.device) * sigma
    return y + noise


# -----------------------------------------------------------------------------
# Phase 2 sweep helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_with_input_noise(
    model: nn.Module,
    x: torch.Tensor,
    y_clean: torch.Tensor,
    sigmas: list[float],
    n_repeats: int = 20,
    seed: int = 0,
) -> dict[float, dict[str, float]]:
    """Sweep input-noise levels and report mean / std test MSE.

    For each sigma we draw ``n_repeats`` independent noise realizations and
    aggregate. The clean targets ``y_clean`` are NOT perturbed — this isolates
    the model's robustness from any change in the optimization target.
    """
    model.eval()
    device = x.device
    g = torch.Generator(device=device).manual_seed(seed)
    criterion = nn.MSELoss()

    results: dict[float, dict[str, float]] = {}
    for sigma in sigmas:
        mses: list[float] = []
        for _ in range(n_repeats):
            x_noisy = add_input_noise(x, sigma, generator=g)
            pred = model(x_noisy)
            mses.append(criterion(pred, y_clean).item())
        t = torch.tensor(mses)
        results[sigma] = {"mean": float(t.mean()), "std": float(t.std(unbiased=False))}
    return results


# -----------------------------------------------------------------------------
# Adversarial attacks  --  stub, completed on Day 2 (Phase 4)
# -----------------------------------------------------------------------------
def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> torch.Tensor:
    """Single-step FGSM attack (l_inf budget ``epsilon``).

    Filled in on Day 2 -- left as a working reference here so Day 1 imports
    don't break.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    x_adv = x.clone().detach().requires_grad_(True)
    pred = model(x_adv)
    loss = loss_fn(pred, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + epsilon * grad.sign()
    if clip is not None:
        x_adv = x_adv.clamp(*clip)
    return x_adv.detach()


def pgd_attack(*args, **kwargs):  # pragma: no cover -- Day 2
    raise NotImplementedError("PGD attack is implemented on Day 2 (Phase 4).")
