"""Noise injection and adversarial perturbation utilities.

Stochastic perturbations (Phase 2 + 3):

* :func:`add_input_noise`            — Gaussian perturbation of inputs, eval time.
* :func:`add_label_noise`            — Gaussian perturbation of training labels.
* :func:`evaluate_with_input_noise`  — sweep-friendly wrapper for Phase 2.

Adversarial attacks (Phase 4):

* :func:`fgsm_attack`                — single-step FGSM, l_inf budget.
* :func:`pgd_attack`                 — PGD-K with optional random restarts.
* :func:`evaluate_adversarial`       — sweep over epsilons; per-attack MSE.
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
# Adversarial attacks (Phase 4)
# -----------------------------------------------------------------------------
def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    clip: tuple[float, float] | None = (0.0, 1.0),
) -> torch.Tensor:
    """Single-step FGSM attack (Goodfellow et al. 2015), l_inf budget ``epsilon``.

    Computes ``x + eps * sign(grad_x L(f(x), y))`` and clips to ``clip``.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    if epsilon == 0.0:
        return x.clone().detach()
    x_adv = x.clone().detach().requires_grad_(True)
    pred = model(x_adv)
    loss = loss_fn(pred, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + epsilon * grad.sign()
    if clip is not None:
        x_adv = x_adv.clamp(*clip)
    return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    n_steps: int = 40,
    step_size: float | None = None,
    n_restarts: int = 3,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    clip: tuple[float, float] | None = (0.0, 1.0),
    seed: int | None = None,
) -> torch.Tensor:
    """PGD-K attack (Madry et al. 2018) with random restarts.

    Performs ``n_restarts`` independent attacks, each initialized with a uniform
    random perturbation in [-eps, eps], runs ``n_steps`` of projected gradient
    ascent at step size ``step_size`` (default ``epsilon / 4``), and returns the
    worst-case x_adv (highest loss per sample).
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss(reduction="none")
    if epsilon == 0.0:
        return x.clone().detach()
    alpha = step_size if step_size is not None else epsilon / 4.0

    if seed is not None:
        gen = torch.Generator(device=x.device).manual_seed(seed)
    else:
        gen = None

    # Track the best (highest-loss) adversarial example found per input row.
    best_x_adv = x.clone().detach()
    best_loss  = torch.full((x.shape[0],), -float("inf"), device=x.device)

    for restart in range(max(n_restarts, 1)):
        # Random start in eps-ball; restart 0 starts from x to include the
        # "no random init" PGD as a safe baseline.
        if restart == 0:
            x_adv = x.clone().detach()
        else:
            delta = (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1) * epsilon
            x_adv = (x + delta).detach()
            if clip is not None:
                x_adv = x_adv.clamp(*clip)

        for _ in range(n_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            pred = model(x_adv)
            loss = loss_fn(pred, y).sum()    # sum so autograd produces per-row grads
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            # Project back to l_inf ball around x.
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            if clip is not None:
                x_adv = x_adv.clamp(*clip)

        # Per-row loss to update worst-case.
        with torch.no_grad():
            pred = model(x_adv)
            row_loss = loss_fn(pred, y).view(x.shape[0], -1).mean(dim=1)
        improved = row_loss > best_loss
        best_loss = torch.where(improved, row_loss, best_loss)
        improved_mask = improved.view(-1, *([1] * (x.dim() - 1)))
        best_x_adv = torch.where(improved_mask, x_adv, best_x_adv)

    return best_x_adv.detach()


def evaluate_adversarial(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilons: list[float],
    attack: str = "fgsm",
    **attack_kwargs,
) -> dict[float, float]:
    """Sweep ``epsilons`` and return ``{eps: adversarial MSE}`` for the chosen attack.

    ``attack`` is one of ``"fgsm"`` or ``"pgd"``; extra kwargs are forwarded.
    """
    model.eval()
    criterion = nn.MSELoss()
    out: dict[float, float] = {}
    for eps in epsilons:
        if attack == "fgsm":
            x_adv = fgsm_attack(model, x, y, eps)
        elif attack == "pgd":
            x_adv = pgd_attack(model, x, y, eps, **attack_kwargs)
        else:
            raise ValueError(f"unknown attack {attack!r}")
        with torch.no_grad():
            out[eps] = float(criterion(model(x_adv), y).item())
    return out
