"""Training loops with deterministic seeding and multi-seed aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .models import ModelBuilder, make_matched_pair
from .targets import make_dataset


@dataclass
class TrainConfig:
    epochs: int = 15_000
    lr: float = 3e-3
    batch_size: int | None = None         # full-batch by default (small data).
    n_train: int = 1200
    k: int = 4
    log_every: int = 3000
    device: str = "cpu"


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    final_loss: float = float("nan")
    n_params: int = 0


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_model(
    model: nn.Module,
    cfg: TrainConfig,
    target_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    x_train: torch.Tensor | None = None,
    y_train: torch.Tensor | None = None,
) -> TrainResult:
    """Train ``model`` on (x_train, y_train); if not supplied, build from cfg.

    ``target_fn`` is used only when y_train is not supplied.
    """
    device = torch.device(cfg.device)
    model = model.to(device)

    if x_train is None or y_train is None:
        x_train, y_train = make_dataset(n=cfg.n_train, k=cfg.k, device=device)
        if target_fn is not None:
            y_train = target_fn(x_train)
    else:
        x_train = x_train.to(device)
        y_train = y_train.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    losses: list[float] = []
    for step in range(cfg.epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if cfg.log_every and step % cfg.log_every == 0:
            print(f"  step {step:6d}   loss = {loss.item():.6e}")

    return TrainResult(
        losses=losses,
        final_loss=losses[-1] if losses else float("nan"),
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def multi_seed_run(
    seeds: Sequence[int],
    cfg: TrainConfig,
    deep_depth: int = 9,
    deep_width: int = 4,
    target_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    """Train the matched (deep, shallow) pair under each seed.

    Returns a dict keyed by 'deep' / 'shallow' whose values are dicts of
    arrays/lists across seeds: ``losses``, ``final_loss``, ``models``.
    """
    out = {
        "deep":    {"losses": [], "final_loss": [], "models": [], "info": []},
        "shallow": {"losses": [], "final_loss": [], "models": [], "info": []},
    }
    for s in seeds:
        _set_seed(s)
        deep, shallow, info = make_matched_pair(
            deep_depth=deep_depth, deep_width=deep_width, seed=s
        )
        print(f"[seed {s}] deep params = {info['deep_params']}, "
              f"shallow params = {info['shallow_params']} "
              f"(width = {info['shallow_width']})")

        print(f"[seed {s}] training deep ...")
        r_deep = train_model(deep, cfg, target_fn=target_fn)
        print(f"[seed {s}] training shallow ...")
        r_shallow = train_model(shallow, cfg, target_fn=target_fn)

        out["deep"]["losses"].append(r_deep.losses)
        out["deep"]["final_loss"].append(r_deep.final_loss)
        out["deep"]["models"].append(deep)
        out["deep"]["info"].append(info)

        out["shallow"]["losses"].append(r_shallow.losses)
        out["shallow"]["final_loss"].append(r_shallow.final_loss)
        out["shallow"]["models"].append(shallow)
        out["shallow"]["info"].append(info)
    return out
