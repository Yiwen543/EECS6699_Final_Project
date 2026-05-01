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
    epochs: int = 30_000
    lr: float = 5e-3
    lr_min: float = 1e-5                  # cosine annealing target
    use_cosine_lr: bool = True            # cosine decay from lr -> lr_min over epochs
    grad_clip: float | None = 1.0         # max grad norm; None disables
    batch_size: int | None = None         # full-batch by default (small data).
    n_train: int = 1200
    k: int = 4
    # Curriculum learning. If set (e.g. [1, 2, 3, 4]), epochs are distributed
    # across stages and the target k is increased stage by stage.
    curriculum_ks: list[int] | None = None
    # Relative epoch weights per curriculum stage. None → equal split.
    # e.g. [1, 1, 2, 4] gives k=4 four times the epochs of k=1.
    curriculum_weights: list[int] | None = None
    # If True, reset lr to cfg.lr and create a fresh cosine schedule at the
    # start of each curriculum stage. Prevents the decayed LR from starving
    # later stages when the target complexity increases.
    reset_lr_per_stage: bool = True
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
    """Train ``model``. Supports curriculum learning via ``cfg.curriculum_ks``.

    Without a curriculum, trains on a single (x_train, y_train) pair (built
    from cfg if not supplied). With ``cfg.curriculum_ks = [k_1, ..., k_S]``,
    splits epochs across S stages and trains on f_{k_s} per stage; optimizer
    and LR scheduler persist across stages.
    """
    from .targets import sawtooth_target  # local import to avoid cycles
    device = torch.device(cfg.device)
    model = model.to(device)

    # Build x_train if not given.
    if x_train is None:
        x_train, _ = make_dataset(n=cfg.n_train, k=cfg.k, device=device)
    else:
        x_train = x_train.to(device)

    # Decide stage schedule.
    if cfg.curriculum_ks:
        ks = list(cfg.curriculum_ks)
        if cfg.curriculum_weights:
            weights = list(cfg.curriculum_weights)
            assert len(weights) == len(ks), "curriculum_weights must match curriculum_ks length"
            total_w = sum(weights)
            epochs_per_stage = [int(cfg.epochs * w / total_w) for w in weights]
        else:
            epochs_per_stage = [cfg.epochs // len(ks)] * len(ks)
        epochs_per_stage[-1] += cfg.epochs - sum(epochs_per_stage)   # absorb remainder
        stage_targets = [sawtooth_target(x_train, k=k) for k in ks]
    else:
        ks = [cfg.k]
        epochs_per_stage = [cfg.epochs]
        if y_train is None:
            y_train = sawtooth_target(x_train, k=cfg.k) if target_fn is None else target_fn(x_train)
        stage_targets = [y_train.to(device)]

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # Scheduler is created per-stage when reset_lr_per_stage=True (curriculum only),
    # otherwise a single global schedule spans all epochs (original behaviour).
    use_per_stage_sched = cfg.use_cosine_lr and bool(cfg.curriculum_ks) and cfg.reset_lr_per_stage
    if cfg.use_cosine_lr and not use_per_stage_sched:
        scheduler: optim.lr_scheduler.LRScheduler | None = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
        )
    else:
        scheduler = None
    criterion = nn.MSELoss()

    losses: list[float] = []
    global_step = 0
    for stage_idx, (k_stage, n_ep, y_stage) in enumerate(zip(ks, epochs_per_stage, stage_targets)):
        if cfg.log_every and len(ks) > 1:
            print(f"  ── stage {stage_idx + 1}/{len(ks)}: k={k_stage}, epochs={n_ep} ──")

        # Retry the first curriculum stage up to 3 times if the network collapses
        # to the constant-predictor minimum (loss ≈ Var(f_1) = 1/12 ≈ 0.083).
        # Subsequent stages are not retried: a collapse there means stage 1 already
        # failed and the check will not trigger.
        max_attempts = 5 if (stage_idx == 0 and len(ks) > 1) else 1
        for attempt in range(max_attempts):
            if use_per_stage_sched:
                # Fresh LR + cosine cycle for every stage so harder targets get full
                # gradient exploration rather than a stale, near-zero learning rate.
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_ep, eta_min=cfg.lr_min
                )
            stage_losses: list[float] = []
            step_start = global_step
            for _ in range(n_ep):
                optimizer.zero_grad()
                pred = model(x_train)
                loss = criterion(pred, y_stage)
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                stage_losses.append(float(loss.item()))
                if cfg.log_every and global_step % cfg.log_every == 0:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    tag = f"k={k_stage} " if len(ks) > 1 else ""
                    print(f"  step {global_step:6d} {tag}  loss = {loss.item():.6e}  lr = {cur_lr:.2e}")
                global_step += 1

            if stage_idx == 0 and stage_losses[-1] > 0.02 and attempt < max_attempts - 1:
                print(f"  [restart] k={k_stage} collapsed (loss={stage_losses[-1]:.4f}), "
                      f"reinit (attempt {attempt + 2}/{max_attempts})")
                model._init_weights()
                optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
                global_step = step_start   # reset step counter for this stage
                continue
            break

        losses.extend(stage_losses)

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
