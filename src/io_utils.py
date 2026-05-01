"""Small persistence helpers — saving and reloading trained models across phases.

Phase 4 reuses Phase 1's trained networks rather than retraining; this module
handles the (de)serialization. Models are saved as PyTorch ``state_dict``s
together with the architecture metadata needed to reconstruct them.
"""
from __future__ import annotations

import json
import pathlib
from typing import Iterable

import torch

from .models import ModelBuilder


def save_models(
    out_dir: str | pathlib.Path,
    tag: str,
    models: Iterable[ModelBuilder],
    metas: Iterable[dict],
    seeds: Iterable[int],
) -> pathlib.Path:
    """Save a list of trained ModelBuilders.

    Layout::

        out_dir/
          tag_seed{S}.pt        # state_dict
          tag_meta.json         # list of dicts with depth/width/seed

    Returns the directory path.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metas_list: list[dict] = []
    for model, meta, seed in zip(models, metas, seeds):
        path = out / f"{tag}_seed{seed}.pt"
        torch.save(model.state_dict(), path)
        metas_list.append({
            "seed": int(seed),
            "depth": int(model.depth),
            "width": int(model.width),
            "params": int(sum(p.numel() for p in model.parameters())),
            **{k: v for k, v in meta.items() if isinstance(v, (int, float, str))},
        })
    (out / f"{tag}_meta.json").write_text(json.dumps(metas_list, indent=2))
    return out


def load_models(
    out_dir: str | pathlib.Path,
    tag: str,
    device: str | torch.device = "cpu",
) -> tuple[list[ModelBuilder], list[dict]]:
    """Load all models with the given ``tag`` from ``out_dir``."""
    out = pathlib.Path(out_dir)
    metas: list[dict] = json.loads((out / f"{tag}_meta.json").read_text())
    models: list[ModelBuilder] = []
    for meta in metas:
        m = ModelBuilder(depth=meta["depth"], width=meta["width"])
        sd = torch.load(out / f"{tag}_seed{meta['seed']}.pt", map_location=device)
        m.load_state_dict(sd)
        m.to(device).eval()
        models.append(m)
    return models, metas
