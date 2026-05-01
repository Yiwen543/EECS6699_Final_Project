"""EECS 6699 Final Project — shared utilities.

Modules:
    targets       — iterated sawtooth target functions f_k(x).
    models        — ReLU MLP builders and parameter-matching helpers.
    train         — multi-seed training loops with deterministic seeding.
    noise         — input/label perturbations; FGSM and PGD attacks (added in Day 2).
    diagnostics   — empirical Lipschitz and linear-region counters (added in Day 3).
"""
from . import targets, models, train, noise, io_utils, diagnostics  # noqa: F401
