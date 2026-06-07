"""Domain lenses — inversion of control for the rule engine.

A *lens* is a small ``{rule_name: multiplier}`` map that scales how strongly
individual rules fire for a given field. ``primer.analyze(df, target=..., lens="genomics")``
nudges the shortlist toward what that discipline's data usually needs, without
forking the engine or hard-coding domain logic into the core.

HONESTY (this matters): the bundled lenses below are **illustrative starting
weights**, not validated, peer-reviewed domain priors. They encode reasonable,
defensible intuitions (genomics is p≫n and sparse; clinical data has structured
missingness and rare outcomes; finance has fat tails and low signal-to-noise) —
but they are a *scaffold for you to tune*, not authoritative knowledge. The real
value is the mechanism: a domain expert can pass their own dict and inject genuine
field expertise. A multiplier of 1.0 is neutral; >1 amplifies a rule, <1 damps it.

Pass either a registered name (str) or your own ``{rule_name: multiplier}`` dict.
"""
from __future__ import annotations

from typing import Dict, Optional, Union

# name -> {description, weights}. Weights multiply rule activation strengths.
LENSES: Dict[str, Dict] = {
    "genomics": {
        "description": "High-dimensional, sparse signal (p ≫ n): emphasise "
                       "regularisation, dimensionality and feature selection.",
        "weights": {
            "high_dimensional": 1.5,
            "low_intrinsic_dim": 1.4,
            "few_informative_features": 1.4,
            "small_sample": 1.3,
        },
    },
    "clinical": {
        "description": "EHR / medical data: structured missingness, rare "
                       "outcomes, robustness to outliers and class overlap.",
        "weights": {
            "missing_values": 1.6,
            "class_imbalance": 1.4,
            "class_overlap": 1.3,
            "outliers_heavy_tails": 1.2,
        },
    },
    "finance": {
        "description": "Markets / risk: heavy tails, correlated drivers, and a "
                       "low signal-to-noise ratio where overfitting is the enemy.",
        "weights": {
            "outliers_heavy_tails": 1.5,
            "multicollinearity": 1.3,
            "low_signal_high_noise": 1.4,
            "linear_suffices": 1.2,
        },
    },
}


def available_lenses() -> Dict[str, str]:
    """Map of lens name -> human description (for discovery / docs)."""
    return {k: v["description"] for k, v in LENSES.items()}


def resolve_lens(lens: Optional[Union[str, Dict[str, float]]]
                 ) -> Optional[Dict[str, float]]:
    """Normalise a lens argument to a ``{rule_name: multiplier}`` dict or None."""
    if lens is None:
        return None
    if isinstance(lens, dict):
        return {str(k): float(v) for k, v in lens.items()}
    if isinstance(lens, str):
        key = lens.strip().lower()
        if key in LENSES:
            return dict(LENSES[key]["weights"])
        raise ValueError(
            f"unknown lens {lens!r}; available: {sorted(LENSES)} "
            "(or pass your own {rule_name: multiplier} dict).")
    raise TypeError("lens must be a name (str), a {rule_name: multiplier} dict, or None.")
