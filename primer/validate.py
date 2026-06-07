"""Layer 6 — cheap-proxy validation (ROADMAP STUB).

This is the highest-value next layer and the real fix for the "20 hours on RF,
then discover XGBoost was better" problem. The design, documented here so it
slots into the existing pipeline without a rewrite:

    Take the top-k recommendations and run them on SMALL sub-samples with TINY
    budgets, using successive halving / Hyperband-style promotion:
        - fit every candidate on, say, 5% of rows;
        - keep the top half, double the data, refit;
        - repeat until one or two survive on (near-)full data.
    Read the learning curves: a candidate still improving steeply as data grows
    is under-fed; a flat one has saturated. The separation between surviving
    curves is what turns the rule engine's *prior* confidence into an
    *empirically calibrated* posterior.

Why it is a stub in v1: it requires actually calling the model libraries
(scikit-learn / xgboost / lightgbm), which we deliberately kept optional to stay
self-contained. When you enable it, implement `ProxyValidator.validate` to return
updated, calibrated confidences and attach measured scores to each
Recommendation.detail.

The interface below is intentionally stable so `core.Primer` can accept a
validator the moment one exists.
"""
from __future__ import annotations

import abc
from typing import List, Optional

import pandas as pd

from .types import DatasetProfile, Recommendation, TaskSpec


class ProxyValidator(abc.ABC):
    """Empirically validates a shortlist on cheap budgets (Layer 6)."""

    available: bool = False

    @abc.abstractmethod
    def validate(self, df: "pd.DataFrame", profile: DatasetProfile, task: TaskSpec,
                 shortlist: List[Recommendation], budget_fracs: Optional[List[float]] = None
                 ) -> List[Recommendation]:
        """Refit `shortlist` on growing sub-samples; return updated recommendations
        with empirically measured scores + calibrated confidence."""
        ...


class NullValidator(ProxyValidator):
    """Default no-op used in v1: returns the shortlist unchanged."""
    available = False

    def validate(self, df, profile, task, shortlist, budget_fracs=None):
        return shortlist
