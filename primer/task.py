"""Layer 2 — task resolution.

Decides regression vs classification vs unsupervised from the target column. This
is deterministic (a tree of cheap checks), not learned — the user's "method of
elimination" applied correctly. Note: reinforcement learning is intentionally
*not* a possible output here, because RL is defined by an environment / reward /
sequential decisions and is not inferable from a static table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .types import ColumnType, DatasetProfile, TaskKind, TaskSpec

# a numeric target with at most this many distinct values is treated as classes
CLASSIFICATION_MAX_UNIQUE = 20
CLASSIFICATION_MAX_FRAC = 0.05


def resolve_task(df: "pd.DataFrame", profile: DatasetProfile) -> TaskSpec:
    target = profile.target
    if target is None:
        return TaskSpec(
            kind=TaskKind.UNSUPERVISED,
            reason="no target column supplied — treating this as structure discovery "
                   "(clustering / dimensionality reduction / anomaly detection).",
        )

    col = profile.column(target)
    y = df[target]
    n = int(y.notna().sum())
    n_unique = int(y.nunique(dropna=True))

    # boolean / two-level target -> binary classification
    if col is not None and col.ctype is ColumnType.BOOLEAN:
        return _classification_spec(y, n_unique, "binary",
                                    "target is boolean / two-valued.")

    # explicit categorical / text target -> classification
    if col is not None and col.ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT):
        sub = "binary" if n_unique == 2 else "multiclass"
        return _classification_spec(y, n_unique, sub,
                                    "target is non-numeric (categorical labels).")

    # numeric target: decide by cardinality
    if pd.api.types.is_numeric_dtype(y):
        few_values = n_unique <= max(CLASSIFICATION_MAX_UNIQUE,
                                     CLASSIFICATION_MAX_FRAC * max(n, 1))
        integer_like = pd.api.types.is_integer_dtype(y) or \
            np.allclose(y.dropna() % 1, 0, atol=1e-9)
        if few_values and integer_like:
            sub = "binary" if n_unique == 2 else "multiclass"
            ordinal = " (values are ordered integers — consider ordinal models)" \
                if n_unique > 2 else ""
            return _classification_spec(
                y, n_unique, sub,
                f"numeric target has only {n_unique} distinct integer values{ordinal}.")
        return TaskSpec(
            kind=TaskKind.REGRESSION, subtype="continuous",
            reason=f"numeric target with {n_unique} distinct values — continuous regression.",
        )

    # fall-through
    return TaskSpec(kind=TaskKind.UNKNOWN,
                    reason="could not confidently resolve the target type.")


def _classification_spec(y: "pd.Series", n_unique: int, subtype: str,
                         reason: str) -> TaskSpec:
    counts = y.value_counts(dropna=True)
    total = counts.sum()
    balance = {str(k): float(v / total) for k, v in counts.items()}
    return TaskSpec(
        kind=TaskKind.CLASSIFICATION, subtype=subtype, n_classes=n_unique,
        class_balance=balance, reason=reason,
    )
