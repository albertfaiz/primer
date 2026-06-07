"""Layer 1 — ingestion and semantic column typing.

Loads a CSV/Parquet path or accepts an in-memory DataFrame, then infers a
semantic type and modelling role for every column. Large files are sub-sampled
(reservoir-style head sample) so profiling stays fast; the design note is that
for genuinely larger-than-memory work you should reach for Polars/DuckDB rather
than hand-rolled chunking — that hook is documented in the roadmap.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

from .types import ColumnProfile, ColumnRole, ColumnType, DatasetProfile

DataLike = Union[str, "pd.DataFrame"]

# heuristic thresholds (kept in one place so they are easy to tune / learn later)
MAX_PROFILE_ROWS = 50_000          # sub-sample above this for speed
TEXT_AVG_LEN = 35                  # avg chars above which an object col is "text"
CATEGORICAL_MAX_UNIQUE = 50        # absolute ceiling for "categorical"
CATEGORICAL_MAX_FRAC = 0.10        # cardinality / n ceiling for "categorical"
ID_UNIQUE_FRAC = 0.98             # uniqueness above which a column looks like an id


def _looks_like_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (s.dtype == object or isinstance(s.dtype, pd.StringDtype)):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def _is_boolean(s: pd.Series, n_unique: int) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if n_unique != 2:
        return False
    vals = set(pd.unique(s.dropna()))
    binary_sets = [{0, 1}, {0.0, 1.0}, {True, False},
                   {"true", "false"}, {"yes", "no"}, {"y", "n"}, {"t", "f"}]
    low = {str(v).strip().lower() for v in vals}
    return low in [{str(x).lower() for x in b} for b in binary_sets]


def _infer_type(s: pd.Series, n_rows: int) -> ColumnType:
    n_unique = int(s.nunique(dropna=True))

    if n_unique <= 1:
        return ColumnType.CONSTANT
    if _is_boolean(s, n_unique):
        return ColumnType.BOOLEAN
    if _looks_like_datetime(s):
        return ColumnType.DATETIME

    if pd.api.types.is_numeric_dtype(s):
        # an integer column that is essentially a unique key reads as an id
        if n_unique >= ID_UNIQUE_FRAC * n_rows and pd.api.types.is_integer_dtype(s):
            return ColumnType.ID
        # few distinct integer codes -> categorical
        if pd.api.types.is_integer_dtype(s) and n_unique <= 20 and n_unique / n_rows < CATEGORICAL_MAX_FRAC:
            return ColumnType.CATEGORICAL
        return ColumnType.NUMERIC

    # object / string / category
    if n_unique >= ID_UNIQUE_FRAC * n_rows:
        return ColumnType.ID
    avg_len = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
    if (n_unique <= CATEGORICAL_MAX_UNIQUE or n_unique / n_rows < CATEGORICAL_MAX_FRAC) \
            and (avg_len or 0) <= TEXT_AVG_LEN:
        return ColumnType.CATEGORICAL
    return ColumnType.TEXT


def load_frame(data: DataLike) -> "pd.DataFrame":
    if isinstance(data, pd.DataFrame):
        return data
    if not isinstance(data, str):
        raise TypeError(f"Expected a path or DataFrame, got {type(data)!r}")
    lower = data.lower()
    if lower.endswith((".parquet", ".pq")):
        return pd.read_parquet(data)
    if lower.endswith((".tsv", ".tab")):
        return pd.read_csv(data, sep="\t")
    if lower.endswith((".json", ".jsonl")):
        return pd.read_json(data, lines=lower.endswith(".jsonl"))
    return pd.read_csv(data)


def profile_dataset(data: DataLike, target: Optional[str] = None,
                    max_rows: int = MAX_PROFILE_ROWS) -> "tuple[pd.DataFrame, DatasetProfile]":
    """Load + sample + type every column. Returns (working_frame, profile)."""
    df_full = load_frame(data)
    n_full = len(df_full)

    sampled = n_full > max_rows
    df = df_full.sample(max_rows, random_state=0).reset_index(drop=True) if sampled else df_full

    if target is not None and target not in df.columns:
        raise KeyError(f"target {target!r} not found. Columns: {list(df.columns)}")

    columns, num, cat, dt, txt, dropped = [], [], [], [], [], []
    for name in df.columns:
        s = df[name]
        ctype = _infer_type(s, len(df))
        n_unique = int(s.nunique(dropna=True))
        n_missing = int(s.isna().sum())
        notes = []

        if name == target:
            role = ColumnRole.TARGET
        elif ctype in (ColumnType.ID,):
            role = ColumnRole.IDENTIFIER
            dropped.append(name)
            notes.append("high-cardinality identifier — excluded from modelling")
        elif ctype is ColumnType.CONSTANT:
            role = ColumnRole.DROPPED
            dropped.append(name)
            notes.append("constant / zero-variance — excluded from modelling")
        else:
            role = ColumnRole.FEATURE
            if ctype is ColumnType.NUMERIC:
                num.append(name)
            elif ctype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
                cat.append(name)
            elif ctype is ColumnType.DATETIME:
                dt.append(name)
            elif ctype is ColumnType.TEXT:
                txt.append(name)

        columns.append(ColumnProfile(
            name=name, ctype=ctype, role=role, dtype=str(s.dtype),
            n_unique=n_unique, n_missing=n_missing,
            missing_frac=float(n_missing / max(len(df), 1)), notes=notes,
        ))

    feature_names = num + cat + dt + txt
    profile = DatasetProfile(
        n_rows=n_full, n_cols=df.shape[1], columns=columns, target=target,
        feature_names=feature_names, numeric_features=num, categorical_features=cat,
        datetime_features=dt, text_features=txt, dropped=dropped,
        sampled=sampled, sample_rows=(len(df) if sampled else None),
    )
    return df, profile
