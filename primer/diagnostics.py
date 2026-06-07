"""Layer 3 — diagnostics.

Surfaces the data problems that quietly wreck studies. For an early researcher
these warnings are often worth more than the model ranking: catching a leaking
column or a temporal-split trap saves more than picking the right estimator ever
will.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from . import numpyx as nx
from .types import DatasetProfile, Diagnostic, Severity, TaskKind, TaskSpec

LEAK_MI_FRAC = 0.95          # normalised MI above which a feature looks like leakage
LEAK_CORR = 0.98            # |corr| with a continuous target that looks like leakage
HIGH_COND = 1e3             # condition number above which multicollinearity is severe
IMBALANCE_WARN = 0.10        # min class share below this -> warn
IMBALANCE_CRIT = 0.02        # ... below this -> critical
HIGH_MISSING = 0.40


def run_diagnostics(df: "pd.DataFrame", profile: DatasetProfile,
                    task: TaskSpec) -> List[Diagnostic]:
    out: List[Diagnostic] = []
    out += _identifier_and_constant(profile)
    out += _missingness(profile)
    out += _duplicates(df)
    out += _temporal(profile)
    if task.kind is TaskKind.CLASSIFICATION:
        out += _imbalance(task)
    if profile.target is not None and task.kind in (TaskKind.REGRESSION,
                                                     TaskKind.CLASSIFICATION):
        out += _leakage(df, profile, task)
    out += _multicollinearity(df, profile)
    out.sort(key=lambda d: d.severity.rank, reverse=True)
    return out


def _identifier_and_constant(profile: DatasetProfile) -> List[Diagnostic]:
    out = []
    ids = [c.name for c in profile.columns if c.ctype.value == "id"]
    consts = [c.name for c in profile.columns if c.ctype.value == "constant"]
    if ids:
        out.append(Diagnostic("identifier_columns", Severity.INFO,
                              f"{len(ids)} identifier-like column(s) excluded from modelling.",
                              columns=ids))
    if consts:
        out.append(Diagnostic("constant_columns", Severity.INFO,
                              f"{len(consts)} constant / zero-variance column(s) dropped.",
                              columns=consts))
    return out


def _missingness(profile: DatasetProfile) -> List[Diagnostic]:
    bad = [(c.name, c.missing_frac) for c in profile.columns
           if c.role.value in ("feature", "target") and c.missing_frac > HIGH_MISSING]
    if not bad:
        return []
    cols = [n for n, _ in bad]
    sev = Severity.CRITICAL if any(n == profile.target for n in cols) else Severity.WARNING
    return [Diagnostic("high_missingness", sev,
                       f"{len(cols)} column(s) are >40% missing — impute carefully or drop.",
                       columns=cols,
                       detail={n: round(f, 3) for n, f in bad})]


def _duplicates(df: "pd.DataFrame") -> List[Diagnostic]:
    dup = int(df.duplicated().sum())
    if dup == 0:
        return []
    frac = dup / max(len(df), 1)
    sev = Severity.WARNING if frac > 0.01 else Severity.INFO
    return [Diagnostic("duplicate_rows", sev,
                       f"{dup} duplicate row(s) ({frac:.1%}) — may inflate CV scores via leakage across folds.",
                       detail={"count": dup, "fraction": round(frac, 4)})]


def _temporal(profile: DatasetProfile) -> List[Diagnostic]:
    if not profile.datetime_features:
        return []
    return [Diagnostic("temporal_columns", Severity.WARNING,
                       "datetime feature(s) present — if observations are a time series, "
                       "use a chronological train/test split (random k-fold leaks the future).",
                       columns=profile.datetime_features)]


def _imbalance(task: TaskSpec) -> List[Diagnostic]:
    if not task.class_balance:
        return []
    min_frac = min(task.class_balance.values())
    minority = min(task.class_balance, key=task.class_balance.get)
    if min_frac < IMBALANCE_CRIT:
        sev = Severity.CRITICAL
    elif min_frac < IMBALANCE_WARN:
        sev = Severity.WARNING
    else:
        return []
    return [Diagnostic("class_imbalance", sev,
                       f"minority class '{minority}' is {min_frac:.1%} of rows — "
                       f"use class weights / resampling and judge with PR-AUC or F1, not accuracy.",
                       detail={"min_class_frac": round(min_frac, 4),
                               "balance": {k: round(v, 4) for k, v in task.class_balance.items()}})]


def _leakage(df: "pd.DataFrame", profile: DatasetProfile,
             task: TaskSpec) -> List[Diagnostic]:
    suspects = []
    target = profile.target
    if task.kind is TaskKind.REGRESSION:
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        for name in profile.numeric_features:
            x = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 8:
                continue
            if np.std(x[m]) < nx.EPS or np.std(y[m]) < nx.EPS:
                continue
            r = abs(np.corrcoef(x[m], y[m])[0, 1])
            if r > LEAK_CORR:
                suspects.append((name, round(float(r), 4)))
    else:  # classification
        yc = df[target].astype("category").cat.codes.to_numpy()
        h_y = nx.entropy_discrete(yc)
        for name in profile.numeric_features:
            x = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            mi = nx.mutual_info_cd(x, yc)
            nmi = nx.normalized_mi(mi, nx.entropy_discrete(
                np.digitize(x[np.isfinite(x)], np.histogram_bin_edges(x[np.isfinite(x)], bins=16)[1:-1])
                if np.isfinite(x).sum() > 8 else np.array([0])), h_y)
            if nmi > LEAK_MI_FRAC:
                suspects.append((name, round(float(nmi), 4)))
        for name in profile.categorical_features:
            xc = df[name].astype("category").cat.codes.to_numpy()
            mi = nx.mutual_info_dd(xc, yc)
            nmi = nx.normalized_mi(mi, nx.entropy_discrete(xc), h_y)
            if nmi > LEAK_MI_FRAC:
                suspects.append((name, round(float(nmi), 4)))

    if not suspects:
        return []
    cols = [n for n, _ in suspects]
    return [Diagnostic("possible_target_leakage", Severity.CRITICAL,
                       f"{len(cols)} feature(s) are almost perfectly predictive of the target — "
                       "verify they are legitimately available at prediction time, not leakage.",
                       columns=cols, detail=dict(suspects))]


def _multicollinearity(df: "pd.DataFrame", profile: DatasetProfile) -> List[Diagnostic]:
    if len(profile.numeric_features) < 2:
        return []
    X = np.column_stack([
        pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        for c in profile.numeric_features])
    cond = nx.condition_number(X)
    if cond < HIGH_COND:
        return []
    return [Diagnostic("multicollinearity", Severity.WARNING,
                       f"high multicollinearity (condition number {cond:.0f}) — "
                       "linear-model coefficients will be unstable; prefer L2/elastic-net or trees.",
                       detail={"condition_number": round(cond, 1)})]
