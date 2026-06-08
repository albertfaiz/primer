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
HIGH_CARDINALITY = 20        # categorical n_unique above this is "high-cardinality"
AUTOCORR_WARN = 0.30         # |lag-1 autocorrelation| of target above this -> sequential
OVERLAP_WARN = 0.50          # class-separation below this -> classes overlap
INTRINSIC_WARN = 0.50        # effective-dim / p below this -> low intrinsic dimensionality
REDUNDANCY_WARN = 0.25       # fraction of highly-correlated feature pairs
HETERO_WARN = 0.40           # |corr(|resid|, prediction)| above this -> heteroskedastic


def run_diagnostics(df: "pd.DataFrame", profile: DatasetProfile,
                    task: TaskSpec, mf: dict = None,
                    landmarks: list = None) -> List[Diagnostic]:
    mf = mf or {}
    landmarks = landmarks or []
    out: List[Diagnostic] = []
    out += _identifier_and_constant(profile)
    out += _high_cardinality(profile)
    out += _missingness(profile)
    out += _duplicates(df)
    out += _temporal(profile)
    out += _sequential_autocorr(mf)
    if task.kind is TaskKind.CLASSIFICATION:
        out += _imbalance(task)
        out += _class_overlap(mf)
    if profile.target is not None and task.kind in (TaskKind.REGRESSION,
                                                     TaskKind.CLASSIFICATION):
        out += _leakage(df, profile, task)
    out += _multicollinearity(df, profile)
    out += _low_intrinsic_dim(profile, mf)
    out += _redundancy(mf)
    out += _heteroskedasticity(task, landmarks)
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


# --------------------------------------------------------------------------- #
# v2 diagnostics
# --------------------------------------------------------------------------- #
def _high_cardinality(profile: DatasetProfile) -> List[Diagnostic]:
    hc = [(c.name, c.n_unique) for c in profile.columns
          if c.ctype.value == "categorical" and c.n_unique > HIGH_CARDINALITY]
    if not hc:
        return []
    cols = [n for n, _ in hc]
    worst = max(u for _, u in hc)
    return [Diagnostic("high_cardinality_categorical", Severity.INFO,
                       f"{len(cols)} high-cardinality categorical column(s) (up to {worst} levels) — "
                       "use native-categorical models (CatBoost/LightGBM) or target/hashing encoding; "
                       "one-hot will explode the feature space, and rare levels risk overfitting.",
                       columns=cols, detail=dict(hc))]


def _sequential_autocorr(mf: dict) -> List[Diagnostic]:
    ac = float(mf.get("target_autocorr", 0.0) or 0.0)
    if abs(ac) < AUTOCORR_WARN:
        return []
    return [Diagnostic("sequential_autocorrelation", Severity.WARNING,
                       f"target shows lag-1 autocorrelation {ac:+.2f} in row order — observations "
                       "are not independent; a random k-fold split will leak. Use a chronological "
                       "or group-wise split and consider time-series validation.",
                       detail={"lag1_autocorr": round(ac, 3)})]


def _class_overlap(mf: dict) -> List[Diagnostic]:
    if "class_separation" not in mf:
        return []
    sep = float(mf["class_separation"])
    lift = float(mf.get("signal_lift", 1.0))
    # Centroid distance measures *linear* separability; if a nonlinear probe
    # still finds signal (e.g. XOR), the classes ARE separable — don't warn.
    if sep >= OVERLAP_WARN or lift >= 0.10:
        return []
    return [Diagnostic("severe_class_overlap", Severity.WARNING,
                       f"classes are geometrically close (standardised centroid distance {sep:.2f}) "
                       "and no cheap probe beats the baseline — they occupy nearly the same region of "
                       "feature space, so no estimator (not even heavy boosting) will separate them. "
                       "Revisit features or labels.",
                       detail={"class_separation": round(sep, 3)})]


def _low_intrinsic_dim(profile: DatasetProfile, mf: dict) -> List[Diagnostic]:
    p = len(profile.numeric_features)
    if p < 4 or "intrinsic_dim_ratio" not in mf:
        return []
    ratio = float(mf["intrinsic_dim_ratio"])
    if ratio >= INTRINSIC_WARN:
        return []
    k = int(mf.get("intrinsic_dim_95", p))
    return [Diagnostic("low_intrinsic_dimensionality", Severity.INFO,
                       f"~{k} components explain 95% of the variance across {p} numeric features — "
                       "the data lives on a lower-dimensional manifold; PCA / regularisation / "
                       "feature selection will help more than added model capacity.",
                       detail={"intrinsic_dim_95": k, "ratio": round(ratio, 3)})]


def _redundancy(mf: dict) -> List[Diagnostic]:
    frac = float(mf.get("frac_highly_correlated_pairs", 0.0) or 0.0)
    if frac < REDUNDANCY_WARN:
        return []
    return [Diagnostic("feature_redundancy", Severity.INFO,
                       f"{frac:.0%} of numeric feature pairs are highly correlated (|r|>0.8) — "
                       "many features are redundant; sparse/L1 models or pruning reduce variance "
                       "without losing signal.",
                       detail={"frac_highly_correlated_pairs": round(frac, 3)})]


def _heteroskedasticity(task: TaskSpec, landmarks: list) -> List[Diagnostic]:
    if task.kind is not TaskKind.REGRESSION:
        return []
    h = 0.0
    for lm in landmarks:
        if lm.name == "linear_ridge":
            h = float(lm.detail.get("resid_pred_corr", 0.0) or 0.0)
    if h < HETERO_WARN:
        return []
    return [Diagnostic("heteroskedasticity", Severity.INFO,
                       f"residual magnitude tracks the prediction (corr {h:.2f}) — the target's noise "
                       "is non-constant; consider a log/Box-Cox transform of the target, or tree models "
                       "which are agnostic to variance scaling.",
                       detail={"resid_pred_corr": round(h, 3)})]
