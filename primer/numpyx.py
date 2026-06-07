"""Fast, dependency-light numerical kernels.

Everything here is pure NumPy and operates on 1-D / 2-D float arrays. These are
the "C-speed under the hood" primitives the rest of the pipeline leans on.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# basic moments
# --------------------------------------------------------------------------- #
def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def skewness(x: np.ndarray) -> float:
    x = _clean(x)
    if x.size < 3:
        return 0.0
    d = x - x.mean()
    m2 = np.mean(d ** 2)
    if m2 < EPS:
        return 0.0
    m3 = np.mean(d ** 3)
    return float(m3 / m2 ** 1.5)


def kurtosis_excess(x: np.ndarray) -> float:
    x = _clean(x)
    if x.size < 4:
        return 0.0
    d = x - x.mean()
    m2 = np.mean(d ** 2)
    if m2 < EPS:
        return 0.0
    m4 = np.mean(d ** 4)
    return float(m4 / m2 ** 2 - 3.0)


def coef_variation(x: np.ndarray) -> float:
    x = _clean(x)
    if x.size == 0:
        return 0.0
    mu = x.mean()
    if abs(mu) < EPS:
        return float("inf") if x.std() > EPS else 0.0
    return float(x.std() / abs(mu))


def outlier_fraction(x: np.ndarray, k: float = 1.5) -> float:
    """Fraction of points outside the Tukey IQR fence."""
    x = _clean(x)
    if x.size < 4:
        return 0.0
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    if iqr < EPS:
        return 0.0
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return float(np.mean((x < lo) | (x > hi)))


# --------------------------------------------------------------------------- #
# information theory
# --------------------------------------------------------------------------- #
def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def entropy_discrete(labels: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    return entropy_from_counts(counts)


def _mi_from_joint(joint: np.ndarray) -> float:
    total = joint.sum()
    if total <= 0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py  # outer product of marginals
    mask = pxy > 0
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask]))
    return float(max(mi, 0.0))


def _auto_bins(n: int) -> int:
    return int(min(32, max(4, round(np.sqrt(n)))))


def mutual_info_cc(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information between two continuous variables (histogram estimate)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 8:
        return 0.0
    b = _auto_bins(x.size)
    joint, _, _ = np.histogram2d(x, y, bins=b)
    return _mi_from_joint(joint)


def mutual_info_cd(x: np.ndarray, y: np.ndarray) -> float:
    """MI between continuous x and discrete label y."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y).ravel()
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if x.size < 8:
        return 0.0
    edges = np.histogram_bin_edges(x, bins=_auto_bins(x.size))
    xb = np.digitize(x, edges[1:-1])
    return mutual_info_dd(xb, y)


def mutual_info_dd(a: np.ndarray, b: np.ndarray) -> float:
    """MI between two discrete variables via a contingency table."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size < 4:
        return 0.0
    ua, ai = np.unique(a, return_inverse=True)
    ub, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((ua.size, ub.size))
    np.add.at(joint, (ai, bi), 1.0)
    return _mi_from_joint(joint)


def normalized_mi(mi: float, h_x: float, h_y: float) -> float:
    denom = np.sqrt(max(h_x, EPS) * max(h_y, EPS))
    if denom < EPS:
        return 0.0
    return float(min(mi / denom, 1.0))


# --------------------------------------------------------------------------- #
# matrix structure
# --------------------------------------------------------------------------- #
def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation, NaN-robust via pairwise mean imputation per column."""
    X = np.asarray(X, float)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(col_means, inds[1])
    std = X.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    Z = (X - X.mean(axis=0)) / std
    n = Z.shape[0]
    C = (Z.T @ Z) / max(n - 1, 1)
    return np.clip(C, -1.0, 1.0)


def condition_number(X: np.ndarray) -> float:
    """Condition number of the correlation matrix (multicollinearity proxy)."""
    if X.shape[1] < 2:
        return 1.0
    C = correlation_matrix(X)
    try:
        ev = np.linalg.eigvalsh(C)
        ev = ev[ev > EPS]
        if ev.size == 0:
            return float("inf")
        return float(ev.max() / ev.min())
    except np.linalg.LinAlgError:
        return float("inf")


# --------------------------------------------------------------------------- #
# preprocessing helpers used by landmark models
# --------------------------------------------------------------------------- #
def impute_mean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float).copy()
    if X.ndim == 1:
        X = X[:, None]
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = np.take(col_means, nan_idx[1])
    return X


def standardize(X: np.ndarray):
    X = np.asarray(X, float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < EPS, 1.0, sd)
    return (X - mu) / sd, mu, sd


def one_hot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    Y = np.zeros((y.size, classes.size))
    lookup = {c: i for i, c in enumerate(classes)}
    for i, v in enumerate(y):
        Y[i, lookup[v]] = 1.0
    return Y
