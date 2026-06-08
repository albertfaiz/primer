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
    """Pearson correlation, NaN-robust via pairwise mean imputation per column.

    Wrapped in errstate + sanitised so extreme-scale or degenerate columns can
    never emit RuntimeWarnings or propagate inf/nan (behaviour was BLAS-dependent
    before — Apple Accelerate vs OpenBLAS differed)."""
    X = np.asarray(X, float)
    with np.errstate(all="ignore"):
        X = np.where(np.isfinite(X), X, np.nan)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(np.isnan(X))
        X = X.copy()
        X[inds] = np.take(col_means, inds[1])
        std = X.std(axis=0)
        std = np.where(std < EPS, 1.0, std)
        Z = (X - X.mean(axis=0)) / std
        Z = np.clip(Z, -1e6, 1e6)            # tame residual extremes pre-matmul
        n = Z.shape[0]
        C = (Z.T @ Z) / max(n - 1, 1)
        C = np.where(np.isfinite(C), C, 0.0)
    return np.clip(C, -1.0, 1.0)


def condition_number(X: np.ndarray) -> float:
    """Condition number of the correlation matrix (multicollinearity proxy)."""
    if X.shape[1] < 2:
        return 1.0
    C = correlation_matrix(X)
    try:
        with np.errstate(all="ignore"):
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


# --------------------------------------------------------------------------- #
# v2 kernels — robust scaling, geometry, signal
# --------------------------------------------------------------------------- #
def robust_standardize(X: np.ndarray, clip: float = 8.0):
    """Centre by median, scale by IQR (≈std for a Gaussian), winsorise to ±clip.

    Robust to astronomical outliers and heavy tails where mean/std scaling fails,
    and it keeps every column on a comparable geometric scale so closed-form
    linear algebra stays numerically stable regardless of raw units."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        scale = iqr / 1.349                          # IQR -> Gaussian-equivalent sigma
        # fall back to std, then to 1, for near-constant columns
        std = np.nanstd(X, axis=0)
        scale = np.where(scale < EPS, std, scale)
        scale = np.where((~np.isfinite(scale)) | (scale < EPS), 1.0, scale)
        med = np.where(np.isfinite(med), med, 0.0)
        Z = (X - med) / scale
        Z = np.where(np.isfinite(Z), Z, 0.0)
        Z = np.clip(Z, -clip, clip)
    return Z, med, scale


def intrinsic_dimensionality(X: np.ndarray, var_target: float = 0.95):
    """Effective dimensionality from PCA/SVD spectrum.

    Returns (k95, ratio, participation_ratio):
      k95   - number of components explaining `var_target` of the variance
      ratio - k95 / p   (≪ 1 means the data lives on a low-dim manifold)
      participation_ratio - (Σλ)² / Σλ²  normalised, a soft effective-rank.
    """
    X = np.asarray(X, float)
    n, p = X.shape if X.ndim == 2 else (X.shape[0], 1)
    if p < 2 or n < 3:
        return p, 1.0, 1.0
    Z, _, _ = robust_standardize(X)
    with np.errstate(all="ignore"):
        try:
            sv = np.linalg.svd(Z, compute_uv=False)
        except np.linalg.LinAlgError:
            return p, 1.0, 1.0
        lam = sv ** 2
        total = lam.sum()
        if total < EPS:
            return p, 1.0, 1.0
        frac = np.cumsum(lam) / total
        k95 = int(np.searchsorted(frac, var_target) + 1)
        pr = (lam.sum() ** 2) / (np.sum(lam ** 2) + EPS)   # participation ratio
    return k95, float(k95 / p), float(pr / p)


def lag1_autocorrelation(y: np.ndarray) -> float:
    """Lag-1 autocorrelation of a sequence (sequential-dependency / leakage proxy).

    High |value| on index-ordered data means random k-fold CV will leak; a
    chronological / group split is required."""
    y = np.asarray(y, float).ravel()
    y = y[np.isfinite(y)]
    if y.size < 8:
        return 0.0
    y0, y1 = y[:-1], y[1:]
    with np.errstate(all="ignore"):
        s0, s1 = y0.std(), y1.std()
        if s0 < EPS or s1 < EPS:
            return 0.0
        r = np.mean((y0 - y0.mean()) * (y1 - y1.mean())) / (s0 * s1)
    return float(r if np.isfinite(r) else 0.0)


def class_separation(X: np.ndarray, y: np.ndarray) -> float:
    """Standardised distance between class centroids (a Mahalanobis-lite signal).

    Near 0 ⇒ classes occupy the same region of feature space and *no* model — not
    even heavy boosting — can separate them; the problem is the features/labels,
    not the estimator. Computed on the two largest classes for a single scalar."""
    X = np.asarray(X, float)
    y = np.asarray(y).ravel()
    if X.ndim == 1:
        X = X[:, None]
    Z, _, _ = robust_standardize(X)
    vals, counts = np.unique(y, return_counts=True)
    if vals.size < 2:
        return 0.0
    top2 = vals[np.argsort(counts)[::-1][:2]]
    with np.errstate(all="ignore"):
        a = Z[y == top2[0]]
        b = Z[y == top2[1]]
        if len(a) < 2 or len(b) < 2:
            return 0.0
        mu_a, mu_b = a.mean(0), b.mean(0)
        pooled = np.sqrt((a.var(0) + b.var(0)) / 2.0) + EPS
        d = np.sqrt(np.sum(((mu_a - mu_b) / pooled) ** 2) / Z.shape[1])
    return float(d if np.isfinite(d) else 0.0)
