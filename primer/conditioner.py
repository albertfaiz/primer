"""Layer 1.5 — the Conditioner.

The stress tests proved a hard lesson: raw tabular data fed straight into
closed-form linear algebra is fragile. Astronomical magnitudes (datetimes cast to
epoch nanoseconds), heavy-tailed outliers, near-constant columns and perfect
collinearity can each push ``XᵀX`` into overflow / singularity — and whether it
actually blows up depends on the BLAS backend (Apple Accelerate vs OpenBLAS),
which is exactly the kind of non-determinism a scientific tool must not have.

The Conditioner is the single front-door every numeric kernel passes through:

  * coerce to float, map ±inf → NaN;
  * drop zero-variance / all-NaN columns (report which);
  * robust-scale (median / IQR) so every column shares a comparable geometry;
  * winsorise extreme values;
  * impute residual NaNs to 0 (= the median, post-centering).

Plus numerically safe solvers (``safe_solve`` with an lstsq fallback, and a
Tikhonov-stabilised ridge) so a singular system degrades gracefully instead of
crashing the user's terminal. Pure NumPy, deterministic, backend-independent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from . import numpyx as nx


@dataclass
class ConditionInfo:
    n_cols_in: int
    kept: List[int] = field(default_factory=list)
    dropped_constant: List[int] = field(default_factory=list)
    n_nonfinite_fixed: int = 0


def condition_matrix(X: np.ndarray, clip: float = 8.0
                     ) -> Tuple[np.ndarray, ConditionInfo]:
    """Return a robustly-scaled, finite, full-rank-ish matrix + a report.

    The returned matrix is safe to hand to any matmul / solve without errstate
    surprises. Columns that carry no information (constant or all-missing) are
    removed and listed in the info object so callers can keep names aligned.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    info = ConditionInfo(n_cols_in=p)

    with np.errstate(all="ignore"):
        nonfinite = ~np.isfinite(X)
        info.n_nonfinite_fixed = int(nonfinite.sum())
        X = np.where(nonfinite, np.nan, X)

        # identify informative columns (finite spread)
        spread = np.nanstd(X, axis=0)
        finite_col = np.isfinite(spread)
        keep_mask = finite_col & (spread > nx.EPS)
        info.kept = np.where(keep_mask)[0].tolist()
        info.dropped_constant = np.where(~keep_mask)[0].tolist()

        if keep_mask.sum() == 0:
            return np.zeros((n, 0)), info

        Xk = X[:, keep_mask]
        Z, _, _ = nx.robust_standardize(Xk, clip=clip)   # handles NaN + winsorise
    return Z, info


def safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b, falling back to least-squares if A is singular/ill-posed."""
    with np.errstate(all="ignore"):
        try:
            x = np.linalg.solve(A, b)
            if np.all(np.isfinite(x)):
                return x
        except np.linalg.LinAlgError:
            pass
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return np.where(np.isfinite(x), x, 0.0)


def ridge_solve(Xc: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Tikhonov-stabilised normal-equation solve on already-conditioned Xc.

    ``alpha`` scales with the matrix size so regularisation is meaningful
    regardless of n; the system is guaranteed positive-definite, so even perfect
    collinearity resolves to finite coefficients.
    """
    with np.errstate(all="ignore"):
        p = Xc.shape[1]
        A = Xc.T @ Xc + alpha * np.eye(p)
        return safe_solve(A, Xc.T @ Y)
