"""Layer 4b — landmarking.

Cheap models fit on a sub-sample whose *scores* become features for the rule
engine. The trick: comparing a linear model to a decision stump to 1-NN tells you
whether the signal is linear, nonlinear, interaction-driven, or locally structured
— without training anything expensive. All models are pure NumPy.

Scores:
  * regression     -> R^2 on a held-out split
  * classification -> balanced accuracy on a held-out split
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import numpyx as nx
from . import conditioner as cond
from .types import LandmarkResult, TaskKind

MAX_LANDMARK_ROWS = 4_000      # cap the sub-sample so this stays in the millisecond range
MAX_KNN_TRAIN = 1_500


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot < nx.EPS:
        return 0.0
    return float(1.0 - np.sum((y - pred) ** 2) / ss_tot)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    classes = np.unique(y)
    recalls = []
    for c in classes:
        mask = y == c
        if mask.sum() == 0:
            continue
        recalls.append(np.mean(pred[mask] == c))
    return float(np.mean(recalls)) if recalls else 0.0


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def _train_test_split(n: int, y: np.ndarray, task: TaskKind, frac: float = 0.3,
                      seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if task is TaskKind.CLASSIFICATION:
        test = []
        for c in np.unique(y):
            ci = idx[y == c]
            rng.shuffle(ci)
            k = max(1, int(round(len(ci) * frac)))
            test.extend(ci[:k].tolist())
        test = np.array(sorted(test))
    else:
        rng.shuffle(idx)
        k = max(1, int(round(n * frac)))
        test = np.sort(idx[:k])
    train = np.setdiff1d(idx, test, assume_unique=False)
    return train, test


# --------------------------------------------------------------------------- #
# models (regression)
# --------------------------------------------------------------------------- #
def _ridge_fit_predict(Xtr, ytr, Xte, alpha=1.0):
    with np.errstate(all="ignore"):
        Xm, ym = Xtr.mean(0), ytr.mean()
        Xc = Xtr - Xm
        A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
        w = cond.safe_solve(A, Xc.T @ (ytr - ym))
        return (Xte - Xm) @ w + ym


def _stump_reg_fit(Xtr, ytr):
    base = np.sum((ytr - ytr.mean()) ** 2)
    best = dict(j=None, t=0.0, left=ytr.mean(), right=ytr.mean(), sse=base)
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, 9)))
        for t in qs:
            left = col <= t
            if left.sum() < 2 or (~left).sum() < 2:
                continue
            lm, rm = ytr[left].mean(), ytr[~left].mean()
            sse = np.sum((ytr[left] - lm) ** 2) + np.sum((ytr[~left] - rm) ** 2)
            if sse < best["sse"]:
                best = dict(j=j, t=float(t), left=float(lm), right=float(rm), sse=float(sse))
    return best


def _stump_reg_predict(model, Xte):
    if model["j"] is None:
        return np.full(Xte.shape[0], model["left"])
    col = Xte[:, model["j"]]
    return np.where(col <= model["t"], model["left"], model["right"])


# --------------------------------------------------------------------------- #
# models (classification)
# --------------------------------------------------------------------------- #
def _ridge_clf_fit_predict(Xtr, ytr, Xte, classes, alpha=1.0):
    with np.errstate(all="ignore"):
        Y = nx.one_hot(ytr, classes)
        Xm = Xtr.mean(0)
        Xc = Xtr - Xm
        Ym = Y.mean(0)
        A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
        W = cond.safe_solve(A, Xc.T @ (Y - Ym))
        scores = (Xte - Xm) @ W + Ym
        return classes[np.argmax(scores, axis=1)]


def _gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


def _stump_clf_fit(Xtr, ytr):
    best = dict(j=None, t=0.0, left=_majority(ytr), right=_majority(ytr), imp=_gini(ytr))
    n = len(ytr)
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        qs = np.unique(np.quantile(col, np.linspace(0.1, 0.9, 9)))
        for t in qs:
            left = col <= t
            nl, nr = left.sum(), (~left).sum()
            if nl < 2 or nr < 2:
                continue
            imp = (nl * _gini(ytr[left]) + nr * _gini(ytr[~left])) / n
            if imp < best["imp"]:
                best = dict(j=j, t=float(t), left=_majority(ytr[left]),
                            right=_majority(ytr[~left]), imp=float(imp))
    return best


def _majority(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]


def _stump_clf_predict(model, Xte):
    if model["j"] is None:
        return np.full(Xte.shape[0], model["left"])
    col = Xte[:, model["j"]]
    return np.where(col <= model["t"], model["left"], model["right"])


def _gnb_fit_predict(Xtr, ytr, Xte, classes):
    log_post = np.zeros((Xte.shape[0], classes.size))
    for k, c in enumerate(classes):
        Xc = Xtr[ytr == c]
        mu = Xc.mean(0)
        var = Xc.var(0) + 1e-9
        prior = np.log(len(Xc) / len(ytr))
        ll = -0.5 * np.sum(((Xte - mu) ** 2) / var + np.log(2 * np.pi * var), axis=1)
        log_post[:, k] = ll + prior
    return classes[np.argmax(log_post, axis=1)]


def _knn1_predict(Xtr, ytr, Xte):
    # squared euclidean via the (a-b)^2 = a^2 + b^2 - 2ab identity (vectorised)
    with np.errstate(all="ignore"):
        g = Xte @ Xtr.T
        te2 = np.sum(Xte ** 2, axis=1)[:, None]
        tr2 = np.sum(Xtr ** 2, axis=1)[None, :]
        d2 = te2 + tr2 - 2 * g
        d2 = np.where(np.isfinite(d2), d2, np.inf)
    return ytr[np.argmin(d2, axis=1)]


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def run_landmarks(Xnum: np.ndarray, y: np.ndarray, task: TaskKind,
                  seed: int = 0) -> List[LandmarkResult]:
    """Run all applicable landmarks on the numeric feature matrix.

    Inputs pass through the Conditioner (robust scaling + finite + full-rank),
    so this is numerically stable on any BLAS regardless of feature scale,
    outliers or collinearity. Scores are sanitised; nothing here can warn or crash.
    """
    results: List[LandmarkResult] = []
    if Xnum is None or Xnum.size == 0 or Xnum.shape[1] == 0:
        return results

    # sub-sample for speed (deterministic)
    rng = np.random.default_rng(seed)
    n = Xnum.shape[0]
    if n > MAX_LANDMARK_ROWS:
        sel = rng.choice(n, MAX_LANDMARK_ROWS, replace=False)
        Xnum, y = Xnum[sel], y[sel]
        n = MAX_LANDMARK_ROWS

    # single robust conditioning, reused by every probe
    X, _info = cond.condition_matrix(Xnum)
    if X.shape[1] == 0:
        return results

    def _clean_score(v: float) -> float:
        return float(v) if np.isfinite(v) else 0.0

    if task is TaskKind.REGRESSION:
        y = np.asarray(y, float)
        tr, te = _train_test_split(n, y, task, seed=seed)
        if len(tr) < 4 or len(te) < 2:
            return results

        results.append(LandmarkResult("baseline_mean", "r2",
                                      _clean_score(r2_score(y[te], np.full(len(te), y[tr].mean())))))
        ridge_pred = _ridge_fit_predict(X[tr], y[tr], X[te])
        # heteroskedasticity probe: does residual magnitude track the prediction?
        hetero = 0.0
        with np.errstate(all="ignore"):
            resid = np.abs(y[te] - ridge_pred)
            if resid.std() > nx.EPS and np.std(ridge_pred) > nx.EPS:
                hetero = abs(float(np.corrcoef(resid, ridge_pred)[0, 1]))
                hetero = hetero if np.isfinite(hetero) else 0.0
        results.append(LandmarkResult("linear_ridge", "r2",
                                      _clean_score(r2_score(y[te], ridge_pred)),
                                      detail={"resid_pred_corr": round(hetero, 3)}))
        st = _stump_reg_fit(X[tr], y[tr])
        results.append(LandmarkResult("decision_stump", "r2",
                                      _clean_score(r2_score(y[te], _stump_reg_predict(st, X[te]))),
                                      detail={"split_feature": st["j"]}))
        ktr = tr if len(tr) <= MAX_KNN_TRAIN else rng.choice(tr, MAX_KNN_TRAIN, replace=False)
        results.append(LandmarkResult("knn1", "r2",
                                      _clean_score(r2_score(y[te], _knn1_predict(X[ktr], y[ktr], X[te])))))

    elif task is TaskKind.CLASSIFICATION:
        y = np.asarray(y)
        classes = np.unique(y)
        tr, te = _train_test_split(n, y, task, seed=seed)
        if len(tr) < max(4, classes.size) or len(te) < 2:
            return results

        maj = _majority(y[tr])
        results.append(LandmarkResult("baseline_majority", "balanced_accuracy",
                                      _clean_score(balanced_accuracy(y[te], np.full(len(te), maj)))))
        results.append(LandmarkResult("linear_ridge_clf", "balanced_accuracy",
                                      _clean_score(balanced_accuracy(y[te], _ridge_clf_fit_predict(X[tr], y[tr], X[te], classes)))))
        st = _stump_clf_fit(X[tr], y[tr])
        results.append(LandmarkResult("decision_stump", "balanced_accuracy",
                                      _clean_score(balanced_accuracy(y[te], _stump_clf_predict(st, X[te]))),
                                      detail={"split_feature": st["j"]}))
        results.append(LandmarkResult("gaussian_nb", "balanced_accuracy",
                                      _clean_score(balanced_accuracy(y[te], _gnb_fit_predict(X[tr], y[tr], X[te], classes)))))
        ktr = tr if len(tr) <= MAX_KNN_TRAIN else rng.choice(tr, MAX_KNN_TRAIN, replace=False)
        results.append(LandmarkResult("knn1", "balanced_accuracy",
                                      _clean_score(balanced_accuracy(y[te], _knn1_predict(X[ktr], y[ktr], X[te])))))

    return results


def landmark_dict(results: List[LandmarkResult]) -> Dict[str, float]:
    return {r.name: r.score for r in results}
