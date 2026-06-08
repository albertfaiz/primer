"""Layer 6 + 7 — cheap empirical validation and honest, evidence-grounded confidence.

This is the layer that turns the rule engine's *prior* into something measured.
It takes the heuristic shortlist and actually trains the candidates — on small,
growing subsamples, with successive-halving (Hyperband-style) promotion:

    fit every candidate on a small budget → keep the top half → grow the data →
    refit the survivors → repeat. The model that survives to the largest budget,
    with the best cross-validated score, wins.

Two design decisions keep this honest and self-contained:

1.  **Optional dependency, zero-dependency core.** Training real models needs
    scikit-learn. The *core* package remains NumPy + pandas; this validator is
    pluggable and simply turns itself off (``available = False``) if sklearn is
    not installed. scikit-learn's ``HistGradientBoosting`` is a faithful,
    dependency-free proxy for the LightGBM/XGBoost/CatBoost family, so we can
    validate almost the whole registry without three heavy boosting libraries.

2.  **No confidence we can't defend (Layer 7).** An uncalibrated probability is
    worse than none: if it says 80% it must be right ~80% of the time, which
    requires held-out *benchmark* calibration we have not done. So we do NOT emit
    a P(win). Instead we report **measured CV scores (mean ± fold std)** — fully
    defensible, because they are measured on the user's own data — and a
    **decisiveness** score that transparently aggregates the three quantities
    that actually warrant trust:
        (a) metafeature decisiveness  — how strongly the prior pointed one way;
        (b) heuristic↔empirical agreement — did the measured winner match the prior;
        (c) proxy-curve separation     — is the leader's margin bigger than the
            fold-to-fold noise?
    The report states, in words, that this is a decisiveness measure and not a
    calibrated success probability. True calibration remains the roadmap.
"""
from __future__ import annotations

import abc
import math
from typing import List, Optional

import numpy as np

from . import registry
from .types import DatasetProfile, Recommendation, TaskKind, TaskSpec

try:                                   # the only place sklearn is (optionally) needed
    import sklearn  # noqa: F401
    _HAS_SKLEARN = True
except Exception:                      # pragma: no cover - environment dependent
    _HAS_SKLEARN = False


class ProxyValidator(abc.ABC):
    """Empirically validates a shortlist on cheap budgets (Layer 6)."""

    available: bool = False

    @abc.abstractmethod
    def validate(self, df, profile: DatasetProfile, task: TaskSpec,
                 shortlist: List[Recommendation],
                 budget_fracs: Optional[List[float]] = None) -> List[Recommendation]:
        ...


class NullValidator(ProxyValidator):
    """Default no-op: returns the shortlist unchanged (pure heuristic mode)."""
    available = False
    last_summary = None

    def validate(self, df, profile, task, shortlist, budget_fracs=None):
        return shortlist


# --------------------------------------------------------------------------- #
# estimator proxies  (sklearn, imported lazily)
# --------------------------------------------------------------------------- #
def _proxy_signature(family: str, key: str, clf: bool) -> str:
    """Distinct trainable proxy per signature. The boosting family shares one
    HistGradientBoosting proxy; linear regression splits Ridge vs ElasticNet."""
    if family == "linear" and not clf:
        return "lin_enet" if key == "elasticnet" else "lin_ridge"
    if family == "linear":
        return "lin_logreg"
    return family


def _make_estimator(family: str, key: str, clf: bool):
    from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                   HistGradientBoostingRegressor,
                                   RandomForestClassifier, RandomForestRegressor)
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy="median")
    scale = StandardScaler()

    if family == "gbt":
        est = (HistGradientBoostingClassifier(random_state=0) if clf
               else HistGradientBoostingRegressor(random_state=0))
        return make_pipeline(imp, est)                     # trees: scaling not needed
    if family == "forest":
        est = (RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1) if clf
               else RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=-1))
        return make_pipeline(imp, est)
    if family == "instance":
        est = KNeighborsClassifier() if clf else KNeighborsRegressor()
        return make_pipeline(imp, scale, est)
    if family == "kernel":
        est = SVC(kernel="rbf", class_weight="balanced") if clf else SVR(kernel="rbf")
        return make_pipeline(imp, scale, est)
    if family == "bayes":
        return make_pipeline(imp, scale, GaussianNB())
    if family == "neural":
        est = (MLPClassifier(max_iter=300, random_state=0) if clf
               else MLPRegressor(max_iter=300, random_state=0))
        return make_pipeline(imp, scale, est)
    # linear
    if clf:
        return make_pipeline(imp, scale, LogisticRegression(max_iter=1000,
                                                            class_weight="balanced"))
    if key == "elasticnet":
        return make_pipeline(imp, scale, ElasticNet(alpha=0.1, random_state=0))
    return make_pipeline(imp, scale, Ridge(alpha=1.0))


# --------------------------------------------------------------------------- #
# the validator
# --------------------------------------------------------------------------- #
class SuccessiveHalvingValidator(ProxyValidator):
    """Empirical validation by successive halving on growing subsamples."""

    name = "successive_halving"

    def __init__(self, budgets=(0.15, 0.35, 0.7), cv: int = 3, top_k: int = 5,
                 max_rows: int = 6000, min_eval: int = 200, seed: int = 0):
        self.available = _HAS_SKLEARN
        self.budgets = tuple(budgets)
        self.cv = cv
        self.top_k = top_k
        self.max_rows = max_rows
        self.min_eval = min_eval
        self.seed = seed
        self.last_summary = None

    # -- helpers -----------------------------------------------------------
    def _subsample(self, n_total, y, n, clf, seed):
        rng = np.random.default_rng(seed)
        if n >= n_total:
            return np.arange(n_total)
        if clf:
            idx = []
            for c in np.unique(y):
                ci = np.where(y == c)[0]
                take = min(len(ci), max(self.cv, int(round(len(ci) * n / n_total))))
                idx.extend(rng.choice(ci, take, replace=False).tolist())
            return np.array(sorted(idx))
        return np.sort(rng.choice(n_total, n, replace=False))

    def _cv_score(self, est, X, y, clf, seed):
        from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                              KFold)
        try:
            if clf:
                splitter = StratifiedKFold(self.cv, shuffle=True, random_state=seed)
                scoring = "balanced_accuracy"
            else:
                splitter = KFold(self.cv, shuffle=True, random_state=seed)
                scoring = "r2"
            s = cross_val_score(est, X, y, cv=splitter, scoring=scoring)
            return float(np.mean(s)), float(np.std(s))
        except Exception:
            return float("-inf"), 0.0

    # -- main --------------------------------------------------------------
    def validate(self, df, profile, task, shortlist, budget_fracs=None):
        self.last_summary = None
        if not self.available or not shortlist or \
           task.kind not in (TaskKind.REGRESSION, TaskKind.CLASSIFICATION):
            return shortlist

        from . import metafeatures as mfx
        clf = task.kind is TaskKind.CLASSIFICATION
        X, _names = mfx.build_numeric_matrix(df, profile)
        if X is None or X.shape[1] == 0:
            return shortlist
        y = mfx._encode_target(df, profile.target, task)
        if len(y) < 30:
            return shortlist

        # cap rows for speed (stratified for classification)
        sel = self._subsample(X.shape[0], y, min(self.max_rows, X.shape[0]),
                              clf, self.seed)
        X, y = X[sel], y[sel]
        N = X.shape[0]
        metric = "balanced_accuracy" if clf else "r2"
        budgets = list(budget_fracs or self.budgets)

        display2key = {m.display: m.key for m in registry._MODELS}
        cands = shortlist[:self.top_k]

        # dedupe by trainable proxy (boosting variants share one HistGBM proxy)
        rep_of = {}                       # signature -> representative model name
        sig_of = {}                       # model name -> signature
        for r in cands:
            key = display2key.get(r.model, "")
            sig = _proxy_signature(r.family, key, clf)
            sig_of[r.model] = sig
            rep_of.setdefault(sig, (r, key))

        survivors = list(rep_of.values())            # [(rec, key), ...] one per proxy
        curves = {r.model: [] for r, _ in survivors}
        results = {}                                  # rep model name -> result dict

        for i, frac in enumerate(budgets):
            n = int(np.clip(round(frac * N), self.min_eval, N))
            sub = self._subsample(N, y, n, clf, self.seed + i + 1)
            Xs, ys = X[sub], y[sub]
            scored = []
            for r, key in survivors:
                est = _make_estimator(r.family, key, clf)
                mean, std = self._cv_score(est, Xs, ys, clf, self.seed)
                curves[r.model].append((round(frac, 3), round(mean, 4)))
                results[r.model] = {"cv_score": round(mean, 4), "cv_std": round(std, 4),
                                    "curve": list(curves[r.model]), "promoted_frac": frac,
                                    "n_rows": int(len(sub)), "metric": metric,
                                    "survived": False}
                scored.append((mean, r, key))
            scored = [t for t in scored if np.isfinite(t[0])]
            if not scored:
                return shortlist                       # everything failed; stay heuristic
            scored.sort(key=lambda t: t[0], reverse=True)
            if i < len(budgets) - 1:
                keep_n = max(1, math.ceil(len(scored) / 2))
                survivors = [(r, key) for _m, r, key in scored[:keep_n]]
            else:
                survivors = [(r, key) for _m, r, key in scored]
                for _m, r, _k in scored:
                    results[r.model]["survived"] = True

        # propagate each representative's measured result to its proxy-mates,
        # then re-rank ALL candidates by (reached-budget, cv-score)
        for r in cands:
            rep_name = rep_of[sig_of[r.model]][0].model
            res = dict(results.get(rep_name, {}))
            if rep_name != r.model and res:
                res["proxied_by"] = rep_name
            r.validation = res or None

        def _rank_key(r):
            v = r.validation or {}
            return (v.get("promoted_frac", 0.0), v.get("cv_score", float("-inf")))

        ranked = sorted(cands, key=_rank_key, reverse=True)
        tail = [r for r in shortlist if r not in cands]
        out = ranked + tail
        for i, r in enumerate(out, 1):
            r.rank = i

        self.last_summary = self._summarize(shortlist, ranked, clf, metric, sig_of)
        return out

    # -- Layer 7: honest decisiveness --------------------------------------
    def _summarize(self, heuristic_order, empirical_order, clf, metric, sig_of):
        base = 0.5 if clf else 0.0
        top = empirical_order[0]
        top_sig = sig_of.get(top.model)
        heuristic_top = heuristic_order[0].model
        # family/proxy-aware agreement: boosting variants share a proxy, so a
        # CatBoost→LightGBM swap within the same proxy still counts as agreement.
        agree = (heuristic_top == top.model) or (sig_of.get(heuristic_top) == top_sig)

        v1 = top.validation or {}
        s1, sd1 = v1.get("cv_score", base), v1.get("cv_std", 0.0)

        # runner-up = best candidate from a DIFFERENT proxy (a proxy-mate tying
        # the leader tells us nothing about how decisive the choice is)
        runner = next((r for r in empirical_order[1:]
                       if sig_of.get(r.model) != top_sig and (r.validation or {})), None)
        if runner is None:
            runner = next((r for r in empirical_order[1:] if (r.validation or {})), None)
        if runner is not None:
            v2 = runner.validation
            s2, sd2 = v2.get("cv_score", base), v2.get("cv_std", 0.0)
        else:
            s2, sd2 = base, 0.0

        sep_sigma = (s1 - s2) / (math.sqrt(sd1 ** 2 + sd2 ** 2) + 1e-9)
        sep_norm = float(np.clip(sep_sigma / 3.0, 0.0, 1.0))     # 3σ gap -> decisive
        prior_dec = float(np.clip(heuristic_order[0].score - heuristic_order[1].score, 0.0, 1.0)) \
            if len(heuristic_order) > 1 else 0.5
        signal = float(np.clip((s1 - base) / 0.4, 0.0, 1.0))

        decisiveness = float(np.clip(
            0.25 * prior_dec + 0.30 * (1.0 if agree else 0.25)
            + 0.30 * sep_norm + 0.15 * signal, 0.05, 0.95))
        label = "high" if decisiveness >= 0.66 else "moderate" if decisiveness >= 0.40 else "low"

        curve = v1.get("curve", [])
        slope = (curve[-1][1] - curve[0][1]) if len(curve) >= 2 else 0.0
        curve_read = ("still climbing as data grows — more data may lift it further"
                      if slope > 0.02 else
                      "flat across budgets — it has largely saturated on this data")

        return {
            "metric": metric,
            "empirical_top": top.model, "heuristic_top": heuristic_top,
            "agreement": agree,
            "top1_score": round(s1, 4), "top1_std": round(sd1, 4),
            "top2_score": round(s2, 4), "top2_std": round(sd2, 4),
            "separation_sigma": round(float(sep_sigma), 2),
            "decisiveness": round(decisiveness, 3), "label": label,
            "curve_read": curve_read,
            "n_evaluated": sum(1 for r in empirical_order if r.validation),
            "budgets": list(self.budgets),
            "note": ("decisiveness aggregates prior strength, heuristic-vs-measured "
                     "agreement, and curve separation on YOUR data — it is NOT a "
                     "calibrated probability of out-of-sample success."),
        }
