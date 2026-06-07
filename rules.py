"""Layer 5 — the rule engine (the 'intelligence').

A rule looks at the metafeatures, landmark scores, and task, decides whether it
*fires* (and how strongly), then deposits signed, human-readable evidence onto
the model capabilities it favours or disfavours. The recommender sums that
evidence per candidate model.

Why this shape:
  * transparent  — every score decomposes into named reasons;
  * extensible   — add a rule = append to RULES, no core change;
  * bridge-ready — the per-capability weights are exactly what a meta-learner
                   would *learn* from OpenML later, so v2 swaps weights for
                   learned ones without changing the interface.

v2 changes (driven by the stress test):
  * `salience` ranks reasons for display, so a structural finding ("linear
    suffices", "categorical-heavy") headlines instead of a generic size prior;
  * weak-signal is measured as **lift over the baseline learner**, which is
    correct for both R² (baseline 0) and balanced accuracy (baseline ~0.5);
  * the sample-size priors were down-weighted so they stop hijacking every brief;
  * an **activation floor** keeps trivial landmark gaps from posing as structure;
  * new rules for **intrinsic dimensionality** and **class overlap**;
  * a **lens** hook multiplies rule strengths by domain-specific weights.

Evidence targets *capabilities*, not model names, so a rule like "signal is
nonlinear" automatically rewards every model that captures nonlinearity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .landmarking import landmark_dict
from .types import Evidence, LandmarkResult, TaskKind, TaskSpec

# landmark gaps below this are noise, not structure — don't fire structural rules
ACTIVATION_FLOOR = 0.03


@dataclass
class RuleContext:
    task: TaskSpec
    mf: Dict[str, float]
    lm: Dict[str, float]

    def f(self, key: str, default: float = 0.0) -> float:
        v = self.mf.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default


# A capability vote: (capability_name, weight). Positive weight rewards models
# that *have* the capability. For "cost" capabilities (needs_scaling) a negative
# weight penalises models that have the cost.
CapabilityVote = Dict[str, float]


@dataclass
class Rule:
    name: str
    description: str
    fires: Callable[[RuleContext], float]          # -> activation strength (0 = off, >0 on)
    votes: Callable[[RuleContext, float], CapabilityVote]
    rationale: Callable[[RuleContext, float], str]
    salience: float = 1.0     # display priority: structural findings > generic priors


# --------------------------------------------------------------------------- #
# helper signal extractors
# --------------------------------------------------------------------------- #
def _linear_score(ctx: RuleContext) -> Optional[float]:
    return ctx.lm.get("linear_ridge", ctx.lm.get("linear_ridge_clf"))


def _stump_score(ctx: RuleContext) -> Optional[float]:
    return ctx.lm.get("decision_stump")


def _knn_score(ctx: RuleContext) -> Optional[float]:
    return ctx.lm.get("knn1")


def _baseline_score(ctx: RuleContext) -> float:
    return ctx.lm.get("baseline_mean", ctx.lm.get("baseline_majority", 0.0)) or 0.0


def _nonbaseline(ctx: RuleContext) -> List[float]:
    return [v for k, v in ctx.lm.items() if not k.startswith("baseline")]


def _best_landmark(ctx: RuleContext) -> float:
    vals = _nonbaseline(ctx)
    return max(vals) if vals else 0.0


def _signal_lift(ctx: RuleContext) -> float:
    """Best probe minus the baseline learner — the universal 'is there signal?'.

    Works across metrics: R² baseline is 0, balanced-accuracy baseline is ~0.5,
    so the *lift* is the comparable quantity (this is the v1 bug that let pure
    noise look learnable in classification)."""
    if not _nonbaseline(ctx):
        return 0.0
    return _best_landmark(ctx) - _baseline_score(ctx)


def _has_landmarks(ctx: RuleContext) -> bool:
    return len(_nonbaseline(ctx)) > 0


# --------------------------------------------------------------------------- #
# the rule set
# --------------------------------------------------------------------------- #
RULES: List[Rule] = [

    # 1. nonlinearity detected by stump >> linear -----------------------------
    Rule(
        "nonlinearity_present",
        "A single threshold (stump) beats the linear model, so the signal is nonlinear.",
        fires=lambda c: (max(0.0, (_stump_score(c) or 0) - (_linear_score(c) or 0))
                         if _stump_score(c) is not None and _linear_score(c) is not None
                         and (_stump_score(c) - _linear_score(c)) > ACTIVATION_FLOOR else 0.0),
        votes=lambda c, s: {"nonlinearity": 3.0 * min(s, 0.4),
                            "interactions": 1.5 * min(s, 0.4)},
        rationale=lambda c, s: f"a single threshold beats the linear model by {s:.2f} — nonlinear structure favours trees/boosting.",
        salience=2.2,
    ),

    # 2. linear model already near the ceiling --------------------------------
    Rule(
        "linear_suffices",
        "The linear model alone explains most of the signal (Occam's razor).",
        fires=lambda c: 1.0 if (_linear_score(c) or 0) >= 0.9
        and ((_stump_score(c) or 0) - (_linear_score(c) or 0)) < 0.05 else 0.0,
        votes=lambda c, s: {"nonlinearity": -1.6, "interactions": -1.2, "interpretable": 1.0},
        rationale=lambda c, s: f"a linear model already scores {(_linear_score(c) or 0):.2f}; added model complexity buys little and costs interpretability.",
        salience=3.0,
    ),

    # 3. local structure: 1-NN beats linear -----------------------------------
    Rule(
        "local_structure",
        "1-NN beats the linear model, indicating local / neighbourhood structure.",
        fires=lambda c: (max(0.0, (_knn_score(c) or 0) - (_linear_score(c) or 0))
                         if _knn_score(c) is not None and _linear_score(c) is not None
                         and (_knn_score(c) - _linear_score(c)) > ACTIVATION_FLOOR else 0.0),
        votes=lambda c, s: {"local_structure": 2.5 * min(s, 0.4),
                            "nonlinearity": 1.0 * min(s, 0.4)},
        rationale=lambda c, s: f"1-NN beats linear by {s:.2f} — neighbourhood structure rewards kNN and tree ensembles.",
        salience=2.2,
    ),

    # 4. weak signal everywhere (high noise) — measured as lift over baseline --
    Rule(
        "low_signal_high_noise",
        "Even the best landmark barely beats the baseline learner — signal is weak / noisy.",
        fires=lambda c: 1.0 if _has_landmarks(c) and _signal_lift(c) < ACTIVATION_FLOOR else 0.0,
        votes=lambda c, s: {"interpretable": 1.2, "fast_train": 1.0,
                            "nonlinearity": -1.0, "interactions": -1.0},
        rationale=lambda c, s: f"the best cheap probe beats the baseline by only {_signal_lift(c):+.2f} — there is little learnable signal; invest in features/data, not a fancier model.",
        salience=3.0,
    ),

    # 5. large n -> favour scalable, penalise quadratic methods ---------------
    Rule(
        "large_sample",
        "Many rows — favour models that scale, penalise kNN/SVM.",
        fires=lambda c: 1.0 if c.f("n_rows") >= 100_000 else
        (0.5 if c.f("n_rows") >= 20_000 else 0.0),
        votes=lambda c, s: {"scales_large_n": 2.0 * s, "fast_train": 1.0 * s},
        rationale=lambda c, s: f"{int(c.f('n_rows')):,} rows — prioritise fast, scalable learners (LightGBM, linear).",
        salience=0.8,
    ),

    # 6. small n -> penalise data-hungry models (down-weighted in v2) ----------
    Rule(
        "small_sample",
        "Few rows — penalise neural nets / deep models, favour regularised & low-n-friendly.",
        fires=lambda c: 1.0 if c.f("n_rows") < 300 else
        (0.4 if c.f("n_rows") < 1_500 else 0.0),
        votes=lambda c, s: {"low_n_friendly": 1.2 * s, "interpretable": 0.4 * s},
        rationale=lambda c, s: f"only {int(c.f('n_rows')):,} rows — data-hungry models overfit; prefer regularised/low-variance learners and strong cross-validation.",
        salience=0.5,
    ),

    # 7. wide data (p large vs n) ---------------------------------------------
    Rule(
        "high_dimensional",
        "Feature count is large relative to rows — favour sparse / high-dim methods.",
        fires=lambda c: 1.0 if c.f("n_to_p_ratio", 1e9) < 5 else
        (0.5 if c.f("n_to_p_ratio", 1e9) < 20 else 0.0),
        votes=lambda c, s: {"high_dim_friendly": 2.0 * s, "low_n_friendly": 0.5 * s},
        rationale=lambda c, s: f"n/p ratio is {c.f('n_to_p_ratio'):.1f} — wide data favours sparse linear (L1) and regularised models; raw deep nets overfit.",
        salience=1.4,
    ),

    # 8. low intrinsic dimensionality (effective rank << p) -------------------
    Rule(
        "low_intrinsic_dim",
        "Few principal components explain the variance — effective dimension is low.",
        fires=lambda c: 1.0 - c.f("intrinsic_dim_ratio", 1.0)
        if c.f("intrinsic_dim_ratio", 1.0) < 0.5 and c.f("n_numeric") >= 4 else 0.0,
        votes=lambda c, s: {"high_dim_friendly": 1.4 * s, "interpretable": 0.5 * s,
                            "nonlinearity": -0.3 * s},
        rationale=lambda c, s: f"~{int(c.f('intrinsic_dim_95'))} components carry 95% of the variance — the data is effectively low-dimensional; regularise or reduce before adding capacity.",
        salience=1.2,
    ),

    # 9. many / high-cardinality categoricals ---------------------------------
    Rule(
        "categorical_heavy",
        "Categorical features dominate, especially high-cardinality ones.",
        fires=lambda c: min(1.0, c.f("frac_categorical")) *
        (1.0 + min(1.0, c.f("n_high_cardinality_cat") / 3.0)),
        votes=lambda c, s: {"native_categorical": 2.4 * min(s, 1.5)},
        rationale=lambda c, s: f"{c.f('frac_categorical'):.0%} of features are categorical"
        + (f" ({int(c.f('n_high_cardinality_cat'))} high-cardinality)" if c.f("n_high_cardinality_cat") else "")
        + " — models with native categorical handling (CatBoost, LightGBM) avoid one-hot blow-up.",
        salience=2.6,
    ),

    # 10. substantial missingness ---------------------------------------------
    Rule(
        "missing_values",
        "Non-trivial missingness — favour models that tolerate it natively.",
        fires=lambda c: 1.0 if c.f("max_missing_frac") > 0.2 else
        (0.5 if c.f("overall_missing_frac") > 0.05 else 0.0),
        votes=lambda c, s: {"robust_missing": 1.8 * s},
        rationale=lambda c, s: f"up to {c.f('max_missing_frac'):.0%} missing in some columns — boosting handles NaNs natively; linear/kNN need imputation first.",
        salience=1.6,
    ),

    # 11. heavy tails / outliers ----------------------------------------------
    Rule(
        "outliers_heavy_tails",
        "Skewed, heavy-tailed features with many outliers.",
        fires=lambda c: 1.0 if c.f("mean_outlier_frac") > 0.05 or c.f("mean_abs_skew") > 2.0 else
        (0.5 if c.f("mean_abs_skew") > 1.0 else 0.0),
        votes=lambda c, s: {"robust_outliers": 1.5 * s, "needs_scaling": -0.8 * s},
        rationale=lambda c, s: f"heavy tails (mean |skew| {c.f('mean_abs_skew'):.1f}, outlier rate {c.f('mean_outlier_frac'):.0%}) — tree models are robust; distance/linear models need transforms.",
        salience=1.0,
    ),

    # 12. multicollinearity ----------------------------------------------------
    Rule(
        "multicollinearity",
        "Strongly correlated features destabilise linear coefficients.",
        fires=lambda c: 1.0 if c.f("condition_number", 1) > 1e3 else
        (0.5 if c.f("frac_highly_correlated_pairs") > 0.1 else 0.0),
        votes=lambda c, s: {"nonlinearity": 0.4 * s},  # gently nudge toward trees/L2
        rationale=lambda c, s: "correlated features make plain linear coefficients unstable — prefer L2/elastic-net or tree models, which are unaffected.",
        salience=1.3,
    ),

    # 13. class imbalance ------------------------------------------------------
    Rule(
        "class_imbalance",
        "Minority class is rare — favour models supporting class weighting.",
        fires=lambda c: 1.0 if c.f("min_class_frac", 1.0) < 0.05 else
        (0.5 if c.f("min_class_frac", 1.0) < 0.15 else 0.0),
        votes=lambda c, s: {"class_weighting": 1.5 * s},
        rationale=lambda c, s: f"minority class is {c.f('min_class_frac'):.0%} — use class weights / resampling and evaluate with PR-AUC or F1.",
        salience=2.6,
    ),

    # 14. severe class overlap — capacity won't help (only when NO signal) ----
    Rule(
        "class_overlap",
        "Class centroids nearly coincide AND no probe finds signal — not separable.",
        fires=lambda c: 1.0 if 0 < c.f("class_separation", 1.0) < 0.5
        and c.f("signal_lift", 1.0) < 0.10 else 0.0,
        votes=lambda c, s: {"interpretable": 0.8, "nonlinearity": -0.6, "robust_outliers": 0.4},
        rationale=lambda c, s: f"class centroids are only {c.f('class_separation'):.2f} apart in standardised space and no cheap probe beats the baseline — the classes overlap, so model capacity won't rescue separability; fix features/labels.",
        salience=2.4,
    ),

    # 15. strong, concentrated signal -> simple models viable -----------------
    Rule(
        "few_informative_features",
        "A handful of features carry most of the dependence with the target.",
        fires=lambda c: 1.0 if 0 < c.f("frac_informative_features", 1.0) < 0.25
        and c.f("max_target_mi") > 0.2 else 0.0,
        votes=lambda c, s: {"interpretable": 1.0, "high_dim_friendly": 0.8},
        rationale=lambda c, s: "only a few features are strongly informative — sparse/interpretable models can match complex ones with far less risk.",
        salience=1.5,
    ),
]


def apply_rules(ctx: RuleContext, rules: Optional[List[Rule]] = None,
                lens: Optional[Dict[str, float]] = None
                ) -> "tuple[Dict[str, float], List[tuple]]":
    """Run all rules. Returns (capability_votes, fired_log).

    capability_votes: capability -> summed signed weight across fired rules
    fired_log: list of (rule_name, activation, votes, rationale, salience)

    `lens` is an optional {rule_name: multiplier} map that scales a rule's
    activation for a given domain (see `primer.lenses`) — the inversion-of-control
    hook that lets a field inject its priors without editing the engine.
    """
    rules = rules or RULES
    lens = lens or {}
    cap_votes: Dict[str, float] = {}
    fired = []
    for rule in rules:
        strength = float(rule.fires(ctx)) * float(lens.get(rule.name, 1.0))
        if strength <= 0:
            continue
        votes = rule.votes(ctx, strength)
        for cap, w in votes.items():
            cap_votes[cap] = cap_votes.get(cap, 0.0) + w
        fired.append((rule.name, strength, votes, rule.rationale(ctx, strength),
                      rule.salience))
    return cap_votes, fired


def build_context(task: TaskSpec, metafeatures: Dict, landmarks: List[LandmarkResult]
                  ) -> RuleContext:
    return RuleContext(task=task, mf=metafeatures, lm=landmark_dict(landmarks))
