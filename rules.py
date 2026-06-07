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

Evidence targets *capabilities*, not model names, so a rule like "signal is
nonlinear" automatically rewards every model that captures nonlinearity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .landmarking import landmark_dict
from .types import Evidence, LandmarkResult, TaskKind, TaskSpec


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


# --------------------------------------------------------------------------- #
# helper signal extractors
# --------------------------------------------------------------------------- #
def _linear_score(ctx: RuleContext) -> Optional[float]:
    return ctx.lm.get("linear_ridge", ctx.lm.get("linear_ridge_clf"))


def _stump_score(ctx: RuleContext) -> Optional[float]:
    return ctx.lm.get("decision_stump")


def _knn_score(ctx: RuleContext) -> Optional[float]:
    return ctx.lm.get("knn1")


def _baseline_score(ctx: RuleContext) -> Optional[float]:
    return ctx.lm.get("baseline_mean", ctx.lm.get("baseline_majority"))


def _best_landmark(ctx: RuleContext) -> float:
    vals = [v for k, v in ctx.lm.items() if not k.startswith("baseline")]
    return max(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# the rule set
# --------------------------------------------------------------------------- #
RULES: List[Rule] = [

    # 1. nonlinearity detected by stump >> linear -----------------------------
    Rule(
        "nonlinearity_present",
        "A single threshold (stump) beats the linear model, so the signal is nonlinear.",
        fires=lambda c: max(0.0, (_stump_score(c) or 0) - (_linear_score(c) or 0))
        if _stump_score(c) is not None and _linear_score(c) is not None else 0.0,
        votes=lambda c, s: {"nonlinearity": 3.0 * min(s, 0.4),
                            "interactions": 1.5 * min(s, 0.4)},
        rationale=lambda c, s: f"stump outperforms the linear model by {s:.2f} — nonlinear structure favours trees/boosting.",
    ),

    # 2. linear model already near the ceiling --------------------------------
    Rule(
        "linear_suffices",
        "The linear model alone explains most of the signal (Occam's razor).",
        fires=lambda c: 1.0 if (_linear_score(c) or 0) >= 0.9
        and ((_stump_score(c) or 0) - (_linear_score(c) or 0)) < 0.05 else 0.0,
        votes=lambda c, s: {"nonlinearity": -1.6, "interactions": -1.2, "interpretable": 1.0},
        rationale=lambda c, s: "a simple linear model already scores very high; complex models add little and cost interpretability.",
    ),

    # 3. local structure: 1-NN beats linear -----------------------------------
    Rule(
        "local_structure",
        "1-NN beats the linear model, indicating local / neighbourhood structure.",
        fires=lambda c: max(0.0, (_knn_score(c) or 0) - (_linear_score(c) or 0))
        if _knn_score(c) is not None and _linear_score(c) is not None else 0.0,
        votes=lambda c, s: {"local_structure": 2.5 * min(s, 0.4),
                            "nonlinearity": 1.0 * min(s, 0.4)},
        rationale=lambda c, s: f"1-NN beats linear by {s:.2f} — neighbourhood structure rewards kNN and tree ensembles.",
    ),

    # 4. weak signal everywhere (high noise) ----------------------------------
    Rule(
        "low_signal_high_noise",
        "Even the best landmark barely beats the baseline — signal is weak / noisy.",
        fires=lambda c: 1.0 if len([k for k in c.lm if not k.startswith("baseline")]) > 0
        and _best_landmark(c) < 0.15 else 0.0,
        votes=lambda c, s: {"interpretable": 1.0, "fast_train": 1.0,
                            "nonlinearity": -0.8, "interactions": -0.8},
        rationale=lambda c, s: "no cheap model finds much signal — favour simple, regularised models and invest in features rather than complex estimators.",
    ),

    # 5. large n -> favour scalable, penalise quadratic methods ---------------
    Rule(
        "large_sample",
        "Many rows — favour models that scale, penalise kNN/SVM.",
        fires=lambda c: 1.0 if c.f("n_rows") >= 100_000 else
        (0.5 if c.f("n_rows") >= 20_000 else 0.0),
        votes=lambda c, s: {"scales_large_n": 2.0 * s, "fast_train": 1.0 * s},
        rationale=lambda c, s: f"{int(c.f('n_rows')):,} rows — prioritise fast, scalable learners (LightGBM, linear).",
    ),

    # 6. small n -> penalise data-hungry models -------------------------------
    Rule(
        "small_sample",
        "Few rows — penalise neural nets / deep models, favour regularised & low-n-friendly.",
        fires=lambda c: 1.0 if c.f("n_rows") < 500 else
        (0.5 if c.f("n_rows") < 2_000 else 0.0),
        votes=lambda c, s: {"low_n_friendly": 1.8 * s, "interpretable": 0.6 * s},
        rationale=lambda c, s: f"only {int(c.f('n_rows')):,} rows — data-hungry models overfit; prefer regularised/low-variance learners and strong cross-validation.",
    ),

    # 7. wide data (p large vs n) ---------------------------------------------
    Rule(
        "high_dimensional",
        "Feature count is large relative to rows — favour sparse / high-dim methods.",
        fires=lambda c: 1.0 if c.f("n_to_p_ratio", 1e9) < 5 else
        (0.5 if c.f("n_to_p_ratio", 1e9) < 20 else 0.0),
        votes=lambda c, s: {"high_dim_friendly": 2.0 * s, "low_n_friendly": 0.5 * s},
        rationale=lambda c, s: f"n/p ratio is {c.f('n_to_p_ratio'):.1f} — wide data favours sparse linear (L1) and regularised models; raw deep nets overfit.",
    ),

    # 8. many / high-cardinality categoricals ---------------------------------
    Rule(
        "categorical_heavy",
        "Categorical features dominate, especially high-cardinality ones.",
        fires=lambda c: min(1.0, c.f("frac_categorical")) *
        (1.0 + min(1.0, c.f("n_high_cardinality_cat") / 3.0)),
        votes=lambda c, s: {"native_categorical": 2.2 * min(s, 1.5)},
        rationale=lambda c, s: f"{c.f('frac_categorical'):.0%} of features are categorical"
        + (f" ({int(c.f('n_high_cardinality_cat'))} high-cardinality)" if c.f("n_high_cardinality_cat") else "")
        + " — models with native categorical handling (CatBoost, LightGBM) avoid one-hot blow-up.",
    ),

    # 9. substantial missingness ----------------------------------------------
    Rule(
        "missing_values",
        "Non-trivial missingness — favour models that tolerate it natively.",
        fires=lambda c: 1.0 if c.f("max_missing_frac") > 0.2 else
        (0.5 if c.f("overall_missing_frac") > 0.05 else 0.0),
        votes=lambda c, s: {"robust_missing": 1.8 * s},
        rationale=lambda c, s: f"up to {c.f('max_missing_frac'):.0%} missing in some columns — boosting handles NaNs natively; linear/kNN need imputation first.",
    ),

    # 10. heavy tails / outliers ----------------------------------------------
    Rule(
        "outliers_heavy_tails",
        "Skewed, heavy-tailed features with many outliers.",
        fires=lambda c: 1.0 if c.f("mean_outlier_frac") > 0.05 or c.f("mean_abs_skew") > 2.0 else
        (0.5 if c.f("mean_abs_skew") > 1.0 else 0.0),
        votes=lambda c, s: {"robust_outliers": 1.5 * s, "needs_scaling": -0.8 * s},
        rationale=lambda c, s: f"heavy tails (mean |skew| {c.f('mean_abs_skew'):.1f}, outlier rate {c.f('mean_outlier_frac'):.0%}) — tree models are robust; distance/linear models need transforms.",
    ),

    # 11. multicollinearity ----------------------------------------------------
    Rule(
        "multicollinearity",
        "Strongly correlated features destabilise linear coefficients.",
        fires=lambda c: 1.0 if c.f("condition_number", 1) > 1e3 else
        (0.5 if c.f("frac_highly_correlated_pairs") > 0.1 else 0.0),
        votes=lambda c, s: {"nonlinearity": 0.4 * s},  # gently nudge toward trees/L2
        rationale=lambda c, s: "correlated features make plain linear coefficients unstable — prefer L2/elastic-net or tree models, which are unaffected.",
    ),

    # 12. class imbalance ------------------------------------------------------
    Rule(
        "class_imbalance",
        "Minority class is rare — favour models supporting class weighting.",
        fires=lambda c: 1.0 if c.f("min_class_frac", 1.0) < 0.05 else
        (0.5 if c.f("min_class_frac", 1.0) < 0.15 else 0.0),
        votes=lambda c, s: {"class_weighting": 1.5 * s},
        rationale=lambda c, s: f"minority class is {c.f('min_class_frac'):.0%} — use class weights / resampling and evaluate with PR-AUC or F1.",
    ),

    # 13. strong, concentrated signal -> simple models viable -----------------
    Rule(
        "few_informative_features",
        "A handful of features carry most of the dependence with the target.",
        fires=lambda c: 1.0 if 0 < c.f("frac_informative_features", 1.0) < 0.25
        and c.f("max_target_mi") > 0.2 else 0.0,
        votes=lambda c, s: {"interpretable": 1.0, "high_dim_friendly": 0.8},
        rationale=lambda c, s: "only a few features are strongly informative — sparse/interpretable models can match complex ones with far less risk.",
    ),
]


def apply_rules(ctx: RuleContext, rules: Optional[List[Rule]] = None
                ) -> "tuple[Dict[str, float], List[tuple]]":
    """Run all rules. Returns (capability_votes, fired_log).

    capability_votes: capability -> summed signed weight across fired rules
    fired_log: list of (rule_name, activation, votes, rationale)
    """
    rules = rules or RULES
    cap_votes: Dict[str, float] = {}
    fired = []
    for rule in rules:
        strength = float(rule.fires(ctx))
        if strength <= 0:
            continue
        votes = rule.votes(ctx, strength)
        for cap, w in votes.items():
            cap_votes[cap] = cap_votes.get(cap, 0.0) + w
        fired.append((rule.name, strength, votes, rule.rationale(ctx, strength)))
    return cap_votes, fired


def build_context(task: TaskSpec, metafeatures: Dict, landmarks: List[LandmarkResult]
                  ) -> RuleContext:
    return RuleContext(task=task, mf=metafeatures, lm=landmark_dict(landmarks))
