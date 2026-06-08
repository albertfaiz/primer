"""Recommendation layer.

`Recommender` is an abstract interface so the rule-based engine here can later be
joined (or replaced) by a meta-learned ranker trained on OpenML — without
touching anything upstream. The rule-based recommender:

  1. runs the rules to get per-capability votes;
  2. scores each candidate model = base prior + sum over capabilities of
     (vote x how much the model has that capability);
  3. attaches the human-readable reasons that moved each model;
  4. normalises scores to [0, 1] and derives an *honest* confidence.

Honesty note baked into the confidence: this is a heuristic *prior*, not a
number calibrated against benchmarks. Confidence reflects how decisively the
evidence separates the top choice from the rest and how strong the landmark
signal is — never a promise of out-of-sample accuracy. The right next step is
the cheap-proxy validation layer (roadmap), which turns this prior into an
empirically calibrated one.
"""
from __future__ import annotations

import abc
import math
from typing import Dict, List, Optional

from . import registry
from .rules import RuleContext, apply_rules, build_context
from .types import (Evidence, LandmarkResult, Recommendation, TaskKind, TaskSpec)

# capabilities that are *costs*: a model "having" them should be penalised when a
# rule votes negatively (handled by sign), and they never grant base credit.
_COST_CAPS = {"needs_scaling"}

# small base prior per family so the ranking is sensible even when no rule fires
_FAMILY_PRIOR = {
    "gbt": 0.55, "forest": 0.45, "linear": 0.40, "kernel": 0.30,
    "instance": 0.28, "bayes": 0.30, "neural": 0.25,
    "clustering": 0.45, "projection": 0.40, "anomaly": 0.40,
}


class Recommender(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def recommend(self, task: TaskSpec, metafeatures: Dict,
                  landmarks: List[LandmarkResult]) -> List[Recommendation]:
        ...


class RuleBasedRecommender(Recommender):
    name = "rule_based"

    def __init__(self, rules=None, lens: Optional[Dict[str, float]] = None):
        self.rules = rules
        self.lens = lens

    def recommend(self, task: TaskSpec, metafeatures: Dict,
                  landmarks: List[LandmarkResult]) -> List[Recommendation]:
        ctx = build_context(task, metafeatures, landmarks)
        cap_votes, fired = apply_rules(ctx, self.rules, lens=self.lens)

        candidates = registry.models_for(task.kind.value)
        if not candidates:
            return []

        # map each fired rule -> (text, votes, salience) so we can attribute
        # contributions back to named rules per model and rank them for display.
        rule_text = {name: rationale for name, _s, _v, rationale, _sal in fired}
        rule_votes = {name: votes for name, _s, votes, _r, _sal in fired}
        rule_sal = {name: sal for name, _s, _v, _r, sal in fired}

        raw_scores: Dict[str, float] = {}
        reasons_by_model: Dict[str, List[Evidence]] = {}

        for m in candidates:
            score = _FAMILY_PRIOR.get(m.family, 0.3)
            reasons: List[Evidence] = []
            # attribute each fired rule's effect on THIS model
            for rname, votes in rule_votes.items():
                contrib = 0.0
                for cap, w in votes.items():
                    have = m.cap(cap)
                    contrib += w * have
                if abs(contrib) > 0.02:
                    reasons.append(Evidence(rule=rname, weight=round(contrib, 3),
                                            text=rule_text[rname],
                                            salience=rule_sal.get(rname, 1.0)))
                    score += contrib
            raw_scores[m.key] = score
            # The SCORE already summed every contribution above. For DISPLAY we
            # order reasons by salience first (structural findings outrank generic
            # size/dimensionality priors) then by absolute impact — this is the v2
            # fix for the small_sample rule hijacking every explanation.
            reasons.sort(key=lambda e: (e.salience, abs(e.weight)), reverse=True)
            reasons_by_model[m.key] = reasons[:4]

        # normalise to [0, 1] via min-max across candidates
        vals = list(raw_scores.values())
        lo, hi = min(vals), max(vals)
        spread = hi - lo if hi > lo else 1.0

        recs: List[Recommendation] = []
        for m in candidates:
            norm = (raw_scores[m.key] - lo) / spread
            recs.append(Recommendation(
                model=m.display, family=m.family, score=round(norm, 4),
                confidence=0.0, reasons=reasons_by_model[m.key],
                caveats=[m.notes] if m.notes else [],
            ))
        recs.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(recs, 1):
            r.rank = i

        _assign_confidence(recs, raw_scores, candidates, landmarks)
        return recs


def _assign_confidence(recs: List[Recommendation], raw_scores: Dict[str, float],
                       candidates, landmarks: List[LandmarkResult]) -> None:
    """Per-recommendation confidence from evidence separation + signal *lift*.

    NOT calibrated probability — a relative, honest signal of decisiveness. The
    v2 change: the signal term is the best probe's *lift over its baseline*
    (R² baseline 0, balanced-accuracy baseline ~0.5), and when that lift is
    near zero the confidence is capped LOW — recommending any model confidently
    on noise would be dishonest, however cleanly the scores happen to separate.
    """
    if len(recs) < 2:
        if recs:
            recs[0].confidence = 0.5
        return

    vals = sorted(raw_scores.values(), reverse=True)
    top, second = vals[0], vals[1]
    rng = (max(vals) - min(vals)) or 1.0
    margin = (top - second) / rng                       # separation of the leader

    # signal LIFT: how far did the best cheap probe get above its own baseline?
    lm_scores = [l.score for l in landmarks if not l.name.startswith("baseline")]
    base_scores = [l.score for l in landmarks if l.name.startswith("baseline")]
    if lm_scores:
        lift = max(lm_scores) - (max(base_scores) if base_scores else 0.0)
        signal = max(0.0, min(1.0, lift / 0.5))         # 0.5 lift -> full credit
    else:
        lift, signal = None, 0.5                         # unsupervised: neutral

    base = 0.45 + 0.35 * _sigmoid(6 * (margin - 0.25)) + 0.20 * signal
    if lift is not None and lift < _WEAK_SIGNAL_LIFT:    # essentially no signal
        base = min(base, 0.33)
    base = max(0.05, min(0.95, base))
    for r in recs:
        # leader gets full confidence; others scaled by their normalised score
        r.confidence = round(base * (0.4 + 0.6 * r.score), 4)


_WEAK_SIGNAL_LIFT = 0.03


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def overall_confidence(recs: List[Recommendation]) -> "tuple[float, str]":
    if not recs:
        return 0.0, "none"
    c = recs[0].confidence
    label = "high" if c >= 0.7 else "moderate" if c >= 0.45 else "low"
    return c, label
