"""Pipeline orchestrator.

`Primer` wires the layers together; `analyze()` is the one-call entry point.
Every component (recommender, validator, rule set) is injectable so the package
stays open to extension: swap in a meta-learned recommender or enable the
cheap-proxy validator without changing call sites.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import pandas as pd

from . import ingest, metafeatures as mfx, task as taskmod
from .diagnostics import run_diagnostics
from .landmarking import run_landmarks
from .recommend import (Recommender, RuleBasedRecommender, overall_confidence)
from .report import render_full
from .types import (PrimerReport, Severity, TaskKind)
from .validate import NullValidator, ProxyValidator

DataLike = Union[str, "pd.DataFrame"]


class Primer:
    def __init__(self,
                 recommender: Optional[Recommender] = None,
                 validator: Optional[ProxyValidator] = None,
                 max_profile_rows: int = ingest.MAX_PROFILE_ROWS):
        self.recommender = recommender or RuleBasedRecommender()
        self.validator = validator or NullValidator()
        self.max_profile_rows = max_profile_rows

    def analyze(self, data: DataLike, target: Optional[str] = None) -> PrimerReport:
        # Layer 1 — ingest + type
        df, profile = ingest.profile_dataset(data, target, self.max_profile_rows)

        # Layer 2 — task
        task = taskmod.resolve_task(df, profile)

        # Layer 3 — diagnostics
        diagnostics = run_diagnostics(df, profile, task)

        # Layer 4a — metafeatures (+ numeric matrix)
        metafeatures = mfx.extract_metafeatures(df, profile, task)

        # Layer 4b — landmarks (supervised tasks only)
        landmarks = []
        if task.kind in (TaskKind.REGRESSION, TaskKind.CLASSIFICATION) and profile.target:
            Xnum, _ = mfx.build_numeric_matrix(df, profile)
            y = mfx._encode_target(df, profile.target, task)
            landmarks = run_landmarks(Xnum, y, task.kind)

        # Layer 5 — recommend
        recommendations = self.recommender.recommend(task, metafeatures, landmarks)

        # Layer 6 — optional cheap-proxy validation (no-op by default)
        if getattr(self.validator, "available", False) and recommendations:
            recommendations = self.validator.validate(
                df, profile, task, recommendations[:3]) + recommendations[3:]

        conf, label = overall_confidence(recommendations)

        report = PrimerReport(
            profile=profile, task=task, diagnostics=diagnostics,
            metafeatures=metafeatures, landmarks=landmarks,
            recommendations=recommendations,
            confidence_overall=conf, confidence_label=label,
        )
        report.directions = _build_directions(report)
        return report

    # convenience
    def render(self, data: DataLike, target: Optional[str] = None) -> str:
        return render_full(self.analyze(data, target))


def _build_directions(report: PrimerReport) -> List[str]:
    """Concrete next steps synthesised from the analysis."""
    d: List[str] = []
    recs = report.recommendations
    task = report.task

    if recs:
        top = recs[0].model
        runners = ", ".join(r.model for r in recs[1:3])
        d.append(f"Start with {top}"
                 + (f", and benchmark it against {runners} before committing compute." if runners else "."))

    # confidence-aware guidance
    if report.confidence_overall < 0.5 and len(recs) >= 2:
        d.append("Confidence is only moderate/low — the top two are close, so run the "
                 "cheap-proxy comparison (small sub-sample, tiny budget) before a full run.")

    # diagnostics -> actions
    for diag in report.diagnostics:
        if diag.severity is Severity.CRITICAL and diag.name == "possible_target_leakage":
            d.append("Resolve the suspected leakage column(s) FIRST — any model will look "
                     "deceptively perfect until you confirm they are available at prediction time.")
        if diag.name == "class_imbalance" and diag.severity in (Severity.WARNING, Severity.CRITICAL):
            d.append("For the imbalance, use class weights or resampling and report PR-AUC / F1 "
                     "rather than raw accuracy.")
        if diag.name == "temporal_columns":
            d.append("Datetime present: validate with a chronological split, not random k-fold.")
        if diag.name == "high_missingness":
            d.append("Address the heavily-missing columns (impute or drop) — or lean on boosting, "
                     "which handles NaNs natively.")

    # task-flavoured next step
    if task.kind is TaskKind.UNSUPERVISED:
        d.append("No target given: decide whether the goal is grouping (clustering), "
                 "compression (PCA/UMAP), or outlier detection (Isolation Forest) — the "
                 "shortlist covers all three.")
    elif report.landmarks:
        best = max((l.score for l in report.landmarks
                    if not l.name.startswith("baseline")), default=0.0)
        if best < 0.15:
            d.append("Cheap models found little signal — invest in feature engineering or "
                     "more data before chasing a fancier estimator.")

    d.append("These are heuristic priors, not benchmarked results — treat the shortlist as "
             "where to *start*, then let a quick empirical comparison settle the winner.")
    return d


# module-level convenience -------------------------------------------------- #
def analyze(data: DataLike, target: Optional[str] = None) -> PrimerReport:
    """One-call entry point: ``primer.analyze('data.csv', target='y')``."""
    return Primer().analyze(data, target)
