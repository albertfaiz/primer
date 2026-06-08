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
from . import lenses as lensmod
from .diagnostics import run_diagnostics
from .landmarking import run_landmarks
from .recommend import (Recommender, RuleBasedRecommender, overall_confidence)
from .report import render_full
from .types import (PrimerReport, Severity, TaskKind)
from .validate import NullValidator, ProxyValidator, SuccessiveHalvingValidator

DataLike = Union[str, "pd.DataFrame"]


class Primer:
    def __init__(self,
                 recommender: Optional[Recommender] = None,
                 validator: Optional[ProxyValidator] = None,
                 lens: Optional[Union[str, Dict[str, float]]] = None,
                 validate: bool = False,
                 max_profile_rows: int = ingest.MAX_PROFILE_ROWS):
        self.lens = lensmod.resolve_lens(lens)
        self.recommender = recommender or RuleBasedRecommender(lens=self.lens)
        if validator is not None:
            self.validator = validator
        elif validate:
            self.validator = SuccessiveHalvingValidator()
            if not self.validator.available:
                import warnings
                warnings.warn(
                    "validate=True requires scikit-learn, which is not installed. "
                    "Proceeding with the heuristic shortlist only (no Layer 6 "
                    "empirical validation). Install it with: pip install scikit-learn",
                    stacklevel=2)
        else:
            self.validator = NullValidator()
        self.max_profile_rows = max_profile_rows

    def analyze(self, data: DataLike, target: Optional[str] = None) -> PrimerReport:
        # Numerical safety net. Every kernel is already conditioned and
        # errstate-guarded individually; this top-level guard additionally
        # silences the spurious matmul RuntimeWarnings that some BLAS builds
        # (notably Apple Accelerate) emit even on benign products — guaranteeing
        # a warning-free run on any platform, not just the one we tested on.
        import numpy as _np
        with _np.errstate(all="ignore"):
            return self._run(data, target)

    def _run(self, data: DataLike, target: Optional[str] = None) -> PrimerReport:
        # Layer 1 — ingest + type (+ memory-safe stratified sampling)
        df, profile = ingest.profile_dataset(data, target, self.max_profile_rows)

        # Layer 2 — task
        task = taskmod.resolve_task(df, profile)

        # Layer 4a — metafeatures (+ numeric matrix). Computed BEFORE diagnostics
        # in v2 so the diagnostic layer can consume intrinsic dimensionality,
        # class overlap, autocorrelation, etc.
        metafeatures = mfx.extract_metafeatures(df, profile, task)

        # Layer 4b — landmarks (supervised tasks only), through the Conditioner
        landmarks = []
        if task.kind in (TaskKind.REGRESSION, TaskKind.CLASSIFICATION) and profile.target:
            Xnum, _ = mfx.build_numeric_matrix(df, profile)
            y = mfx._encode_target(df, profile.target, task)
            landmarks = run_landmarks(Xnum, y, task.kind)
            # record the universal signal measure for the brief
            nb = [l.score for l in landmarks if not l.name.startswith("baseline")]
            bl = [l.score for l in landmarks if l.name.startswith("baseline")]
            if nb:
                metafeatures["signal_lift"] = round(max(nb) - (max(bl) if bl else 0.0), 4)
            for l in landmarks:
                if l.name == "linear_ridge":
                    metafeatures["heteroskedasticity"] = float(
                        l.detail.get("resid_pred_corr", 0.0) or 0.0)

        # Layer 3 — diagnostics (now metafeature- and landmark-aware)
        diagnostics = run_diagnostics(df, profile, task, metafeatures, landmarks)

        # Layer 5 — recommend (lens-aware)
        recommendations = self.recommender.recommend(task, metafeatures, landmarks)

        # Layer 6 — optional empirical validation (successive halving). The
        # validator returns the shortlist re-ranked by MEASURED CV score and
        # attaches per-model results; it no-ops if unavailable (no sklearn) or
        # for unsupervised tasks.
        validated = False
        val_summary = None
        if getattr(self.validator, "available", False) and recommendations and \
           task.kind in (TaskKind.REGRESSION, TaskKind.CLASSIFICATION):
            recommendations = self.validator.validate(df, profile, task, recommendations)
            val_summary = getattr(self.validator, "last_summary", None)
            validated = val_summary is not None

        # Layer 7 — confidence. When validation ran, the overall figure is the
        # empirical *decisiveness* (honest, not a calibrated probability); else
        # it falls back to the heuristic separation signal.
        if validated:
            conf, label = float(val_summary["decisiveness"]), val_summary["label"]
        else:
            conf, label = overall_confidence(recommendations)

        report = PrimerReport(
            profile=profile, task=task, diagnostics=diagnostics,
            metafeatures=metafeatures, landmarks=landmarks,
            recommendations=recommendations,
            confidence_overall=conf, confidence_label=label,
            validated=validated, validation_summary=val_summary,
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
        if report.validated and recs[0].validation:
            v = recs[0].validation
            s = report.validation_summary or {}
            metric = v.get("metric", "score")
            tail = ("the heuristic prior agreed." if s.get("agreement")
                    else f"this overturned the prior favourite ({s.get('heuristic_top')}) — "
                         "trust the measurement.")
            d.append(f"Empirically, {top} scored best under subsample cross-validation "
                     f"({v.get('cv_score'):.3f} {metric}, ±{v.get('cv_std'):.3f}); {tail}")
        else:
            runners = ", ".join(r.model for r in recs[1:3])
            d.append(f"Start with {top}"
                     + (f", and benchmark it against {runners} before committing compute." if runners else "."))

    # confidence-aware guidance
    if report.validated and report.confidence_overall < 0.45:
        d.append("Empirical decisiveness is low — the top candidates are within fold-to-fold "
                 "noise of each other. Treat them as a tie and pick on secondary criteria "
                 "(speed, interpretability, maintenance).")
    elif (not report.validated) and report.confidence_overall < 0.5 and len(recs) >= 2:
        d.append("Confidence is only moderate/low — the top two are close, so run validate=True "
                 "(cheap-proxy comparison) before committing to a full run.")

    # diagnostics -> actions
    for diag in report.diagnostics:
        if diag.severity is Severity.CRITICAL and diag.name == "possible_target_leakage":
            d.append("Resolve the suspected leakage column(s) FIRST — any model will look "
                     "deceptively perfect until you confirm they are available at prediction time.")
        if diag.name == "class_imbalance" and diag.severity in (Severity.WARNING, Severity.CRITICAL):
            d.append("For the imbalance, use class weights or resampling and report PR-AUC / F1 "
                     "rather than raw accuracy.")
        if diag.name in ("temporal_columns", "sequential_autocorrelation"):
            d.append("Validate with a chronological / group-wise split, not random k-fold — "
                     "the rows are not independent.")
        if diag.name == "high_missingness":
            d.append("Address the heavily-missing columns (impute or drop) — or lean on boosting, "
                     "which handles NaNs natively.")
        if diag.name == "high_cardinality_categorical":
            d.append("Encode the high-cardinality column(s) with target/hashing encoding, or use "
                     "CatBoost/LightGBM directly rather than one-hot.")
        if diag.name == "severe_class_overlap":
            d.append("The classes overlap geometrically — before tuning models, check whether the "
                     "features actually carry class-separating information.")
        if diag.name == "low_intrinsic_dimensionality":
            d.append("Effective dimensionality is low — try PCA or L1/L2 regularisation before "
                     "adding model capacity.")

    # task-flavoured next step
    if task.kind is TaskKind.UNSUPERVISED:
        d.append("No target given: decide whether the goal is grouping (clustering), "
                 "compression (PCA/UMAP), or outlier detection (Isolation Forest) — the "
                 "shortlist covers all three.")
    elif report.landmarks:
        lift = report.metafeatures.get("signal_lift", None)
        if lift is not None and lift < 0.03:
            d.append("Cheap probes barely beat the baseline — there is little learnable signal. "
                     "Invest in feature engineering or more/better data before chasing a fancier "
                     "estimator; a complex model here would only overfit noise.")

    d.append("These are heuristic priors, not benchmarked results — treat the shortlist as "
             "where to *start*, then let a quick empirical comparison settle the winner.")
    return d


# module-level convenience -------------------------------------------------- #
def analyze(data: DataLike, target: Optional[str] = None,
            lens: Optional[Union[str, Dict[str, float]]] = None,
            validate: bool = False) -> PrimerReport:
    """One-call entry point: ``primer.analyze('data.csv', target='y')``.

    Optional ``lens`` applies a domain weight profile (see ``primer.lenses``).
    Optional ``validate=True`` runs Layer 6 — empirical successive-halving
    validation of the shortlist (requires scikit-learn; no-ops without it) — and
    reports measured CV scores plus an honest decisiveness figure.
    """
    return Primer(lens=lens, validate=validate).analyze(data, target)
