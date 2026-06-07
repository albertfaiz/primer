"""primer — a fast, transparent model-selection brief for tabular ML.

Profile a dataset, resolve the task, flag data problems, probe structure with
cheap landmark models, and return a confidence-scored, *explained* shortlist of
models to try — in seconds, before you spend hours training.

Quick start
-----------
    import primer
    report = primer.analyze("data.csv", target="price")
    print(report)                 # one-screen brief
    print(report.render())        # full report
    report.recommendations[0]     # top pick, with reasons + confidence
    report.to_dict()              # everything, JSON-ready

Design
------
Layered and injectable (see `core.Primer`):
    1 ingest      2 task        3 diagnostics
    4 metafeatures + landmarks  5 recommend (rule engine)
    6 cheap-proxy validation (roadmap)   7 calibrated confidence (roadmap)
The recommender is an interface, so a meta-learned ranker can join the
rule-based one later without changing your code.
"""
from __future__ import annotations

from . import registry
from .core import Primer, analyze
from .recommend import Recommender, RuleBasedRecommender
from .registry import ModelSpec, register
from .rules import RULES, Rule
from .types import (ColumnProfile, ColumnRole, ColumnType, DatasetProfile,
                    Diagnostic, Evidence, LandmarkResult, PrimerReport,
                    Recommendation, Severity, TaskKind, TaskSpec)
from .validate import ProxyValidator

__version__ = "0.1.0"

__all__ = [
    "analyze", "Primer",
    "Recommender", "RuleBasedRecommender", "ProxyValidator",
    "Rule", "RULES", "ModelSpec", "register", "registry",
    "PrimerReport", "Recommendation", "Diagnostic", "TaskSpec", "TaskKind",
    "DatasetProfile", "ColumnProfile", "ColumnType", "ColumnRole",
    "LandmarkResult", "Evidence", "Severity",
    "__version__",
]
