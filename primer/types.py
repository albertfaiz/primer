"""Shared data structures for the primer pipeline.

These are intentionally plain dataclasses so the whole report is trivially
serialisable (``to_dict``) and easy to extend. Nothing here imports numpy or
pandas, keeping this module a dependency-free foundation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ColumnType(str, Enum):
    """Semantic type of a column, inferred during ingestion."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"                # high-cardinality identifier, not a feature
    CONSTANT = "constant"    # zero / near-zero variance


class ColumnRole(str, Enum):
    TARGET = "target"
    FEATURE = "feature"
    IDENTIFIER = "identifier"
    DROPPED = "dropped"      # excluded from modelling (constant / id / leak)


class TaskKind(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    UNSUPERVISED = "unsupervised"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]


@dataclass
class ColumnProfile:
    name: str
    ctype: ColumnType
    role: ColumnRole
    dtype: str
    n_unique: int
    n_missing: int
    missing_frac: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ctype"] = self.ctype.value
        d["role"] = self.role.value
        return d


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    columns: List[ColumnProfile]
    target: Optional[str]
    feature_names: List[str] = field(default_factory=list)
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    datetime_features: List[str] = field(default_factory=list)
    text_features: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    sampled: bool = False
    sample_rows: Optional[int] = None

    def column(self, name: str) -> Optional[ColumnProfile]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target": self.target,
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "datetime_features": self.datetime_features,
            "text_features": self.text_features,
            "dropped": self.dropped,
            "sampled": self.sampled,
            "sample_rows": self.sample_rows,
            "columns": [c.to_dict() for c in self.columns],
        }


@dataclass
class TaskSpec:
    kind: TaskKind
    subtype: Optional[str] = None          # e.g. "binary", "multiclass", "ordinal"
    n_classes: Optional[int] = None
    class_balance: Optional[Dict[str, float]] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass
class Diagnostic:
    name: str
    severity: Severity
    message: str
    columns: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class LandmarkResult:
    """Score of one cheap 'landmark' model on a held-out split."""
    name: str
    metric: str
    score: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """A single signed contribution from a rule toward one model family."""
    rule: str
    weight: float            # signed; positive = supports, negative = against
    text: str
    salience: float = 1.0    # display priority; structural findings outrank priors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    model: str
    family: str
    score: float                       # normalised suitability in [0, 1]
    confidence: float                  # heuristic confidence in [0, 1]
    rank: int = 0
    reasons: List[Evidence] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reasons"] = [e.to_dict() for e in self.reasons]
        return d


@dataclass
class PrimerReport:
    profile: DatasetProfile
    task: TaskSpec
    diagnostics: List[Diagnostic]
    metafeatures: Dict[str, Any]
    landmarks: List[LandmarkResult]
    recommendations: List[Recommendation]
    directions: List[str] = field(default_factory=list)
    confidence_overall: float = 0.0
    confidence_label: str = "unknown"

    # --- convenience -----------------------------------------------------
    @property
    def top(self) -> Optional[Recommendation]:
        return self.recommendations[0] if self.recommendations else None

    def shortlist(self, k: int = 3) -> List[Recommendation]:
        return self.recommendations[:k]

    def render(self) -> str:
        """Full human-readable report."""
        from .report import render_full
        return render_full(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "task": self.task.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "metafeatures": self.metafeatures,
            "landmarks": [l.to_dict() for l in self.landmarks],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "directions": self.directions,
            "confidence_overall": self.confidence_overall,
            "confidence_label": self.confidence_label,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        from .report import render_summary
        return render_summary(self)
