"""Report rendering — turns a PrimerReport into readable text.

Kept separate from the data structures so output formatting can evolve (or gain a
rich/HTML variant) without touching the analysis.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import Severity

if TYPE_CHECKING:
    from .types import PrimerReport

_BAR = "─" * 64
_SEV_MARK = {Severity.CRITICAL: "[!!]", Severity.WARNING: "[! ]", Severity.INFO: "[ i]"}


def _conf_bar(x: float, width: int = 20) -> str:
    filled = int(round(x * width))
    return "█" * filled + "·" * (width - filled)


def render_summary(report: "PrimerReport") -> str:
    p, t = report.profile, report.task
    lines = [_BAR, "  PRIMER — model-selection brief", _BAR]
    smp = f"  (profiled on {p.sample_rows:,} sampled rows)" if p.sampled else ""
    lines.append(f"  data      : {p.n_rows:,} rows x {p.n_cols} cols{smp}")
    lines.append(f"  task      : {t.kind.value}"
                 + (f" / {t.subtype}" if t.subtype else "")
                 + (f"  ({t.n_classes} classes)" if t.n_classes else ""))
    crit = sum(1 for d in report.diagnostics if d.severity is Severity.CRITICAL)
    warn = sum(1 for d in report.diagnostics if d.severity is Severity.WARNING)
    lines.append(f"  flags     : {crit} critical, {warn} warning")
    lines.append(f"  confidence: {report.confidence_label.upper()} "
                 f"({report.confidence_overall:.0%})  {_conf_bar(report.confidence_overall)}")
    lines.append("")
    lines.append("  recommended models:")
    for r in report.recommendations[:3]:
        lines.append(f"    {r.rank}. {r.model:<28} score {r.score:.2f}  conf {r.confidence:.0%}")
        if r.reasons:
            lines.append(f"       └ {r.reasons[0].text}")
    lines.append(_BAR)
    return "\n".join(lines)


def render_full(report: "PrimerReport") -> str:
    p, t = report.profile, report.task
    L = [render_summary(report), ""]

    # task reasoning
    L.append("TASK")
    L.append(f"  {t.reason}")
    if t.class_balance:
        bal = ", ".join(f"{k}:{v:.0%}" for k, v in t.class_balance.items())
        L.append(f"  class balance — {bal}")
    L.append("")

    # diagnostics
    L.append("DIAGNOSTICS")
    if not report.diagnostics:
        L.append("  none — data looks clean.")
    for d in report.diagnostics:
        L.append(f"  {_SEV_MARK[d.severity]} {d.message}")
        if d.columns:
            shown = ", ".join(d.columns[:6]) + (" …" if len(d.columns) > 6 else "")
            L.append(f"        columns: {shown}")
    L.append("")

    # recommendations
    L.append("MODEL SHORTLIST")
    for r in report.recommendations:
        L.append(f"  {r.rank}. {r.model}   [{r.family}]   "
                 f"score {r.score:.2f} | confidence {r.confidence:.0%}")
        for e in r.reasons:
            sign = "+" if e.weight >= 0 else "−"
            L.append(f"       {sign} {e.text}")
        for c in r.caveats:
            L.append(f"       · note: {c}")
    L.append("")

    # landmarks
    L.append("LANDMARK PROBES  (cheap models — structure detectors)")
    if not report.landmarks:
        L.append("  (skipped — no usable numeric/encoded features)")
    for lm in report.landmarks:
        L.append(f"    {lm.name:<22} {lm.metric:<20} {lm.score:+.3f}")
    L.append("")

    # key metafeatures
    L.append("KEY METAFEATURES")
    mf = report.metafeatures
    keys = ["n_rows", "n_features", "n_to_p_ratio", "frac_categorical",
            "overall_missing_frac", "mean_abs_skew", "max_abs_correlation",
            "condition_number", "mean_target_mi", "max_target_mi",
            "frac_informative_features", "min_class_frac", "imbalance_ratio"]
    for k in keys:
        if k in mf and mf[k] is not None:
            v = mf[k]
            v = f"{v:.3f}" if isinstance(v, float) else str(v)
            L.append(f"    {k:<28} {v}")
    L.append("")

    # directions
    L.append("DIRECTIONS")
    for d in report.directions:
        L.append(f"  → {d}")
    L.append(_BAR)
    return "\n".join(L)
