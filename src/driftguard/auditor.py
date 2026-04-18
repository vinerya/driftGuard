from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from driftguard.detector import DriftDetector
from driftguard.schema import DriftResult


@dataclass
class CorpusComparison:
    """Result of comparing two reference corpora.

    Useful for detecting domain shift between prompt versions, model upgrades,
    or dataset changes.  ``centroid_shift`` is cosine distance (0 = identical,
    1 = orthogonal); values above 0.05 are considered significant by default.
    """

    centroid_shift: float
    threshold_delta: float
    nn_threshold_delta: float
    size_delta: int
    is_significant: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "centroid_shift": round(self.centroid_shift, 6),
            "threshold_delta": round(self.threshold_delta, 6),
            "nn_threshold_delta": round(self.nn_threshold_delta, 6),
            "size_delta": self.size_delta,
            "is_significant": self.is_significant,
        }


@dataclass
class AuditReport:
    """Summary of a batch drift audit."""

    total: int
    passed: int
    flagged: int
    pass_rate: float
    drift_rate: float
    centroid_similarity_mean: float
    centroid_similarity_p5: float
    centroid_similarity_p25: float
    centroid_similarity_p50: float
    centroid_similarity_p75: float
    centroid_similarity_p95: float
    threshold: float
    nn_threshold: float
    outliers: list[DriftResult]
    results: list[DriftResult]
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_results(cls, results: list[DriftResult]) -> "AuditReport":
        if not results:
            raise ValueError("Cannot build an AuditReport from an empty result list.")
        sims = np.array([r.centroid_similarity for r in results])
        flagged = [r for r in results if r.is_drift]
        passed = [r for r in results if not r.is_drift]
        return cls(
            total=len(results),
            passed=len(passed),
            flagged=len(flagged),
            pass_rate=len(passed) / len(results),
            drift_rate=len(flagged) / len(results),
            centroid_similarity_mean=float(sims.mean()),
            centroid_similarity_p5=float(np.percentile(sims, 5)),
            centroid_similarity_p25=float(np.percentile(sims, 25)),
            centroid_similarity_p50=float(np.percentile(sims, 50)),
            centroid_similarity_p75=float(np.percentile(sims, 75)),
            centroid_similarity_p95=float(np.percentile(sims, 95)),
            threshold=results[0].threshold,
            nn_threshold=results[0].nn_threshold,
            outliers=flagged,
            results=results,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "flagged": self.flagged,
                "pass_rate": round(self.pass_rate, 4),
                "drift_rate": round(self.drift_rate, 4),
                "threshold": round(self.threshold, 4),
                "nn_threshold": round(self.nn_threshold, 4),
            },
            "distribution": {
                "mean": round(self.centroid_similarity_mean, 4),
                "p5": round(self.centroid_similarity_p5, 4),
                "p25": round(self.centroid_similarity_p25, 4),
                "p50": round(self.centroid_similarity_p50, 4),
                "p75": round(self.centroid_similarity_p75, 4),
                "p95": round(self.centroid_similarity_p95, 4),
            },
            "outliers": [
                {
                    "text": r.text[:300],
                    "centroid_similarity": round(r.centroid_similarity, 4),
                    "max_reference_similarity": round(r.max_reference_similarity, 4),
                    "metadata": r.metadata,
                }
                for r in self.outliers[:100]
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self) -> str:
        import datetime as dt

        ts = dt.datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        pass_color = "#28a745" if self.pass_rate >= 0.9 else "#ffc107" if self.pass_rate >= 0.7 else "#dc3545"
        drift_color = "#dc3545" if self.drift_rate > 0.1 else "#ffc107" if self.drift_rate > 0.0 else "#28a745"

        outlier_rows = ""
        for r in self.outliers[:100]:
            outlier_rows += (
                f"<tr>"
                f"<td>{r.text[:120].replace('<', '&lt;').replace('>', '&gt;')}</td>"
                f"<td class='score-low'>{r.centroid_similarity:.4f}</td>"
                f"<td>{r.max_reference_similarity:.4f}</td>"
                f"</tr>\n"
            )
        if not outlier_rows:
            outlier_rows = "<tr><td colspan='3' style='color:#666;font-style:italic;'>No drift detected.</td></tr>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Drift Audit Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ color: #1a1a2e; margin-bottom: 4px; }}
  .ts {{ color: #888; font-size: 0.9em; margin-bottom: 32px; }}
  .cards {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .card {{ background: #f8f9fa; border-radius: 8px; padding: 18px 22px; flex: 1; min-width: 130px; }}
  .card .val {{ font-size: 2em; font-weight: 700; }}
  .card .lbl {{ font-size: 0.8em; color: #666; margin-top: 2px; }}
  h2 {{ font-size: 1.1em; color: #444; border-bottom: 1px solid #eee; padding-bottom: 6px; margin-top: 32px; }}
  .dist {{ display: flex; gap: 0; margin-bottom: 8px; border-radius: 6px; overflow: hidden; }}
  .dist-item {{ flex: 1; text-align: center; padding: 10px 4px; background: #e9ecef; font-size: 0.8em; }}
  .dist-item .dv {{ font-weight: 700; font-size: 1.1em; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #f1f3f4; text-align: left; padding: 10px 12px; font-size: 0.82em; color: #555; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #f0f0f0; font-size: 0.88em; vertical-align: top; }}
  .score-low {{ color: #dc3545; font-weight: 600; }}
</style>
</head>
<body>
<h1>Drift Audit Report</h1>
<p class="ts">Generated {ts} &nbsp;·&nbsp; threshold {self.threshold:.4f} &nbsp;·&nbsp; nn_threshold {self.nn_threshold:.4f}</p>

<div class="cards">
  <div class="card"><div class="val">{self.total}</div><div class="lbl">Total responses</div></div>
  <div class="card"><div class="val" style="color:{pass_color}">{self.pass_rate:.1%}</div><div class="lbl">Pass rate</div></div>
  <div class="card"><div class="val" style="color:{drift_color}">{self.drift_rate:.1%}</div><div class="lbl">Drift rate</div></div>
  <div class="card"><div class="val" style="color:{drift_color}">{self.flagged}</div><div class="lbl">Flagged</div></div>
</div>

<h2>Centroid similarity distribution</h2>
<div class="dist">
  <div class="dist-item"><div class="dv">{self.centroid_similarity_p5:.3f}</div>p5</div>
  <div class="dist-item"><div class="dv">{self.centroid_similarity_p25:.3f}</div>p25</div>
  <div class="dist-item"><div class="dv">{self.centroid_similarity_p50:.3f}</div>p50</div>
  <div class="dist-item"><div class="dv">{self.centroid_similarity_mean:.3f}</div>mean</div>
  <div class="dist-item"><div class="dv">{self.centroid_similarity_p75:.3f}</div>p75</div>
  <div class="dist-item"><div class="dv">{self.centroid_similarity_p95:.3f}</div>p95</div>
</div>
<p style="font-size:0.8em;color:#888">Threshold: {self.threshold:.4f} — responses below this line are flagged.</p>

<h2>Flagged responses ({self.flagged})</h2>
<table>
  <thead><tr><th>Text</th><th>Centroid sim</th><th>Max ref sim</th></tr></thead>
  <tbody>{outlier_rows}</tbody>
</table>
</body>
</html>"""


class Auditor:
    """Batch drift auditor.

    Runs drift detection over a list of responses and returns a structured
    ``AuditReport`` with summary statistics, score distributions, and a list
    of flagged outliers.

    Usage::

        auditor = Auditor(detector)
        report = auditor.run(responses)

        print(f"Pass rate: {report.pass_rate:.1%}")
        print(report.to_json())
        open("report.html", "w").write(report.to_html())
    """

    def __init__(self, detector: DriftDetector) -> None:
        self._detector = detector

    def run(self, texts: Sequence[str]) -> AuditReport:
        """Check all texts and return an AuditReport."""
        results = [self._detector.check(t) for t in texts]
        return AuditReport.from_results(results)

    async def arun(self, texts: Sequence[str]) -> AuditReport:
        """Async variant — checks all texts concurrently."""
        results = await asyncio.gather(*[self._detector.acheck(t) for t in texts])
        return AuditReport.from_results(list(results))
