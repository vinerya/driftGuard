from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DriftResult:
    """Result from a single drift check."""

    is_drift: bool
    centroid_similarity: float
    max_reference_similarity: float
    threshold: float
    nn_threshold: float
    text: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WindowDriftResult:
    """Result from a windowed (distribution-level) drift check."""

    is_drift: bool
    window_centroid_similarity: float
    drift_fraction: float
    window_size: int
    threshold: float
    drift_fraction_threshold: float
    timestamp: float = field(default_factory=time.time)


class DriftError(Exception):
    """Raised when drift is detected in blocking mode."""

    def __init__(self, result: DriftResult) -> None:
        self.result = result
        super().__init__(
            f"Drift detected: centroid_similarity={result.centroid_similarity:.4f} "
            f"< threshold={result.threshold:.4f}"
        )
