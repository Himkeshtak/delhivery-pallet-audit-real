from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from .contracts import Evidence


@dataclass(frozen=True)
class TimedEvidence:
    timestamp_s: float
    evidence: Evidence


class TemporalEvidenceFusion:
    """Confidence-weighted evidence fusion per externally supplied track identity."""

    def __init__(self, window_size: int = 15, half_life_s: float = 1.0, minimum_frames: int = 3):
        if window_size < 1 or half_life_s <= 0 or minimum_frames < 1:
            raise ValueError("invalid temporal fusion policy")
        self.window_size = window_size
        self.half_life_s = half_life_s
        self.minimum_frames = minimum_frames
        self._history: defaultdict[tuple[str, str], deque[TimedEvidence]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(
        self, track_id: str, feature_name: str, timestamp_s: float, evidence: Evidence
    ) -> Evidence | None:
        if not isinstance(evidence.value, (int, float)):
            return evidence if evidence.confidence > 0 else None
        key = (track_id, feature_name)
        history = self._history[key]
        history.append(TimedEvidence(timestamp_s, evidence))
        if len(history) < self.minimum_frames:
            return None

        values = np.array([float(item.evidence.value) for item in history], dtype=float)
        ages = np.array([max(0.0, timestamp_s - item.timestamp_s) for item in history])
        recency = np.exp(-math.log(2.0) * ages / self.half_life_s)
        observation_weights = np.array(
            [item.evidence.confidence * item.evidence.coverage for item in history], dtype=float
        )
        weights = recency * observation_weights
        if float(weights.sum()) <= 1e-12:
            return None
        mean = float(np.average(values, weights=weights))
        temporal_variance = float(np.average((values - mean) ** 2, weights=weights))
        measurement_variance = float(
            np.average(
                [
                    (item.evidence.standard_deviation or 0.0) ** 2
                    for item in history
                ],
                weights=weights,
            )
        )
        confidence = float(
            np.clip(
                np.average([item.evidence.confidence for item in history], weights=recency),
                0,
                1,
            )
        )
        coverage = float(
            np.clip(np.average([item.evidence.coverage for item in history], weights=recency), 0, 1)
        )
        return Evidence(
            value=mean,
            confidence=confidence,
            coverage=coverage,
            standard_deviation=math.sqrt(temporal_variance + measurement_variance),
            source=f"temporal:{history[-1].evidence.source}",
            units=history[-1].evidence.units,
            metadata={"frames": len(history), "half_life_s": self.half_life_s},
        )

    def evict_track(self, track_id: str) -> None:
        for key in [key for key in self._history if key[0] == track_id]:
            del self._history[key]
