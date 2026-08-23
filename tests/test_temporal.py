from pallet_audit.contracts import Evidence
from pallet_audit.temporal import TemporalEvidenceFusion


def test_temporal_fusion_waits_then_reduces_noise() -> None:
    fusion = TemporalEvidenceFusion(window_size=5, half_life_s=2.0, minimum_frames=3)
    assert fusion.update("p1", "overhang", 0.0, Evidence(0.01, 0.9, 1.0, 0.002)) is None
    assert fusion.update("p1", "overhang", 0.1, Evidence(0.03, 0.9, 1.0, 0.002)) is None
    result = fusion.update("p1", "overhang", 0.2, Evidence(0.02, 0.9, 1.0, 0.002))
    assert result is not None
    assert 0.01 < float(result.value) < 0.03
    assert result.standard_deviation is not None and result.standard_deviation > 0.002
    assert result.metadata["frames"] == 3


def test_track_eviction_prevents_identity_leakage() -> None:
    fusion = TemporalEvidenceFusion(minimum_frames=2)
    fusion.update("p1", "damage", 0.0, Evidence(0.1, 0.9))
    fusion.evict_track("p1")
    assert fusion.update("p1", "damage", 1.0, Evidence(0.9, 0.9)) is None

