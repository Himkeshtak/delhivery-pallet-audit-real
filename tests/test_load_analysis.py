import math

from pallet_audit.contracts import PoseEstimate, Verdict
from pallet_audit.load_analysis import (
    VisibleBoxGeometry,
    VisibleLoadGeometry,
    derive_visible_load_evidence,
)
from pallet_audit.sop import evaluate_sop


def _rotated_box(
    center: tuple[float, float], width: float, height: float, angle_deg: float
) -> tuple[tuple[float, float], ...]:
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    points = []
    for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                 (width / 2, height / 2), (-width / 2, height / 2)):
        points.append(
            (center[0] + x * cosine - y * sine, center[1] + x * sine + y * cosine)
        )
    return tuple(points)


def _pose() -> PoseEstimate:
    return PoseEstimate(
        available=True,
        x_m=0.0,
        y_m=0.0,
        theta_deg=0.0,
        visible_face="front",
        covariance_xy_m2=[[0.0001, 0.0], [0.0, 0.0001]],
        theta_std_deg=1.0,
        confidence=0.95,
        reprojection_rmse_px=0.3,
    )


def test_visible_geometry_emits_partial_metric_evidence() -> None:
    geometry = VisibleLoadGeometry(
        pallet_polygon_xy_m=((-0.6, -0.5), (0.6, -0.5), (0.6, 0.5), (-0.6, 0.5)),
        load_polygon_xy_m=((-0.5, -0.4), (0.5, -0.4), (0.5, 0.4), (-0.5, 0.4)),
        pallet_theta_deg=0.0,
        boxes=(
            VisibleBoxGeometry(_rotated_box((-0.2, 0), 0.40, 0.20, 0), 0, 0.95),
            VisibleBoxGeometry(_rotated_box((0.2, 0), 0.20, 0.10, 0), 1, 0.95),
        ),
        visible_coverage=0.9,
        pallet_mask_confidence=0.95,
        load_mask_confidence=0.9,
        boundary_std_m=0.002,
        orientation_std_deg=1.0,
    )
    evidence = derive_visible_load_evidence(geometry)
    assert evidence["overhang_m"].value == 0.0
    assert evidence["centroid_offset_m"].value < 1e-8
    assert evidence["maximum_box_rotation_deg"].value < 1e-6
    assert evidence["size_inversion_probability"].value == 0.0
    checks = evaluate_sop(evidence, _pose())
    assert checks[0].verdict is Verdict.PASS
    assert checks[2].verdict is Verdict.PASS
    assert checks[3].verdict is Verdict.MANUAL_REVIEW
    assert checks[6].verdict is Verdict.PASS
    assert checks[1].verdict is Verdict.MANUAL_REVIEW


def test_visible_geometry_detects_overhang_rotation_and_size_inversion() -> None:
    geometry = VisibleLoadGeometry(
        pallet_polygon_xy_m=((-0.6, -0.5), (0.6, -0.5), (0.6, 0.5), (-0.6, 0.5)),
        load_polygon_xy_m=((-0.5, -0.4), (0.65, -0.4), (0.65, 0.4), (-0.5, 0.4)),
        pallet_theta_deg=0.0,
        boxes=(
            VisibleBoxGeometry(_rotated_box((0, 0), 0.20, 0.10, 20), 0, 0.95),
            VisibleBoxGeometry(_rotated_box((0, 0), 0.50, 0.25, 0), 1, 0.95),
        ),
        visible_coverage=0.9,
        pallet_mask_confidence=0.95,
        load_mask_confidence=0.9,
        boundary_std_m=0.002,
        orientation_std_deg=1.0,
    )
    evidence = derive_visible_load_evidence(geometry)
    assert 0.049 < evidence["overhang_m"].value < 0.051
    assert 19.9 < evidence["maximum_box_rotation_deg"].value < 20.1
    assert evidence["size_inversion_probability"].value == 1.0
