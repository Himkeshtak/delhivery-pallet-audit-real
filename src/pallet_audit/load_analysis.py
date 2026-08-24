from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .contracts import Evidence


@dataclass(frozen=True)
class VisibleBoxGeometry:
    """One visible box mask projected into a common metric plane.

    ``layer_index`` starts at zero for the bottom visible layer. A layer is an
    annotation/inference result, not something inferred from image height alone.
    """

    polygon_xy_m: tuple[tuple[float, float], ...]
    layer_index: int | None
    confidence: float


@dataclass(frozen=True)
class VisibleLoadGeometry:
    pallet_polygon_xy_m: tuple[tuple[float, float], ...]
    load_polygon_xy_m: tuple[tuple[float, float], ...]
    pallet_theta_deg: float
    boxes: tuple[VisibleBoxGeometry, ...] = ()
    visible_coverage: float = 0.0
    pallet_mask_confidence: float = 0.0
    load_mask_confidence: float = 0.0
    boundary_std_m: float | None = None
    orientation_std_deg: float | None = None


def _polygon(points: tuple[tuple[float, float], ...], name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] != 2:
        raise ValueError(f"{name} must contain at least three 2D points")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if abs(float(cv2.contourArea(array))) < 1e-9:
        raise ValueError(f"{name} has zero area")
    return array


def _centroid(polygon: np.ndarray) -> np.ndarray:
    moments = cv2.moments(polygon)
    if abs(float(moments["m00"])) < 1e-12:
        raise ValueError("polygon centroid is undefined")
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=float,
    )


def _principal_axis_deg(polygon: np.ndarray) -> tuple[float, float]:
    centered = polygon.astype(float) - polygon.astype(float).mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    ordered = np.sort(np.maximum(eigenvalues, 0.0))
    anisotropy = 1.0 - math.sqrt(ordered[0] / max(ordered[1], 1e-12))
    angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return angle, float(np.clip(anisotropy, 0.0, 1.0))


def _axis_misalignment_deg(box_axis_deg: float, pallet_axis_deg: float) -> float:
    difference = (box_axis_deg - pallet_axis_deg + 45.0) % 90.0 - 45.0
    return abs(difference)


def _visible_size_inversion_score(boxes: tuple[VisibleBoxGeometry, ...]) -> tuple[float, int]:
    by_layer: dict[int, list[float]] = {}
    for box in boxes:
        if box.layer_index is None:
            continue
        polygon = _polygon(box.polygon_xy_m, "box polygon")
        by_layer.setdefault(box.layer_index, []).append(abs(float(cv2.contourArea(polygon))))
    ordered_layers = sorted(by_layer)
    comparisons = 0
    inversions = 0
    for lower, upper in zip(ordered_layers, ordered_layers[1:], strict=False):
        for lower_area in by_layer[lower]:
            for upper_area in by_layer[upper]:
                comparisons += 1
                if upper_area > lower_area * 1.05:
                    inversions += 1
    return (inversions / comparisons if comparisons else 0.0), comparisons


def derive_visible_load_evidence(geometry: VisibleLoadGeometry) -> dict[str, Evidence]:
    """Convert visible metric polygons into the four geometry-backed SOP signals.

    The function intentionally emits no height, wrap, or damage evidence. Those
    checks remain ``MANUAL_REVIEW`` until their dedicated measurements/models
    exist. Coverage describes only the visible side and must be propagated into
    the policy rather than interpreted as complete 3D observability.
    """
    pallet = _polygon(geometry.pallet_polygon_xy_m, "pallet polygon")
    load = _polygon(geometry.load_polygon_xy_m, "load polygon")
    coverage = float(np.clip(geometry.visible_coverage, 0.0, 1.0))
    mask_confidence = float(
        np.clip(min(geometry.pallet_mask_confidence, geometry.load_mask_confidence), 0.0, 1.0)
    )
    distances = [
        cv2.pointPolygonTest(pallet, (float(point[0]), float(point[1])), True)
        for point in load
    ]
    overhang_m = max(0.0, max(-float(distance) for distance in distances))
    centroid_offset_m = float(np.linalg.norm(_centroid(load) - _centroid(pallet)))
    standard_deviation = geometry.boundary_std_m
    evidence: dict[str, Evidence] = {
        "overhang_m": Evidence(
            overhang_m,
            mask_confidence,
            coverage=coverage,
            standard_deviation=standard_deviation,
            source="projected pallet/load masks",
            units="m",
            metadata={"scope": "visible load footprint only"},
        ),
        "centroid_offset_m": Evidence(
            centroid_offset_m,
            mask_confidence,
            coverage=coverage,
            standard_deviation=standard_deviation,
            source="projected pallet/load masks",
            units="m",
            metadata={"centroid_type": "geometric, not mass"},
        ),
    }

    rotations: list[float] = []
    rotation_confidences: list[float] = []
    usable_boxes = 0
    for box in geometry.boxes:
        polygon = _polygon(box.polygon_xy_m, "box polygon")
        axis_deg, anisotropy = _principal_axis_deg(polygon)
        if anisotropy < 0.10:
            continue
        usable_boxes += 1
        rotations.append(_axis_misalignment_deg(axis_deg, geometry.pallet_theta_deg))
        rotation_confidences.append(float(np.clip(box.confidence * anisotropy, 0.0, 1.0)))
    if rotations:
        evidence["maximum_box_rotation_deg"] = Evidence(
            max(rotations),
            min(rotation_confidences),
            coverage=coverage * usable_boxes / max(len(geometry.boxes), 1),
            standard_deviation=geometry.orientation_std_deg,
            source="visible projected box masks",
            units="deg",
            metadata={"usable_boxes": usable_boxes, "visible_boxes": len(geometry.boxes)},
        )

    inversion_score, comparisons = _visible_size_inversion_score(geometry.boxes)
    if comparisons:
        box_confidence = min(float(np.clip(box.confidence, 0.0, 1.0)) for box in geometry.boxes)
        evidence["size_inversion_probability"] = Evidence(
            inversion_score,
            box_confidence,
            coverage=coverage,
            source="visible metric box areas and reviewed layer indices",
            metadata={"pairwise_comparisons": comparisons, "score_not_calibrated": True},
        )
    return evidence
