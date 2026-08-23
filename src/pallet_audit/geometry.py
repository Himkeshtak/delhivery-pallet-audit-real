from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contracts import KeypointObservation, PoseEstimate

FOOTPRINT_KEYPOINTS = (
    "front_left_floor",
    "front_right_floor",
    "rear_left_floor",
    "rear_right_floor",
)


@dataclass(frozen=True)
class PosePolicy:
    minimum_keypoint_confidence: float = 0.45
    assumed_keypoint_sigma_px: float = 2.0
    maximum_reprojection_rmse_px: float = 1.5
    maximum_position_std_m: float = 0.02
    maximum_yaw_std_deg: float = 3.0
    plausible_width_m: tuple[float, float] = (0.70, 1.30)
    plausible_depth_m: tuple[float, float] = (0.70, 1.40)


@dataclass(frozen=True)
class HomographyCalibration:
    calibration_id: str
    image_to_floor: np.ndarray
    reprojection_rmse_px: float
    floor_frame: str
    valid_floor_polygon_m: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.image_to_floor, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("image_to_floor must be a finite 3x3 matrix")
        if abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("image_to_floor must be invertible")
        object.__setattr__(self, "image_to_floor", matrix)

    def project(self, x_px: float, y_px: float) -> np.ndarray:
        homogeneous = self.image_to_floor @ np.array([x_px, y_px, 1.0])
        if abs(homogeneous[2]) < 1e-12:
            raise ValueError("point projects to infinity")
        return homogeneous[:2] / homogeneous[2]

    def jacobian(self, x_px: float, y_px: float) -> np.ndarray:
        h = self.image_to_floor
        denominator = h[2, 0] * x_px + h[2, 1] * y_px + h[2, 2]
        numerator_x = h[0, 0] * x_px + h[0, 1] * y_px + h[0, 2]
        numerator_y = h[1, 0] * x_px + h[1, 1] * y_px + h[1, 2]
        denominator_sq = denominator**2
        return np.array(
            [
                [
                    (h[0, 0] * denominator - numerator_x * h[2, 0]) / denominator_sq,
                    (h[0, 1] * denominator - numerator_x * h[2, 1]) / denominator_sq,
                ],
                [
                    (h[1, 0] * denominator - numerator_y * h[2, 0]) / denominator_sq,
                    (h[1, 1] * denominator - numerator_y * h[2, 1]) / denominator_sq,
                ],
            ]
        )


def _point_in_polygon(point: np.ndarray, polygon: tuple[tuple[float, float], ...]) -> bool:
    if not polygon:
        return True
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            boundary_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < boundary_x:
                inside = not inside
        previous = current
    return inside


def _angle_degrees(vector: np.ndarray) -> float:
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    return (angle + 180.0) % 360.0 - 180.0


def estimate_floor_pose(
    observations: Mapping[str, KeypointObservation],
    calibration: HomographyCalibration,
    policy: PosePolicy | None = None,
) -> PoseEstimate:
    policy = policy or PosePolicy()
    if calibration.reprojection_rmse_px > policy.maximum_reprojection_rmse_px:
        return PoseEstimate.unavailable(
            "calibration_reprojection_error_above_policy",
            f"rmse_px={calibration.reprojection_rmse_px:.3f}",
        )

    missing = [name for name in FOOTPRINT_KEYPOINTS if name not in observations]
    if missing:
        return PoseEstimate.unavailable("missing_footprint_keypoints", *missing)

    weak = [
        name
        for name in FOOTPRINT_KEYPOINTS
        if observations[name].confidence < policy.minimum_keypoint_confidence
    ]
    if weak:
        return PoseEstimate.unavailable("low_keypoint_confidence", *weak)

    floor_points: dict[str, np.ndarray] = {}
    point_covariances: dict[str, np.ndarray] = {}
    for name in FOOTPRINT_KEYPOINTS:
        observation = observations[name]
        try:
            floor_points[name] = calibration.project(observation.x_px, observation.y_px)
        except ValueError:
            return PoseEstimate.unavailable("homography_projection_failed", name)
        sigma_px = policy.assumed_keypoint_sigma_px / max(observation.confidence, 0.05)
        jacobian = calibration.jacobian(observation.x_px, observation.y_px)
        point_covariances[name] = jacobian @ (np.eye(2) * sigma_px**2) @ jacobian.T

    front = (floor_points["front_left_floor"] + floor_points["front_right_floor"]) / 2
    rear = (floor_points["rear_left_floor"] + floor_points["rear_right_floor"]) / 2
    center = np.mean(np.stack(list(floor_points.values())), axis=0)
    front_width = float(
        np.linalg.norm(floor_points["front_right_floor"] - floor_points["front_left_floor"])
    )
    rear_width = float(
        np.linalg.norm(floor_points["rear_right_floor"] - floor_points["rear_left_floor"])
    )
    left_depth = float(
        np.linalg.norm(floor_points["front_left_floor"] - floor_points["rear_left_floor"])
    )
    right_depth = float(
        np.linalg.norm(floor_points["front_right_floor"] - floor_points["rear_right_floor"])
    )
    width = (front_width + rear_width) / 2
    depth = (left_depth + right_depth) / 2
    if not policy.plausible_width_m[0] <= width <= policy.plausible_width_m[1]:
        return PoseEstimate.unavailable("implausible_pallet_width", f"width_m={width:.3f}")
    if not policy.plausible_depth_m[0] <= depth <= policy.plausible_depth_m[1]:
        return PoseEstimate.unavailable("implausible_pallet_depth", f"depth_m={depth:.3f}")
    if not _point_in_polygon(center, calibration.valid_floor_polygon_m):
        return PoseEstimate.unavailable("pose_outside_calibrated_floor_polygon")

    direction = front - rear
    direction_norm_sq = float(direction @ direction)
    if direction_norm_sq < 1e-8:
        return PoseEstimate.unavailable("front_rear_direction_degenerate")

    covariance_center = sum(point_covariances.values()) / 16.0
    covariance_front = (
        point_covariances["front_left_floor"] + point_covariances["front_right_floor"]
    ) / 4.0
    covariance_rear = (
        point_covariances["rear_left_floor"] + point_covariances["rear_right_floor"]
    ) / 4.0
    direction_covariance = covariance_front + covariance_rear
    theta_gradient = np.array([-direction[1], direction[0]]) / direction_norm_sq
    theta_variance_rad = float(theta_gradient @ direction_covariance @ theta_gradient.T)
    theta_std_deg = math.degrees(math.sqrt(max(theta_variance_rad, 0.0)))
    position_std_m = math.sqrt(max(float(np.linalg.eigvalsh(covariance_center).max()), 0.0))

    mean_keypoint_confidence = float(
        np.mean([observations[name].confidence for name in FOOTPRINT_KEYPOINTS])
    )
    geometry_consistency = math.exp(-abs(front_width - rear_width) / max(width, 1e-6))
    calibration_factor = math.exp(-calibration.reprojection_rmse_px / 2.0)
    confidence = float(
        np.clip(mean_keypoint_confidence * geometry_consistency * calibration_factor, 0.0, 1.0)
    )

    reasons: list[str] = []
    available = True
    if position_std_m > policy.maximum_position_std_m:
        available = False
        reasons.append(f"position_std_m={position_std_m:.4f}_above_policy")
    if theta_std_deg > policy.maximum_yaw_std_deg:
        available = False
        reasons.append(f"theta_std_deg={theta_std_deg:.3f}_above_policy")

    return PoseEstimate(
        available=available,
        x_m=float(center[0]) if available else None,
        y_m=float(center[1]) if available else None,
        theta_deg=_angle_degrees(direction) if available else None,
        visible_face="front" if available else None,
        covariance_xy_m2=covariance_center.tolist() if available else None,
        theta_std_deg=theta_std_deg if available else None,
        confidence=confidence if available else 0.0,
        reprojection_rmse_px=calibration.reprojection_rmse_px,
        reasons=reasons,
    )
