import numpy as np

from pallet_audit.contracts import KeypointObservation
from pallet_audit.geometry import HomographyCalibration, PosePolicy, estimate_floor_pose


def _observations() -> dict[str, KeypointObservation]:
    return {
        "front_left_floor": KeypointObservation(0.0, 100.0, 0.95),
        "front_right_floor": KeypointObservation(100.0, 100.0, 0.95),
        "rear_left_floor": KeypointObservation(0.0, 0.0, 0.95),
        "rear_right_floor": KeypointObservation(100.0, 0.0, 0.95),
    }


def test_metric_pose_and_directed_orientation() -> None:
    calibration = HomographyCalibration(
        "identity-scale",
        np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]], dtype=float),
        reprojection_rmse_px=0.2,
        floor_frame="test-floor",
        valid_floor_polygon_m=((-1, -1), (2, -1), (2, 2), (-1, 2)),
    )
    pose = estimate_floor_pose(_observations(), calibration)
    assert pose.available
    assert pose.x_m == 0.5
    assert pose.y_m == 0.5
    assert pose.theta_deg == 90.0
    assert pose.visible_face == "front"
    assert pose.theta_std_deg is not None and pose.theta_std_deg < 3.0


def test_pose_abstains_when_keypoint_is_weak() -> None:
    observations = _observations()
    observations["front_left_floor"] = KeypointObservation(0.0, 100.0, 0.1)
    calibration = HomographyCalibration(
        "identity-scale",
        np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]], dtype=float),
        reprojection_rmse_px=0.2,
        floor_frame="test-floor",
    )
    pose = estimate_floor_pose(observations, calibration)
    assert not pose.available
    assert "low_keypoint_confidence" in pose.reasons


def test_pose_abstains_outside_empirical_uncertainty_bar() -> None:
    calibration = HomographyCalibration(
        "noisy-scale",
        np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]], dtype=float),
        reprojection_rmse_px=0.2,
        floor_frame="test-floor",
    )
    policy = PosePolicy(assumed_keypoint_sigma_px=20.0)
    pose = estimate_floor_pose(_observations(), calibration, policy)
    assert not pose.available
    assert any(reason.startswith("position_std_m") for reason in pose.reasons)

