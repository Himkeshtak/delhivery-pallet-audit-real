from pallet_audit.evaluation import (
    box_iou_xywh,
    distribution,
    evaluate_coco_predictions,
    evaluate_pose_records,
    wrapped_angle_error_deg,
)


def test_distribution_keeps_raw_values_and_quantiles() -> None:
    report = distribution([1, 2, 3, 4, 5])
    assert report["count"] == 5
    assert report["p50"] == 3
    assert report["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_wrapped_angle_error_handles_boundary() -> None:
    assert wrapped_angle_error_deg(-179, 179) == 2


def test_pose_report_separates_translation_rotation_and_availability() -> None:
    report = evaluate_pose_records(
        [
            {
                "sample_id": "ok",
                "available": "true",
                "gt_x_m": 0,
                "gt_y_m": 0,
                "gt_theta_deg": 179,
                "pred_x_m": 0.01,
                "pred_y_m": 0,
                "pred_theta_deg": -179,
                "range_m": 2,
                "view_yaw_deg": 15,
            },
            {"sample_id": "bad", "available": "false", "failure_reason": "occluded"},
        ]
    )
    assert report["availability_rate"] == 0.5
    assert report["translation_error_m"]["values"] == [0.01]
    assert report["rotation_error_deg"]["values"] == [2.0]
    assert report["joint_2cm_3deg_rate_when_available"] == 1.0


def test_detection_and_localization_are_reported_separately() -> None:
    ground_truth = {
        "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "pallet"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}],
    }
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}]
    report = evaluate_coco_predictions(ground_truth, predictions)
    assert report["detection"]["classes"]["pallet"]["by_iou_threshold"]["0.50"][
        "true_positives"
    ] == 1
    assert report["localization_matched_at_iou_0_50"]["iou"]["p50"] == 1.0
    assert report["localization_matched_at_iou_0_50"]["center_error_px"]["p95"] == 0.0
    assert box_iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]) == 1 / 3

