from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"count": 0, "values": []}
    if not np.isfinite(array).all():
        raise ValueError("distribution values must be finite")
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def wrapped_angle_error_deg(predicted: float, truth: float) -> float:
    return abs((predicted - truth + 180.0) % 360.0 - 180.0)


def evaluate_pose_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    successful_bar: list[bool] = []
    evaluated: list[dict[str, Any]] = []
    unavailable_reasons: defaultdict[str, int] = defaultdict(int)

    for row in rows:
        available = str(row.get("available", "true")).lower() in {"1", "true", "yes"}
        if not available:
            unavailable_reasons[str(row.get("failure_reason", "unspecified"))] += 1
            continue
        required = ("gt_x_m", "gt_y_m", "gt_theta_deg", "pred_x_m", "pred_y_m", "pred_theta_deg")
        missing = [key for key in required if row.get(key) in {None, ""}]
        if missing:
            raise ValueError(f"available pose record is missing: {missing}")
        translation = math.hypot(
            float(row["pred_x_m"]) - float(row["gt_x_m"]),
            float(row["pred_y_m"]) - float(row["gt_y_m"]),
        )
        rotation = wrapped_angle_error_deg(
            float(row["pred_theta_deg"]), float(row["gt_theta_deg"])
        )
        meets_bar = translation <= 0.02 and rotation <= 3.0
        translation_errors.append(translation)
        rotation_errors.append(rotation)
        successful_bar.append(meets_bar)
        evaluated.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "translation_error_m": translation,
                "rotation_error_deg": rotation,
                "meets_2cm_3deg": meets_bar,
                "range_m": float(row["range_m"]) if row.get("range_m") not in {None, ""} else None,
                "view_yaw_deg": (
                    float(row["view_yaw_deg"])
                    if row.get("view_yaw_deg") not in {None, ""}
                    else None
                ),
            }
        )

    worst = sorted(
        evaluated,
        key=lambda item: item["translation_error_m"] / 0.02 + item["rotation_error_deg"] / 3.0,
        reverse=True,
    )[:3]
    return {
        "evaluation_schema_version": 1,
        "total_samples": len(rows),
        "available_samples": len(evaluated),
        "availability_rate": len(evaluated) / len(rows) if rows else 0.0,
        "unavailable_reasons": dict(sorted(unavailable_reasons.items())),
        "translation_error_m": distribution(translation_errors),
        "rotation_error_deg": distribution(rotation_errors),
        "joint_2cm_3deg_rate_when_available": (
            float(np.mean(successful_bar)) if successful_bar else 0.0
        ),
        "worst_cases": worst,
        "samples": evaluated,
    }


def box_iou_xywh(left: Iterable[float], right: Iterable[float]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    intersection_width = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_height = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_width * intersection_height
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        precision[recall >= point].max() if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    return float(np.mean(interpolated))


def evaluate_coco_predictions(
    ground_truth: Mapping[str, Any],
    predictions: Iterable[Mapping[str, Any]],
    iou_thresholds: Iterable[float] = tuple(np.arange(0.5, 1.0, 0.05)),
) -> dict[str, Any]:
    category_names = {int(item["id"]): str(item["name"]) for item in ground_truth["categories"]}
    annotations_by_category: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_category[int(annotation["category_id"])].append(dict(annotation))
    predictions_by_category: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        category_id = int(prediction["category_id"])
        if category_id not in category_names:
            raise ValueError(f"prediction has unknown category_id={category_id}")
        predictions_by_category[category_id].append(dict(prediction))

    thresholds = [round(float(value), 2) for value in iou_thresholds]
    class_reports: dict[str, Any] = {}
    all_aps: list[float] = []
    localization_ious: list[float] = []
    localization_center_px: list[float] = []
    localization_center_normalized: list[float] = []

    for category_id, category_name in sorted(category_names.items()):
        ground = annotations_by_category[category_id]
        predicted = sorted(
            predictions_by_category[category_id],
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        threshold_reports: dict[str, Any] = {}
        for threshold in thresholds:
            matched: set[int] = set()
            true_positive: list[float] = []
            false_positive: list[float] = []
            for prediction in predicted:
                candidate_indexes = [
                    index
                    for index, annotation in enumerate(ground)
                    if index not in matched
                    and int(annotation["image_id"]) == int(prediction["image_id"])
                ]
                overlaps = [
                    box_iou_xywh(prediction["bbox"], ground[index]["bbox"])
                    for index in candidate_indexes
                ]
                best_position = int(np.argmax(overlaps)) if overlaps else -1
                best_iou = overlaps[best_position] if overlaps else 0.0
                if best_iou >= threshold:
                    matched_index = candidate_indexes[best_position]
                    matched.add(matched_index)
                    true_positive.append(1.0)
                    false_positive.append(0.0)
                    if threshold == 0.5:
                        gt_box = ground[matched_index]["bbox"]
                        pred_box = prediction["bbox"]
                        gt_center = np.array(
                            [gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3] / 2],
                            dtype=float,
                        )
                        pred_center = np.array(
                            [pred_box[0] + pred_box[2] / 2, pred_box[1] + pred_box[3] / 2],
                            dtype=float,
                        )
                        center_error = float(np.linalg.norm(pred_center - gt_center))
                        diagonal = math.hypot(float(gt_box[2]), float(gt_box[3]))
                        localization_ious.append(best_iou)
                        localization_center_px.append(center_error)
                        localization_center_normalized.append(center_error / max(diagonal, 1e-9))
                else:
                    true_positive.append(0.0)
                    false_positive.append(1.0)

            tp_cumulative = np.cumsum(true_positive)
            fp_cumulative = np.cumsum(false_positive)
            recall = tp_cumulative / max(len(ground), 1)
            precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-9)
            average_precision = _average_precision(recall, precision) if ground else 0.0
            all_aps.append(average_precision)
            threshold_reports[f"{threshold:.2f}"] = {
                "average_precision": average_precision,
                "true_positives": int(sum(true_positive)),
                "false_positives": int(sum(false_positive)),
                "false_negatives": len(ground) - int(sum(true_positive)),
                "precision_curve": precision.tolist(),
                "recall_curve": recall.tolist(),
            }
        class_reports[category_name] = {
            "ground_truth_instances": len(ground),
            "predictions": len(predicted),
            "by_iou_threshold": threshold_reports,
        }

    return {
        "evaluation_schema_version": 1,
        "detection": {
            "iou_thresholds": thresholds,
            "mean_ap_50_95": float(np.mean(all_aps)) if all_aps else 0.0,
            "classes": class_reports,
        },
        "localization_matched_at_iou_0_50": {
            "iou": distribution(localization_ious),
            "center_error_px": distribution(localization_center_px),
            "center_error_fraction_of_gt_diagonal": distribution(localization_center_normalized),
        },
    }
