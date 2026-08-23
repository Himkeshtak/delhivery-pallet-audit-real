from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .evaluation import distribution


def box_iou_xyxy(left: Iterable[float], right: Iterable[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(
    ground_truth: list[Mapping[str, Any]],
    predictions: list[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    matched_ground: set[int] = set()
    matches: list[dict[str, Any]] = []
    false_positive_indexes: list[int] = []
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: float(predictions[index].get("confidence", 0.0)),
        reverse=True,
    )
    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]
        candidates = [
            index
            for index, truth in enumerate(ground_truth)
            if index not in matched_ground and int(truth["class_id"]) == int(prediction["class_id"])
        ]
        overlaps = [
            box_iou_xyxy(prediction["box_xyxy"], ground_truth[index]["box_xyxy"])
            for index in candidates
        ]
        if not overlaps or max(overlaps) < iou_threshold:
            false_positive_indexes.append(prediction_index)
            continue
        best_position = max(range(len(overlaps)), key=overlaps.__getitem__)
        ground_index = candidates[best_position]
        matched_ground.add(ground_index)
        gt_box = [float(value) for value in ground_truth[ground_index]["box_xyxy"]]
        pred_box = [float(value) for value in prediction["box_xyxy"]]
        gt_center = ((gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2)
        pred_center = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
        center_error = math.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1])
        gt_diagonal = math.hypot(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1])
        matches.append(
            {
                "ground_truth_index": ground_index,
                "prediction_index": prediction_index,
                "class_id": int(prediction["class_id"]),
                "iou": overlaps[best_position],
                "center_error_px": center_error,
                "center_error_fraction_of_gt_diagonal": center_error
                / max(gt_diagonal, 1e-9),
            }
        )
    return {
        "matches": matches,
        "false_positive_indexes": false_positive_indexes,
        "false_negative_indexes": [
            index for index in range(len(ground_truth)) if index not in matched_ground
        ],
    }


def localization_report(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    overall_iou: list[float] = []
    overall_center: list[float] = []
    overall_normalized_center: list[float] = []
    per_class: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"iou": [], "center": [], "normalized_center": []}
    )
    for sample in samples:
        truth = list(sample["ground_truth"])
        predictions = list(sample["predictions"])
        matching = greedy_match(truth, predictions)
        for match in matching["matches"]:
            iou = float(match["iou"])
            center = float(match["center_error_px"])
            normalized = float(match["center_error_fraction_of_gt_diagonal"])
            overall_iou.append(iou)
            overall_center.append(center)
            overall_normalized_center.append(normalized)
            values = per_class[int(match["class_id"])]
            values["iou"].append(iou)
            values["center"].append(center)
            values["normalized_center"].append(normalized)
        false_positives = len(matching["false_positive_indexes"])
        false_negatives = len(matching["false_negative_indexes"])
        median_iou = (
            sorted(float(item["iou"]) for item in matching["matches"])[
                len(matching["matches"]) // 2
            ]
            if matching["matches"]
            else 0.0
        )
        denominator = max(len(truth), 1)
        severity = (false_positives + false_negatives) / denominator + (1.0 - median_iou)
        if false_negatives > false_positives:
            hypothesis = "missed instances; inspect scale, occlusion, and domain shift"
        elif false_positives > false_negatives:
            hypothesis = "false positives; inspect pallet-like background structure"
        else:
            hypothesis = "localization error or balanced false-positive/false-negative failure"
        evaluated.append(
            {
                "image": str(sample["image"]),
                "ground_truth_instances": len(truth),
                "predictions": len(predictions),
                "matches_at_iou_0_50": len(matching["matches"]),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "matched_median_iou": median_iou,
                "severity": severity,
                "automated_root_cause_hypothesis": hypothesis,
            }
        )
    worst = sorted(evaluated, key=lambda item: item["severity"], reverse=True)[:3]
    return {
        "matching_iou_threshold": 0.5,
        "matched_instances": len(overall_iou),
        "iou": distribution(overall_iou),
        "center_error_px": distribution(overall_center),
        "center_error_fraction_of_gt_diagonal": distribution(overall_normalized_center),
        "per_class": {
            str(class_id): {
                "iou": distribution(values["iou"]),
                "center_error_px": distribution(values["center"]),
                "center_error_fraction_of_gt_diagonal": distribution(
                    values["normalized_center"]
                ),
            }
            for class_id, values in sorted(per_class.items())
        },
        "worst_cases": worst,
        "samples": evaluated,
    }
