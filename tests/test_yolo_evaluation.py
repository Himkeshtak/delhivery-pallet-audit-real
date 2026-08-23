from pallet_audit.yolo_evaluation import greedy_match, localization_report


def test_greedy_match_reports_localization_and_failures() -> None:
    truth = [{"class_id": 1, "box_xyxy": [0, 0, 10, 10]}]
    predictions = [
        {"class_id": 1, "confidence": 0.9, "box_xyxy": [1, 1, 11, 11]},
        {"class_id": 0, "confidence": 0.8, "box_xyxy": [0, 0, 10, 10]},
    ]
    result = greedy_match(truth, predictions)
    assert len(result["matches"]) == 1
    assert len(result["false_positive_indexes"]) == 1
    assert result["false_negative_indexes"] == []


def test_localization_report_ranks_worst_case() -> None:
    samples = [
        {
            "image": "good.jpg",
            "ground_truth": [{"class_id": 0, "box_xyxy": [0, 0, 10, 10]}],
            "predictions": [
                {"class_id": 0, "confidence": 0.9, "box_xyxy": [0, 0, 10, 10]}
            ],
        },
        {
            "image": "bad.jpg",
            "ground_truth": [{"class_id": 0, "box_xyxy": [0, 0, 10, 10]}],
            "predictions": [],
        },
    ]
    report = localization_report(samples)
    assert report["matched_instances"] == 1
    assert report["worst_cases"][0]["image"] == "bad.jpg"
