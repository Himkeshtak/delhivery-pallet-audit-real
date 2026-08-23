from pathlib import Path

import pytest

from tools.train_anomaly import validate_folder_layout


def test_anomaly_layout_requires_real_bad_holdout(tmp_path: Path) -> None:
    (tmp_path / "train" / "good").mkdir(parents=True)
    (tmp_path / "test" / "good").mkdir(parents=True)
    with pytest.raises(ValueError, match="abnormal_test"):
        validate_folder_layout(tmp_path)

