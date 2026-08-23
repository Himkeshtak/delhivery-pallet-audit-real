import hashlib
import re
from pathlib import Path
from urllib.parse import unquote


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_committed_model_weights_match_model_cards() -> None:
    expected = {
        "detect-real-v1.pt": (
            5_427_738,
            "36273b60fb817a0d869188714c90ca8156a3370bb6da1d3536d71f7359cb7662",
        ),
        "segment-real-v1.pt": (
            5_961_764,
            "79aaaa6c147ef5662f624e5db607f25fce0dea209b573f75c8c67554dccdcd0c",
        ),
    }
    for name, (expected_bytes, expected_hash) in expected.items():
        path = Path("weights/releases") / name
        assert path.stat().st_size == expected_bytes
        assert _sha256(path) == expected_hash


def test_readme_local_links_resolve() -> None:
    markdown = Path("README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", markdown)
    local_targets = [target for target in targets if "://" not in target]
    assert local_targets
    for target in local_targets:
        clean = unquote(target.split("#", 1)[0])
        assert (Path(clean)).is_file(), f"README link does not resolve: {target}"
