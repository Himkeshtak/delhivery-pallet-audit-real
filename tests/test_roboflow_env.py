from pathlib import Path

from pallet_audit.data.roboflow import _local_env_api_key


def test_private_key_can_be_loaded_from_gitignored_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# private\nROBOFLOW_API_KEY='secret-value'\n", encoding="utf-8")
    assert _local_env_api_key(env_file) == "secret-value"

