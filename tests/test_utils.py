import pytest
from utils import load_config


def test_load_config_reads_default_yaml():
    config = load_config()

    assert config["SIMULATION"]["AGENTS"] == 10
    assert config["MEMORY"]["SIZE"] == 10
    assert config["MEMORY"]["PREFERENCE_FRACTION"] == 0.5


def test_load_config_custom_path(tmp_path):
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text("SIMULATION:\n  AGENTS: 5\n", encoding="utf-8")

    config = load_config(custom_yaml)
    assert config["SIMULATION"]["AGENTS"] == 5


def test_load_config_nonexistent_path(tmp_path):
    missing_yaml = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing_yaml)
