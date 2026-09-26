from utils import load_config


def test_load_config_reads_default_yaml():
    config = load_config()

    assert config["SIMULATION"]["AGENTS"] == 10
    assert config["MEMORY"]["SIZE"] == 10
    assert config["MEMORY"]["PREFERENCE_FRACTION"] == 0.5
