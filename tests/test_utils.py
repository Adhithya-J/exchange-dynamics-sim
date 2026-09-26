from utils import load_config


def test_load_config_reads_default_yaml():
    config = load_config()

    assert config["ENV_INIT"]["N_AGENTS"] == 10
    assert config["MEMORY"]["MEMORY_SIZE"] == 10
