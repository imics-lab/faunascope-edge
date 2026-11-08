import pytest

from main import load_config


def test_values_come_from_the_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEVICE_ID", raising=False)
    monkeypatch.delenv("CAMERA_FPS", raising=False)
    (tmp_path / ".env").write_text("# comment\nDEVICE_ID=node_7\nCAMERA_FPS=30\n")
    (tmp_path / "config.yaml").write_text('animal_analyzer:\n  device_id: "${DEVICE_ID}"\n  fps: ${CAMERA_FPS}\n')

    config = load_config("config.yaml")
    assert config["animal_analyzer"] == {"device_id": "node_7", "fps": 30}


def test_the_dotenv_file_overrides_the_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEVICE_ID", "from_shell")
    (tmp_path / ".env").write_text("DEVICE_ID=from_dotenv\n")
    (tmp_path / "config.yaml").write_text("device_id: ${DEVICE_ID}\n")

    assert load_config("config.yaml") == {"device_id": "from_dotenv"}


@pytest.mark.xfail(strict=True, reason="Audit M17: an unset variable is left in the config as literal text.")
def test_an_unset_variable_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SERVER_URL", raising=False)
    (tmp_path / "config.yaml").write_text("server_url: ${SERVER_URL}\n")

    with pytest.raises((SystemExit, KeyError, ValueError)):
        load_config("config.yaml")


def test_the_committed_config_parses(monkeypatch):
    monkeypatch.chdir(__import__("pathlib").Path(__file__).parent.parent)
    config = load_config("config/config.yaml")
    assert {"motion_detector", "animal_analyzer", "uploader"} <= set(config)
