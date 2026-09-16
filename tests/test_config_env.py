import os

from config import load_env_file


def test_load_env_file_sets_missing_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("HIVER_TEST_ENV_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("HIVER_TEST_ENV_KEY=fromfile\n# comment\nEMPTY=\n", encoding="utf-8")
    load_env_file(path, override=False)
    assert os.environ["HIVER_TEST_ENV_KEY"] == "fromfile"
    assert "EMPTY" not in os.environ or os.environ.get("EMPTY") != ""


def test_load_env_file_does_not_override_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVER_TEST_ENV_KEY", "fromshell")
    path = tmp_path / ".env"
    path.write_text("HIVER_TEST_ENV_KEY=fromfile\n", encoding="utf-8")
    load_env_file(path, override=False)
    assert os.environ["HIVER_TEST_ENV_KEY"] == "fromshell"
