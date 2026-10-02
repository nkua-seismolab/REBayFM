"""Tests for rebayfm.config."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rebayfm import __main__ as cli
from rebayfm.config import ConfigError, load_config

VALID_MINIMUM = "seiscomp: {}\n"


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "text",
    [
        "false",
        "[]",
        "seiscomp: false",
        "seiscomp: []",
        "seiscomp: null",
        "seiscomp: {wait_time: true}",
        "seiscomp: {wait_time: -1}",
        "seiscomp: {reprocess: 'false'}",
        "logging: {level: nonsense}",
        "logging: [",
    ],
)
def test_invalid_config_rejected(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_informational_cli_needs_no_config(monkeypatch, option):
    monkeypatch.setattr(cli.sys, "argv", ["rebayfm", option, "--config", "/missing.yaml"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0


@pytest.mark.parametrize("args", [["--wait-time", "-1"], ["-w-1"], ["--config"]])
def test_invalid_cli_rejected(args):
    with pytest.raises(SystemExit) as result:
        cli._config_path_from_argv(["rebayfm", *args])
    assert result.value.code == 2


def test_cli_precedence(tmp_path):
    config = load_config(write(tmp_path, "seiscomp: {host: yaml-host, database: yaml-db}"))
    for arguments in (
        ["-H", "cli-host", "-d", "cli-db"],
        ["-Hcli-host", "-dcli-db"],
        ["--host=cli-host", "--database=cli-db"],
    ):
        argv = ["rebayfm", *arguments]
        assert cli._inject_connection_args(argv, config) == argv


def test_worker_failure_stops_before_listener(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cli.sys, "argv", ["rebayfm", "--config", str(write(tmp_path, ""))])
    monkeypatch.setattr(cli.faulthandler, "register", Mock())
    executor = Mock()
    executor.submit.return_value.result.side_effect = RuntimeError("failed")
    monkeypatch.setattr(cli, "ProcessPoolExecutor", Mock(return_value=executor))
    listener = Mock()
    monkeypatch.setitem(
        cli.sys.modules, "rebayfm.app", SimpleNamespace(make_dispatch=Mock(), make_finalize=Mock())
    )
    monkeypatch.setitem(
        cli.sys.modules, "rebayfm.worker", SimpleNamespace(init_worker=Mock(), run_batch=Mock())
    )
    monkeypatch.setitem(
        cli.sys.modules, "rebayfm.scclient.listener", SimpleNamespace(EventListenerApp=listener)
    )
    assert cli.main() == 1
    listener.assert_not_called()
    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
    assert "Worker initialization failed" in caplog.text
    assert "Worker process ready" not in caplog.text


def test_defaults(tmp_path):
    config = load_config(write(tmp_path, VALID_MINIMUM))
    assert config.seiscomp.wait_time == 300
    assert config.logging.level == "DEBUG"
    assert config.bayfm.min_polarities == 8
    assert config.bayfm.use_spol is True
    assert config.bayfm.use_nulls is False
    assert config.bayfm.spol_sigma_deg == 15.0
    assert config.bayfm.f_spol == 3
    assert config.bayfm.hpd_level == 0.68
    assert config.output.directory == "output"
    assert config.output.save_figures is True


def test_partial_override(tmp_path):
    config = load_config(
        write(tmp_path, "seiscomp:\n  wait_time: 60\nbayfm:\n  min_polarities: 10\n")
    )
    assert config.seiscomp.wait_time == 60
    assert config.bayfm.min_polarities == 10
    # untouched sections keep defaults
    assert config.bayfm.w_spol == 0.2


def test_empty_config_is_valid(tmp_path):
    assert load_config(write(tmp_path, "")).bayfm.min_polarities == 8


@pytest.mark.parametrize(
    "key", ["bogus", "subscriptions", "pick_group", "focmech_group", "load_inventory"]
)
def test_unknown_key_rejected(tmp_path, key):
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(write(tmp_path, f"seiscomp:\n  {key}: 1\n"))


def test_unknown_section_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown top-level"):
        load_config(write(tmp_path, "bogus:\n  a: 1\n"))


def test_empty_velocity_model_rejected(tmp_path):
    with pytest.raises(ConfigError, match="path"):
        load_config(write(tmp_path, VALID_MINIMUM + "velocity_model:\n  path: ''\n"))


def test_invalid_f_spol_rejected(tmp_path):
    with pytest.raises(ConfigError, match="f_spol"):
        load_config(write(tmp_path, VALID_MINIMUM + "bayfm:\n  f_spol: 4\n"))


def test_invalid_hpd_level_rejected(tmp_path):
    with pytest.raises(ConfigError, match="hpd_level"):
        load_config(write(tmp_path, VALID_MINIMUM + "bayfm:\n  hpd_level: 1.5\n"))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_repo_config_is_valid():
    config = load_config("config.example.yaml")
    assert config.logging.level == "DEBUG"
