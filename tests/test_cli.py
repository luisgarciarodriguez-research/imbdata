"""
tests.test_cli — Unit tests for the ``imbdata`` command-line interface.

Covers argument parsing, subcommand output, and exit codes for
:class:`imbdata.cli.CLI`, using a service bound to a temporary store so the
tests never mutate the user's data store.

Author:
    Luis García Rodríguez
    Doctorado en Ciencia e Ingeniería de la Computación (DCIC)
    IIMAS — Universidad Nacional Autónoma de México (UNAM)
    CVU: 905206 · ORCID: 0009-0004-9514-5508

Project:
    imbdata v0.3.0 — Imbalanced Classification Dataset Repository
    Advisor: Dr. José Antonio Neme Castillo
    Research Group: Anomalocaris
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imbdata import __version__
from imbdata.api import DatasetService
from imbdata.cli import EXIT_ERROR, EXIT_OK, CLI, main
from imbdata.config import StoreConfig
from imbdata.registry import DatasetRegistry


@pytest.fixture()
def cli(temp_store: StoreConfig) -> CLI:
    """Return a CLI bound to an isolated, empty data store.

    Args:
        temp_store: Temporary store configuration.

    Returns:
        A CLI instance whose service writes only inside the temporary store.
    """
    return CLI(service=DatasetService(config=temp_store))


def test_no_command_prints_help(cli: CLI, capsys: pytest.CaptureFixture[str]) -> None:
    """Running with no subcommand shows usage and exits successfully."""
    assert cli.run([]) == EXIT_OK
    assert "usage: imbdata" in capsys.readouterr().out


def test_list_prints_every_dataset(cli: CLI, capsys: pytest.CaptureFixture[str]) -> None:
    """`imbdata list` reports the full registry count."""
    assert cli.run(["list"]) == EXIT_OK
    assert "30 dataset(s)" in capsys.readouterr().out


def test_list_filters_by_domain(cli: CLI, capsys: pytest.CaptureFixture[str]) -> None:
    """`--domain` restricts the listing to that domain."""
    assert cli.run(["list", "--domain", "bioinformatics"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "ecoli_imu" in output and "credit_card_fraud" not in output


def test_list_cached_on_an_empty_store_reports_nothing(
    cli: CLI, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--cached` against an empty store matches no dataset."""
    assert cli.run(["list", "--cached"]) == EXIT_OK
    assert "no datasets match" in capsys.readouterr().out


def test_info_prints_metadata(cli: CLI, capsys: pytest.CaptureFixture[str]) -> None:
    """`imbdata info` prints aligned key/value metadata lines."""
    assert cli.run(["info", "spambase"]) == EXIT_OK
    assert "digital_communications" in capsys.readouterr().out


def test_info_on_unknown_dataset_exits_with_error(
    cli: CLI, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unregistered key is reported on stderr with a non-zero exit code."""
    assert cli.run(["info", "not_a_dataset"]) == EXIT_ERROR
    assert "Unknown dataset" in capsys.readouterr().err


def test_download_without_targets_exits_with_error(
    cli: CLI, capsys: pytest.CaptureFixture[str]
) -> None:
    """`imbdata download` needs keys, --domain, or --all."""
    assert cli.run(["download"]) == EXIT_ERROR
    assert "--all" in capsys.readouterr().err


def test_download_reports_per_dataset_failures(
    temp_store: StoreConfig, synthetic_registry_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unresolvable dataset is reported, not raised.

    Runs against the synthetic registry, whose entries declare no
    `download.files`, so nothing is fetched over the network.
    """
    cli = CLI(service=DatasetService(
        config=temp_store, registry=DatasetRegistry(synthetic_registry_file)
    ))
    assert cli.run(["download", "alpha_set"]) == EXIT_ERROR
    assert "0/1 dataset(s) ready" in capsys.readouterr().out


def test_verify_on_an_empty_store_finds_nothing_to_check(
    cli: CLI, capsys: pytest.CaptureFixture[str]
) -> None:
    """With nothing cached, verification succeeds and reports zero checks."""
    assert cli.run(["verify"]) == EXIT_OK
    assert "0/0 cached dataset(s) OK" in capsys.readouterr().out


def test_status_reports_the_store_location(
    cli: CLI, temp_store: StoreConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """`imbdata status` shows the resolved store path and its counters."""
    assert cli.run(["status"]) == EXIT_OK
    output = capsys.readouterr().out
    assert str(temp_store.store_path()) in output
    assert "registered   : 30 dataset(s)" in output


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    """`--version` reports the package version and exits via SystemExit(0).

    Asserts against ``__version__`` rather than a literal so that a release
    bump does not need this test edited alongside it.
    """
    with pytest.raises(SystemExit) as exit_info:
        CLI.build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert f"imbdata {__version__}" in capsys.readouterr().out


def test_version_is_reported_consistently_everywhere() -> None:
    """The CLI, the User-Agent and the installed metadata agree on one version.

    They used to hardcode the string separately, so a bump could leave the
    HTTP User-Agent claiming a release that no longer matched the package.
    """
    from importlib.metadata import version

    from imbdata.download import USER_AGENT

    assert version("imbdata") == __version__
    assert USER_AGENT.startswith(f"imbdata/{__version__} ")


def test_main_dispatches_to_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """The console-script entry point returns the CLI's exit code."""
    monkeypatch.setattr(CLI, "run", lambda self, argv=None: EXIT_OK)
    assert main(["list"]) == EXIT_OK


def test_human_readable_sizes_use_binary_units() -> None:
    """Byte counts are rendered with binary unit suffixes."""
    assert CLI._human(0) == "0.0 B"
    assert CLI._human(1536) == "1.5 KiB"
    assert CLI._human(5 * 1024 ** 3) == "5.0 GiB"
