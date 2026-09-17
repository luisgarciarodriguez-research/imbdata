"""
tests.test_config — Unit tests for :class:`imbdata.config.StoreConfig`.

Covers store path resolution priority, directory creation, per-dataset path
construction (including variants), and atomic JSON document handling.

Author:
    Luis García Rodríguez
    Doctorado en Ciencia e Ingeniería de la Computación (DCIC)
    IIMAS — Universidad Nacional Autónoma de México (UNAM)
    CVU: 905206 · ORCID: 0009-0004-9514-5508

Project:
    imbdata v0.3.1 — Imbalanced Classification Dataset Repository
    Advisor: Dr. José Antonio Neme Castillo
    Research Group: Anomalocaris
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imbdata.config import DEFAULT_STORE_PATH, STORE_ENV_VAR, StoreConfig


def test_explicit_path_takes_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A constructor argument outranks the environment variable."""
    monkeypatch.setenv(STORE_ENV_VAR, str(tmp_path / "from_env"))
    config = StoreConfig(store_path=tmp_path / "explicit")
    assert config.store_path() == tmp_path / "explicit"


def test_environment_variable_is_used_when_no_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IMBDATA_STORE relocates the store when no explicit path is given."""
    monkeypatch.setenv(STORE_ENV_VAR, str(tmp_path / "from_env"))
    assert StoreConfig().store_path() == tmp_path / "from_env"


def test_default_path_is_used_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without overrides the store resolves to ~/.imbdata."""
    monkeypatch.delenv(STORE_ENV_VAR, raising=False)
    monkeypatch.setattr(StoreConfig, "DEFAULT_STORE_PATH", DEFAULT_STORE_PATH)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    assert StoreConfig().store_path() == DEFAULT_STORE_PATH


def test_user_home_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A '~'-prefixed store path is expanded to an absolute path."""
    monkeypatch.setenv(STORE_ENV_VAR, "~/imbdata_expanded")
    assert StoreConfig().store_path() == Path.home() / "imbdata_expanded"


def test_ensure_dirs_creates_the_store_tree(tmp_path: Path) -> None:
    """ensure_dirs() creates root, raw/, and processed/ and returns them."""
    dirs = StoreConfig(store_path=tmp_path / "store").ensure_dirs()
    assert set(dirs) == {"root", "raw", "processed"}
    assert all(path.is_dir() for path in dirs.values())


def test_ensure_dirs_is_idempotent(temp_store: StoreConfig) -> None:
    """Calling ensure_dirs() twice does not fail on existing directories."""
    first = temp_store.ensure_dirs()
    assert first == temp_store.ensure_dirs()


def test_raw_dir_creates_a_per_dataset_directory(temp_store: StoreConfig) -> None:
    """raw_dir() returns <store>/raw/<key> and creates it by default."""
    path = temp_store.raw_dir("spambase")
    assert path == temp_store.store_path() / "raw" / "spambase"
    assert path.is_dir()


def test_raw_dir_can_skip_creation(temp_store: StoreConfig) -> None:
    """raw_dir(create=False) resolves the path without touching the disk."""
    path = temp_store.raw_dir("not_yet", create=False)
    assert not path.exists()


def test_processed_path_names_the_canonical_parquet(temp_store: StoreConfig) -> None:
    """The default variant is stored as <key>.parquet."""
    assert temp_store.processed_path("spambase").name == "spambase.parquet"


def test_processed_path_separates_variants(temp_store: StoreConfig) -> None:
    """Variants live beside the default file as <key>__<variant>.parquet."""
    assert temp_store.processed_path("tcga_brca", "full").name == "tcga_brca__full.parquet"


def test_manifest_round_trips_through_json(temp_store: StoreConfig) -> None:
    """A written manifest is read back unchanged."""
    payload = {"spambase": {"sha256_processed": "abc", "size_bytes": 7}}
    temp_store.write_manifest(payload)
    assert temp_store.read_manifest() == payload


def test_reading_an_absent_manifest_returns_empty(temp_store: StoreConfig) -> None:
    """A store without a manifest reads as an empty mapping, not an error."""
    assert temp_store.read_manifest() == {}


def test_reading_a_corrupt_manifest_returns_empty(temp_store: StoreConfig) -> None:
    """Invalid JSON degrades to an empty mapping with a warning."""
    temp_store.manifest_path().write_text("{not json", encoding="utf-8")
    assert temp_store.read_manifest() == {}


def test_write_leaves_no_temporary_file_behind(temp_store: StoreConfig) -> None:
    """The atomic write renames its scratch file rather than leaving it."""
    temp_store.write_manifest({"a": 1})
    assert not list(temp_store.store_path().glob("*.tmp"))


def test_registry_path_points_at_the_bundled_yaml() -> None:
    """The bundled datasets.yaml ships inside the installed package."""
    path = StoreConfig.registry_path()
    assert path.name == "datasets.yaml" and path.is_file()
