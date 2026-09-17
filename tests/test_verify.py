"""
tests.test_verify — Unit tests for hashing and manifest management.

Covers :func:`imbdata.verify.compute_sha256` and the CRUD plus verification
surface of :class:`imbdata.verify.ManifestManager`.

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

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from imbdata.config import StoreConfig
from imbdata.exceptions import IntegrityError
from imbdata.verify import (
    STATUS_MISMATCH,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_UNKNOWN,
    ManifestManager,
    compute_sha256,
    manifest_key,
)


def _write_parquet(config: StoreConfig, name: str, frame: pd.DataFrame) -> Path:
    """Materialize a canonical frame at the store's processed path.

    Args:
        config: Store configuration naming the destination.
        name: Dataset key.
        frame: Frame to serialize.

    Returns:
        Path to the written parquet file.
    """
    path = config.processed_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, compression="snappy", index=False)
    return path


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    """The chunked digest equals hashlib's digest of the whole payload."""
    payload = b"imbdata" * 5000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert compute_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_compute_sha256_is_stable_across_chunk_sizes(tmp_path: Path) -> None:
    """The digest does not depend on the read chunk size."""
    path = tmp_path / "blob.bin"
    path.write_bytes(b"0123456789" * 999)
    assert compute_sha256(path, chunk_size=7) == compute_sha256(path, chunk_size=1 << 20)


def test_compute_sha256_on_missing_file_raises_integrity_error(tmp_path: Path) -> None:
    """Hashing an absent file raises IntegrityError, not OSError."""
    with pytest.raises(IntegrityError, match="Cannot hash"):
        compute_sha256(tmp_path / "absent.bin")


def test_manifest_key_appends_the_variant() -> None:
    """Variants get their own manifest key so they never overwrite the default."""
    assert manifest_key("tcga_brca") == "tcga_brca"
    assert manifest_key("tcga_brca", "full") == "tcga_brca__full"


def test_update_records_hash_size_and_timestamp(
    temp_store: StoreConfig, canonical_frame: pd.DataFrame
) -> None:
    """update() stores the digest, the byte size, and an ISO-8601 timestamp."""
    path = _write_parquet(temp_store, "alpha", canonical_frame)
    entry = ManifestManager(temp_store).update("alpha", path)

    assert entry["sha256_processed"] == compute_sha256(path)
    assert entry["size_bytes"] == path.stat().st_size
    assert entry["updated_utc"].endswith("+00:00")


def test_update_accepts_extra_fields(
    temp_store: StoreConfig, canonical_frame: pd.DataFrame
) -> None:
    """Caller-supplied statistics are stored verbatim in the entry."""
    path = _write_parquet(temp_store, "alpha", canonical_frame)
    entry = ManifestManager(temp_store).update("alpha", path, n_rows=6, domain="medicine")
    assert entry["n_rows"] == 6 and entry["domain"] == "medicine"


def test_update_on_missing_file_raises_integrity_error(temp_store: StoreConfig) -> None:
    """Recording a dataset that was never written raises IntegrityError."""
    with pytest.raises(IntegrityError, match="no processed file"):
        ManifestManager(temp_store).update("ghost", temp_store.processed_path("ghost"))


def test_entry_returns_none_for_unrecorded_datasets(temp_store: StoreConfig) -> None:
    """Querying an absent entry yields None rather than raising."""
    assert ManifestManager(temp_store).entry("ghost") is None


def test_remove_deletes_the_entry_and_reports_it(
    temp_store: StoreConfig, canonical_frame: pd.DataFrame
) -> None:
    """remove() returns True the first time and False afterwards."""
    manager = ManifestManager(temp_store)
    manager.update("alpha", _write_parquet(temp_store, "alpha", canonical_frame))
    assert manager.remove("alpha") is True
    assert manager.remove("alpha") is False


def test_verify_returns_ok_for_an_untouched_file(
    temp_store: StoreConfig, canonical_frame: pd.DataFrame
) -> None:
    """A file that has not changed since recording verifies as OK."""
    manager = ManifestManager(temp_store)
    manager.update("alpha", _write_parquet(temp_store, "alpha", canonical_frame))
    assert manager.verify("alpha") == STATUS_OK


def test_verify_detects_a_modified_file(
    temp_store: StoreConfig, canonical_frame: pd.DataFrame
) -> None:
    """Rewriting the parquet with different content is reported as a mismatch."""
    manager = ManifestManager(temp_store)
    manager.update("alpha", _write_parquet(temp_store, "alpha", canonical_frame))
    _write_parquet(temp_store, "alpha", canonical_frame.iloc[:4])
    assert manager.verify("alpha") == STATUS_MISMATCH


def test_verify_reports_missing_when_nothing_is_cached(temp_store: StoreConfig) -> None:
    """A dataset with no processed file is MISSING, not an error."""
    assert ManifestManager(temp_store).verify("alpha") == STATUS_MISSING


def test_verify_reports_unknown_for_unrecorded_cached_files(
    temp_store: StoreConfig, canonical_frame: pd.DataFrame
) -> None:
    """A cached file that was never recorded is NOT_IN_MANIFEST."""
    _write_parquet(temp_store, "alpha", canonical_frame)
    assert ManifestManager(temp_store).verify("alpha") == STATUS_UNKNOWN


def test_verify_all_covers_every_requested_key(
    temp_store: StoreConfig, canonical_frame: pd.DataFrame
) -> None:
    """verify_all() reports one status per requested dataset."""
    manager = ManifestManager(temp_store)
    manager.update("alpha", _write_parquet(temp_store, "alpha", canonical_frame))
    assert manager.verify_all(["alpha", "beta"]) == {
        "alpha": STATUS_OK,
        "beta": STATUS_MISSING,
    }


def test_require_raises_when_verification_fails(temp_store: StoreConfig) -> None:
    """require() converts any non-OK status into an IntegrityError."""
    with pytest.raises(IntegrityError, match="MISSING"):
        ManifestManager(temp_store).require("alpha")
