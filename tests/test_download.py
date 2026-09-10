"""
tests.test_download — Unit tests for the download engines.

Covers engine selection, registry-spec parsing, URL normalization, archive
extraction (including the ``extract: false`` opt-out and the path-traversal
guard), and the idempotency check that skips completed downloads.

Author:
    Luis García Rodríguez
    Doctorado en Ciencia e Ingeniería de la Computación (DCIC)
    IIMAS — Universidad Nacional Autónoma de México (UNAM)
    CVU: 905206 · ORCID: 0009-0004-9514-5508

Project:
    imbdata v0.2.0 — Imbalanced Classification Dataset Repository
    Advisor: Dr. José Antonio Neme Castillo
    Research Group: Anomalocaris
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pytest

from imbdata.download import (
    BaseDownloader,
    DataverseDownloader,
    DownloadManager,
    GitHubDownloader,
    HttpDownloader,
    KaggleDownloader,
    OpenMLDownloader,
)
from imbdata.exceptions import DownloadError


class _RecordingDownloader(BaseDownloader):
    """A downloader that records calls instead of touching the network."""

    SOURCES = ("test",)

    def __init__(self) -> None:
        """Initialize the recorder."""
        super().__init__(progress=False)
        self.calls: list[str] = []

    def _fetch(self, name: str, meta: dict[str, Any], target_dir: Path) -> None:
        """Record the call and write the file the registry declares."""
        self.calls.append(name)
        for filename in self.expected_files(meta) or ["placeholder.txt"]:
            (target_dir / filename).write_text("data", encoding="utf-8")


# ── Engine selection ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("uci", HttpDownloader),
        ("libsvm", HttpDownloader),
        ("github", GitHubDownloader),
        ("harvard_dataverse", DataverseDownloader),
        ("openml", OpenMLDownloader),
        ("kaggle", KaggleDownloader),
    ],
)
def test_manager_selects_the_engine_for_each_source(source: str, expected: type) -> None:
    """Every registry source maps to the engine that serves it."""
    assert isinstance(DownloadManager()._select_engine(source), expected)


def test_manager_rejects_an_unknown_source() -> None:
    """An unserviceable source names the sources that are known."""
    with pytest.raises(DownloadError, match="No download engine"):
        DownloadManager()._select_engine("carrier_pigeon")


# ── Registry-spec parsing ─────────────────────────────────────────────

def test_expected_files_reads_the_download_expect_list() -> None:
    """`download.expect` names the files a successful download must produce."""
    meta = {"download": {"expect": ["a.csv", "b.csv"]}}
    assert BaseDownloader.expected_files(meta) == ["a.csv", "b.csv"]


def test_expected_files_falls_back_to_the_registry_filename() -> None:
    """Entries without a download block fall back to their `filename` field."""
    assert BaseDownloader.expected_files({"filename": "wdbc.data"}) == ["wdbc.data"]


def test_expected_files_ignores_globs() -> None:
    """A glob cannot be checked for existence, so it is not an expected file."""
    assert BaseDownloader.expected_files({"filename": "*.csv"}) == []


def test_expected_files_skips_archives_that_will_be_extracted() -> None:
    """An archive's own name is not expected to survive extraction."""
    meta = {"download": {"files": [{"url": "u", "filename": "x.zip", "extract": True}]}}
    assert BaseDownloader.expected_files(meta) == []


def test_filename_is_derived_from_the_url_when_unspecified() -> None:
    """Percent-encoded URL segments are decoded into the local filename."""
    assert BaseDownloader.filename_from_url("https://h/p/KDDTrain%2B.txt") == "KDDTrain+.txt"


def test_github_blob_urls_are_rewritten_to_raw() -> None:
    """Registry entries may cite either the blob or the raw GitHub URL."""
    assert GitHubDownloader.to_raw_url(
        "https://github.com/o/r/blob/main/a.csv"
    ) == "https://raw.githubusercontent.com/o/r/main/a.csv"


def test_raw_github_urls_are_left_alone() -> None:
    """An already-raw URL passes through unchanged."""
    url = "https://raw.githubusercontent.com/o/r/main/a.csv"
    assert GitHubDownloader.to_raw_url(url) == url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", "mlg-ulb/creditcardfraud"),
        ("https://www.kaggle.com/c/ieee-fraud-detection", "ieee-fraud-detection"),
        ("https://example.org/not-kaggle", None),
    ],
)
def test_kaggle_slug_extraction(url: str, expected: str | None) -> None:
    """Dataset and competition URLs both yield their Kaggle slug."""
    assert KaggleDownloader.slug_from_url(url) == expected


# ── Archive handling ──────────────────────────────────────────────────

def _make_zip(path: Path, members: dict[str, str]) -> Path:
    """Write a zip archive containing the given members.

    Args:
        path: Archive path to create.
        members: Mapping of member name to its text content.

    Returns:
        The archive path.
    """
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)
    return path


def test_extract_expands_a_zip_and_removes_it(tmp_path: Path) -> None:
    """Extraction writes the members and deletes the archive by default."""
    archive = _make_zip(tmp_path / "bundle.zip", {"a.csv": "1", "b.csv": "2"})
    extracted = HttpDownloader(progress=False).extract(archive, tmp_path)
    assert sorted(p.name for p in extracted) == ["a.csv", "b.csv"]
    assert not archive.exists()


def test_extract_can_keep_the_archive(tmp_path: Path) -> None:
    """`keep_archive` leaves the original in place."""
    archive = _make_zip(tmp_path / "bundle.zip", {"a.csv": "1"})
    HttpDownloader(progress=False).extract(archive, tmp_path, keep_archive=True)
    assert archive.exists()


def test_extract_refuses_path_traversal_members(tmp_path: Path) -> None:
    """An archive escaping its target directory is rejected."""
    archive = _make_zip(tmp_path / "evil.zip", {"../escaped.csv": "1"})
    with pytest.raises(DownloadError, match="unsafe member"):
        HttpDownloader(progress=False).extract(archive, tmp_path)


def test_extract_rejects_a_corrupt_archive(tmp_path: Path) -> None:
    """A file that is not a supported archive is reported, not silently kept."""
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not an archive")
    with pytest.raises(DownloadError, match="Unsupported archive"):
        HttpDownloader(progress=False).extract(archive, tmp_path)


def test_is_archive_recognizes_the_supported_suffixes() -> None:
    """Archive detection covers the compressed formats the registry cites."""
    assert HttpDownloader.is_archive(Path("x.tar.gz"))
    assert HttpDownloader.is_archive(Path("x.zip"))
    assert not HttpDownloader.is_archive(Path("x.csv"))


# ── Download flow ─────────────────────────────────────────────────────

def test_download_skips_when_the_expected_files_are_present(tmp_path: Path) -> None:
    """A completed download is not repeated."""
    engine = _RecordingDownloader()
    meta = {"download": {"expect": ["a.csv"]}}
    engine.download("alpha", meta, tmp_path)
    engine.download("alpha", meta, tmp_path)
    assert engine.calls == ["alpha"]


def test_download_finds_expected_files_nested_by_extraction(tmp_path: Path) -> None:
    """Files an archive placed in a subdirectory still satisfy the check."""
    nested = tmp_path / "bundle"
    nested.mkdir()
    (nested / "a.csv").write_text("1", encoding="utf-8")

    engine = _RecordingDownloader()
    engine.download("alpha", {"download": {"expect": ["a.csv"]}}, tmp_path)
    assert engine.calls == []


def test_http_engine_without_file_urls_explains_the_manual_fallback(tmp_path: Path) -> None:
    """An entry with no direct URLs tells the user where to place files by hand."""
    with pytest.raises(DownloadError, match="place the raw files manually"):
        HttpDownloader(progress=False).download("alpha", {"filename": "a.csv"}, tmp_path)


def test_kaggle_engine_without_credentials_or_mirror_raises(tmp_path: Path) -> None:
    """Missing credentials and no mirror is a credentials error with guidance."""
    engine = KaggleDownloader(credentials_path=tmp_path / "absent.json", progress=False)
    if engine.has_credentials():  # the environment may supply KAGGLE_* variables
        pytest.skip("Kaggle credentials are configured in this environment")
    with pytest.raises(DownloadError, match="Create New API Token"):
        engine.download("alpha", {"filename": "a.csv"}, tmp_path)
