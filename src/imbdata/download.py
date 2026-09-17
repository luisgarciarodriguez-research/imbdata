"""
imbdata.download — Dataset download engines.

Provides download engines for Kaggle, UCI/HTTP, OpenML, GitHub, and Harvard
Dataverse sources. Each engine implements the :class:`DownloadEngine` protocol
and manages retry logic, progress reporting, archive extraction, and credential
discovery. :class:`DownloadManager` selects the appropriate engine from the
``source`` (or ``download.engine``) field of a dataset's registry entry.

Every engine writes into ``<store>/raw/<dataset_key>/`` and is idempotent: a
dataset whose expected files are already present is not downloaded again.

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

import gzip
import logging
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse

import requests
from tqdm import tqdm

from imbdata import __version__
from imbdata.config import StoreConfig, default_config
from imbdata.exceptions import CredentialsError, DownloadError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0
TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1 << 16  # 64 KiB
USER_AGENT = (
    f"imbdata/{__version__} "
    f"(+https://github.com/luisgarciarodriguez-research/imbdata)"
)

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".gz")

__all__ = [
    "DownloadEngine",
    "BaseDownloader",
    "HttpDownloader",
    "GitHubDownloader",
    "DataverseDownloader",
    "OpenMLDownloader",
    "KaggleDownloader",
    "DownloadManager",
    "download_dataset",
]


@runtime_checkable
class DownloadEngine(Protocol):
    """Structural interface implemented by every download engine."""

    def supports(self, source_type: str) -> bool:
        """Return whether this engine can serve datasets of ``source_type``.

        Args:
            source_type: Value of the registry ``source`` field, or the
                explicit ``download.engine`` override.

        Returns:
            ``True`` when this engine handles that source type.
        """
        ...

    def download(self, name: str, meta: dict[str, Any], target_dir: Path) -> Path:
        """Download a dataset's raw files into ``target_dir``.

        Args:
            name: Dataset key.
            meta: Registry metadata for the dataset.
            target_dir: Destination directory.

        Returns:
            Path to the directory holding the downloaded files.
        """
        ...


class BaseDownloader(ABC):
    """Common machinery shared by every concrete download engine.

    Subclasses declare the registry source types they serve in ``SOURCES`` and
    implement :meth:`_fetch`. The base class handles the idempotency check,
    directory creation, retrying HTTP transfers, progress reporting, and
    archive extraction.

    Attributes:
        max_retries: Number of attempts per file before giving up.
        timeout: Per-request timeout in seconds.
        progress: Whether to render a tqdm progress bar during transfers.

    Example:
        >>> downloader = HttpDownloader()
        >>> downloader.supports("uci")
        True
    """

    SOURCES: tuple[str, ...] = ()

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT_SECONDS,
        progress: bool = True,
    ) -> None:
        """Initialize the engine.

        Args:
            max_retries: Number of attempts per file before raising.
            timeout: Per-request timeout in seconds.
            progress: Whether to display a progress bar.
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.progress = progress
        self._session: requests.Session | None = None

    def __repr__(self) -> str:
        """Return an unambiguous representation of the engine."""
        return f"{type(self).__name__}(max_retries={self.max_retries}, timeout={self.timeout})"

    # ── Protocol surface ───────────────────────────────────────────────

    def supports(self, source_type: str) -> bool:
        """Return whether this engine serves ``source_type``.

        Args:
            source_type: Registry source name (e.g. ``'uci'``, ``'kaggle'``).

        Returns:
            ``True`` when ``source_type`` is declared in ``SOURCES``.
        """
        return source_type in self.SOURCES

    def download(self, name: str, meta: dict[str, Any], target_dir: Path) -> Path:
        """Download a dataset's raw files, skipping work already done.

        Args:
            name: Dataset key.
            meta: Registry metadata for the dataset.
            target_dir: Destination directory; created when absent.

        Returns:
            Path to ``target_dir``.

        Raises:
            DownloadError: If the transfer fails or yields no usable files.
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        expected = self.expected_files(meta)
        if expected and self._has_all(target_dir, expected):
            logger.debug(f"'{name}' already present in {target_dir}; skipping download")
            return target_dir

        logger.info(f"Downloading '{name}' via {type(self).__name__}")
        self._fetch(name, meta, target_dir)

        if not any(target_dir.iterdir()):
            raise DownloadError(f"Download of '{name}' produced no files in {target_dir}")
        missing = [f for f in expected if not self._find(target_dir, f)]
        if missing:
            raise DownloadError(
                f"Download of '{name}' is missing expected file(s): {', '.join(missing)}"
            )
        return target_dir

    @abstractmethod
    def _fetch(self, name: str, meta: dict[str, Any], target_dir: Path) -> None:
        """Perform the source-specific transfer into ``target_dir``.

        Args:
            name: Dataset key.
            meta: Registry metadata for the dataset.
            target_dir: Destination directory, guaranteed to exist.

        Raises:
            DownloadError: If the transfer fails.
        """

    # ── Shared helpers ─────────────────────────────────────────────────

    @staticmethod
    def download_spec(meta: dict[str, Any]) -> dict[str, Any]:
        """Return the ``download`` block of a registry entry.

        Args:
            meta: Registry metadata for a dataset.

        Returns:
            The ``download`` mapping, or an empty mapping when absent.
        """
        spec = meta.get("download") or {}
        return spec if isinstance(spec, dict) else {}

    @classmethod
    def expected_files(cls, meta: dict[str, Any]) -> list[str]:
        """List the filenames a successful download must produce.

        Reads ``download.expect`` when present, then the per-file ``expect``
        and ``filename`` keys, and finally falls back to the top-level
        ``filename`` field of the registry entry.

        Args:
            meta: Registry metadata for a dataset.

        Returns:
            Filenames (without directories); empty when nothing is declared or
            the declaration uses a glob.
        """
        spec = cls.download_spec(meta)
        names: list[str] = []
        names.extend(cls._as_list(spec.get("expect")))
        for item in spec.get("files") or []:
            if isinstance(item, dict):
                names.extend(cls._as_list(item.get("expect")))
                if item.get("filename") and not item.get("extract"):
                    names.append(str(item["filename"]))
        if not names:
            names.extend(cls._as_list(meta.get("filename")))
        return [n for n in dict.fromkeys(names) if "*" not in n]

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        """Normalize a scalar-or-list registry field into a list of strings.

        Args:
            value: A string, a list of strings, or ``None``.

        Returns:
            A list of strings; empty when ``value`` is ``None``.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    @staticmethod
    def _find(directory: Path, filename: str) -> Path | None:
        """Locate ``filename`` directly in ``directory`` or in any subdirectory.

        Args:
            directory: Directory to search.
            filename: Base name to look for.

        Returns:
            The first matching path, or ``None`` when not found.
        """
        direct = directory / filename
        if direct.is_file():
            return direct
        return next(iter(sorted(directory.rglob(filename))), None)

    @classmethod
    def _has_all(cls, directory: Path, filenames: list[str]) -> bool:
        """Return whether every name in ``filenames`` exists under ``directory``."""
        return all(cls._find(directory, name) is not None for name in filenames)

    @property
    def session(self) -> requests.Session:
        """Return the lazily-created HTTP session used for transfers."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": USER_AGENT})
        return self._session

    def close(self) -> None:
        """Release the underlying HTTP session, if one was opened."""
        if self._session is not None:
            self._session.close()
            self._session = None

    @staticmethod
    def filename_from_url(url: str) -> str:
        """Derive a local filename from a URL path.

        Args:
            url: Absolute URL.

        Returns:
            The percent-decoded final path segment, or ``'download'`` when the
            URL has no usable path.
        """
        candidate = unquote(Path(urlparse(url).path).name)
        return candidate or "download"

    def fetch_url(self, url: str, destination: Path, label: str | None = None) -> Path:
        """Download a single URL to ``destination`` with retries and progress.

        The transfer is written to a ``.part`` sibling and renamed on success,
        so an interrupted run never leaves a truncated file in place.

        Args:
            url: Absolute URL to retrieve.
            destination: Local file to write.
            label: Short description shown on the progress bar.

        Returns:
            Path to the written file.

        Raises:
            DownloadError: If every attempt fails.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.part")
        description = label or destination.name
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with self.session.get(url, stream=True, timeout=self.timeout) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("Content-Length") or 0)
                    with open(partial, "wb") as handle:
                        bar = tqdm(
                            total=total or None,
                            unit="B",
                            unit_scale=True,
                            desc=description,
                            disable=not self.progress,
                            leave=False,
                        )
                        with bar:
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                if not chunk:
                                    continue
                                handle.write(chunk)
                                bar.update(len(chunk))
                partial.replace(destination)
                return destination
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                partial.unlink(missing_ok=True)
                if attempt < self.max_retries:
                    delay = BACKOFF_SECONDS * attempt
                    logger.warning(
                        f"Attempt {attempt}/{self.max_retries} for {url} failed ({exc}); "
                        f"retrying in {delay:.0f}s"
                    )
                    time.sleep(delay)

        raise DownloadError(f"Failed to download {url} after {self.max_retries} attempts: "
                            f"{last_error}")

    def fetch_files(self, files: list[dict[str, Any]], target_dir: Path) -> list[Path]:
        """Download every entry of a registry ``download.files`` list.

        Args:
            files: File specifications, each with ``url`` and optionally
                ``filename`` and ``extract``. Archives are expanded unless the
                specification sets ``extract: false``.
            target_dir: Destination directory.

        Returns:
            Paths of the files written (archives are replaced by their
            extracted members).

        Raises:
            DownloadError: If a specification lacks a URL or a transfer fails.
        """
        written: list[Path] = []
        for item in files:
            if not isinstance(item, dict) or not item.get("url"):
                raise DownloadError(f"Malformed download specification: {item!r}")
            url = str(item["url"])
            filename = str(item.get("filename") or self.filename_from_url(url))
            destination = target_dir / filename
            self.fetch_url(url, destination)
            # An explicit `extract: false` opts out; otherwise archives are
            # expanded by default. Large tarballs read lazily by a preprocessor
            # are better left packed.
            extract = item.get("extract")
            if extract is None:
                extract = self.is_archive(destination)
            if extract:
                written.extend(self.extract(destination, target_dir))
            else:
                written.append(destination)
        return written

    @staticmethod
    def is_archive(path: Path) -> bool:
        """Return whether ``path`` looks like a supported archive."""
        return path.name.lower().endswith(ARCHIVE_SUFFIXES)

    def extract(self, archive: Path, target_dir: Path, keep_archive: bool = False) -> list[Path]:
        """Extract a zip, tar, or gzip archive into ``target_dir``.

        Args:
            archive: Archive file to expand.
            target_dir: Directory receiving the members.
            keep_archive: Whether to keep the archive after extraction.

        Returns:
            Paths of the extracted members.

        Raises:
            DownloadError: If the archive is corrupt or of an unknown format.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        name = archive.name.lower()
        try:
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as bundle:
                    members = [m for m in bundle.namelist() if not m.endswith("/")]
                    self._reject_unsafe(members, archive)
                    bundle.extractall(target_dir)
                extracted = [target_dir / m for m in members]
            elif tarfile.is_tarfile(archive):
                with tarfile.open(archive) as bundle:
                    members = [m.name for m in bundle.getmembers() if m.isfile()]
                    self._reject_unsafe(members, archive)
                    bundle.extractall(target_dir, filter="data")
                extracted = [target_dir / m for m in members]
            elif name.endswith(".gz"):
                destination = target_dir / archive.name[: -len(".gz")]
                with gzip.open(archive, "rb") as source, open(destination, "wb") as sink:
                    shutil.copyfileobj(source, sink)
                extracted = [destination]
            else:
                raise DownloadError(f"Unsupported archive format: {archive.name}")
        except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as exc:
            raise DownloadError(f"Cannot extract {archive}: {exc}") from exc

        if not keep_archive:
            archive.unlink(missing_ok=True)
        logger.debug(f"Extracted {len(extracted)} member(s) from {archive.name}")
        return extracted

    @staticmethod
    def _reject_unsafe(members: list[str], archive: Path) -> None:
        """Guard against path traversal in archive member names.

        Args:
            members: Member names declared by the archive.
            archive: The archive being expanded, used for the error message.

        Raises:
            DownloadError: If any member escapes the extraction directory.
        """
        for member in members:
            if member.startswith("/") or ".." in Path(member).parts:
                raise DownloadError(f"Refusing unsafe member '{member}' in {archive.name}")


class HttpDownloader(BaseDownloader):
    """Download engine for plain HTTP/HTTPS sources.

    Serves UCI, LIBSVM, KEEL, PROMISE, UNB/CIC, UNSW, CWRU, and LinkedOmics
    datasets, plus any entry whose ``download.engine`` is ``'http'``. Handles
    retries with exponential backoff, progress reporting, and automatic
    extraction of zip/tar/gzip archives.

    Example:
        >>> engine = HttpDownloader()
        >>> engine.download("spambase", meta, Path("~/.imbdata/raw/spambase"))
    """

    SOURCES = (
        "http",
        "uci",
        "libsvm",
        "keel",
        "promise",
        "unb",
        "cic_unb",
        "unsw",
        "cwru",
        "linkedomics",
    )

    def _fetch(self, name: str, meta: dict[str, Any], target_dir: Path) -> None:
        """Download every file declared in ``download.files``.

        Args:
            name: Dataset key.
            meta: Registry metadata for the dataset.
            target_dir: Destination directory.

        Raises:
            DownloadError: If the entry declares no direct file URLs.
        """
        spec = self.download_spec(meta)
        files = spec.get("files")
        if not files:
            raise DownloadError(
                f"Dataset '{name}' has no 'download.files' entry in datasets.yaml. "
                f"Add direct file URLs, or place the raw files manually in {target_dir}."
            )
        self.fetch_files(list(files), target_dir)


class GitHubDownloader(HttpDownloader):
    """Download engine for files served from GitHub repositories.

    Accepts the same ``download.files`` specification as
    :class:`HttpDownloader` and additionally rewrites ``github.com/.../blob/...``
    URLs into their ``raw.githubusercontent.com`` equivalents, so registry
    entries may cite either form.
    """

    SOURCES = ("github",)

    def _fetch(self, name: str, meta: dict[str, Any], target_dir: Path) -> None:
        """Download the declared GitHub files, normalizing blob URLs first.

        Args:
            name: Dataset key.
            meta: Registry metadata for the dataset.
            target_dir: Destination directory.

        Raises:
            DownloadError: If the entry declares no file URLs.
        """
        spec = self.download_spec(meta)
        files = spec.get("files")
        if not files:
            raise DownloadError(
                f"Dataset '{name}' has no 'download.files' entry in datasets.yaml. "
                f"Add raw GitHub URLs, or place the raw files manually in {target_dir}."
            )
        normalized = [
            {**item, "url": self.to_raw_url(str(item["url"]))} if item.get("url") else item
            for item in files
        ]
        self.fetch_files(normalized, target_dir)

    @staticmethod
    def to_raw_url(url: str) -> str:
        """Rewrite a GitHub web URL into its raw-content equivalent.

        Args:
            url: A ``github.com`` blob URL, or an already-raw URL.

        Returns:
            The ``raw.githubusercontent.com`` URL, or ``url`` unchanged when no
            rewrite applies.

        Example:
            >>> GitHubDownloader.to_raw_url("https://github.com/o/r/blob/main/a.csv")
            'https://raw.githubusercontent.com/o/r/main/a.csv'
        """
        if "github.com" in url and "/blob/" in url:
            return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/", 1)
        return url


class DataverseDownloader(HttpDownloader):
    """Download engine for Harvard Dataverse datasets.

    Resolves a dataset DOI to the Dataverse archival bundle endpoint when the
    registry entry supplies ``download.doi`` instead of explicit file URLs.

    Attributes:
        BASE_URL: Root of the Dataverse installation serving the datasets.
    """

    SOURCES = ("harvard_dataverse", "dataverse")
    BASE_URL = "https://dataverse.harvard.edu"

    def _fetch(self, name: str, meta: dict[str, Any], target_dir: Path) -> None:
        """Download the declared files, or the whole dataset bundle by DOI.

        Args:
            name: Dataset key.
            meta: Registry metadata for the dataset.
            target_dir: Destination directory.

        Raises:
            DownloadError: If neither ``download.files`` nor ``download.doi``
                is declared.
        """
        spec = self.download_spec(meta)
        if spec.get("files"):
            self.fetch_files(list(spec["files"]), target_dir)
            return

        doi = spec.get("doi")
        if not doi:
            raise DownloadError(
                f"Dataset '{name}' needs 'download.files' or 'download.doi' in datasets.yaml."
            )
        url = f"{self.BASE_URL}/api/access/dataset/:persistentId/?persistentId={doi}"
        archive = target_dir / f"{name}_dataverse.zip"
        self.fetch_url(url, archive, label=f"{name} (dataverse)")
        self.extract(archive, target_dir)


class OpenMLDownloader(BaseDownloader):
    """Download engine for OpenML-hosted datasets.

    Prefers the official ``openml`` client when it is installed, and otherwise
    falls back to the public OpenML REST endpoint, which serves the dataset's
    ARFF file over plain HTTP without authentication.

    Attributes:
        API_URL: REST template used by the no-dependency fallback path.

    Example:
        >>> OpenMLDownloader().supports("openml")
        True
    """

    SOURCES = ("openml",)
    API_URL = "https://api.openml.org/data/v1/download/{file_id}"
    DATASET_URL = "https://api.openml.org/api/v1/json/data/{dataset_id}"

    def _fetch(self, name: str, meta: dict[str, Any], target_dir: Path) -> None:
        """Retrieve the dataset's ARFF/CSV payload from OpenML.

        Args:
            name: Dataset key.
            meta: Registry metadata; must carry ``openml_id``.
            target_dir: Destination directory.

        Raises:
            DownloadError: If ``openml_id`` is missing or the transfer fails.
        """
        dataset_id = meta.get("openml_id") or self.download_spec(meta).get("openml_id")
        if dataset_id is None:
            raise DownloadError(f"Dataset '{name}' has no 'openml_id' in datasets.yaml.")

        if self._fetch_with_client(int(dataset_id), name, target_dir):
            return
        self._fetch_with_rest(int(dataset_id), name, target_dir)

    def _fetch_with_client(self, dataset_id: int, name: str, target_dir: Path) -> bool:
        """Try to materialize the dataset through the ``openml`` package.

        Args:
            dataset_id: Numeric OpenML dataset identifier.
            name: Dataset key, used to name the output CSV.
            target_dir: Destination directory.

        Returns:
            ``True`` when the client produced a CSV, ``False`` when the
            ``openml`` package is unavailable.

        Raises:
            DownloadError: If the client is installed but the fetch fails.
        """
        try:
            import openml
        except ImportError:
            logger.debug("openml package unavailable; using the REST fallback")
            return False

        try:
            openml.config.set_root_cache_directory(str(target_dir / ".openml_cache"))
            dataset = openml.datasets.get_dataset(dataset_id)
            frame, *_ = dataset.get_data(dataset_format="dataframe")
        except Exception as exc:  # openml raises a broad family of errors
            raise DownloadError(f"OpenML client failed for dataset {dataset_id}: {exc}") from exc

        destination = target_dir / f"{name}.csv"
        frame.to_csv(destination, index=False)
        logger.debug(f"OpenML client wrote {destination}")
        return True

    def _fetch_with_rest(self, dataset_id: int, name: str, target_dir: Path) -> None:
        """Download the dataset's raw file through the OpenML REST API.

        Args:
            dataset_id: Numeric OpenML dataset identifier.
            name: Dataset key, used to name the output file.
            target_dir: Destination directory.

        Raises:
            DownloadError: If the metadata query or the transfer fails.
        """
        info_url = self.DATASET_URL.format(dataset_id=dataset_id)
        try:
            response = self.session.get(info_url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            file_url = payload["data_set_description"]["url"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise DownloadError(
                f"Cannot resolve OpenML dataset {dataset_id} via {info_url}: {exc}"
            ) from exc

        suffix = Path(urlparse(file_url).path).suffix or ".arff"
        self.fetch_url(file_url, target_dir / f"{name}{suffix}", label=f"{name} (openml)")


class KaggleDownloader(BaseDownloader):
    """Download engine for Kaggle-hosted datasets and competitions.

    Authenticates through ``~/.kaggle/kaggle.json`` (or the ``KAGGLE_USERNAME``
    and ``KAGGLE_KEY`` environment variables) and shells out to the ``kaggle``
    CLI, which handles the API contract and the zip payload. When credentials
    are absent and the registry entry declares a ``download.mirror``, the
    mirror is used instead, so credential-free datasets stay reachable.

    Attributes:
        credentials_path: Path to the Kaggle API credentials file.
        max_retries: Maximum number of download retry attempts.
        timeout: Request timeout in seconds.

    Example:
        >>> downloader = KaggleDownloader()
        >>> downloader.supports("kaggle")
        True
    """

    SOURCES = ("kaggle",)
    CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"

    def __init__(
        self,
        credentials_path: str | Path | None = None,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT_SECONDS,
        progress: bool = True,
    ) -> None:
        """Initialize the engine.

        Args:
            credentials_path: Location of ``kaggle.json``. Defaults to
                ``~/.kaggle/kaggle.json``.
            max_retries: Number of attempts before raising.
            timeout: Per-request timeout in seconds.
            progress: Whether to display a progress bar.
        """
        super().__init__(max_retries=max_retries, timeout=timeout, progress=progress)
        self.credentials_path = (
            Path(credentials_path) if credentials_path is not None else self.CREDENTIALS_PATH
        )

    def has_credentials(self) -> bool:
        """Return whether Kaggle credentials are discoverable.

        Returns:
            ``True`` when ``kaggle.json`` exists or both ``KAGGLE_USERNAME``
            and ``KAGGLE_KEY`` are set in the environment.
        """
        if self.credentials_path.is_file():
            return True
        return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    def _fetch(self, name: str, meta: dict[str, Any], target_dir: Path) -> None:
        """Download a Kaggle dataset, falling back to a declared mirror.

        The mirror declared under ``download.mirror`` covers both failure modes
        of the Kaggle path: absent credentials, and credentials the API rejects
        (expired token, revoked key, changed authentication).

        Args:
            name: Dataset key.
            meta: Registry metadata for the dataset.
            target_dir: Destination directory.

        Raises:
            CredentialsError: If credentials are missing and no mirror exists.
            DownloadError: If the Kaggle transfer fails and no mirror exists.
        """
        spec = self.download_spec(meta)
        mirror = spec.get("mirror")

        if not self.has_credentials():
            if mirror:
                logger.warning(
                    f"No Kaggle credentials at {self.credentials_path}; "
                    f"using the declared mirror for '{name}'."
                )
                self._fetch_mirror(name, mirror, target_dir)
                return
            raise CredentialsError(
                f"Kaggle credentials not found for '{name}'. Create "
                f"{self.credentials_path} (Kaggle → Account → Create New API Token), "
                f"or set KAGGLE_USERNAME and KAGGLE_KEY."
            )

        slug = spec.get("slug") or self.slug_from_url(str(meta.get("url", "")))
        if not slug:
            raise DownloadError(f"Dataset '{name}' has no Kaggle slug in datasets.yaml.")

        competition = bool(spec.get("competition")) or "/c/" in str(meta.get("url", ""))
        try:
            self._run_kaggle_cli(name, slug, target_dir, competition=competition)
        except DownloadError as exc:
            if not mirror:
                raise
            logger.warning(
                f"Kaggle transfer of '{name}' failed ({exc}); "
                f"falling back to the declared mirror."
            )
            self._fetch_mirror(name, mirror, target_dir)
            return
        self._expand_archives(target_dir)

    def _fetch_mirror(self, name: str, mirror: dict[str, Any], target_dir: Path) -> None:
        """Download a dataset from its credential-free mirror.

        Args:
            name: Dataset key.
            mirror: The ``download.mirror`` mapping from the registry.
            target_dir: Destination directory.

        Raises:
            DownloadError: If the mirror declares no files.
        """
        files = mirror.get("files")
        if not files:
            raise DownloadError(f"Mirror for '{name}' declares no files in datasets.yaml.")
        self.fetch_files(list(files), target_dir)

    def _run_kaggle_cli(
        self,
        name: str,
        slug: str,
        target_dir: Path,
        competition: bool = False,
    ) -> None:
        """Invoke the ``kaggle`` CLI to download a dataset or competition.

        Args:
            name: Dataset key, used in error messages.
            slug: Kaggle identifier (``'owner/dataset'`` or a competition name).
            target_dir: Destination directory.
            competition: Whether ``slug`` names a competition rather than a
                public dataset.

        Raises:
            DownloadError: If the CLI is missing, times out, or exits non-zero.
        """
        executable = shutil.which("kaggle")
        if executable is None:
            raise DownloadError(
                f"The 'kaggle' CLI is required to download '{name}'. "
                f"Install it with: pip install 'imbdata[kaggle]'"
            )
        subcommand = ["competitions", "download", "-c", slug] if competition else [
            "datasets", "download", "-d", slug
        ]
        command = [executable, *subcommand, "-p", str(target_dir), "--force"]
        logger.info(f"Running: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout * 30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DownloadError(f"Kaggle download of '{name}' failed: {exc}") from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise DownloadError(f"Kaggle download of '{name}' failed: {detail}")

    def _expand_archives(self, target_dir: Path) -> None:
        """Extract every archive the Kaggle CLI left in ``target_dir``.

        Args:
            target_dir: Directory holding the downloaded payload.
        """
        for archive in sorted(target_dir.glob("*.zip")):
            self.extract(archive, target_dir)

    @staticmethod
    def slug_from_url(url: str) -> str | None:
        """Extract the Kaggle slug from a dataset or competition URL.

        Args:
            url: A ``kaggle.com`` URL.

        Returns:
            ``'owner/dataset'`` for dataset URLs, the competition name for
            competition URLs, or ``None`` when the URL is not recognized.

        Example:
            >>> KaggleDownloader.slug_from_url(
            ...     "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            'mlg-ulb/creditcardfraud'
        """
        parts = [p for p in urlparse(url).path.split("/") if p]
        if "datasets" in parts:
            index = parts.index("datasets")
            owner_and_name = parts[index + 1: index + 3]
            if len(owner_and_name) == 2:
                return "/".join(owner_and_name)
        if "c" in parts:
            index = parts.index("c")
            if len(parts) > index + 1:
                return parts[index + 1]
        return None


class DownloadManager:
    """Routes each dataset to the download engine that serves its source.

    The manager owns one instance of every engine and dispatches on the
    ``download.engine`` override when present, and on the registry ``source``
    field otherwise.

    Attributes:
        config: Store configuration used to resolve raw directories.
        engines: The engine instances available for dispatch.

    Example:
        >>> manager = DownloadManager()
        >>> manager.download("spambase")
        PosixPath('/home/luis/.imbdata/raw/spambase')
    """

    def __init__(
        self,
        config: StoreConfig | None = None,
        engines: list[BaseDownloader] | None = None,
        progress: bool = True,
    ) -> None:
        """Initialize the manager and its engines.

        Args:
            config: Store configuration. Defaults to the process-wide one.
            engines: Engine instances to dispatch across. Defaults to one of
                each built-in engine.
            progress: Whether engines should render progress bars.
        """
        self.config = config if config is not None else default_config()
        self.engines: list[BaseDownloader] = engines if engines is not None else [
            HttpDownloader(progress=progress),
            GitHubDownloader(progress=progress),
            DataverseDownloader(progress=progress),
            OpenMLDownloader(progress=progress),
            KaggleDownloader(progress=progress),
        ]

    def __repr__(self) -> str:
        """Return an unambiguous representation of the manager."""
        names = ", ".join(type(engine).__name__ for engine in self.engines)
        return f"{type(self).__name__}(engines=[{names}])"

    def _select_engine(self, source: str) -> BaseDownloader:
        """Return the engine that serves ``source``.

        Args:
            source: Registry ``source`` value or ``download.engine`` override.

        Returns:
            The first engine whose :meth:`~BaseDownloader.supports` accepts it.

        Raises:
            DownloadError: If no registered engine serves ``source``.
        """
        for engine in self.engines:
            if engine.supports(source):
                return engine
        known = sorted({s for engine in self.engines for s in engine.SOURCES})
        raise DownloadError(
            f"No download engine for source '{source}'. Known sources: {', '.join(known)}"
        )

    def download(
        self,
        name: str,
        meta: dict[str, Any] | None = None,
        target_dir: Path | None = None,
    ) -> Path:
        """Download a dataset's raw files into the data store.

        Args:
            name: Dataset key.
            meta: Registry metadata. Looked up from the registry when omitted.
            target_dir: Destination directory. Defaults to
                ``<store>/raw/<name>``.

        Returns:
            Path to the directory holding the raw files.

        Raises:
            DatasetNotFoundError: If ``name`` is not registered.
            DownloadError: If no engine serves the source or the transfer fails.

        Example:
            >>> DownloadManager().download("ecoli_imu").name
            'ecoli_imu'
        """
        if meta is None:
            from imbdata.registry import get_dataset_meta

            meta = get_dataset_meta(name)
        if target_dir is None:
            target_dir = self.config.raw_dir(name)

        spec = BaseDownloader.download_spec(meta)
        source = str(spec.get("engine") or meta.get("source") or "http")
        engine = self._select_engine(source)
        return engine.download(name, meta, Path(target_dir))

    def close(self) -> None:
        """Release the HTTP sessions held by every engine."""
        for engine in self.engines:
            engine.close()


# ── Functional facade ──────────────────────────────────────────────────

def download_dataset(
    name: str,
    meta: dict[str, Any] | None = None,
    target_dir: Path | None = None,
) -> Path:
    """Download a dataset's raw files using a default :class:`DownloadManager`.

    Args:
        name: Dataset key.
        meta: Registry metadata. Looked up from the registry when omitted.
        target_dir: Destination directory. Defaults to ``<store>/raw/<name>``.

    Returns:
        Path to the directory holding the raw files.

    Raises:
        DownloadError: If the download fails.
    """
    return DownloadManager().download(name, meta, target_dir)
