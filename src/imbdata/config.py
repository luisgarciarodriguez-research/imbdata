"""
imbdata.config — Data store path resolution and configuration management.

Encapsulates every filesystem location used by the package in the
:class:`StoreConfig` class. Module-level convenience functions delegate to a
lazily-created default instance so that simple consumers never need to build a
configuration object explicitly.

Data store layout::

    ~/.imbdata/
    ├── config.json      # User configuration overrides
    ├── manifest.json    # SHA-256 hashes and versioning info
    ├── raw/             # Original downloaded files (per-dataset subdirectories)
    └── processed/       # Canonical parquet files (ready to load)

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

import json
import logging
import os
from pathlib import Path
from typing import Any

from imbdata.exceptions import ImbdataError

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path.home() / ".imbdata"
STORE_ENV_VAR = "IMBDATA_STORE"
CONFIG_FILENAME = "config.json"
MANIFEST_FILENAME = "manifest.json"
RAW_DIRNAME = "raw"
PROCESSED_DIRNAME = "processed"
REGISTRY_FILENAME = "datasets.yaml"

__all__ = [
    "StoreConfig",
    "DEFAULT_STORE_PATH",
    "STORE_ENV_VAR",
    "store_path",
    "raw_dir",
    "processed_path",
    "manifest_path",
    "registry_path",
    "ensure_dirs",
    "read_manifest",
    "write_manifest",
    "default_config",
]


class StoreConfig:
    """Resolves and manages the layout of the local ``imbdata`` data store.

    The store location is resolved once at construction time, in priority order:

    1. The ``store_path`` argument passed to the constructor.
    2. The ``IMBDATA_STORE`` environment variable.
    3. The ``store_path`` key of ``~/.imbdata/config.json``.
    4. The default location, ``~/.imbdata``.

    Attributes:
        DEFAULT_STORE_PATH: Fallback store location (``~/.imbdata``).
        STORE_ENV_VAR: Name of the environment variable that overrides the store.

    Example:
        >>> config = StoreConfig()
        >>> config.store_path().name
        '.imbdata'
        >>> config.processed_path("spambase").name
        'spambase.parquet'
    """

    DEFAULT_STORE_PATH = DEFAULT_STORE_PATH
    STORE_ENV_VAR = STORE_ENV_VAR

    def __init__(self, store_path: str | Path | None = None) -> None:
        """Initialize the configuration and resolve the store location.

        Args:
            store_path: Explicit store location. When omitted, the location is
                resolved from the environment, the user config file, or the
                default path, in that order.
        """
        self._store_path = self._resolve_store_path(store_path)

    def __repr__(self) -> str:
        """Return an unambiguous representation of the configuration."""
        return f"{type(self).__name__}(store_path={str(self._store_path)!r})"

    # ── Path resolution ────────────────────────────────────────────────

    @classmethod
    def _resolve_store_path(cls, override: str | Path | None = None) -> Path:
        """Resolve the store location from the documented priority order.

        Args:
            override: Explicit location supplied by the caller, or ``None``.

        Returns:
            The absolute path of the data store root.
        """
        if override is not None:
            return Path(override).expanduser()

        from_env = os.environ.get(cls.STORE_ENV_VAR)
        if from_env:
            return Path(from_env).expanduser()

        user_config = cls.DEFAULT_STORE_PATH / CONFIG_FILENAME
        if user_config.is_file():
            try:
                payload = json.loads(user_config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"Ignoring unreadable config file {user_config}: {exc}")
            else:
                configured = payload.get("store_path")
                if configured:
                    return Path(configured).expanduser()

        return cls.DEFAULT_STORE_PATH

    def store_path(self) -> Path:
        """Return the resolved root directory of the data store.

        Returns:
            Path to the store root (e.g. ``PosixPath('/home/luis/.imbdata')``).
        """
        return self._store_path

    def raw_dir(self, dataset_key: str, create: bool = True) -> Path:
        """Return the ``raw/`` subdirectory that holds a dataset's source files.

        Args:
            dataset_key: Dataset key in snake_case (e.g. ``'spambase'``).
            create: Whether to create the directory when it does not exist.

        Returns:
            Path to ``<store>/raw/<dataset_key>``.
        """
        target = self._store_path / RAW_DIRNAME / dataset_key
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def processed_dir(self, create: bool = False) -> Path:
        """Return the directory holding canonical parquet files.

        Args:
            create: Whether to create the directory when it does not exist.

        Returns:
            Path to ``<store>/processed``.
        """
        target = self._store_path / PROCESSED_DIRNAME
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def processed_path(self, dataset_key: str, variant: str | None = None) -> Path:
        """Return the path of the canonical parquet file for a dataset.

        Args:
            dataset_key: Dataset key in snake_case (e.g. ``'tcga_brca'``).
            variant: Optional variant name. Variants are stored side by side as
                ``<dataset_key>__<variant>.parquet``.

        Returns:
            Path to the parquet file, whether or not it already exists.
        """
        suffix = f"__{variant}" if variant else ""
        return self.processed_dir() / f"{dataset_key}{suffix}.parquet"

    def manifest_path(self) -> Path:
        """Return the path of the global manifest file."""
        return self._store_path / MANIFEST_FILENAME

    def config_path(self) -> Path:
        """Return the path of the user configuration file."""
        return self._store_path / CONFIG_FILENAME

    @staticmethod
    def registry_path() -> Path:
        """Return the path of the ``datasets.yaml`` bundled with the package."""
        return Path(__file__).parent / REGISTRY_FILENAME

    def ensure_dirs(self) -> dict[str, Path]:
        """Create the store directory tree if it does not already exist.

        Returns:
            Mapping of logical names (``'root'``, ``'raw'``, ``'processed'``) to
            their paths.

        Raises:
            ImbdataError: If the directories cannot be created.
        """
        dirs = {
            "root": self._store_path,
            "raw": self._store_path / RAW_DIRNAME,
            "processed": self._store_path / PROCESSED_DIRNAME,
        }
        try:
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ImbdataError(f"Cannot create data store at {self._store_path}: {exc}") from exc
        return dirs

    # ── JSON documents ─────────────────────────────────────────────────

    def read_config(self) -> dict[str, Any]:
        """Read the user configuration file.

        Returns:
            The parsed configuration, or an empty dict when absent or invalid.
        """
        return self._read_json(self.config_path())

    def write_config(self, data: dict[str, Any]) -> None:
        """Write the user configuration file atomically.

        Args:
            data: Configuration payload to serialize.
        """
        self._write_json(self.config_path(), data)

    def read_manifest(self) -> dict[str, Any]:
        """Read the global manifest.

        Returns:
            The parsed manifest, or an empty dict when absent or invalid.
        """
        return self._read_json(self.manifest_path())

    def write_manifest(self, data: dict[str, Any]) -> None:
        """Write the global manifest atomically.

        Args:
            data: Manifest payload to serialize.
        """
        self._write_json(self.manifest_path(), data)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        """Read a JSON document, tolerating absence and corruption.

        Args:
            path: File to read.

        Returns:
            Parsed mapping, or an empty dict if the file is missing or invalid.
        """
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Ignoring unreadable JSON file {path}: {exc}")
            return {}

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        """Write a JSON document atomically via a temporary sibling file.

        Args:
            path: Destination file.
            data: Mapping to serialize.

        Raises:
            ImbdataError: If the file cannot be written.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            raise ImbdataError(f"Cannot write {path}: {exc}") from exc


# ── Functional facade over a lazily-created default instance ───────────

_DEFAULT_CONFIG: StoreConfig | None = None


def default_config(refresh: bool = False) -> StoreConfig:
    """Return the process-wide default :class:`StoreConfig` instance.

    Args:
        refresh: Rebuild the instance, re-reading the environment and user
            config file. Useful in tests that relocate the store.

    Returns:
        The shared configuration object.
    """
    global _DEFAULT_CONFIG
    if refresh or _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = StoreConfig()
    return _DEFAULT_CONFIG


def store_path() -> Path:
    """Return the resolved data store path of the default configuration."""
    return default_config().store_path()


def raw_dir(dataset_key: str) -> Path:
    """Return (and create) the raw directory of ``dataset_key``."""
    return default_config().raw_dir(dataset_key)


def processed_path(dataset_key: str, variant: str | None = None) -> Path:
    """Return the canonical parquet path of ``dataset_key``."""
    return default_config().processed_path(dataset_key, variant)


def manifest_path() -> Path:
    """Return the path of the global manifest file."""
    return default_config().manifest_path()


def registry_path() -> Path:
    """Return the path of the bundled ``datasets.yaml``."""
    return StoreConfig.registry_path()


def ensure_dirs() -> dict[str, Path]:
    """Create the store directory tree and return its paths."""
    return default_config().ensure_dirs()


def read_manifest() -> dict[str, Any]:
    """Read the global manifest of the default configuration."""
    return default_config().read_manifest()


def write_manifest(data: dict[str, Any]) -> None:
    """Write the global manifest of the default configuration."""
    default_config().write_manifest(data)
