"""
imbdata.api — Core public API for loading and managing datasets.

Implements the five calls a consumer project needs — :func:`load`,
:func:`list_datasets`, :func:`info`, :func:`ensure`, and :func:`verify` — on top
of the four collaborating classes of the package: :class:`~imbdata.config.StoreConfig`,
:class:`~imbdata.registry.DatasetRegistry`, :class:`~imbdata.download.DownloadManager`,
:class:`~imbdata.preprocess.DatasetPreprocessor`, and
:class:`~imbdata.verify.ManifestManager`.

:class:`DatasetService` owns the orchestration; the module-level functions are a
thin facade over a lazily-created default service, so ``import imbdata`` stays a
two-call API while remaining fully injectable in tests.

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

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from imbdata.config import StoreConfig, default_config
from imbdata.download import DownloadManager
from imbdata.exceptions import ImbdataError
from imbdata.preprocess import TARGET_COLUMN, DatasetPreprocessor
from imbdata.registry import DatasetRegistry, default_registry
from imbdata.verify import ManifestManager

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetService",
    "load",
    "list_datasets",
    "info",
    "ensure",
    "verify",
    "store_path",
    "default_service",
]


class DatasetService:
    """Orchestrates registry lookup, download, preprocessing, and caching.

    A dataset is materialized lazily: :meth:`load` downloads the raw files,
    runs the dataset's preprocessing routine, records the SHA-256 of the
    resulting parquet in the manifest, and then reads it back. Subsequent calls
    hit the cached parquet directly.

    Attributes:
        config: Store configuration resolving all filesystem paths.
        registry: Dataset metadata source.
        downloader: Engine dispatcher used to fetch raw files.
        preprocessor: Per-dataset preprocessing dispatch.
        manifest: SHA-256 manifest reader/writer.

    Example:
        >>> service = DatasetService()
        >>> X, y = service.load("spambase")
        >>> X.shape, int(y.sum())
        ((4601, 57), 1813)
    """

    def __init__(
        self,
        config: StoreConfig | None = None,
        registry: DatasetRegistry | None = None,
        downloader: DownloadManager | None = None,
        preprocessor: DatasetPreprocessor | None = None,
        manifest: ManifestManager | None = None,
    ) -> None:
        """Initialize the service and its collaborators.

        Args:
            config: Store configuration. Defaults to the process-wide one.
            registry: Dataset registry. Defaults to the bundled registry.
            downloader: Download dispatcher. Defaults to a new manager bound to
                ``config``.
            preprocessor: Preprocessing dispatch. Defaults to a new instance.
            manifest: Manifest manager. Defaults to one bound to ``config``.
        """
        self.config = config if config is not None else default_config()
        self.registry = registry if registry is not None else default_registry()
        self.downloader = downloader if downloader is not None else DownloadManager(self.config)
        self.preprocessor = preprocessor if preprocessor is not None else DatasetPreprocessor()
        self.manifest = manifest if manifest is not None else ManifestManager(self.config)

    def __repr__(self) -> str:
        """Return an unambiguous representation of the service."""
        return f"{type(self).__name__}(store={str(self.config.store_path())!r})"

    # ── Discovery ──────────────────────────────────────────────────────

    def store_path(self) -> Path:
        """Return the resolved data store path."""
        return self.config.store_path()

    def list_datasets(self, domain: str | None = None) -> list[str]:
        """List available dataset keys, optionally filtered by domain.

        Args:
            domain: If provided, only return datasets from this domain.

        Returns:
            Sorted list of dataset keys.
        """
        return self.registry.filter(domain=domain) if domain else self.registry.list_all()

    def info(self, name: str, variant: str | None = None) -> dict[str, Any]:
        """Return metadata for a dataset without loading its features.

        Registry metadata is always available. Statistics (``N``, ``d``, ``IR``)
        are computed from the cached parquet and are therefore present only once
        the dataset has been materialized.

        Args:
            name: Dataset key (e.g. ``'credit_card_fraud'``).
            variant: Optional variant name.

        Returns:
            Mapping with ``name``, ``domain``, ``source``, ``status``,
            ``is_cached``, ``store_path``, and — when cached — ``N``, ``d``,
            ``n_minority``, ``n_majority``, ``IR``, ``minority_pct``, and the
            recorded ``sha256``.

        Raises:
            DatasetNotFoundError: If ``name`` is not registered.
        """
        meta = self.registry.get(name)
        processed = self.config.processed_path(name, variant)

        result: dict[str, Any] = {
            "name": name,
            "domain": meta.get("domain"),
            "source": meta.get("source"),
            "status": meta.get("status"),
            "variant": variant,
            "is_cached": processed.is_file(),
            "is_implemented": self.preprocessor.supports(name),
            "store_path": str(self.config.store_path()),
            "processed_path": str(processed),
        }
        if meta.get("notes"):
            result["notes"] = meta["notes"]

        if processed.is_file():
            result.update(self._cached_statistics(processed))
            entry = self.manifest.entry(name, variant)
            if entry:
                result["sha256"] = entry.get("sha256_processed")
                result["updated_utc"] = entry.get("updated_utc")
        return result

    @staticmethod
    def _cached_statistics(processed: Path) -> dict[str, Any]:
        """Compute size and imbalance statistics from a cached parquet file.

        Args:
            processed: Path to a canonical parquet file.

        Returns:
            Mapping with ``N``, ``d``, ``n_minority``, ``n_majority``, ``IR``,
            and ``minority_pct``.
        """
        frame = pd.read_parquet(processed)
        target = frame[TARGET_COLUMN]
        n_minority = int(target.sum())
        n_majority = int(len(target) - n_minority)
        return {
            "N": len(frame),
            "d": frame.shape[1] - 1,
            "n_minority": n_minority,
            "n_majority": n_majority,
            "IR": round(n_majority / n_minority, 2) if n_minority else float("inf"),
            "minority_pct": round(100.0 * n_minority / len(frame), 3) if len(frame) else 0.0,
        }

    # ── Materialization ────────────────────────────────────────────────

    def load(
        self,
        name: str,
        variant: str | None = None,
        return_meta: bool = False,
        force: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
        """Load a dataset, materializing it on first access.

        Args:
            name: Dataset key (e.g. ``'credit_card_fraud'``).
            variant: Optional variant (e.g. ``'full'`` for ``tcga_brca``).
            return_meta: Return ``(X, y, meta)`` instead of ``(X, y)``.
            force: Re-download and re-preprocess even when a cached parquet
                exists.

        Returns:
            ``(X, y)`` where ``X`` is a ``float64`` feature frame and ``y`` an
            ``int64`` Series with ``0`` = majority and ``1`` = minority; or
            ``(X, y, meta)`` when ``return_meta`` is set.

        Raises:
            DatasetNotFoundError: If ``name`` is not registered.
            DownloadError: If the raw files cannot be retrieved.
            PreprocessingError: If preprocessing fails or violates the contract.

        Example:
            >>> X, y = load("ecoli_imu")
            >>> y.value_counts().to_dict()
            {0: 301, 1: 35}
        """
        processed = self.ensure_one(name, variant=variant, force=force)
        frame = pd.read_parquet(processed)
        y = frame.pop(TARGET_COLUMN)
        y.name = TARGET_COLUMN

        if return_meta:
            return frame, y, self.info(name, variant)
        return frame, y

    def ensure_one(self, name: str, variant: str | None = None, force: bool = False) -> Path:
        """Guarantee that a dataset's canonical parquet exists on disk.

        Runs the full download → preprocess → record pipeline when the file is
        absent, and returns immediately when it is already cached.

        Args:
            name: Dataset key.
            variant: Optional variant name.
            force: Rebuild even when a cached parquet exists.

        Returns:
            Path to the canonical parquet file.

        Raises:
            DownloadError: If the raw files cannot be retrieved.
            PreprocessingError: If preprocessing fails.
            ImbdataError: If the pipeline completes without producing the file.
        """
        processed = self.config.processed_path(name, variant)
        if processed.is_file() and not force:
            return processed

        meta = self.registry.get(name)
        self.config.ensure_dirs()

        raw_dir = self.downloader.download(name, meta, self.config.raw_dir(name))
        self.preprocessor.preprocess(name, raw_dir, processed, meta, variant)

        if not processed.is_file():
            raise ImbdataError(
                f"Preprocessing of '{name}' completed without writing {processed}."
            )

        statistics = self._cached_statistics(processed)
        self.manifest.update(
            name,
            processed,
            variant=variant,
            source=meta.get("source"),
            domain=meta.get("domain"),
            n_rows=statistics["N"],
            n_features=statistics["d"],
            n_minority=statistics["n_minority"],
            imbalance_ratio=statistics["IR"],
        )
        logger.info(f"'{name}' ready at {processed}")
        return processed

    def ensure(
        self,
        names: list[str] | str | None = None,
        domain: str | None = None,
        force: bool = False,
    ) -> dict[str, str]:
        """Pre-download and preprocess several datasets in bulk.

        Failures are captured per dataset rather than aborting the batch, so a
        single unreachable source does not block the rest.

        Args:
            names: Dataset keys to materialize. A bare string is accepted.
            domain: Materialize every dataset of this domain instead.
            force: Rebuild datasets that are already cached.

        Returns:
            Mapping of dataset key to ``'OK (cached)'``, ``'DOWNLOADED'``, or
            ``'ERROR: <reason>'``.

        Raises:
            ValueError: If neither ``names`` nor ``domain`` is supplied.
        """
        if domain is not None:
            names = self.list_datasets(domain=domain)
        elif isinstance(names, str):
            names = [names]
        if not names:
            raise ValueError("Provide either `names` or `domain`.")

        results: dict[str, str] = {}
        for name in names:
            try:
                cached = self.config.processed_path(name).is_file()
                self.ensure_one(name, force=force)
                results[name] = "OK (cached)" if cached and not force else "DOWNLOADED"
            except (ImbdataError, OSError, ValueError) as exc:
                results[name] = f"ERROR: {exc}"
                logger.error(f"Failed to ensure '{name}': {exc}")
        return results

    def verify(self, names: list[str] | None = None) -> dict[str, str]:
        """Verify cached datasets against the manifest's SHA-256 records.

        Args:
            names: Dataset keys to check. Defaults to the full registry.

        Returns:
            Mapping of dataset key to ``'OK'``, ``'MISSING'``,
            ``'NOT_IN_MANIFEST'``, or ``'HASH_MISMATCH'``.
        """
        return self.manifest.verify_all(names)


# ── Functional facade over a lazily-created default service ────────────

_DEFAULT_SERVICE: DatasetService | None = None


def default_service(refresh: bool = False) -> DatasetService:
    """Return the process-wide default :class:`DatasetService`.

    Args:
        refresh: Rebuild the service, re-resolving the store configuration.

    Returns:
        The shared service instance.
    """
    global _DEFAULT_SERVICE
    if refresh or _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = DatasetService()
    return _DEFAULT_SERVICE


def store_path() -> Path:
    """Return the resolved data store path.

    Returns:
        Path to the store root (e.g. ``PosixPath('/home/luis/.imbdata')``).
    """
    return default_service().store_path()


def list_datasets(domain: str | None = None) -> list[str]:
    """List available dataset keys, optionally filtered by domain.

    Args:
        domain: If provided, only return datasets from this domain.

    Returns:
        Sorted list of dataset keys.

    Example:
        >>> list_datasets(domain="financial_fraud")
        ['baf', 'credit_card_fraud', 'elliptic_bitcoin', 'ieee_cis_fraud', 'paysim']
    """
    return default_service().list_datasets(domain=domain)


def info(name: str, variant: str | None = None) -> dict[str, Any]:
    """Return metadata for a dataset without loading its features.

    Args:
        name: Dataset key.
        variant: Optional variant name.

    Returns:
        Metadata mapping; size and imbalance statistics are included once the
        dataset is cached.

    Raises:
        DatasetNotFoundError: If ``name`` is not registered.
    """
    return default_service().info(name, variant)


def load(
    name: str,
    variant: str | None = None,
    return_meta: bool = False,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load a dataset, downloading and preprocessing it on first access.

    Args:
        name: Dataset key.
        variant: Optional variant name.
        return_meta: Return ``(X, y, meta)`` instead of ``(X, y)``.
        force: Rebuild the dataset even when a cached parquet exists.

    Returns:
        ``(X, y)`` or ``(X, y, meta)``, where ``X`` holds ``float64`` features
        and ``y`` is an ``int64`` Series with ``0`` = majority, ``1`` = minority.

    Raises:
        DatasetNotFoundError: If ``name`` is not registered.
        DownloadError: If the raw files cannot be retrieved.
        PreprocessingError: If preprocessing fails.

    Example:
        >>> X, y = load("breast_cancer_wisconsin")
        >>> X.shape
        (569, 30)
    """
    return default_service().load(name, variant, return_meta=return_meta, force=force)


def ensure(
    names: list[str] | str | None = None,
    domain: str | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Pre-download and preprocess datasets in bulk.

    Args:
        names: Dataset keys to materialize.
        domain: Materialize every dataset of this domain instead.
        force: Rebuild datasets that are already cached.

    Returns:
        Mapping of dataset key to its outcome string.

    Raises:
        ValueError: If neither ``names`` nor ``domain`` is supplied.

    Example:
        >>> ensure(["spambase", "ecoli_imu"])
        {'spambase': 'OK (cached)', 'ecoli_imu': 'DOWNLOADED'}
    """
    return default_service().ensure(names, domain=domain, force=force)


def verify(names: list[str] | None = None) -> dict[str, str]:
    """Verify integrity of cached datasets against the manifest.

    Args:
        names: Dataset keys to check. Defaults to the full registry.

    Returns:
        Mapping of dataset key to its verification status.
    """
    return default_service().verify(names)
