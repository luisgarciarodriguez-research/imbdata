"""
imbdata.verify — SHA-256 integrity verification and manifest management.

Combines a pure hashing function (:func:`compute_sha256`) with the stateful
:class:`ManifestManager`, which owns reading, updating, and verifying the
``manifest.json`` document at the root of the data store. The manifest is what
lets two machines prove they hold byte-identical processed datasets.

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
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from imbdata.config import StoreConfig, default_config
from imbdata.exceptions import IntegrityError

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 20  # 1 MiB

STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_UNKNOWN = "NOT_IN_MANIFEST"
STATUS_MISMATCH = "HASH_MISMATCH"

__all__ = [
    "compute_sha256",
    "manifest_key",
    "ManifestManager",
    "sha256_file",
    "update_manifest",
    "verify_all",
]


# ── Pure functions ─────────────────────────────────────────────────────

def compute_sha256(path: str | Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Compute the SHA-256 digest of a file.

    Reads the file in fixed-size chunks so that multi-gigabyte datasets can be
    hashed without loading them into memory.

    Args:
        path: File to hash.
        chunk_size: Number of bytes read per iteration.

    Returns:
        The lowercase hexadecimal digest.

    Raises:
        IntegrityError: If the file cannot be read.

    Example:
        >>> compute_sha256("processed/spambase.parquet")[:8]
        '3f2a9c1d'
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IntegrityError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def manifest_key(name: str, variant: str | None = None) -> str:
    """Build the manifest key of a dataset, accounting for variants.

    Args:
        name: Dataset key (e.g. ``'tcga_brca'``).
        variant: Optional variant name (e.g. ``'full'``).

    Returns:
        ``name`` for the default variant, ``'name__variant'`` otherwise.
    """
    return f"{name}__{variant}" if variant else name


# ── Stateful manifest management ───────────────────────────────────────

class ManifestManager:
    """Reads, updates, and verifies the data store manifest.

    Every processed parquet file is recorded in ``manifest.json`` with its
    SHA-256 digest, size, row/column counts, and the UTC timestamp of the
    preprocessing run. :meth:`verify` compares the digest on disk against the
    recorded one.

    Attributes:
        config: The :class:`~imbdata.config.StoreConfig` naming the store.

    Example:
        >>> manager = ManifestManager()
        >>> manager.update("spambase", Path("~/.imbdata/processed/spambase.parquet"))
        >>> manager.verify("spambase")
        'OK'
    """

    def __init__(self, config: StoreConfig | None = None) -> None:
        """Initialize the manager.

        Args:
            config: Store configuration to operate on. Defaults to the
                process-wide configuration.
        """
        self.config = config if config is not None else default_config()

    def __repr__(self) -> str:
        """Return an unambiguous representation of the manager."""
        return f"{type(self).__name__}(config={self.config!r})"

    # ── I/O ────────────────────────────────────────────────────────────

    def read(self) -> dict[str, Any]:
        """Read the manifest document.

        Returns:
            Mapping of manifest key to its recorded entry; empty when the
            manifest does not exist yet.
        """
        return self.config.read_manifest()

    def write(self, data: dict[str, Any]) -> None:
        """Write the manifest document atomically.

        Args:
            data: Full manifest payload to persist.
        """
        self.config.write_manifest(data)

    def entry(self, name: str, variant: str | None = None) -> dict[str, Any] | None:
        """Return the manifest entry of a dataset, if recorded.

        Args:
            name: Dataset key.
            variant: Optional variant name.

        Returns:
            The recorded entry, or ``None`` when the dataset is not in the
            manifest.
        """
        return self.read().get(manifest_key(name, variant))

    # ── Mutation ───────────────────────────────────────────────────────

    def update(
        self,
        name: str,
        path: str | Path,
        variant: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Record the SHA-256 digest and statistics of a processed dataset.

        Args:
            name: Dataset key.
            path: Path to the processed parquet file.
            variant: Optional variant name.
            **extra: Additional fields to store verbatim in the entry
                (e.g. ``n_rows``, ``n_features``, ``sha256_raw``).

        Returns:
            The entry that was written.

        Raises:
            IntegrityError: If the file does not exist or cannot be hashed.
        """
        path = Path(path)
        if not path.is_file():
            raise IntegrityError(f"Cannot record '{name}': no processed file at {path}")

        entry: dict[str, Any] = {
            "sha256_processed": compute_sha256(path),
            "size_bytes": path.stat().st_size,
            "path": str(path),
            "variant": variant,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        entry.update(extra)

        manifest = self.read()
        manifest[manifest_key(name, variant)] = entry
        self.write(manifest)
        logger.debug(f"Manifest updated for '{manifest_key(name, variant)}'")
        return entry

    def remove(self, name: str, variant: str | None = None) -> bool:
        """Delete a dataset's manifest entry.

        Args:
            name: Dataset key.
            variant: Optional variant name.

        Returns:
            ``True`` if an entry was removed, ``False`` if there was none.
        """
        manifest = self.read()
        if manifest.pop(manifest_key(name, variant), None) is None:
            return False
        self.write(manifest)
        return True

    # ── Verification ───────────────────────────────────────────────────

    def verify(self, name: str, variant: str | None = None) -> str:
        """Verify one processed dataset against its recorded digest.

        Args:
            name: Dataset key.
            variant: Optional variant name.

        Returns:
            One of ``'OK'``, ``'MISSING'`` (no processed file cached),
            ``'NOT_IN_MANIFEST'`` (cached but never recorded), or
            ``'HASH_MISMATCH'`` (cached content differs from the manifest).
        """
        processed = self.config.processed_path(name, variant)
        if not processed.is_file():
            return STATUS_MISSING

        recorded = self.entry(name, variant)
        if recorded is None or "sha256_processed" not in recorded:
            return STATUS_UNKNOWN

        actual = compute_sha256(processed)
        if actual != recorded["sha256_processed"]:
            logger.warning(f"SHA-256 mismatch for '{manifest_key(name, variant)}' at {processed}")
            return STATUS_MISMATCH
        return STATUS_OK

    def verify_all(self, names: list[str] | None = None) -> dict[str, str]:
        """Verify every registered dataset against the manifest.

        Args:
            names: Dataset keys to check. Defaults to the full registry.

        Returns:
            Mapping of dataset key to verification status.
        """
        from imbdata.registry import default_registry

        keys = names if names is not None else default_registry().list_all()
        return {name: self.verify(name) for name in sorted(keys)}

    def require(self, name: str, variant: str | None = None) -> None:
        """Raise unless a dataset verifies as ``OK``.

        Args:
            name: Dataset key.
            variant: Optional variant name.

        Raises:
            IntegrityError: If verification returns anything other than ``OK``.
        """
        status = self.verify(name, variant)
        if status != STATUS_OK:
            raise IntegrityError(f"Integrity check failed for '{name}': {status}")


# ── Functional facade ──────────────────────────────────────────────────

def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 digest of a file (alias of :func:`compute_sha256`)."""
    return compute_sha256(path)


def update_manifest(name: str, processed_path: str | Path, variant: str | None = None,
                    **extra: Any) -> dict[str, Any]:
    """Record a processed dataset in the default store's manifest.

    Args:
        name: Dataset key.
        processed_path: Path to the processed parquet file.
        variant: Optional variant name.
        **extra: Additional fields stored verbatim in the entry.

    Returns:
        The entry that was written.
    """
    return ManifestManager().update(name, processed_path, variant=variant, **extra)


def verify_all(names: list[str] | None = None) -> dict[str, str]:
    """Verify all cached datasets in the default store against the manifest.

    Args:
        names: Dataset keys to check. Defaults to the full registry.

    Returns:
        Mapping of dataset key to verification status.
    """
    return ManifestManager().verify_all(names)
