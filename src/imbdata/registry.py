"""
imbdata.registry — Dataset registry backed by ``datasets.yaml``.

Parses the bundled dataset registry and exposes metadata lookup, domain
filtering, and structural validation through the :class:`DatasetRegistry`
class. Module-level helpers wrap a lazily-created default instance so that
``api.py`` and consumer code can query the registry without managing state.

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
from typing import Any, Iterator

import yaml

from imbdata.config import StoreConfig
from imbdata.exceptions import DatasetNotFoundError, RegistryError

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("domain", "source")

__all__ = [
    "DatasetRegistry",
    "get_registry",
    "get_dataset_meta",
    "list_domains",
]


class DatasetRegistry:
    """Read-only view over the dataset metadata declared in ``datasets.yaml``.

    The YAML file maps each dataset key to a metadata mapping describing its
    domain, download source, target column, binarization rule, encoding, and
    imputation policy. The registry parses the file once and caches it.

    Attributes:
        path: Location of the YAML registry file being served.

    Example:
        >>> registry = DatasetRegistry()
        >>> len(registry.list_all())
        28
        >>> registry.get("spambase")["domain"]
        'digital_communications'
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize the registry.

        Args:
            path: Alternative registry file. Defaults to the ``datasets.yaml``
                bundled with the installed package.
        """
        self.path = Path(path) if path is not None else StoreConfig.registry_path()
        self._entries: dict[str, dict[str, Any]] | None = None

    def __repr__(self) -> str:
        """Return an unambiguous representation of the registry."""
        return f"{type(self).__name__}(path={str(self.path)!r})"

    def __len__(self) -> int:
        """Return the number of registered datasets."""
        return len(self.entries)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered dataset keys in sorted order."""
        return iter(sorted(self.entries))

    def __contains__(self, name: object) -> bool:
        """Return whether ``name`` is a registered dataset key."""
        return name in self.entries

    # ── Parsing ────────────────────────────────────────────────────────

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        """Return the parsed registry, loading it on first access.

        Returns:
            Mapping of dataset key to its metadata mapping.

        Raises:
            RegistryError: If the file is missing, unreadable, or malformed.
        """
        if self._entries is None:
            self._entries = self._load()
        return self._entries

    def _load(self) -> dict[str, dict[str, Any]]:
        """Parse the YAML registry file.

        Returns:
            Mapping of dataset key to its metadata mapping.

        Raises:
            RegistryError: If the file is missing, unreadable, or malformed.
        """
        if not self.path.is_file():
            raise RegistryError(f"Dataset registry not found at {self.path}")
        try:
            payload = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RegistryError(f"Cannot parse dataset registry {self.path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise RegistryError(
                f"Dataset registry {self.path} must define a mapping of dataset keys, "
                f"got {type(payload).__name__}"
            )

        entries: dict[str, dict[str, Any]] = {}
        for key, meta in payload.items():
            if not isinstance(meta, dict):
                raise RegistryError(
                    f"Entry '{key}' in {self.path} must be a mapping, "
                    f"got {type(meta).__name__}"
                )
            entries[str(key)] = meta
        logger.debug(f"Loaded {len(entries)} dataset entries from {self.path}")
        return entries

    def reload(self) -> None:
        """Discard the cached registry so the next access re-reads the file."""
        self._entries = None

    # ── Lookup ─────────────────────────────────────────────────────────

    def get(self, name: str) -> dict[str, Any]:
        """Return the metadata of a single dataset.

        Args:
            name: Dataset key (e.g. ``'credit_card_fraud'``).

        Returns:
            A copy of the dataset's metadata mapping, with the key injected
            under ``'key'``.

        Raises:
            DatasetNotFoundError: If ``name`` is not a registered dataset key.

        Example:
            >>> DatasetRegistry().get("ecoli_imu")["domain"]
            'bioinformatics'
        """
        entries = self.entries
        if name not in entries:
            suggestion = self._suggest(name)
            hint = f" Did you mean '{suggestion}'?" if suggestion else ""
            raise DatasetNotFoundError(
                f"Unknown dataset '{name}'. {len(entries)} datasets are registered.{hint}"
            )
        meta = dict(entries[name])
        meta.setdefault("key", name)
        return meta

    def list_all(self) -> list[str]:
        """Return every registered dataset key, sorted alphabetically."""
        return sorted(self.entries)

    def filter(
        self,
        domain: str | None = None,
        source: str | None = None,
        status: str | None = None,
    ) -> list[str]:
        """Return the dataset keys matching every supplied criterion.

        Args:
            domain: Restrict to datasets of this domain (e.g. ``'medicine'``).
            source: Restrict to datasets served by this source (e.g. ``'uci'``).
            status: Restrict to datasets with this registry status.

        Returns:
            Sorted list of matching dataset keys; empty when nothing matches.

        Example:
            >>> DatasetRegistry().filter(domain="financial_fraud")
            ['baf', 'credit_card_fraud', 'elliptic_bitcoin', 'ieee_cis_fraud', 'paysim']
        """
        criteria = {"domain": domain, "source": source, "status": status}
        active = {field: value for field, value in criteria.items() if value is not None}
        return sorted(
            key
            for key, meta in self.entries.items()
            if all(meta.get(field) == value for field, value in active.items())
        )

    def domains(self) -> list[str]:
        """Return the sorted list of distinct domains present in the registry."""
        return sorted({meta["domain"] for meta in self.entries.values() if "domain" in meta})

    def sources(self) -> list[str]:
        """Return the sorted list of distinct download sources in the registry."""
        return sorted({meta["source"] for meta in self.entries.values() if "source" in meta})

    def validate(self) -> bool:
        """Check that every entry declares the mandatory metadata fields.

        Returns:
            ``True`` when the registry is structurally valid.

        Raises:
            RegistryError: If any entry is missing a required field.
        """
        problems: list[str] = []
        for key, meta in self.entries.items():
            missing = [field for field in REQUIRED_FIELDS if not meta.get(field)]
            if missing:
                problems.append(f"'{key}' is missing {', '.join(missing)}")
        if problems:
            raise RegistryError(
                f"Invalid dataset registry {self.path}: " + "; ".join(problems)
            )
        return True

    def _suggest(self, name: str) -> str | None:
        """Return the closest registered key to ``name``, if any is close enough.

        Args:
            name: The unknown key typed by the caller.

        Returns:
            The best matching dataset key, or ``None`` when nothing is similar.
        """
        import difflib

        matches = difflib.get_close_matches(name, self.entries, n=1, cutoff=0.6)
        return matches[0] if matches else None


# ── Functional facade over a lazily-created default instance ───────────

_DEFAULT_REGISTRY: DatasetRegistry | None = None


def default_registry(refresh: bool = False) -> DatasetRegistry:
    """Return the process-wide default :class:`DatasetRegistry` instance.

    Args:
        refresh: Rebuild the instance, re-reading ``datasets.yaml``.

    Returns:
        The shared registry object.
    """
    global _DEFAULT_REGISTRY
    if refresh or _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = DatasetRegistry()
    return _DEFAULT_REGISTRY


def get_registry() -> dict[str, dict[str, Any]]:
    """Return the full registry as a plain mapping.

    Returns:
        Mapping of dataset key to metadata mapping.

    Raises:
        RegistryError: If ``datasets.yaml`` is missing or malformed.
    """
    return default_registry().entries


def get_dataset_meta(name: str) -> dict[str, Any]:
    """Return the metadata of a single dataset.

    Args:
        name: Dataset key (e.g. ``'spambase'``).

    Returns:
        The dataset's metadata mapping.

    Raises:
        DatasetNotFoundError: If ``name`` is not registered.
    """
    return default_registry().get(name)


def list_domains() -> list[str]:
    """Return the sorted list of distinct domains in the registry."""
    return default_registry().domains()
