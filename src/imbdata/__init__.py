"""
imbdata — Centralized dataset repository for imbalanced classification benchmarks.

Manages a single local data store (``~/.imbdata/``) that downloads, preprocesses,
caches, versions, and verifies the benchmark datasets shared across several
doctoral projects. Every dataset is served in one canonical format: ``float64``
features plus an ``int64`` ``target`` column with ``0`` = majority and
``1`` = minority, free of missing values and categorical columns.

Usage:
    import imbdata

    # Load a dataset (downloads + preprocesses on first call, cached after)
    X, y = imbdata.load("credit_card_fraud")

    # List all available datasets, or filter by domain
    names = imbdata.list_datasets()
    financial = imbdata.list_datasets(domain="financial_fraud")

    # Inspect metadata without loading the features
    meta = imbdata.info("credit_card_fraud")

    # Pre-download in bulk
    imbdata.ensure(["credit_card_fraud", "paysim"])
    imbdata.ensure(domain="financial_fraud")

    # Verify integrity of everything cached, and locate the store
    report = imbdata.verify()
    path = imbdata.store_path()

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

__version__ = "0.3.0"

from imbdata.api import (
    DatasetService,
    ensure,
    info,
    list_datasets,
    load,
    store_path,
    verify,
)
from imbdata.config import StoreConfig
from imbdata.download import DownloadManager
from imbdata.exceptions import (
    CredentialsError,
    DatasetNotFoundError,
    DownloadError,
    ImbdataError,
    IntegrityError,
    PreprocessingError,
    RegistryError,
)
from imbdata.preprocess import DatasetPreprocessor
from imbdata.registry import DatasetRegistry
from imbdata.verify import ManifestManager, compute_sha256

__all__ = [
    # Public API
    "load",
    "list_datasets",
    "info",
    "ensure",
    "verify",
    "store_path",
    # Domain classes
    "DatasetService",
    "StoreConfig",
    "DatasetRegistry",
    "DownloadManager",
    "DatasetPreprocessor",
    "ManifestManager",
    "compute_sha256",
    # Exceptions
    "ImbdataError",
    "DatasetNotFoundError",
    "DownloadError",
    "CredentialsError",
    "PreprocessingError",
    "IntegrityError",
    "RegistryError",
    "__version__",
]
