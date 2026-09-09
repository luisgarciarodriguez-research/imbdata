"""
imbdata.exceptions — Domain-specific exception hierarchy.

Defines the exception types raised across the package. Every error surfaced by
``imbdata`` derives from :class:`ImbdataError`, so a consumer project can catch
the whole family with a single ``except ImbdataError``.

Author:
    Luis García Rodríguez
    Doctorado en Ciencia e Ingeniería de la Computación (DCIC)
    IIMAS — Universidad Nacional Autónoma de México (UNAM)
    CVU: 905206 · ORCID: 0009-0004-9514-5508

Project:
    imbdata v0.1.0 — Imbalanced Classification Dataset Repository
    Advisor: Dr. José Antonio Neme Castillo
    Research Group: Anomalocaris
"""

from __future__ import annotations

__all__ = [
    "ImbdataError",
    "DatasetNotFoundError",
    "DownloadError",
    "IntegrityError",
    "PreprocessingError",
    "RegistryError",
    "CredentialsError",
]


class ImbdataError(Exception):
    """Base exception for all imbdata errors."""


class RegistryError(ImbdataError):
    """Raised when ``datasets.yaml`` is missing, unreadable, or malformed."""


class DatasetNotFoundError(ImbdataError):
    """Raised when a dataset key is not in the registry."""


class DownloadError(ImbdataError):
    """Raised when a dataset download fails."""


class CredentialsError(DownloadError):
    """Raised when credentials for an authenticated source are unavailable."""


class PreprocessingError(ImbdataError):
    """Raised when a dataset cannot be preprocessed into the canonical format."""


class IntegrityError(ImbdataError):
    """Raised when SHA-256 verification fails."""
