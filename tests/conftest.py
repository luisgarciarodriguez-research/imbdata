"""
tests.conftest — Shared pytest fixtures for the imbdata test suite.

Provides an isolated temporary data store, a synthetic dataset registry, and
helpers that let the unit tests run offline. Integration tests that need real
data are skipped unless the corresponding dataset is already cached in the
user's store.

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

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from imbdata.config import StoreConfig

PILOT_DATASETS = [
    "breast_cancer_wisconsin",
    "pima_diabetes",
    "spambase",
    "svmguide1",
    "ecoli_imu",
]

BATCH_A_DATASETS = [
    "mammography",
    "iranian_churn",
    "ozone_level",
    "adult_census",
    "wine_quality_red",
    "wine_quality_white",
    "abalone_19",
    "secom",
    "nsl_kdd",
    "satimage",
]

BATCH_C_DATASETS = [
    "yeast_me3",
    "nasa_pc1",
    "nasa_jm1",
    "tcga_brca",
    "unsw_nb15",
    "cic_ids_2017",
    "swan_sf",
    "cwru_bearing",
    "seu_gearbox",
]

BATCH_B_DATASETS = [
    "credit_card_fraud",
    "paysim",
    "ieee_cis_fraud",
    "elliptic_bitcoin",
    "baf",
    "vehicle_insurance_fraud",
]

IMPLEMENTED_DATASETS = (
    PILOT_DATASETS + BATCH_A_DATASETS + BATCH_B_DATASETS + BATCH_C_DATASETS
)

SYNTHETIC_REGISTRY = """
alpha_set:
  domain: medicine
  status: existing
  source: uci
  target_column: label
  minority_value: 1
beta_set:
  domain: medicine
  status: new
  source: kaggle
  target_column: y
  minority_value: 1
gamma_set:
  domain: cybersecurity
  status: new
  source: openml
  openml_id: 310
  target_column: class
  minority_value: 1
"""


# Datasets whose processed parquet runs to hundreds of megabytes or more.
# Reading and hashing these dominates a test run, so their parametrized cases
# carry the `slow` marker and are skipped unless the run asks for them.
LARGE_DATASETS = frozenset({
    "baf",
    "cic_ids_2017",
    "credit_card_fraud",
    "ieee_cis_fraud",
    "nsl_kdd",
    "paysim",
    "swan_sf",
    "tcga_brca",
    "unsw_nb15",
})


def dataset_params(names: list[str]) -> list[Any]:
    """Turn dataset keys into parametrize arguments, marking the large ones slow.

    Args:
        names: Dataset keys to parametrize over.

    Returns:
        A list of plain strings and :func:`pytest.param` entries, the latter
        carrying ``pytest.mark.slow`` for the datasets in
        :data:`LARGE_DATASETS`.
    """
    return [
        pytest.param(name, marks=pytest.mark.slow) if name in LARGE_DATASETS else name
        for name in names
    ]


@pytest.fixture()
def temp_store(tmp_path: Path) -> StoreConfig:
    """Return a :class:`StoreConfig` rooted in an empty temporary directory.

    Args:
        tmp_path: Pytest-provided scratch directory.

    Returns:
        A configuration whose store tree has already been created.
    """
    config = StoreConfig(store_path=tmp_path / "store")
    config.ensure_dirs()
    return config


@pytest.fixture()
def synthetic_registry_file(tmp_path: Path) -> Path:
    """Write a small three-dataset registry and return its path.

    Args:
        tmp_path: Pytest-provided scratch directory.

    Returns:
        Path to the YAML file.
    """
    path = tmp_path / "synthetic.yaml"
    path.write_text(SYNTHETIC_REGISTRY, encoding="utf-8")
    return path


@pytest.fixture()
def canonical_frame() -> pd.DataFrame:
    """Return a small frame that satisfies the canonical format contract.

    Returns:
        Six rows, two ``float64`` features, and a binary ``target``.
    """
    return pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "target": [0, 0, 0, 0, 1, 1],
        }
    ).astype({"f1": "float64", "f2": "float64", "target": "int64"})


def requires_cached(name: str) -> pytest.MarkDecorator:
    """Build a skip marker for tests that need a materialized dataset.

    Args:
        name: Dataset key that must already be cached in the default store.

    Returns:
        A ``pytest.mark.skipif`` marker that skips when the parquet is absent.
    """
    cached = StoreConfig().processed_path(name).is_file()
    return pytest.mark.skipif(
        not cached,
        reason=f"'{name}' is not cached; run `imbdata download {name}` first",
    )
