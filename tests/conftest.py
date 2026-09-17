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
    imbdata v0.4.0 — Imbalanced Classification Dataset Repository
    Advisor: Dr. José Antonio Neme Castillo
    Research Group: Anomalocaris
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from imbdata.config import StoreConfig
from imbdata.exceptions import DownloadError

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
    "saml_d",
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
    "saml_d",
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


# ── Synthetic raw files for the fraud endpoint (0.4.0) ────────────────

FRAUD_DATASETS = [
    "credit_card_fraud",
    "elliptic_bitcoin",
    "ieee_cis_fraud",
    "paysim",
    "saml_d",
]


def _cycle(values: list[Any], rows: int) -> list[Any]:
    """Repeat ``values`` until ``rows`` items are produced."""
    return [values[index % len(values)] for index in range(rows)]


def write_synthetic_credit_card_fraud(raw_dir: Path, rows: int = 20) -> Path:
    """Write a stand-in ``creditcard.csv``: elapsed seconds, 3 components, amount."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "Time": [float(index) for index in range(rows)],
            "V1": [0.1 * index for index in range(rows)],
            "V2": [-0.2 * index for index in range(rows)],
            "V3": [0.3 * index for index in range(rows)],
            "Amount": [10.0 + index for index in range(rows)],
            "Class": [1 if index % 5 == 0 else 0 for index in range(rows)],
        }
    )
    path = raw_dir / "creditcard.csv"
    frame.to_csv(path, index=False)
    return path


def write_synthetic_paysim(raw_dir: Path, rows: int = 20) -> Path:
    """Write a stand-in PaySim log, account identifiers included."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "step": [1 + index // 4 for index in range(rows)],
            "type": _cycle(["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"], rows),
            "amount": [100.0 * (index + 1) for index in range(rows)],
            "nameOrig": _cycle([f"C{1000 + index}" for index in range(5)], rows),
            "oldbalanceOrg": [500.0] * rows,
            "newbalanceOrig": [400.0] * rows,
            "nameDest": _cycle([f"C{1003 + index}" for index in range(5)], rows),
            "oldbalanceDest": [0.0] * rows,
            "newbalanceDest": [100.0] * rows,
            "isFraud": [1 if index % 5 == 0 else 0 for index in range(rows)],
            "isFlaggedFraud": [0] * rows,
        }
    )
    path = raw_dir / "PS_20174392719_1491204439457_log.csv"
    frame.to_csv(path, index=False)
    return path


def write_synthetic_ieee_cis_fraud(raw_dir: Path, rows: int = 20) -> Path:
    """Write stand-in IEEE-CIS transaction and identity tables."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    transactions = pd.DataFrame(
        {
            "TransactionID": [2987000 + index for index in range(rows)],
            "isFraud": [1 if index % 5 == 0 else 0 for index in range(rows)],
            "TransactionDT": [86400 + 60 * index for index in range(rows)],
            "TransactionAmt": [25.0 + index for index in range(rows)],
            "ProductCD": _cycle(["W", "C"], rows),
            "card1": [1000 + index for index in range(rows)],
        }
    )
    identities = pd.DataFrame(
        {
            "TransactionID": [2987000 + index for index in range(0, rows, 2)],
            "id_01": [-5.0 * index for index in range(0, rows, 2)],
            "DeviceType": ["mobile"] * len(range(0, rows, 2)),
        }
    )
    transactions.to_csv(raw_dir / "train_transaction.csv", index=False)
    identities.to_csv(raw_dir / "train_identity.csv", index=False)
    return raw_dir / "train_transaction.csv"


def write_synthetic_elliptic_bitcoin(raw_dir: Path) -> Path:
    """Write a stand-in Elliptic graph: 4 illicit, 6 licit and 3 unknown nodes.

    The edge list deliberately links labelled nodes to unknown ones, which is
    what the fraud endpoint must preserve while the canonical frame drops them.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    node_ids = list(range(100, 113))
    classes = ["1"] * 4 + ["2"] * 6 + ["unknown"] * 3
    features = pd.DataFrame(
        {
            0: node_ids,
            1: _cycle([1, 2, 3], len(node_ids)),
            2: [0.5 * index for index in range(len(node_ids))],
            3: [-0.25 * index for index in range(len(node_ids))],
            4: [1.0] * len(node_ids),
        }
    )
    features.to_csv(raw_dir / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame({"txId": node_ids, "class": classes}).to_csv(
        raw_dir / "elliptic_txs_classes.csv", index=False
    )
    pd.DataFrame(
        {
            "txId1": [100, 101, 102, 110, 104],
            "txId2": [101, 110, 103, 111, 105],
        }
    ).to_csv(raw_dir / "elliptic_txs_edgelist.csv", index=False)
    return raw_dir / "elliptic_txs_features.csv"


def write_synthetic_saml_d(raw_dir: Path, rows: int = 20) -> Path:
    """Write a stand-in ``SAML-D.csv`` with the published header."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "Time": [f"10:{index // 60:02d}:{index % 60:02d}" for index in range(rows)],
            "Date": _cycle(["2022-10-07", "2022-10-08"], rows),
            "Sender_account": _cycle([8724731955 + index for index in range(5)], rows),
            "Receiver_account": _cycle([2769355426 + index for index in range(5)], rows),
            "Amount": [1000.0 + 10 * index for index in range(rows)],
            "Payment_currency": _cycle(["UK pounds", "Dirham"], rows),
            "Received_currency": _cycle(["UK pounds", "Euro"], rows),
            "Sender_bank_location": _cycle(["UK", "UAE"], rows),
            "Receiver_bank_location": _cycle(["UK", "Germany"], rows),
            "Payment_type": _cycle(["Cash Deposit", "Cheque", "ACH", "Cross-border"], rows),
            "Is_laundering": [1 if index % 5 == 0 else 0 for index in range(rows)],
            "Laundering_type": _cycle(
                ["Smurfing", "Normal_Fan_Out", "Normal_Fan_In", "Normal_Group"], rows
            ),
        }
    )
    path = raw_dir / "SAML-D.csv"
    frame.to_csv(path, index=False)
    return path


SYNTHETIC_FRAUD_WRITERS = {
    "credit_card_fraud": write_synthetic_credit_card_fraud,
    "elliptic_bitcoin": write_synthetic_elliptic_bitcoin,
    "ieee_cis_fraud": write_synthetic_ieee_cis_fraud,
    "paysim": write_synthetic_paysim,
    "saml_d": write_synthetic_saml_d,
}


class OfflineDownloadManager:
    """Stand-in for :class:`~imbdata.download.DownloadManager` that never downloads.

    The unit tests run against synthetic raw files that are already in the
    temporary store. Injecting this manager makes a missing file an immediate
    error instead of a silent multi-gigabyte download from the real source.

    Attributes:
        config: Store configuration naming the raw directories.
    """

    def __init__(self, config: StoreConfig) -> None:
        """Initialize the manager.

        Args:
            config: Store configuration whose ``raw/`` holds the fixtures.
        """
        self.config = config

    def download(self, name: str, meta: dict[str, Any], target_dir: Path) -> Path:
        """Return the raw directory, provided the fixtures are already there.

        Args:
            name: Dataset key.
            meta: Registry metadata (unused).
            target_dir: Raw directory of the dataset.

        Returns:
            ``target_dir``.

        Raises:
            DownloadError: If the directory holds no file.
        """
        target_dir = Path(target_dir)
        if not any(target_dir.glob("*")):
            raise DownloadError(
                f"Offline test downloader: no raw fixture for '{name}' in {target_dir}"
            )
        return target_dir

    def close(self) -> None:
        """Match the real manager's interface; nothing to release."""


@pytest.fixture()
def fraud_raw_store(tmp_path: Path) -> StoreConfig:
    """Return a temporary store whose ``raw/`` holds synthetic fraud datasets.

    The bundled registry is used unchanged, so the ``fraud`` blocks under test
    are the real ones; only the data is synthetic and the store is temporary.

    Args:
        tmp_path: Pytest-provided scratch directory.

    Returns:
        A configuration rooted in the temporary store.
    """
    config = StoreConfig(store_path=tmp_path / "fraud_store")
    config.ensure_dirs()
    for name, writer in SYNTHETIC_FRAUD_WRITERS.items():
        writer(config.raw_dir(name))
    return config
