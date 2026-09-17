"""
tests.test_api — Integration tests for the public API.

Exercises :class:`imbdata.api.DatasetService` and the module-level facade:
listing, filtering, metadata lookup, caching, bulk ``ensure``, and the
end-to-end download → preprocess → verify → load pipeline on the pilot
datasets. Tests that need real data are skipped when the dataset is not yet
cached, so the suite stays green offline.

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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import imbdata
from imbdata.api import DatasetService
from imbdata.config import StoreConfig
from imbdata.exceptions import DatasetNotFoundError
from imbdata.preprocess import DatasetPreprocessor
from imbdata.registry import DatasetRegistry
from imbdata.verify import STATUS_OK, ManifestManager

from .conftest import (
    IMPLEMENTED_DATASETS,
    PILOT_DATASETS,
    dataset_params,
    requires_cached,
)

EXPECTED_DATASET_COUNT = 30

# Reference values from the PLAN.md dataset table. IR is checked only where the
# table's figure was reproduced; svmguide1's is recorded in STATUS.md instead.
PILOT_ROWS: dict[str, int] = {}  # populated from PILOT_SHAPES below

PILOT_SHAPES = {
    "breast_cancer_wisconsin": (569, 30),
    "pima_diabetes": (768, 8),
    "spambase": (4601, 57),
    "svmguide1": (7089, 4),
    "ecoli_imu": (336, 7),
}

PILOT_ROWS.update({name: rows for name, (rows, _) in PILOT_SHAPES.items()})

# Row counts for Batch A. Feature counts are omitted where preprocessing
# expands the raw dimensionality (one-hot encoding) or contracts it (SECOM's
# >50%-missing filter), since PLAN.md quotes the raw d in those rows.
BATCH_C_ROWS = {
    "yeast_me3": 1484,
    "nasa_pc1": 1109,
    "nasa_jm1": 10885,
    "tcga_brca": 826,
    "unsw_nb15": 257673,
    # 2,830,743 raw flows minus the 2,867 rows carrying Inf/NaN flow rates.
    "cic_ids_2017": 2827876,
    # All five temporal partitions of the Harvard Dataverse record.
    "swan_sf": 331185,
    # 10 CWRU conditions x 230 windows, less the 44 windows that condition
    # IR_014_1 cannot supply from its own X175 recording.
    "cwru_bearing": 2256,
    # 10 SEU gearset recordings x 1,000 windows.
    "seu_gearbox": 10000,
}

BATCH_B_ROWS = {
    "credit_card_fraud": 284807,
    "paysim": 6362620,
    "ieee_cis_fraud": 590540,
    "elliptic_bitcoin": 46564,
    "baf": 1000000,
    "vehicle_insurance_fraud": 15420,
}

BATCH_A_ROWS = {
    "mammography": 11183,
    "iranian_churn": 3150,
    "ozone_level": 2536,   # 1-hour horizon; the eighthr variant has 2,534
    "adult_census": 48842,
    "wine_quality_red": 1599,
    "wine_quality_white": 4898,
    "abalone_19": 4177,
    "secom": 1567,
    "nsl_kdd": 148517,
    # Statlog Landsat Satellite: sat.trn (4,435) + sat.tst (2,000).
    "satimage": 6435,
}


# ── Discovery ─────────────────────────────────────────────────────────

def test_list_datasets_returns_the_full_registry() -> None:
    """The public listing exposes all 30 registered datasets, sorted."""
    names = imbdata.list_datasets()
    assert len(names) == EXPECTED_DATASET_COUNT
    assert names == sorted(names)


def test_list_datasets_filters_by_domain() -> None:
    """Domain filtering matches the financial-fraud group of PLAN.md."""
    assert imbdata.list_datasets(domain="financial_fraud") == [
        "baf",
        "credit_card_fraud",
        "elliptic_bitcoin",
        "ieee_cis_fraud",
        "paysim",
    ]


def test_list_datasets_with_unknown_domain_is_empty() -> None:
    """An unknown domain yields an empty list rather than raising."""
    assert imbdata.list_datasets(domain="astronomy") == []


def test_store_path_is_the_resolved_store_root() -> None:
    """The public store_path() agrees with the configuration object."""
    assert imbdata.store_path() == StoreConfig().store_path()


# ── Metadata ──────────────────────────────────────────────────────────

def test_info_reports_registry_metadata_without_downloading() -> None:
    """Metadata is available for datasets that have never been materialized."""
    details = imbdata.info("paysim")
    assert details["domain"] == "financial_fraud"
    assert details["source"] == "kaggle"
    assert "name" in details and details["name"] == "paysim"


def test_info_exposes_the_license_block() -> None:
    """The registry's `license` block reaches info() unchanged."""
    details = imbdata.info("credit_card_fraud")
    assert details["license"]["spdx"] == "DbCL-1.0"
    assert details["license"]["status"] == "declared"


def test_info_returns_a_license_callers_cannot_mutate() -> None:
    """Editing the returned block, list included, leaves the registry intact."""
    imbdata.info("baf")["license"]["restrictions"].append("tampered")
    assert "tampered" not in imbdata.info("baf")["license"]["restrictions"]


def test_info_omits_license_when_the_registry_declares_none(
    temp_store: StoreConfig, synthetic_registry_file: Path
) -> None:
    """A user registry without `license` is served without the key, not rejected."""
    service = DatasetService(
        config=temp_store, registry=DatasetRegistry(synthetic_registry_file)
    )
    assert "license" not in service.info("alpha_set")


def test_info_on_unknown_dataset_raises() -> None:
    """An unregistered key raises DatasetNotFoundError."""
    with pytest.raises(DatasetNotFoundError):
        imbdata.info("not_a_dataset")


def test_info_flags_every_registered_dataset_as_implemented() -> None:
    """`is_implemented` is true across the registry now that all 30 have routines."""
    missing = [n for n in imbdata.list_datasets() if not imbdata.info(n)["is_implemented"]]
    assert missing == []


@requires_cached("ecoli_imu")
def test_info_includes_statistics_once_cached() -> None:
    """A materialized dataset reports N, d, IR, and its recorded digest."""
    details = imbdata.info("ecoli_imu")
    assert details["is_cached"] is True
    assert (details["N"], details["d"]) == PILOT_SHAPES["ecoli_imu"]
    assert details["n_minority"] == 35
    assert details["IR"] == pytest.approx(8.6, abs=0.05)
    assert len(details["sha256"]) == 64


# ── Loading ───────────────────────────────────────────────────────────

ALL_ROWS = {**BATCH_A_ROWS, **BATCH_B_ROWS, **BATCH_C_ROWS}


@pytest.mark.parametrize("name", dataset_params(IMPLEMENTED_DATASETS))
def test_dataset_satisfies_the_canonical_contract(name: str) -> None:
    """Every implemented dataset honours the whole format contract.

    The clauses are asserted together rather than in separate parametrized
    tests so that each dataset - some of which run to millions of rows - is
    read from disk once per session instead of once per clause.
    """
    if not StoreConfig().processed_path(name).is_file():
        pytest.skip(f"'{name}' is not cached; run `imbdata download {name}` first")

    X, y = imbdata.load(name)

    # Dtypes: float64 features, int64 binary target named 'target'.
    assert set(map(str, X.dtypes)) == {"float64"}
    assert str(y.dtype) == "int64"
    assert set(y.unique()) == {0, 1}
    assert y.name == "target"

    # No missing values, no infinities, default RangeIndex.
    assert int(X.isnull().sum().sum()) == 0
    assert bool(np.isfinite(X.to_numpy()).all())
    assert isinstance(X.index, pd.RangeIndex)

    # Class 1 is the minority, as the contract requires.
    assert 0 < int(y.sum()) <= len(y) - int(y.sum())

    # Row count matches the reference value recorded for this dataset.
    expected = {**PILOT_ROWS, **ALL_ROWS}.get(name)
    if expected is not None:
        assert len(y) == expected


@pytest.mark.parametrize("name", PILOT_DATASETS)
def test_load_matches_the_planned_dimensions(name: str) -> None:
    """N and d match the reference table in PLAN.md."""
    if not StoreConfig().processed_path(name).is_file():
        pytest.skip(f"'{name}' is not cached; run `imbdata download {name}` first")

    X, y = imbdata.load(name)
    assert (len(y), X.shape[1]) == PILOT_SHAPES[name]


@requires_cached("cwru_bearing")
def test_cwru_bearing_uses_each_files_own_recording() -> None:
    """Condition IR_014_1 reads X175, not the stray X217 bundled in file 175.

    CWRU ships file 175 with a second, unrelated recording. Selecting it would
    silently give that condition 230 windows instead of the 186 its own
    recording supports.
    """
    X, y = imbdata.load("cwru_bearing")
    assert X.shape[1] == 9
    assert int(y.sum()) == 230          # healthy windows, the minority class
    assert len(y) == 2256               # 2,300 minus the 44 X217-only windows


@requires_cached("seu_gearbox")
def test_seu_gearbox_has_the_declared_128_spectral_features() -> None:
    """A 256-sample window yields 128 FFT magnitude bins, healthy as minority."""
    X, y = imbdata.load("seu_gearbox")
    assert X.shape[1] == 128
    assert int(y.sum()) == 2000         # 2 healthy recordings x 1,000 windows
    assert len(y) == 10000


@requires_cached("ozone_level")
def test_ozone_level_horizons_share_features_and_differ_in_imbalance() -> None:
    """The 1-hour default and the 8-hour variant vary only the labelling.

    Both horizons label the same 72 meteorological predictors over the same
    days, so the pair isolates the effect of the imbalance ratio.
    """
    default, y_default = imbdata.load("ozone_level")
    assert int(y_default.sum()) == 73          # ozone days under the 1-hour standard

    variant_path = StoreConfig().processed_path("ozone_level", "eighthr")
    if not variant_path.is_file():
        pytest.skip("the 'eighthr' variant is not cached")

    eighthr, y_eighthr = imbdata.load("ozone_level", variant="eighthr")
    assert list(eighthr.columns) == list(default.columns)
    assert int(y_eighthr.sum()) == 160         # ozone days under the 8-hour standard


@requires_cached("swan_sf")
def test_swan_sf_has_the_declared_192_mvts_features() -> None:
    """24 magnetic field parameters x 8 statistics = the 192 declared features."""
    X, y = imbdata.load("swan_sf")
    assert X.shape[1] == 192
    assert int(y.sum()) == 6234  # M- and X-class flares across all five partitions


@requires_cached("elliptic_bitcoin")
def test_elliptic_bitcoin_marks_illicit_as_the_minority() -> None:
    """Class 1 (illicit, 4,545 nodes) is the positive class, not class 2.

    The registry's original comment had the mapping inverted, which produced
    an IR of 0.1:1 with licit transactions as the minority.
    """
    _, y = imbdata.load("elliptic_bitcoin")
    assert int(y.sum()) == 4545
    assert len(y) - int(y.sum()) == 42019


@requires_cached("tcga_brca")
def test_tcga_brca_variants_are_stored_separately() -> None:
    """The variance-filtered default and the full matrix coexist in the store."""
    default, y_default = imbdata.load("tcga_brca")
    assert default.shape[1] == 5000

    full_path = StoreConfig().processed_path("tcga_brca", "full")
    if not full_path.is_file():
        pytest.skip("the 'full' variant is not cached")

    full, y_full = imbdata.load("tcga_brca", variant="full")
    assert full.shape[1] > default.shape[1]
    assert y_full.equals(y_default)


@requires_cached("ecoli_imu")
def test_load_with_return_meta_appends_the_metadata() -> None:
    """return_meta=True yields the (X, y, meta) triple."""
    X, y, meta = imbdata.load("ecoli_imu", return_meta=True)
    assert meta["N"] == len(y) and meta["d"] == X.shape[1]


@requires_cached("ecoli_imu")
def test_load_is_idempotent_across_calls() -> None:
    """Loading twice returns identical frames from the cached parquet."""
    first, _ = imbdata.load("ecoli_imu")
    second, _ = imbdata.load("ecoli_imu")
    assert first.equals(second)


def test_load_unknown_dataset_raises() -> None:
    """Loading an unregistered key fails before touching the network."""
    with pytest.raises(DatasetNotFoundError):
        imbdata.load("not_a_dataset")


# ── Bulk operations ───────────────────────────────────────────────────

def test_ensure_without_arguments_raises() -> None:
    """ensure() needs either names or a domain."""
    with pytest.raises(ValueError, match="names.*domain"):
        imbdata.ensure()


def test_ensure_reports_cached_datasets_without_rebuilding() -> None:
    """A dataset already on disk is reported as cached, not re-downloaded."""
    cached = [n for n in PILOT_DATASETS if StoreConfig().processed_path(n).is_file()]
    if not cached:
        pytest.skip("no pilot dataset is cached")
    assert imbdata.ensure(cached[:1]) == {cached[0]: "OK (cached)"}


def test_ensure_captures_failures_per_dataset(
    temp_store: StoreConfig, synthetic_registry_file: Path
) -> None:
    """One unresolvable dataset does not abort the whole batch.

    Runs against the synthetic registry, whose entries declare no
    `download.files` and have no preprocessing routine, so the failure is
    raised locally without any network access.
    """
    service = DatasetService(
        config=temp_store, registry=DatasetRegistry(synthetic_registry_file)
    )
    results = service.ensure(["alpha_set", "beta_set"])
    assert set(results) == {"alpha_set", "beta_set"}
    assert all(outcome.startswith("ERROR") for outcome in results.values())


# ── Verification ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_verify_reports_a_status_for_every_dataset() -> None:
    """verify() covers the whole registry, marking uncached ones as MISSING.

    Marked slow: it re-hashes every cached parquet, several gigabytes in all.
    """
    report = imbdata.verify()
    assert len(report) == EXPECTED_DATASET_COUNT
    assert set(report.values()) <= {"OK", "MISSING", "NOT_IN_MANIFEST", "HASH_MISMATCH"}


@pytest.mark.parametrize("name", dataset_params(IMPLEMENTED_DATASETS))
def test_cached_datasets_verify_ok(name: str) -> None:
    """Each cached pilot parquet still matches its recorded SHA-256."""
    if not StoreConfig().processed_path(name).is_file():
        pytest.skip(f"'{name}' is not cached; run `imbdata download {name}` first")
    assert imbdata.verify([name])[name] == STATUS_OK


# ── Service wiring ────────────────────────────────────────────────────

def test_service_uses_the_injected_store(temp_store: StoreConfig) -> None:
    """A service built on a temporary store never touches the user's store."""
    service = DatasetService(config=temp_store)
    assert service.store_path() == temp_store.store_path()
    assert service.info("spambase")["is_cached"] is False


def test_service_composes_the_documented_collaborators() -> None:
    """The service wires together the classes named in the PLAN.md architecture."""
    service = DatasetService()
    assert isinstance(service.config, StoreConfig)
    assert isinstance(service.preprocessor, DatasetPreprocessor)
    assert isinstance(service.manifest, ManifestManager)


def test_end_to_end_pipeline_on_an_isolated_store(
    temp_store: StoreConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download → preprocess → manifest → load runs in a throwaway store.

    The raw files are copied from the user's store when available, so the test
    exercises the full pipeline without repeating the network transfer.
    """
    name = "ecoli_imu"
    source_raw = StoreConfig().raw_dir(name, create=False)
    if not source_raw.is_dir() or not any(source_raw.iterdir()):
        pytest.skip(f"raw files for '{name}' are not available locally")

    import shutil

    shutil.copytree(source_raw, temp_store.raw_dir(name), dirs_exist_ok=True)

    service = DatasetService(config=temp_store)
    processed = service.ensure_one(name)

    assert processed.is_file()
    assert service.verify([name])[name] == STATUS_OK

    X, y = service.load(name)
    assert (len(y), X.shape[1]) == PILOT_SHAPES[name]
    assert int(y.sum()) == 35


def test_processed_files_land_inside_the_store(temp_store: StoreConfig) -> None:
    """Canonical parquets are written under <store>/processed/."""
    path = temp_store.processed_path("anything")
    assert path.parent == temp_store.store_path() / "processed"
