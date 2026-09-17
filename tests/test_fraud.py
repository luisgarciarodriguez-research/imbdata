"""
tests.test_fraud — Unit tests for the fraud endpoint (0.4.0).

Every test runs offline against synthetic raw files in a temporary store, with
the bundled registry, so the ``fraud`` blocks under test are the real ones.

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

import pandas as pd
import pytest

from imbdata.api import DatasetService
from imbdata.config import StoreConfig
from imbdata.exceptions import (
    DatasetNotFoundError,
    NotAFraudDatasetError,
    PreprocessingError,
    RegistryError,
)
from imbdata.fraud import (
    CONTEXT_SCHEMA,
    EDGES_SCHEMA,
    NODES_SCHEMA,
    FraudService,
    encode_accounts,
    manifest_key,
    to_seconds,
)
from imbdata.registry import DatasetRegistry
from imbdata.verify import (
    STATUS_MISMATCH,
    STATUS_MISSING,
    STATUS_OK,
    ManifestManager,
)

from .conftest import FRAUD_DATASETS, OfflineDownloadManager

# Native temporal column of each served dataset, as its `fraud` block declares.
TIME_COLUMN = {
    "credit_card_fraud": "Time",
    "elliptic_bitcoin": "time_step",
    "ieee_cis_fraud": "TransactionDT",
    "paysim": "step",
    "saml_d": "timestamp",
}


def offline_service(config: StoreConfig, registry: DatasetRegistry | None = None
                    ) -> FraudService:
    """Build a fraud service that can only ever read the store's fixtures.

    Args:
        config: Temporary store holding the synthetic raw files.
        registry: Optional registry override.

    Returns:
        A service whose downloader refuses to reach the network.
    """
    downloader = OfflineDownloadManager(config)
    return FraudService(
        config=config,
        registry=registry,
        datasets=DatasetService(config=config, registry=registry, downloader=downloader),
        downloader=downloader,
    )


@pytest.fixture()
def service(fraud_raw_store: StoreConfig) -> FraudService:
    """Return a fraud service bound to the synthetic temporary store."""
    return offline_service(fraud_raw_store)


# ── Pure functions ────────────────────────────────────────────────────

def test_to_seconds_scales_hours_by_3600() -> None:
    """An hourly step becomes 3,600 seconds."""
    assert to_seconds(pd.Series([1.0, 2.0]), "hour").tolist() == [3600.0, 7200.0]


def test_to_seconds_leaves_seconds_unchanged() -> None:
    """A column already in seconds is returned as it is."""
    assert to_seconds(pd.Series([0.0, 42.0]), "second").tolist() == [0.0, 42.0]


def test_to_seconds_on_an_unconvertible_unit_is_all_nan() -> None:
    """Elliptic's `step` has no published duration, so it yields NaN."""
    result = to_seconds(pd.Series([1.0, 49.0]), "step")
    assert result.isna().all()
    assert str(result.dtype) == "float64"


def test_encode_accounts_shares_one_code_space() -> None:
    """The same account has the same code as payer and as payee."""
    src = pd.Series(["C2", "C1", "C3"])
    dst = pd.Series(["C1", "C3", "C2"])
    src_id, dst_id = encode_accounts(src, dst)
    mapping = dict(zip(src.tolist() + dst.tolist(), src_id.tolist() + dst_id.tolist()))
    assert mapping["C1"] == 0 and mapping["C2"] == 1 and mapping["C3"] == 2
    assert src_id.tolist() == [1, 0, 2]
    assert dst_id.tolist() == [0, 2, 1]


def test_encode_accounts_is_invariant_to_row_order() -> None:
    """Sorting the union makes the codes independent of the row order."""
    src = pd.Series(["C2", "C1", "C3"])
    dst = pd.Series(["C1", "C3", "C2"])
    forward_src, forward_dst = encode_accounts(src, dst)

    order = [2, 0, 1]
    shuffled_src, shuffled_dst = encode_accounts(
        src.iloc[order].reset_index(drop=True), dst.iloc[order].reset_index(drop=True)
    )
    assert shuffled_src.tolist() == forward_src.iloc[order].tolist()
    assert shuffled_dst.tolist() == forward_dst.iloc[order].tolist()


def test_encode_accounts_returns_nullable_integers() -> None:
    """The codes are Int64, the dtype the context schema declares."""
    src_id, dst_id = encode_accounts(pd.Series(["a"]), pd.Series(["b"]))
    assert str(src_id.dtype) == "Int64" and str(dst_id.dtype) == "Int64"


def test_manifest_key_namespaces_the_artefacts() -> None:
    """Artefact keys live under `fraud/`, away from the canonical datasets."""
    assert manifest_key("paysim", "context") == "fraud/paysim.context"


# ── Discovery and metadata ────────────────────────────────────────────

def test_list_datasets_returns_the_five_declared_keys(service: FraudService) -> None:
    """Only datasets with a `fraud:` block are served."""
    assert service.list_datasets() == FRAUD_DATASETS


def test_info_appends_the_fraud_block(service: FraudService) -> None:
    """fraud.info() is imbdata.info() plus the registry's fraud block."""
    details = service.info("saml_d")
    assert details["domain"] == "financial_fraud"
    assert details["fraud"]["time"]["calendar"] is True
    assert details["fraud"]["drift_provenance"] == "researcher_constructed"
    assert details["fraud_artefacts"] == {"context": False}


def test_info_returns_a_fraud_block_callers_cannot_mutate(service: FraudService) -> None:
    """The block is deep-copied, so editing it leaves the registry intact."""
    service.info("paysim")["fraud"]["graph"]["src"] = "tampered"
    assert service.info("paysim")["fraud"]["graph"]["src"] == "nameOrig"


def test_info_on_a_non_fraud_dataset_raises(service: FraudService) -> None:
    """A registered dataset without the block cannot be served."""
    with pytest.raises(NotAFraudDatasetError, match="spambase"):
        service.info("spambase")


# ── Context schema ────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FRAUD_DATASETS)
def test_context_has_the_declared_schema_and_dtypes(
    service: FraudService, name: str
) -> None:
    """Every dataset yields the same context columns, order, and dtypes."""
    dataset = service.load(name)
    assert list(dataset.context.columns) == list(CONTEXT_SCHEMA)
    assert [str(dtype) for dtype in dataset.context.dtypes] == list(CONTEXT_SCHEMA.values())
    assert isinstance(dataset.context.index, pd.RangeIndex)


@pytest.mark.parametrize("name", FRAUD_DATASETS)
def test_context_aligns_row_by_row_with_the_canonical_frame(
    service: FraudService, name: str
) -> None:
    """context has one row per row of X, and its `t` is X's temporal column."""
    dataset = service.load(name)
    assert len(dataset.context) == len(dataset.X)
    assert dataset.context["t"].notna().all()

    column = TIME_COLUMN[name]
    assert dataset.context["t"].tolist() == dataset.X[column].tolist()


@pytest.mark.parametrize("name", FRAUD_DATASETS)
def test_load_serves_the_canonical_x_and_y_unchanged(
    service: FraudService, name: str
) -> None:
    """X and y are exactly what the canonical service returns."""
    dataset = service.load(name)
    X, y = service.datasets.load(name)
    assert dataset.X.equals(X)
    assert dataset.y.equals(y)


def test_context_of_credit_card_fraud_has_amount_but_no_graph(
    service: FraudService,
) -> None:
    """The ULB dataset publishes time and amount only."""
    context = service.load("credit_card_fraud").context
    assert context["amount"].notna().all()
    assert context["t_seconds"].equals(context["t"])
    assert context["event_time"].isna().all()
    assert context[["src_id", "dst_id", "entity_id", "node_id", "typology"]].isna().all().all()
    assert context["currency"].isna().all()


def test_context_of_paysim_carries_the_account_graph(service: FraudService) -> None:
    """PaySim's rows are the edges: both endpoints and the payer entity."""
    context = service.load("paysim").context
    assert context[["src_id", "dst_id", "entity_id"]].notna().all().all()
    # entity is nameOrig, the payer, so it reuses the payer's code.
    assert context["entity_id"].equals(context["src_id"])
    # step is hourly, so t_seconds is t x 3600.
    assert context["t_seconds"].equals(context["t"] * 3600.0)
    assert context["event_time"].isna().all()


def test_context_of_ieee_cis_has_no_graph_or_entity(service: FraudService) -> None:
    """No counterparty is published, so the graph columns stay null."""
    context = service.load("ieee_cis_fraud").context
    assert context["amount"].notna().all()
    assert context[["src_id", "dst_id", "entity_id", "node_id"]].isna().all().all()


def test_context_of_saml_d_has_calendar_time_currency_and_typology(
    service: FraudService,
) -> None:
    """SAML-D publishes a calendar instant, a currency, accounts, and a typology."""
    dataset = service.load("saml_d")
    context = dataset.context
    assert context["event_time"].notna().all()
    assert str(context["event_time"].dtype) == "datetime64[ns]"
    # event_time is the calendar reading of the same instant as t.
    assert context["event_time"].equals(pd.to_datetime(context["t"], unit="s"))
    assert set(context["currency"].dropna()) == {"UK pounds", "Dirham"}
    assert context[["src_id", "dst_id", "entity_id"]].notna().all().all()

    assert context["typology"].notna().all()
    assert "Laundering_type" not in dataset.X.columns
    assert "typology" not in dataset.X.columns


# ── Elliptic: the graph keeps its unknown nodes ───────────────────────

def test_elliptic_context_covers_only_the_labelled_nodes(service: FraudService) -> None:
    """The 3 unknown nodes are absent from X and from the context."""
    dataset = service.load("elliptic_bitcoin")
    assert len(dataset.context) == len(dataset.X) == 10
    assert dataset.context["node_id"].notna().all()
    assert dataset.context["t_seconds"].isna().all()   # unit: step
    assert dataset.context["amount"].isna().all()      # no amount is published


def test_elliptic_nodes_keep_unknowns_with_a_null_row(service: FraudService) -> None:
    """nodes holds every node; the unlabelled ones map to no row of X."""
    nodes = service.load("elliptic_bitcoin").nodes
    assert list(nodes.columns) == list(NODES_SCHEMA)
    assert len(nodes) == 13
    assert int(nodes["row"].notna().sum()) == 10
    unknown = nodes[nodes["label"].isna()]
    assert len(unknown) == 3
    assert unknown["row"].isna().all()
    # Illicit is class 1 of the registry, mapped to the canonical label 1.
    assert int((nodes["label"] == 1).sum()) == 4
    assert int((nodes["label"] == 0).sum()) == 6


def test_elliptic_edges_to_unlabelled_nodes_are_kept(service: FraudService) -> None:
    """An edge is kept even when one endpoint has no row in X."""
    edges = service.load("elliptic_bitcoin").edges
    assert list(edges.columns) == list(EDGES_SCHEMA)
    assert len(edges) == 5
    crossing = edges[edges["dst_row"].isna() | edges["src_row"].isna()]
    assert len(crossing) == 2
    assert set(crossing["dst_node"]) == {110, 111}


def test_elliptic_rows_match_the_canonical_positions(service: FraudService) -> None:
    """A node's `row` indexes the very row of X that carries its features."""
    dataset = service.load("elliptic_bitcoin")
    nodes = dataset.nodes.dropna(subset=["row"])
    for node_id, row in zip(nodes["node_id"], nodes["row"]):
        assert dataset.context.loc[int(row), "node_id"] == node_id


def test_non_graph_datasets_have_no_nodes_or_edges(service: FraudService) -> None:
    """Where the rows are the edges, the extra tables would be redundant."""
    for name in ("credit_card_fraud", "paysim", "saml_d", "ieee_cis_fraud"):
        dataset = service.load(name)
        assert dataset.nodes is None and dataset.edges is None


# ── Materialization, caching and verification ─────────────────────────

def test_ensure_builds_then_reports_the_cache(service: FraudService) -> None:
    """The first call builds the artefacts; the second finds them cached."""
    assert service.ensure(["credit_card_fraud"]) == {"credit_card_fraud": "BUILT"}
    assert service.ensure(["credit_card_fraud"]) == {"credit_card_fraud": "OK (cached)"}


def test_ensure_all_builds_every_served_dataset(service: FraudService) -> None:
    """A bare ensure() covers the whole endpoint."""
    assert service.ensure() == {name: "BUILT" for name in FRAUD_DATASETS}
    assert set(service.verify().values()) == {STATUS_OK}


def test_ensure_reports_the_error_per_dataset(fraud_raw_store: StoreConfig) -> None:
    """A failure is captured per key instead of aborting the batch."""
    registry = DatasetRegistry()
    entry = registry.entries["credit_card_fraud"]
    registry.entries["credit_card_fraud"] = {
        **entry,
        "fraud": {**entry["fraud"], "amount": {"column": "NotAColumn"}},
    }
    outcome = offline_service(fraud_raw_store, registry).ensure(["credit_card_fraud"])
    assert outcome["credit_card_fraud"].startswith("ERROR")
    assert "NotAColumn" in outcome["credit_card_fraud"]


def test_ensure_without_the_raw_files_does_not_reach_the_network(
    service: FraudService,
) -> None:
    """A missing raw fixture is an error, never a silent download."""
    for path in service.config.raw_dir("saml_d").glob("*"):
        path.unlink()
    assert service.ensure(["saml_d"])["saml_d"].startswith("ERROR")


def test_verify_reports_missing_before_anything_is_built(service: FraudService) -> None:
    """Nothing is cached in a fresh store."""
    assert set(service.verify().values()) == {STATUS_MISSING}


def test_verify_detects_a_tampered_artefact(service: FraudService) -> None:
    """Rewriting an artefact behind the manifest's back is a hash mismatch."""
    service.ensure(["paysim"])
    path = service.store.path("paysim", "context")
    frame = pd.read_parquet(path)
    frame.loc[0, "amount"] = 999999.0
    frame.to_parquet(path, index=False)
    assert service.verify(["paysim"])["fraud/paysim.context"] == STATUS_MISMATCH


def test_force_rebuilds_the_artefacts(service: FraudService) -> None:
    """force=True restores an artefact edited on disk."""
    service.ensure(["paysim"])
    path = service.store.path("paysim", "context")
    original = pd.read_parquet(path)
    original.assign(amount=0.0).to_parquet(path, index=False)

    rebuilt = service.load("paysim", force=True).context
    assert rebuilt["amount"].equals(original["amount"])
    assert service.verify(["paysim"])["fraud/paysim.context"] == STATUS_OK


def test_artefacts_are_recorded_in_the_manifest(service: FraudService) -> None:
    """Each artefact gets a `fraud/` entry with its digest and row count."""
    service.ensure(["elliptic_bitcoin"])
    manifest = ManifestManager(service.config).read()
    for part, rows in (("context", 10), ("nodes", 13), ("edges", 5)):
        entry = manifest[f"fraud/elliptic_bitcoin.{part}"]
        assert entry["n_rows"] == rows
        assert entry["dataset"] == "elliptic_bitcoin" and entry["part"] == part
        assert len(entry["sha256_processed"]) == 64
        assert Path(entry["path"]).is_file()


def test_canonical_verify_is_unchanged_by_the_fraud_artefacts(
    service: FraudService,
) -> None:
    """imbdata.verify() reports exactly the canonical datasets, before and after."""
    datasets = service.datasets
    datasets.ensure(FRAUD_DATASETS)
    before = datasets.verify(FRAUD_DATASETS)
    assert set(before.values()) == {STATUS_OK}

    service.ensure()
    after = datasets.verify(FRAUD_DATASETS)
    assert before == after
    assert set(after) == set(FRAUD_DATASETS)
    assert not [key for key in after if key.startswith("fraud/")]


def test_artefacts_live_under_processed_fraud(service: FraudService) -> None:
    """The canonical listing keeps naming only what imbdata.load serves."""
    service.ensure(["saml_d"])
    fraud_dir = service.config.fraud_dir()
    assert fraud_dir == service.config.processed_dir() / "fraud"
    assert (fraud_dir / "saml_d.context.parquet").is_file()
    assert sorted(p.name for p in service.config.processed_dir().glob("*.parquet")) == [
        "saml_d.parquet"
    ]


# ── Errors ────────────────────────────────────────────────────────────

def test_load_of_a_non_fraud_dataset_raises(service: FraudService) -> None:
    """A registered dataset without a `fraud:` block is refused."""
    with pytest.raises(NotAFraudDatasetError):
        service.load("spambase")


def test_load_of_an_unknown_key_raises(service: FraudService) -> None:
    """An unregistered key keeps raising the registry's own error."""
    with pytest.raises(DatasetNotFoundError):
        service.load("not_a_dataset")


def test_a_malformed_fraud_block_fails_registry_validation(tmp_path: Path) -> None:
    """An unknown time unit is a registry error, not a missing context column."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "broken_set:\n"
        "  domain: financial_fraud\n"
        "  source: kaggle\n"
        "  fraud:\n"
        "    time: {column: t, unit: fortnight, calendar: false}\n"
        "    amount: null\n"
        "    graph: null\n"
        "    entity: null\n"
        "    synthetic: false\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="time.unit"):
        DatasetRegistry(path).validate()


def test_a_fraud_block_missing_required_fields_fails_validation(tmp_path: Path) -> None:
    """The block must declare time, amount, graph, entity and synthetic."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "broken_set:\n"
        "  domain: financial_fraud\n"
        "  source: kaggle\n"
        "  fraud:\n"
        "    time: {column: t, unit: second, calendar: false}\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="is missing amount"):
        DatasetRegistry(path).validate()


def test_an_account_graph_without_endpoints_fails_validation(tmp_path: Path) -> None:
    """A graph of accounts needs both `src` and `dst` to be encodable."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "broken_set:\n"
        "  domain: financial_fraud\n"
        "  source: kaggle\n"
        "  fraud:\n"
        "    time: {column: t, unit: second, calendar: false}\n"
        "    amount: null\n"
        "    graph: {kind: account, src: payer}\n"
        "    entity: null\n"
        "    synthetic: true\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="needs `src` and `dst`"):
        DatasetRegistry(path).validate()


def test_a_context_declaring_an_absent_column_raises(
    fraud_raw_store: StoreConfig,
) -> None:
    """A block pointing at a column the raw frame lacks fails loudly."""
    registry = DatasetRegistry()
    entries = registry.entries
    entries["credit_card_fraud"] = {
        **entries["credit_card_fraud"],
        "fraud": {
            **entries["credit_card_fraud"]["fraud"],
            "amount": {"column": "NotAColumn"},
        },
    }
    with pytest.raises(PreprocessingError, match="NotAColumn"):
        offline_service(fraud_raw_store, registry).load("credit_card_fraud")


def test_an_unknown_artefact_name_is_refused(service: FraudService) -> None:
    """Only context, nodes and edges can be written."""
    from imbdata.exceptions import ImbdataError

    with pytest.raises(ImbdataError, match="Unknown fraud artefact"):
        service.store.write("paysim", {"sidecar": pd.DataFrame({"a": [1]})})


# ══════════════════════════════════════════════════════════════════════
# Integration — the real store (B5 of the 0.4.0 handoff)
# ══════════════════════════════════════════════════════════════════════

FRAUD_REFERENCE = {
    # key: (N, n_minority, has_graph)
    "credit_card_fraud": (284807, 492, False),
    "elliptic_bitcoin": (46564, 4545, True),
    "ieee_cis_fraud": (590540, 20663, False),
    "paysim": (6362620, 8213, False),
    "saml_d": (9504852, 9873, False),
}

ELLIPTIC_NODES = 203769
ELLIPTIC_EDGES = 234355


def cached(name: str) -> bool:
    """Return whether a dataset's canonical parquet and context are both cached."""
    config = StoreConfig()
    return (
        config.processed_path(name).is_file()
        and config.fraud_path(name, "context").is_file()
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", FRAUD_DATASETS)
def test_real_context_matches_the_canonical_frame(name: str) -> None:
    """On the real store, X, y and the context agree row by row."""
    if not cached(name):
        pytest.skip(f"'{name}' has no cached fraud context; run `imbdata fraud download {name}`")

    import imbdata
    from imbdata import fraud as fraud_module

    dataset = fraud_module.load(name)
    X, y = imbdata.load(name)
    rows, minority, has_graph = FRAUD_REFERENCE[name]

    assert dataset.X.equals(X) and dataset.y.equals(y)
    assert len(dataset.context) == len(X) == rows
    assert int(y.sum()) == minority
    assert dataset.context["t"].notna().all()
    assert (dataset.nodes is not None) is has_graph
    assert (dataset.edges is not None) is has_graph


@pytest.mark.slow
def test_real_elliptic_graph_keeps_every_node() -> None:
    """The graph tables hold all 203,769 nodes and 234,355 edges."""
    if not cached("elliptic_bitcoin"):
        pytest.skip("elliptic_bitcoin has no cached fraud context")

    from imbdata import fraud as fraud_module

    dataset = fraud_module.load("elliptic_bitcoin")
    assert len(dataset.nodes) == ELLIPTIC_NODES
    assert len(dataset.edges) == ELLIPTIC_EDGES
    assert int(dataset.nodes["row"].notna().sum()) == 46564
    assert int(dataset.nodes["label"].isna().sum()) == ELLIPTIC_NODES - 46564


@pytest.mark.slow
def test_real_saml_d_context_is_calendar_dated() -> None:
    """SAML-D's context carries the calendar instant and the typology."""
    if not cached("saml_d"):
        pytest.skip("saml_d has no cached fraud context")

    from imbdata import fraud as fraud_module

    dataset = fraud_module.load("saml_d")
    context = dataset.context
    assert context["event_time"].notna().all()
    assert str(context["event_time"].min().date()) == "2022-10-07"
    assert str(context["event_time"].max().date()) == "2023-08-23"
    assert context["typology"].nunique() == 28
    assert dataset.meta["fraud"]["drift_provenance"] == "researcher_constructed"
    assert context["src_id"].max() < 855460 and context["dst_id"].max() < 855460
