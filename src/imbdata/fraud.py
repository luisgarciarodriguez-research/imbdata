"""
imbdata.fraud — Fraud endpoint: per-row context beside the canonical format.

The canonical format is deliberately narrow: ``float64`` features plus a binary
``target``. That contract drops what a fraud study needs to reason about *when*
a transaction happened, *how much* it moved, and *who* paid *whom* — the
surrogate account keys, the calendar, the typology, and the graph. This module
serves those facts as a parallel artefact, aligned row by row with the
canonical frame, without touching the contract or the data any dataset key
already serves.

Only datasets declaring a ``fraud`` block in ``datasets.yaml`` are served. The
context is built from the very frame the canonical preprocessing consumes
(``DatasetPreprocessor.read_raw``), so the two cannot drift out of alignment.

Usage:
    from imbdata import fraud

    fraud.list_datasets()          # keys with a `fraud:` block
    ds = fraud.load("paysim")      # FraudDataset
    ds.X.equals(imbdata.load("paysim")[0])   # True
    ds.context[["t", "amount", "src_id", "dst_id"]].head()

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

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imbdata.api import DatasetService
from imbdata.config import StoreConfig, default_config
from imbdata.download import DownloadManager
from imbdata.exceptions import (
    ImbdataError,
    NotAFraudDatasetError,
    PreprocessingError,
)
from imbdata.preprocess import TARGET_COLUMN, DatasetPreprocessor, normalize_target
from imbdata.registry import DatasetRegistry, default_registry
from imbdata.verify import (
    STATUS_MISMATCH,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_UNKNOWN,
    ManifestManager,
    compute_sha256,
)

logger = logging.getLogger(__name__)

PARQUET_COMPRESSION = "snappy"

# Fixed schema of `context`. Every column exists for every dataset; a concept a
# dataset does not publish is all-null and declared as such in its `fraud`
# block, so a consumer can write one pipeline for all of them.
CONTEXT_SCHEMA: dict[str, str] = {
    "t": "float64",
    "t_seconds": "float64",
    "event_time": "datetime64[ns]",
    "amount": "float64",
    "currency": "string",
    "src_id": "Int64",
    "dst_id": "Int64",
    "entity_id": "Int64",
    "node_id": "Int64",
    "typology": "string",
}

NODES_SCHEMA: dict[str, str] = {
    "node_id": "Int64",
    "t": "float64",
    "label": "Int64",
    "row": "Int64",
}

EDGES_SCHEMA: dict[str, str] = {
    "src_node": "Int64",
    "dst_node": "Int64",
    "src_row": "Int64",
    "dst_row": "Int64",
}

# Seconds per native time unit. ``step`` is intentionally absent: Elliptic's
# two-week windows have no published duration, and inventing one would put a
# fabricated number in a column consumers would take as a fact.
SECONDS_PER_UNIT: dict[str, float] = {"second": 1.0, "hour": 3600.0}

PART_CONTEXT = "context"
PART_NODES = "nodes"
PART_EDGES = "edges"
PARTS = (PART_CONTEXT, PART_NODES, PART_EDGES)

__all__ = [
    "CONTEXT_SCHEMA",
    "NODES_SCHEMA",
    "EDGES_SCHEMA",
    "SECONDS_PER_UNIT",
    "FraudDataset",
    "FraudContextBuilder",
    "FraudStore",
    "FraudService",
    "encode_accounts",
    "to_seconds",
    "manifest_key",
    "list_datasets",
    "info",
    "load",
    "ensure",
    "verify",
    "default_service",
]


# ══════════════════════════════════════════════════════════════════════
# Pure functions
# ══════════════════════════════════════════════════════════════════════

def to_seconds(t: pd.Series, unit: str) -> pd.Series:
    """Convert a native time column to seconds, where the unit allows it.

    Args:
        t: Native time values.
        unit: Native unit: ``'second'``, ``'hour'``, or ``'step'``.

    Returns:
        A ``float64`` Series of seconds on the same origin as ``t``, or all
        ``NaN`` when the unit has no published duration (``'step'``).

    Example:
        >>> to_seconds(pd.Series([1.0, 2.0]), "hour").tolist()
        [3600.0, 7200.0]
        >>> to_seconds(pd.Series([1.0]), "step").isna().all()
        True
    """
    values = pd.to_numeric(t, errors="coerce").astype("float64")
    factor = SECONDS_PER_UNIT.get(unit)
    if factor is None:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    return (values * factor).astype("float64")


def encode_accounts(src: pd.Series, dst: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Encode payer and payee identifiers into one shared integer space.

    Both sides are factorized together over the sorted union of their values,
    so the same account carries the same code whether it pays or is paid, and
    the code does not depend on the order of the rows. No randomness is
    involved.

    Args:
        src: Payer identifiers.
        dst: Payee identifiers, aligned with ``src``.

    Returns:
        ``(src_id, dst_id)`` as ``Int64`` Series aligned with the inputs.

    Example:
        >>> a, b = encode_accounts(pd.Series(["C2", "C1"]), pd.Series(["C1", "C3"]))
        >>> a.tolist(), b.tolist()
        ([1, 0], [0, 2])
    """
    combined = pd.concat([src, dst], ignore_index=True)
    codes, _ = pd.factorize(combined, sort=True)
    encoded = pd.Series(codes, dtype="Int64")
    half = len(src)
    return (
        pd.Series(encoded.iloc[:half].to_numpy(), index=src.index, dtype="Int64"),
        pd.Series(encoded.iloc[half:].to_numpy(), index=dst.index, dtype="Int64"),
    )


def manifest_key(name: str, part: str) -> str:
    """Build the manifest key of one fraud artefact.

    Args:
        name: Dataset key.
        part: Artefact name (``'context'``, ``'nodes'``, ``'edges'``).

    Returns:
        ``'fraud/<name>.<part>'``, the namespace that keeps these entries out
        of the canonical datasets' verification report.
    """
    return f"fraud/{name}.{part}"


def _empty_column(length: int, dtype: str) -> pd.Series:
    """Return an all-null column of the requested dtype and length.

    Args:
        length: Number of rows.
        dtype: Target dtype from one of the schemas.

    Returns:
        A Series of nulls (``NaN``, ``NaT`` or ``pd.NA`` as the dtype dictates).
    """
    if dtype == "datetime64[ns]":
        return pd.Series(pd.NaT, index=pd.RangeIndex(length), dtype=dtype)
    if dtype == "float64":
        return pd.Series(np.nan, index=pd.RangeIndex(length), dtype=dtype)
    return pd.Series(pd.NA, index=pd.RangeIndex(length), dtype=dtype)


def _conform(frame: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Return ``frame`` with exactly the schema's columns, order, and dtypes.

    Args:
        frame: Frame under construction.
        schema: Mapping of column name to dtype.

    Returns:
        A new frame carrying every schema column, in schema order, with a
        default ``RangeIndex``.
    """
    conformed = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    for column, dtype in schema.items():
        if column in frame.columns:
            values = frame[column].reset_index(drop=True)
            conformed[column] = values.astype(dtype)
        else:
            conformed[column] = _empty_column(len(frame), dtype)
    return conformed


# ══════════════════════════════════════════════════════════════════════
# Domain entity
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FraudDataset:
    """A fraud dataset: the canonical frame plus its per-row context.

    ``X`` and ``y`` are exactly what :func:`imbdata.load` returns for the same
    key. ``context`` has one row per row of ``X``, in the same order, with the
    fixed schema of :data:`CONTEXT_SCHEMA`.

    Warning:
        ``context['typology']`` is derived from the label (SAML-D's
        ``Laundering_type`` names the generator that produced each suspicious
        transaction). It is served for stratification and error analysis and
        must never be used as a feature.

    Attributes:
        name: Dataset key.
        X: Canonical ``float64`` feature frame.
        y: Canonical ``int64`` target (``0`` = majority, ``1`` = minority).
        context: Per-row native context, aligned with ``X``.
        nodes: Full node table when the graph is a transaction graph
            (Elliptic), otherwise ``None``.
        edges: Edge list when the graph is not implicit in ``context``
            (Elliptic), otherwise ``None``.
        meta: The mapping :func:`info` returns for this dataset.

    Example:
        >>> ds = load("paysim")
        >>> len(ds.context) == len(ds.X)
        True
    """

    name: str
    X: pd.DataFrame
    y: pd.Series
    context: pd.DataFrame
    nodes: pd.DataFrame | None
    edges: pd.DataFrame | None
    meta: dict[str, Any]

    def __repr__(self) -> str:
        """Return an unambiguous representation of the dataset."""
        graph = "graph" if self.edges is not None else "no-graph"
        return (
            f"{type(self).__name__}(name={self.name!r}, N={len(self.X)}, "
            f"d={self.X.shape[1]}, {graph})"
        )


# ══════════════════════════════════════════════════════════════════════
# Context construction
# ══════════════════════════════════════════════════════════════════════

class FraudContextBuilder:
    """Builds the per-row context of a fraud dataset from its raw frame.

    One ``_context_<dataset_key>`` method exists per served dataset, dispatched
    like :class:`~imbdata.preprocess.DatasetPreprocessor`'s routines. Each one
    receives the frame of the dataset's shared reader — the same frame the
    canonical parquet is built from — so ``context`` and ``X`` share a row
    order by construction rather than by convention.

    Attributes:
        registry: Registry supplying the ``fraud`` block of each dataset.
        preprocessor: Owner of the raw readers, including the Elliptic edge
            list that no canonical routine touches.

    Example:
        >>> builder = FraudContextBuilder()
        >>> builder.supports("paysim")
        True
    """

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        preprocessor: DatasetPreprocessor | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            registry: Dataset registry. Defaults to the bundled registry.
            preprocessor: Preprocessor owning the raw readers. Defaults to a
                new instance.
        """
        self.registry = registry if registry is not None else default_registry()
        self.preprocessor = (
            preprocessor if preprocessor is not None else DatasetPreprocessor()
        )

    def __repr__(self) -> str:
        """Return an unambiguous representation of the builder."""
        return f"{type(self).__name__}(registry={self.registry!r})"

    # ── Dispatch ───────────────────────────────────────────────────────

    def method_for(self, name: str):
        """Return the context routine bound to a dataset key, if any.

        Args:
            name: Dataset key.

        Returns:
            The bound ``_context_<name>`` method, or ``None``.
        """
        return getattr(self, f"_context_{name}", None)

    def supports(self, name: str) -> bool:
        """Return whether a context routine exists for ``name``."""
        return self.method_for(name) is not None

    def build(
        self,
        name: str,
        frame: pd.DataFrame,
        meta: dict[str, Any],
        raw_dir: Path | str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Build the artefacts of one dataset from its raw frame.

        Args:
            name: Dataset key.
            frame: Frame returned by the dataset's shared reader.
            meta: Registry metadata, including the ``fraud`` block.
            raw_dir: Directory holding the raw files. Required only by the
                datasets whose graph lives in a file of its own (Elliptic).

        Returns:
            Mapping with a ``'context'`` frame and, for a transaction graph,
            ``'nodes'`` and ``'edges'`` frames.

        Raises:
            PreprocessingError: If the dataset has no context routine, the raw
                frame lacks a column the ``fraud`` block declares, or a needed
                raw directory is missing.
        """
        method = self.method_for(name)
        if method is None:
            raise PreprocessingError(
                f"No fraud context routine for '{name}'. Add a _context_{name}() method "
                f"to FraudContextBuilder."
            )
        logger.info(f"Building the fraud context of '{name}'")
        return method(frame, meta, Path(raw_dir) if raw_dir is not None else None)

    # ── Shared construction ────────────────────────────────────────────

    @staticmethod
    def _block(meta: dict[str, Any]) -> dict[str, Any]:
        """Return the ``fraud`` block of a metadata mapping.

        Args:
            meta: Registry metadata for a dataset.

        Returns:
            The ``fraud`` block.

        Raises:
            PreprocessingError: If the entry declares none.
        """
        block = meta.get("fraud")
        if not isinstance(block, dict):
            raise PreprocessingError(
                f"'{meta.get('key')}' declares no fraud block; it cannot be served by "
                f"imbdata.fraud."
            )
        return block

    def _generic_context(
        self,
        frame: pd.DataFrame,
        meta: dict[str, Any],
        keep: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Build the context columns every dataset shares.

        Args:
            frame: Raw frame from the shared reader.
            meta: Registry metadata, including the ``fraud`` block.
            keep: Optional boolean mask selecting the rows that survive into
                the canonical frame. ``None`` keeps every row.

        Returns:
            A frame with the schema of :data:`CONTEXT_SCHEMA`.

        Raises:
            PreprocessingError: If a declared column is absent from ``frame``.
        """
        block = self._block(meta)
        time = block["time"]
        columns = [time["column"]]
        amount = block.get("amount") or {}
        graph = block.get("graph") or {}
        entity = block.get("entity")
        typology = block.get("typology")
        for declared in (
            amount.get("column"), amount.get("currency"),
            graph.get("src"), graph.get("dst"), entity, typology,
        ):
            if declared:
                columns.append(str(declared))
        missing = [column for column in dict.fromkeys(columns) if column not in frame.columns]
        if missing:
            raise PreprocessingError(
                f"'{meta.get('key')}': the fraud block declares column(s) {missing}, which "
                f"the raw frame does not carry."
            )

        selected = frame if keep is None else frame.loc[keep]
        built = pd.DataFrame(index=pd.RangeIndex(len(selected)))

        t = pd.to_numeric(selected[time["column"]], errors="coerce").astype("float64")
        t = t.reset_index(drop=True)
        built["t"] = t
        built["t_seconds"] = to_seconds(t, str(time["unit"]))
        if time.get("calendar"):
            built["event_time"] = pd.to_datetime(built["t_seconds"], unit="s")

        if amount.get("column"):
            built["amount"] = (
                pd.to_numeric(selected[amount["column"]], errors="coerce")
                .astype("float64").reset_index(drop=True)
            )
        if amount.get("currency"):
            built["currency"] = (
                selected[amount["currency"]].astype("string").reset_index(drop=True)
            )

        src_id: pd.Series | None = None
        dst_id: pd.Series | None = None
        if graph.get("kind") == "account":
            src_id, dst_id = encode_accounts(
                selected[graph["src"]].reset_index(drop=True),
                selected[graph["dst"]].reset_index(drop=True),
            )
            built["src_id"] = src_id
            built["dst_id"] = dst_id

        if entity:
            if src_id is not None and entity == graph.get("src"):
                built["entity_id"] = src_id
            elif dst_id is not None and entity == graph.get("dst"):
                built["entity_id"] = dst_id
            else:
                codes, _ = pd.factorize(selected[entity].reset_index(drop=True), sort=True)
                built["entity_id"] = pd.Series(codes, dtype="Int64")

        if typology:
            built["typology"] = selected[typology].astype("string").reset_index(drop=True)

        context = _conform(built, CONTEXT_SCHEMA)
        if context["t"].isna().any():
            raise PreprocessingError(
                f"'{meta.get('key')}': {int(context['t'].isna().sum())} row(s) have no "
                f"native time value."
            )
        return context

    # ── Per-dataset routines ───────────────────────────────────────────

    def _context_credit_card_fraud(
        self, frame: pd.DataFrame, meta: dict[str, Any], raw_dir: Path | None = None
    ) -> dict[str, pd.DataFrame]:
        """Build the ULB credit card context: elapsed seconds and amount only.

        Args:
            frame: Raw frame from ``_read_credit_card_fraud``.
            meta: Registry metadata for the dataset.
            raw_dir: Unused; the context comes entirely from ``frame``.

        Returns:
            Mapping with the ``'context'`` frame.
        """
        return {PART_CONTEXT: self._generic_context(frame, meta)}

    def _context_paysim(
        self, frame: pd.DataFrame, meta: dict[str, Any], raw_dir: Path | None = None
    ) -> dict[str, pd.DataFrame]:
        """Build the PaySim context, whose rows are themselves the graph edges.

        ``nameOrig``/``nameDest`` become ``src_id``/``dst_id`` in one shared
        code space, so no separate node or edge table is needed.

        Args:
            frame: Raw frame from ``_read_paysim``.
            meta: Registry metadata for the dataset.
            raw_dir: Unused; the context comes entirely from ``frame``.

        Returns:
            Mapping with the ``'context'`` frame.
        """
        return {PART_CONTEXT: self._generic_context(frame, meta)}

    def _context_ieee_cis_fraud(
        self, frame: pd.DataFrame, meta: dict[str, Any], raw_dir: Path | None = None
    ) -> dict[str, pd.DataFrame]:
        """Build the IEEE-CIS context: seconds from an unpublished origin, plus amount.

        Args:
            frame: Raw frame from ``_read_ieee_cis_fraud``.
            meta: Registry metadata for the dataset.
            raw_dir: Unused; the context comes entirely from ``frame``.

        Returns:
            Mapping with the ``'context'`` frame.
        """
        return {PART_CONTEXT: self._generic_context(frame, meta)}

    def _context_saml_d(
        self, frame: pd.DataFrame, meta: dict[str, Any], raw_dir: Path | None = None
    ) -> dict[str, pd.DataFrame]:
        """Build the SAML-D context: calendar time, currency, accounts, typology.

        Args:
            frame: Raw frame from ``_read_saml_d``.
            meta: Registry metadata for the dataset.
            raw_dir: Unused; the context comes entirely from ``frame``.

        Returns:
            Mapping with the ``'context'`` frame.
        """
        return {PART_CONTEXT: self._generic_context(frame, meta)}

    def _context_elliptic_bitcoin(
        self, frame: pd.DataFrame, meta: dict[str, Any], raw_dir: Path | None = None
    ) -> dict[str, pd.DataFrame]:
        """Build the Elliptic context and its full transaction graph.

        The canonical frame keeps only the 46,564 labelled nodes, so the
        context is the same subset in the same order, with ``node_id`` carrying
        ``txId``. ``nodes`` and ``edges`` instead keep every node, ``unknown``
        included: dropping them would cut the graph the consumer measures. The
        ``row`` columns map a node to its position in ``X``, or ``<NA>`` when
        the node carries no label.

        Args:
            frame: Raw frame from ``_read_elliptic_bitcoin`` (all nodes).
            meta: Registry metadata for the dataset.
            raw_dir: Directory holding ``elliptic_txs_edgelist.csv``.

        Returns:
            Mapping with the ``'context'``, ``'nodes'`` and ``'edges'`` frames.

        Raises:
            PreprocessingError: If the edge list is absent or malformed.
        """
        block = self._block(meta)
        target_column = str(meta.get("target_column", "class"))
        labels = frame[target_column].astype(str).str.strip()
        label = normalize_target(
            labels,
            str(meta.get("minority_value", "1")),
            majority_value=str(meta.get("majority_value", "2")),
        )
        keep = label.notna()

        context = self._generic_context(frame, meta, keep=keep)
        context["node_id"] = (
            pd.to_numeric(frame.loc[keep, "txId"], errors="coerce")
            .astype("Int64").reset_index(drop=True)
        )
        context = _conform(context, CONTEXT_SCHEMA)

        # Position of every node in X/context, <NA> for the unlabelled ones.
        row = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        row.loc[keep] = np.arange(int(keep.sum()))
        nodes = _conform(
            pd.DataFrame(
                {
                    "node_id": pd.to_numeric(frame["txId"], errors="coerce"),
                    "t": pd.to_numeric(frame[block["time"]["column"]], errors="coerce"),
                    "label": label.astype("Int64"),
                    "row": row,
                }
            ),
            NODES_SCHEMA,
        )

        if raw_dir is None:
            raise PreprocessingError(
                "'elliptic_bitcoin': the raw directory is required to read the edge list."
            )
        edge_list = self.preprocessor.read_elliptic_edges(raw_dir)
        position = nodes.set_index("node_id")["row"]
        src_node = pd.to_numeric(edge_list["txId1"], errors="coerce").astype("Int64")
        dst_node = pd.to_numeric(edge_list["txId2"], errors="coerce").astype("Int64")
        edges = _conform(
            pd.DataFrame(
                {
                    "src_node": src_node,
                    "dst_node": dst_node,
                    "src_row": src_node.map(position).astype("Int64"),
                    "dst_row": dst_node.map(position).astype("Int64"),
                }
            ),
            EDGES_SCHEMA,
        )
        logger.info(
            f"elliptic_bitcoin: {len(context)} labelled rows, {len(nodes)} nodes, "
            f"{len(edges)} edges"
        )
        return {PART_CONTEXT: context, PART_NODES: nodes, PART_EDGES: edges}


# ══════════════════════════════════════════════════════════════════════
# Storage
# ══════════════════════════════════════════════════════════════════════

class FraudStore:
    """Owns the fraud artefacts on disk: their paths, writing, and verification.

    Artefacts live in ``<store>/processed/fraud/`` and are recorded in the
    store's single ``manifest.json`` under the ``fraud/`` namespace, so
    ``imbdata.verify()`` keeps reporting exactly the canonical datasets.
    Like the canonical parquet files they are immutable once written and are
    only rebuilt on an explicit ``force``.

    Attributes:
        config: Store configuration resolving all filesystem paths.
        manifest: Manifest manager recording digests.

    Example:
        >>> store = FraudStore()
        >>> store.path("paysim", "context").name
        'paysim.context.parquet'
    """

    def __init__(
        self,
        config: StoreConfig | None = None,
        manifest: ManifestManager | None = None,
    ) -> None:
        """Initialize the store.

        Args:
            config: Store configuration. Defaults to the process-wide one.
            manifest: Manifest manager. Defaults to one bound to ``config``.
        """
        self.config = config if config is not None else default_config()
        self.manifest = manifest if manifest is not None else ManifestManager(self.config)

    def __repr__(self) -> str:
        """Return an unambiguous representation of the store."""
        return f"{type(self).__name__}(store={str(self.config.store_path())!r})"

    def path(self, name: str, part: str) -> Path:
        """Return the path of one artefact.

        Args:
            name: Dataset key.
            part: Artefact name (``'context'``, ``'nodes'``, ``'edges'``).

        Returns:
            Path to the parquet file, whether or not it exists.
        """
        return self.config.fraud_path(name, part)

    def parts_of(self, name: str, meta: dict[str, Any]) -> list[str]:
        """Return the artefacts a dataset is expected to have.

        Args:
            name: Dataset key.
            meta: Registry metadata, including the ``fraud`` block.

        Returns:
            ``['context']``, plus ``'nodes'`` and ``'edges'`` when the dataset
            declares a transaction graph.
        """
        graph = (meta.get("fraud") or {}).get("graph") or {}
        if graph.get("kind") == "transaction":
            return [PART_CONTEXT, PART_NODES, PART_EDGES]
        return [PART_CONTEXT]

    def has_all(self, name: str, meta: dict[str, Any]) -> bool:
        """Return whether every expected artefact of a dataset is on disk.

        Args:
            name: Dataset key.
            meta: Registry metadata, including the ``fraud`` block.

        Returns:
            ``True`` when nothing needs to be built.
        """
        return all(self.path(name, part).is_file() for part in self.parts_of(name, meta))

    def write(self, name: str, artefacts: dict[str, pd.DataFrame]) -> dict[str, Path]:
        """Serialize a dataset's artefacts and record them in the manifest.

        Args:
            name: Dataset key.
            artefacts: Mapping of part name to frame.

        Returns:
            Mapping of part name to the path written.

        Raises:
            ImbdataError: If an unknown part name is supplied.
        """
        unknown = [part for part in artefacts if part not in PARTS]
        if unknown:
            raise ImbdataError(f"Unknown fraud artefact(s) {unknown} for '{name}'.")

        self.config.fraud_dir(create=True)
        written: dict[str, Path] = {}
        for part, frame in artefacts.items():
            path = self.path(name, part)
            frame.to_parquet(path, compression=PARQUET_COMPRESSION, index=False)
            self.manifest.update(
                manifest_key(name, part),
                path,
                dataset=name,
                part=part,
                n_rows=len(frame),
            )
            written[part] = path
            logger.info(f"Wrote {path.name}: {len(frame)} row(s)")
        return written

    def read(self, name: str, part: str) -> pd.DataFrame:
        """Read one artefact from disk.

        Args:
            name: Dataset key.
            part: Artefact name.

        Returns:
            The artefact as stored.

        Raises:
            ImbdataError: If the file is not cached.
        """
        path = self.path(name, part)
        if not path.is_file():
            raise ImbdataError(f"Fraud artefact '{manifest_key(name, part)}' is not cached.")
        return pd.read_parquet(path)

    def verify(self, name: str, part: str) -> str:
        """Verify one artefact against its recorded digest.

        Resolves the file through the manifest entry's own ``path``, because
        :meth:`imbdata.verify.ManifestManager.verify` resolves canonical
        dataset paths, which these artefacts are not.

        Args:
            name: Dataset key.
            part: Artefact name.

        Returns:
            ``'OK'``, ``'MISSING'``, ``'NOT_IN_MANIFEST'`` or
            ``'HASH_MISMATCH'``.
        """
        key = manifest_key(name, part)
        recorded = self.manifest.read().get(key)
        path = Path(recorded["path"]) if recorded and recorded.get("path") else self.path(
            name, part
        )
        if not path.is_file():
            return STATUS_MISSING
        if recorded is None or "sha256_processed" not in recorded:
            return STATUS_UNKNOWN
        if compute_sha256(path) != recorded["sha256_processed"]:
            logger.warning(f"SHA-256 mismatch for '{key}' at {path}")
            return STATUS_MISMATCH
        return STATUS_OK


# ══════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════

class FraudService:
    """Serves fraud datasets: canonical frame, context, and graph.

    Materialization is lazy and reuses everything already cached: the
    canonical parquet through :class:`~imbdata.api.DatasetService`, the raw
    files through :class:`~imbdata.download.DownloadManager`, and the context
    artefacts through :class:`FraudStore`. The raw files are only re-read when
    an artefact is missing or ``force`` is set.

    Attributes:
        config: Store configuration.
        registry: Dataset metadata source.
        datasets: Canonical dataset service.
        downloader: Engine dispatcher used to fetch raw files.
        preprocessor: Owner of the shared raw readers.
        builder: Context builder.
        store: Fraud artefact store.

    Example:
        >>> service = FraudService()
        >>> "saml_d" in service.list_datasets()
        True
    """

    def __init__(
        self,
        config: StoreConfig | None = None,
        registry: DatasetRegistry | None = None,
        datasets: DatasetService | None = None,
        downloader: DownloadManager | None = None,
        preprocessor: DatasetPreprocessor | None = None,
        builder: FraudContextBuilder | None = None,
        store: FraudStore | None = None,
    ) -> None:
        """Initialize the service and its collaborators.

        Args:
            config: Store configuration. Defaults to the process-wide one.
            registry: Dataset registry. Defaults to the bundled registry.
            datasets: Canonical dataset service. Defaults to one bound to
                ``config`` and ``registry``.
            downloader: Download dispatcher. Defaults to a new manager.
            preprocessor: Preprocessor owning the shared readers.
            builder: Context builder. Defaults to a new instance.
            store: Fraud artefact store. Defaults to one bound to ``config``.
        """
        self.config = config if config is not None else default_config()
        self.registry = registry if registry is not None else default_registry()
        self.datasets = datasets if datasets is not None else DatasetService(
            config=self.config, registry=self.registry
        )
        self.downloader = downloader if downloader is not None else DownloadManager(self.config)
        self.preprocessor = (
            preprocessor if preprocessor is not None else self.datasets.preprocessor
        )
        self.builder = builder if builder is not None else FraudContextBuilder(
            self.registry, self.preprocessor
        )
        self.store = store if store is not None else FraudStore(self.config)

    def __repr__(self) -> str:
        """Return an unambiguous representation of the service."""
        return f"{type(self).__name__}(store={str(self.config.store_path())!r})"

    # ── Discovery ──────────────────────────────────────────────────────

    def list_datasets(self) -> list[str]:
        """Return the dataset keys the fraud endpoint serves, sorted.

        Returns:
            The keys declaring a ``fraud`` block in the registry.
        """
        return self.registry.filter(fraud=True)

    def _fraud_meta(self, name: str) -> dict[str, Any]:
        """Return the registry metadata of a served dataset.

        Args:
            name: Dataset key.

        Returns:
            The metadata mapping, ``fraud`` block included.

        Raises:
            DatasetNotFoundError: If ``name`` is not registered.
            NotAFraudDatasetError: If it declares no ``fraud`` block.
        """
        meta = self.registry.get(name)
        if not meta.get("fraud"):
            raise NotAFraudDatasetError(
                f"'{name}' is registered but declares no fraud block, so imbdata.fraud "
                f"cannot serve it. Served datasets: {self.list_datasets()}"
            )
        return meta

    def info(self, name: str) -> dict[str, Any]:
        """Return a served dataset's metadata plus its ``fraud`` block.

        Args:
            name: Dataset key.

        Returns:
            Everything :func:`imbdata.info` reports, plus ``'fraud'`` (the
            registry block) and ``'fraud_artefacts'`` (part name to
            ``True``/``False`` for cached or not).

        Raises:
            DatasetNotFoundError: If ``name`` is not registered.
            NotAFraudDatasetError: If it declares no ``fraud`` block.
        """
        meta = self._fraud_meta(name)
        details = self.datasets.info(name)
        details["fraud"] = copy.deepcopy(meta["fraud"])
        details["fraud_artefacts"] = {
            part: self.store.path(name, part).is_file()
            for part in self.store.parts_of(name, meta)
        }
        return details

    # ── Materialization ────────────────────────────────────────────────

    def ensure_one(self, name: str, force: bool = False) -> dict[str, Path]:
        """Guarantee that a dataset's canonical parquet and context exist.

        Args:
            name: Dataset key.
            force: Rebuild the context artefacts even when cached.

        Returns:
            Mapping of part name to its path on disk.

        Raises:
            DatasetNotFoundError: If ``name`` is not registered.
            NotAFraudDatasetError: If it declares no ``fraud`` block.
            DownloadError: If the raw files cannot be retrieved.
            PreprocessingError: If the context cannot be built.
        """
        meta = self._fraud_meta(name)
        self.datasets.ensure_one(name, force=force)

        parts = self.store.parts_of(name, meta)
        if not force and self.store.has_all(name, meta):
            return {part: self.store.path(name, part) for part in parts}

        raw_dir = self.downloader.download(name, meta, self.config.raw_dir(name))
        frame = self.preprocessor.read_raw(name, raw_dir, meta)
        artefacts = self.builder.build(name, frame, meta, raw_dir=raw_dir)
        del frame

        missing = [part for part in parts if part not in artefacts]
        if missing:
            raise PreprocessingError(
                f"'{name}': the context routine produced no {', '.join(missing)} artefact."
            )
        return self.store.write(name, artefacts)

    def load(self, name: str, force: bool = False) -> FraudDataset:
        """Load a fraud dataset, materializing what is missing.

        Args:
            name: Dataset key.
            force: Rebuild the context artefacts even when cached.

        Returns:
            A :class:`FraudDataset` whose ``X`` and ``y`` are identical to
            :func:`imbdata.load`'s for the same key.

        Raises:
            DatasetNotFoundError: If ``name`` is not registered.
            NotAFraudDatasetError: If it declares no ``fraud`` block.
            DownloadError: If the raw files cannot be retrieved.
            PreprocessingError: If the context cannot be built.
            ImbdataError: If a written artefact cannot be read back.

        Example:
            >>> ds = load("credit_card_fraud")
            >>> ds.context["amount"].notna().all()
            True
        """
        meta = self._fraud_meta(name)
        self.ensure_one(name, force=force)

        frame = pd.read_parquet(self.config.processed_path(name))
        y = frame.pop(TARGET_COLUMN)
        y.name = TARGET_COLUMN

        context = self.store.read(name, PART_CONTEXT)
        if len(context) != len(frame):
            raise PreprocessingError(
                f"'{name}': the context has {len(context)} row(s) against {len(frame)} in the "
                f"canonical frame. Rebuild it with force=True."
            )
        parts = self.store.parts_of(name, meta)
        nodes = self.store.read(name, PART_NODES) if PART_NODES in parts else None
        edges = self.store.read(name, PART_EDGES) if PART_EDGES in parts else None

        return FraudDataset(
            name=name,
            X=frame,
            y=y,
            context=context,
            nodes=nodes,
            edges=edges,
            meta=self.info(name),
        )

    def ensure(self, names: list[str] | str | None = None, force: bool = False
               ) -> dict[str, str]:
        """Pre-build the context of several datasets in bulk.

        Failures are captured per dataset rather than aborting the batch.

        Args:
            names: Dataset keys. A bare string is accepted. Defaults to every
                served dataset.
            force: Rebuild artefacts that are already cached.

        Returns:
            Mapping of dataset key to ``'OK (cached)'``, ``'BUILT'``, or
            ``'ERROR: <reason>'``.
        """
        if names is None:
            names = self.list_datasets()
        elif isinstance(names, str):
            names = [names]

        results: dict[str, str] = {}
        for name in names:
            try:
                meta = self._fraud_meta(name)
                cached = self.store.has_all(name, meta) and self.config.processed_path(
                    name
                ).is_file()
                self.ensure_one(name, force=force)
                results[name] = "OK (cached)" if cached and not force else "BUILT"
            except (ImbdataError, OSError, ValueError) as exc:
                results[name] = f"ERROR: {exc}"
                logger.error(f"Failed to ensure the fraud context of '{name}': {exc}")
        return results

    def verify(self, names: list[str] | None = None) -> dict[str, str]:
        """Verify the fraud artefacts against the manifest's digests.

        Args:
            names: Dataset keys to check. Defaults to every served dataset.

        Returns:
            Mapping of artefact key (``'fraud/<name>.<part>'``) to ``'OK'``,
            ``'MISSING'``, ``'NOT_IN_MANIFEST'`` or ``'HASH_MISMATCH'``.
        """
        keys = names if names is not None else self.list_datasets()
        report: dict[str, str] = {}
        for name in sorted(keys):
            meta = self._fraud_meta(name)
            for part in self.store.parts_of(name, meta):
                report[manifest_key(name, part)] = self.store.verify(name, part)
        return report


# ── Functional facade over a lazily-created default service ────────────

_DEFAULT_SERVICE: FraudService | None = None


def default_service(refresh: bool = False) -> FraudService:
    """Return the process-wide default :class:`FraudService`.

    Args:
        refresh: Rebuild the service, re-resolving the store configuration.

    Returns:
        The shared service instance.
    """
    global _DEFAULT_SERVICE
    if refresh or _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = FraudService()
    return _DEFAULT_SERVICE


def list_datasets() -> list[str]:
    """Return the dataset keys the fraud endpoint serves.

    Returns:
        Sorted keys declaring a ``fraud`` block.

    Example:
        >>> list_datasets()
        ['credit_card_fraud', 'elliptic_bitcoin', 'ieee_cis_fraud', 'paysim', 'saml_d']
    """
    return default_service().list_datasets()


def info(name: str) -> dict[str, Any]:
    """Return a served dataset's metadata plus its ``fraud`` block.

    Args:
        name: Dataset key.

    Returns:
        The metadata mapping.

    Raises:
        DatasetNotFoundError: If ``name`` is not registered.
        NotAFraudDatasetError: If it declares no ``fraud`` block.
    """
    return default_service().info(name)


def load(name: str, force: bool = False) -> FraudDataset:
    """Load a fraud dataset with its per-row context.

    Args:
        name: Dataset key.
        force: Rebuild the context artefacts even when cached.

    Returns:
        A :class:`FraudDataset`.

    Raises:
        DatasetNotFoundError: If ``name`` is not registered.
        NotAFraudDatasetError: If it declares no ``fraud`` block.
        DownloadError: If the raw files cannot be retrieved.
        PreprocessingError: If the context cannot be built.
    """
    return default_service().load(name, force=force)


def ensure(names: list[str] | str | None = None, force: bool = False) -> dict[str, str]:
    """Pre-build the fraud context of several datasets.

    Args:
        names: Dataset keys. Defaults to every served dataset.
        force: Rebuild artefacts that are already cached.

    Returns:
        Mapping of dataset key to its outcome string.

    Example:
        >>> ensure(["credit_card_fraud"])
        {'credit_card_fraud': 'OK (cached)'}
    """
    return default_service().ensure(names, force=force)


def verify(names: list[str] | None = None) -> dict[str, str]:
    """Verify the cached fraud artefacts against the manifest.

    Args:
        names: Dataset keys to check. Defaults to every served dataset.

    Returns:
        Mapping of artefact key to its verification status.
    """
    return default_service().verify(names)
