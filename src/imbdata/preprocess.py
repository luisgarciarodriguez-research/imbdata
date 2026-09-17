"""
imbdata.preprocess — Per-dataset preprocessing pipelines.

Turns the heterogeneous raw files under ``<store>/raw/`` into the canonical
parquet format that every consumer project depends on:

* all feature columns are ``float64``;
* a single ``target`` column of ``int64`` with ``0`` = majority, ``1`` = minority;
* no missing values, no infinities, no categorical columns, default RangeIndex.

The stateful dispatch lives in :class:`DatasetPreprocessor`, which owns one
``_preprocess_<key>`` method per dataset. The transformations those methods
compose (:func:`binarize_column`, :func:`onehot_encode`, :func:`impute_median`,
and friends) are pure functions: they take frames and return new frames without
touching the filesystem or mutating their inputs.

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

import logging
import tarfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from imbdata.exceptions import PreprocessingError

logger = logging.getLogger(__name__)

TARGET_COLUMN = "target"
PARQUET_COMPRESSION = "snappy"
MISSING_TOKENS = ("?", "NA", "N/A", "na", "nan", "NaN", "", " ", "-")

# The 43 columns of NSL-KDD's KDDTrain+/KDDTest+ flat files: 41 features, the
# attack label, and the difficulty score the registry drops.
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

__all__ = [
    "TARGET_COLUMN",
    "binarize_column",
    "binarize_at_most",
    "normalize_target",
    "onehot_encode",
    "ordinal_encode",
    "impute_median",
    "impute_mode",
    "drop_high_missing",
    "drop_constant_columns",
    "select_top_variance",
    "extract_mvts_features",
    "extract_spectral_features",
    "extract_time_domain_features",
    "segment_signal",
    "TIME_DOMAIN_STATISTICS",
    "MVTS_STATISTICS",
    "to_numeric_frame",
    "assemble_canonical",
    "DatasetPreprocessor",
    "preprocess_dataset",
]


# ══════════════════════════════════════════════════════════════════════
# Pure transformations
# ══════════════════════════════════════════════════════════════════════

def binarize_column(series: pd.Series, minority_value: Any) -> pd.Series:
    """Map a label column to the canonical binary encoding.

    Rows equal to ``minority_value`` become ``1``; every other row becomes
    ``0``. Comparison is attempted on the native dtype first and falls back to
    a whitespace-stripped string comparison, which absorbs the trailing spaces
    and quoting habits of the UCI and KEEL flat files.

    Args:
        series: Raw label column.
        minority_value: Value that identifies the positive (minority) class.

    Returns:
        An ``int64`` Series of zeros and ones, aligned with ``series``.

    Raises:
        PreprocessingError: If no row matches ``minority_value``.

    Example:
        >>> binarize_column(pd.Series(["B", "M", "B"]), "M").tolist()
        [0, 1, 0]
    """
    mask = series == minority_value
    if not mask.any():
        as_text = series.astype(str).str.strip().str.strip("'\"")
        mask = as_text == str(minority_value).strip()
    if not mask.any():
        observed = sorted(series.astype(str).str.strip().unique())[:12]
        raise PreprocessingError(
            f"Minority value {minority_value!r} never occurs in the target column. "
            f"Observed values: {observed}"
        )
    return mask.astype("int64")


def binarize_at_most(series: pd.Series, threshold: float) -> pd.Series:
    """Map values at or below ``threshold`` to 1 and every other row to 0.

    Complements :func:`binarize_column`, which tests equality against a single
    value. Ordinal scores such as the UCI wine quality grade define their
    minority class as a tail rather than one level, and a tail cannot be
    expressed as an equality.

    Args:
        series: Raw ordinal label column.
        threshold: Highest value that still belongs to the minority class.

    Returns:
        An ``int64`` Series of zeros and ones, aligned with ``series``.

    Raises:
        PreprocessingError: If no row falls at or below ``threshold``.

    Example:
        >>> binarize_at_most(pd.Series([3, 5, 8, 4]), 4).tolist()
        [1, 0, 0, 1]
    """
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric <= threshold
    if not mask.any():
        raise PreprocessingError(
            f"No row scores at or below {threshold}; "
            f"observed range {numeric.min()}-{numeric.max()}"
        )
    return mask.astype("int64")


def normalize_target(
    y: pd.Series,
    minority_value: Any,
    majority_value: Any = None,
) -> pd.Series:
    """Binarize a label column, optionally restricting it to two known labels.

    When ``majority_value`` is supplied, rows carrying neither label are marked
    with ``pandas.NA`` so the caller can drop them together with their features
    (this is how ``elliptic_bitcoin`` discards its unlabeled nodes).

    Args:
        y: Raw label column.
        minority_value: Value identifying the positive (minority) class.
        majority_value: Value identifying the negative class. When ``None``,
            every non-minority row is treated as majority.

    Returns:
        A Series of ``0``/``1``, with ``pandas.NA`` for rows belonging to
        neither declared label.

    Example:
        >>> normalize_target(pd.Series([1, 2, 3]), 2, majority_value=1).tolist()
        [0, 1, <NA>]
    """
    binary = binarize_column(y, minority_value)
    if majority_value is None:
        return binary

    known = (y == minority_value) | (y == majority_value)
    if not known.any():
        as_text = y.astype(str).str.strip().str.strip("'\"")
        known = as_text.isin({str(minority_value).strip(), str(majority_value).strip()})
    return binary.astype("Int64").where(known)


def onehot_encode(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """One-hot encode the requested categorical columns.

    Columns absent from ``df`` are ignored, so a registry entry may list the
    superset of categoricals that appear across a dataset's variants.

    Args:
        df: Feature frame.
        columns: Names of the columns to expand.

    Returns:
        A new frame where each listed column is replaced by its indicator
        columns, named ``'<column>=<level>'``.

    Example:
        >>> onehot_encode(pd.DataFrame({"t": ["a", "b"]}), ["t"]).columns.tolist()
        ['t=a', 't=b']
    """
    present = [column for column in columns if column in df.columns]
    if not present:
        return df.copy()
    encoded = pd.get_dummies(df, columns=present, prefix=present, prefix_sep="=", dtype="float64")
    return encoded


def ordinal_encode(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ordinal-encode the requested categorical columns.

    Levels are numbered by sorted order so that the mapping is deterministic
    across machines and runs.

    Args:
        df: Feature frame.
        columns: Names of the columns to encode in place.

    Returns:
        A new frame with the listed columns replaced by their integer codes.

    Example:
        >>> ordinal_encode(pd.DataFrame({"s": ["M", "F", "I"]}), ["s"])["s"].tolist()
        [2.0, 0.0, 1.0]
    """
    encoded = df.copy()
    for column in columns:
        if column not in encoded.columns:
            continue
        levels = sorted(encoded[column].astype(str).unique())
        mapping = {level: float(index) for index, level in enumerate(levels)}
        encoded[column] = encoded[column].astype(str).map(mapping)
    return encoded


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values in every numeric column with its median.

    Columns that are entirely missing have no median; they are filled with
    ``0.0`` so that the canonical contract (no NaN) still holds.

    Args:
        df: Numeric feature frame.

    Returns:
        A new frame with no missing values.

    Example:
        >>> impute_median(pd.DataFrame({"a": [1.0, None, 3.0]}))["a"].tolist()
        [1.0, 2.0, 3.0]
    """
    filled = df.copy()
    medians = filled.median(numeric_only=True)
    for column in filled.columns:
        median = medians.get(column)
        if median is not None and pd.notna(median):
            filled[column] = filled[column].fillna(median)
        elif filled[column].isna().all():
            filled[column] = 0.0
    return filled


def impute_mode(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Replace missing values with the most frequent value of each column.

    Args:
        df: Feature frame, typically holding categorical columns.
        columns: Columns to impute. Defaults to every column of ``df``.

    Returns:
        A new frame with the listed columns imputed.

    Example:
        >>> impute_mode(pd.DataFrame({"c": ["a", None, "a"]}))["c"].tolist()
        ['a', 'a', 'a']
    """
    filled = df.copy()
    for column in list(columns) if columns is not None else list(filled.columns):
        if column not in filled.columns:
            continue
        modes = filled[column].mode(dropna=True)
        if not modes.empty:
            filled[column] = filled[column].fillna(modes.iloc[0])
    return filled


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns whose missing-value fraction exceeds ``threshold``.

    Args:
        df: Feature frame.
        threshold: Maximum tolerated fraction of missing values, in ``[0, 1]``.

    Returns:
        A new frame without the over-sparse columns.

    Raises:
        ValueError: If ``threshold`` is outside ``[0, 1]``.

    Example:
        >>> drop_high_missing(pd.DataFrame({"a": [1, 2], "b": [None, None]})).columns.tolist()
        ['a']
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    if df.empty:
        return df.copy()
    keep = df.columns[df.isna().mean() <= threshold]
    dropped = len(df.columns) - len(keep)
    if dropped:
        logger.debug(f"Dropped {dropped} column(s) with >{threshold:.0%} missing values")
    return df[keep].copy()


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that carry a single distinct value.

    Zero-variance columns are uninformative for every complexity measure and
    break correlation-based ones, so they are removed before serialization.

    Args:
        df: Feature frame.

    Returns:
        A new frame without the constant columns.

    Example:
        >>> drop_constant_columns(pd.DataFrame({"a": [1, 2], "k": [7, 7]})).columns.tolist()
        ['a']
    """
    if df.empty:
        return df.copy()
    varying = [column for column in df.columns if df[column].nunique(dropna=False) > 1]
    dropped = len(df.columns) - len(varying)
    if dropped:
        logger.debug(f"Dropped {dropped} constant column(s)")
    return df[varying].copy()


def to_numeric_frame(df: pd.DataFrame, missing_tokens: Sequence[str] = MISSING_TOKENS
                     ) -> pd.DataFrame:
    """Coerce every column to ``float64``, mapping missing tokens to NaN.

    Flat files from UCI and KEEL encode missing values as ``'?'`` and pad
    fields with spaces; both are normalized here before the numeric cast.

    Args:
        df: Raw frame with possibly textual numeric columns.
        missing_tokens: Strings to treat as missing.

    Returns:
        A new frame with every column of dtype ``float64``.
    """
    numeric = df.copy()
    for column in numeric.columns:
        series = numeric[column]
        if series.dtype == object:
            series = series.astype(str).str.strip().replace(list(missing_tokens), np.nan)
        numeric[column] = pd.to_numeric(series, errors="coerce").astype("float64")
    return numeric


def select_top_variance(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep the ``top_k`` columns with the highest variance.

    Used for the high-dimensional gene-expression datasets, where distance-based
    complexity measures need a tractable feature count. Ties are broken by
    column order, so the selection is deterministic across runs.

    Args:
        df: Numeric feature frame.
        top_k: Number of columns to retain. Values at or above the current
            width return the frame unchanged.

    Returns:
        A new frame holding the selected columns, in their original order.

    Raises:
        ValueError: If ``top_k`` is not positive.

    Example:
        >>> select_top_variance(pd.DataFrame({"a": [1, 1], "b": [0, 9]}), 1).columns.tolist()
        ['b']
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if top_k >= df.shape[1]:
        return df.copy()
    variances = df.var(numeric_only=True)
    keep = set(variances.nlargest(top_k).index)
    return df[[column for column in df.columns if column in keep]].copy()


MVTS_STATISTICS = ("mean", "std", "min", "max", "skewness", "kurtosis", "slope", "autocorr_lag1")


def extract_mvts_features(mvts: np.ndarray, statistics: Sequence[str] = MVTS_STATISTICS
                          ) -> np.ndarray:
    """Summarize a multivariate time series into per-parameter statistics.

    Collapses a ``(timesteps, parameters)`` slice into one scalar per
    (parameter, statistic) pair, which is how SWAN-SF's 24 magnetic field
    parameters over 60 timesteps become the 192 tabular features the registry
    declares. Missing timesteps are ignored per parameter; a parameter observed
    fewer than two times yields zeros rather than NaN, so the canonical
    contract survives sparse instances.

    Args:
        mvts: Array of shape ``(timesteps, parameters)``.
        statistics: Statistics to compute, in output order. Supported names are
            ``mean``, ``std``, ``min``, ``max``, ``skewness``, ``kurtosis``,
            ``slope`` (least-squares trend over the timestep index), and
            ``autocorr_lag1`` (lag-1 Pearson autocorrelation).

    Returns:
        A 1-D array of length ``parameters * len(statistics)``, ordered
        parameter-major: all statistics of parameter 1, then of parameter 2, …

    Raises:
        ValueError: If ``mvts`` is not two-dimensional, or a statistic is not
            supported.

    Example:
        >>> extract_mvts_features(np.arange(6.0).reshape(3, 2), ["mean"]).tolist()
        [2.0, 3.0]
    """
    series = np.asarray(mvts, dtype="float64")
    if series.ndim != 2:
        raise ValueError(f"mvts must be 2-D (timesteps, parameters), got shape {series.shape}")

    unsupported = [name for name in statistics if name not in MVTS_STATISTICS]
    if unsupported:
        raise ValueError(
            f"Unsupported statistic(s) {unsupported}; supported: {list(MVTS_STATISTICS)}"
        )

    features: list[float] = []
    for column in range(series.shape[1]):
        observed = series[:, column]
        observed = observed[np.isfinite(observed)]
        features.extend(_column_statistics(observed, statistics))
    return np.asarray(features, dtype="float64")


def _column_statistics(values: np.ndarray, statistics: Sequence[str]) -> list[float]:
    """Compute the requested statistics of a single parameter's observations.

    Args:
        values: Finite observations of one parameter, in time order.
        statistics: Statistic names to compute, in output order.

    Returns:
        One float per requested statistic; all zeros when fewer than two
        observations are available.
    """
    if values.size < 2:
        return [0.0] * len(statistics)

    mean = float(values.mean())
    std = float(values.std())
    computed: list[float] = []
    for name in statistics:
        if name == "mean":
            computed.append(mean)
        elif name == "std":
            computed.append(std)
        elif name == "min":
            computed.append(float(values.min()))
        elif name == "max":
            computed.append(float(values.max()))
        elif name == "skewness":
            computed.append(float(((values - mean) ** 3).mean() / std ** 3) if std else 0.0)
        elif name == "kurtosis":
            computed.append(float(((values - mean) ** 4).mean() / std ** 4) if std else 0.0)
        elif name == "slope":
            index = np.arange(values.size, dtype="float64")
            variance = float(((index - index.mean()) ** 2).sum())
            covariance = float(((index - index.mean()) * (values - mean)).sum())
            computed.append(covariance / variance if variance else 0.0)
        else:  # autocorr_lag1
            head, tail = values[:-1], values[1:]
            denominator = float(head.std() * tail.std())
            if denominator:
                covariance = float(((head - head.mean()) * (tail - tail.mean())).mean())
                computed.append(covariance / denominator)
            else:
                computed.append(0.0)
    return [value if np.isfinite(value) else 0.0 for value in computed]


TIME_DOMAIN_STATISTICS = (
    "max", "min", "mean", "sd", "rms", "skewness", "kurtosis", "crest", "form",
)


def segment_signal(
    signal: np.ndarray,
    window: int,
    overlap: int = 0,
    limit: int | None = None,
) -> np.ndarray:
    """Cut a 1-D signal into fixed-length windows.

    Vibration datasets ship one continuous recording per operating condition;
    the rows of the tabular dataset only come into existence once a window
    length is chosen, so this function is where a recording's N is decided.

    Args:
        signal: One-dimensional recording.
        window: Number of samples per window.
        overlap: Number of samples shared by consecutive windows. ``0`` gives
            non-overlapping segmentation.
        limit: Keep at most this many leading windows. ``None`` keeps all.

    Returns:
        An array of shape ``(n_windows, window)``.

    Raises:
        ValueError: If ``window`` is not positive, ``overlap`` is negative or
            not smaller than ``window``, or the signal is shorter than one
            window.

    Example:
        >>> segment_signal(np.arange(10.0), window=4).shape
        (2, 4)
    """
    series = np.asarray(signal, dtype="float64").ravel()
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if not 0 <= overlap < window:
        raise ValueError(f"overlap must lie in [0, window), got {overlap}")
    if series.size < window:
        raise ValueError(f"signal of {series.size} sample(s) is shorter than one window")

    step = window - overlap
    n_windows = 1 + (series.size - window) // step
    if limit is not None:
        n_windows = min(n_windows, limit)
    offsets = np.arange(n_windows) * step
    return np.stack([series[start: start + window] for start in offsets])


def extract_time_domain_features(
    segments: np.ndarray,
    statistics: Sequence[str] = TIME_DOMAIN_STATISTICS,
) -> np.ndarray:
    """Summarize vibration windows with time-domain condition indicators.

    Implements the nine indicators used by the COMIA bearing pipeline. The
    standardized moments divide by the **sample** standard deviation (``ddof=1``)
    and ``kurtosis`` is excess (three subtracted), which is what reproduces that
    project's ``feature_time_48k_2048_load_1.csv``.

    Args:
        segments: Array of shape ``(n_windows, window)``.
        statistics: Indicators to compute, in output order. Supported names are
            ``max``, ``min``, ``mean``, ``sd``, ``rms``, ``skewness``,
            ``kurtosis``, ``crest`` (``max / rms``) and ``form``
            (``rms / mean``).

    Returns:
        An array of shape ``(n_windows, len(statistics))``.

    Raises:
        ValueError: If ``segments`` is not two-dimensional, a window holds
            fewer than four samples, or a statistic is not supported.

    Example:
        >>> extract_time_domain_features(np.array([[1.0, 2.0, 3.0, 4.0]]), ["mean"])
        array([[2.5]])
    """
    windows = np.asarray(segments, dtype="float64")
    if windows.ndim != 2:
        raise ValueError(f"segments must be 2-D (n_windows, window), got {windows.shape}")
    if windows.shape[1] < 4:
        raise ValueError(f"a window needs at least 4 samples, got {windows.shape[1]}")

    unsupported = [name for name in statistics if name not in TIME_DOMAIN_STATISTICS]
    if unsupported:
        raise ValueError(
            f"Unsupported statistic(s) {unsupported}; "
            f"supported: {list(TIME_DOMAIN_STATISTICS)}"
        )

    mean = windows.mean(axis=1)
    sd = windows.std(axis=1, ddof=1)
    rms = np.sqrt((windows ** 2).mean(axis=1))
    maximum = windows.max(axis=1)
    centered = windows - mean[:, None]
    safe_sd = np.where(sd == 0.0, 1.0, sd)
    safe_mean = np.where(mean == 0.0, np.nan, mean)
    safe_rms = np.where(rms == 0.0, np.nan, rms)

    available = {
        "max": maximum,
        "min": windows.min(axis=1),
        "mean": mean,
        "sd": sd,
        "rms": rms,
        "skewness": (centered ** 3).mean(axis=1) / safe_sd ** 3,
        "kurtosis": (centered ** 4).mean(axis=1) / safe_sd ** 4 - 3.0,
        "crest": maximum / safe_rms,
        "form": rms / safe_mean,
    }
    return np.column_stack([available[name] for name in statistics])


def extract_spectral_features(segments: np.ndarray) -> np.ndarray:
    """Turn vibration windows into their FFT magnitude spectra.

    Each window of ``w`` samples becomes ``w // 2`` real-valued magnitude bins,
    which is the frequency-domain representation the COMIA gearbox pipeline
    uses. The Nyquist bin is dropped so that a 256-sample window yields exactly
    128 features.

    Args:
        segments: Array of shape ``(n_windows, window)``.

    Returns:
        An array of shape ``(n_windows, window // 2)`` of magnitudes.

    Raises:
        ValueError: If ``segments`` is not two-dimensional or its windows are
            shorter than two samples.

    Example:
        >>> extract_spectral_features(np.ones((3, 8))).shape
        (3, 4)
    """
    windows = np.asarray(segments, dtype="float64")
    if windows.ndim != 2:
        raise ValueError(f"segments must be 2-D (n_windows, window), got {windows.shape}")
    width = windows.shape[1]
    if width < 2:
        raise ValueError(f"a window needs at least 2 samples, got {width}")
    return np.abs(np.fft.rfft(windows, n=width))[:, : width // 2]


def assemble_canonical(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Join features and target into the canonical frame.

    Rows whose target is missing are dropped, feature columns are cast to
    ``float64``, the target to ``int64``, column names are stringified and
    de-duplicated, and the index is reset.

    Args:
        X: Feature frame.
        y: Binary target aligned with ``X``.

    Returns:
        A frame of ``float64`` features plus an ``int64`` ``target`` column.

    Raises:
        PreprocessingError: If nothing remains after dropping unlabeled rows.
    """
    labelled = y.notna()
    features = X.loc[labelled].copy()
    target = y.loc[labelled]

    if features.empty:
        raise PreprocessingError("No labelled rows remain after target normalization.")

    features.columns = _unique_names(features.columns)
    features = features.astype("float64")
    features[TARGET_COLUMN] = np.asarray(target, dtype="int64")
    return features.reset_index(drop=True)


def _unique_names(columns: pd.Index) -> list[str]:
    """Stringify column names and disambiguate duplicates.

    Args:
        columns: Original column index.

    Returns:
        A list of unique string names, suffixed ``'_1'``, ``'_2'``, … on
        collision.
    """
    seen: dict[str, int] = {}
    names: list[str] = []
    for column in columns:
        name = str(column).strip() or "feature"
        if name == TARGET_COLUMN:
            name = f"{name}_feature"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


# ══════════════════════════════════════════════════════════════════════
# Stateful dispatch
# ══════════════════════════════════════════════════════════════════════

class DatasetPreprocessor:
    """Dispatches each dataset to its dedicated preprocessing routine.

    One ``_preprocess_<dataset_key>`` method exists per registered dataset.
    Every method reads the dataset's raw files, applies the binarization,
    encoding, and imputation declared in ``datasets.yaml``, and writes the
    canonical parquet through :meth:`write`.

    Attributes:
        validate: Whether :meth:`write` enforces the canonical format contract
            before serializing.

    Example:
        >>> preprocessor = DatasetPreprocessor()
        >>> preprocessor.supports("spambase")
        True
        >>> preprocessor.preprocess("spambase", raw_dir, out_path, meta)
    """

    def __init__(self, validate: bool = True) -> None:
        """Initialize the preprocessor.

        Args:
            validate: Enforce the canonical format contract before writing.
        """
        self.validate = validate

    def __repr__(self) -> str:
        """Return an unambiguous representation of the preprocessor."""
        return f"{type(self).__name__}(validate={self.validate})"

    # ── Dispatch ───────────────────────────────────────────────────────

    def method_for(self, name: str):
        """Return the preprocessing method bound to a dataset key.

        Args:
            name: Dataset key.

        Returns:
            The bound ``_preprocess_<name>`` method, or ``None`` when the
            dataset has no implementation yet.
        """
        return getattr(self, f"_preprocess_{name}", None)

    def supports(self, name: str) -> bool:
        """Return whether a preprocessing routine exists for ``name``."""
        return self.method_for(name) is not None

    def implemented(self) -> list[str]:
        """Return the sorted keys of every implemented preprocessing routine."""
        prefix = "_preprocess_"
        return sorted(
            attribute[len(prefix):]
            for attribute in dir(self)
            if attribute.startswith(prefix)
        )

    def preprocess(
        self,
        name: str,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any] | None = None,
        variant: str | None = None,
    ) -> Path:
        """Preprocess one dataset into its canonical parquet file.

        Args:
            name: Dataset key.
            raw_dir: Directory holding the dataset's raw files.
            output_path: Destination parquet file.
            meta: Registry metadata. Looked up from the registry when omitted.
            variant: Optional variant name (e.g. ``'full'``).

        Returns:
            Path to the written parquet file.

        Raises:
            PreprocessingError: If the dataset has no routine, or the routine
                fails, or its output violates the canonical contract.
        """
        method = self.method_for(name)
        if method is None:
            raise PreprocessingError(
                f"No preprocessing routine for '{name}'. Add a "
                f"_preprocess_{name}() method to DatasetPreprocessor."
            )
        if meta is None:
            from imbdata.registry import get_dataset_meta

            meta = get_dataset_meta(name)

        logger.info(f"Preprocessing '{name}'" + (f" (variant={variant})" if variant else ""))
        method(Path(raw_dir), Path(output_path), meta, variant)
        return Path(output_path)

    # ── Shared I/O helpers ─────────────────────────────────────────────

    @staticmethod
    def locate(raw_dir: Path, *candidates: str) -> Path:
        """Find the first of ``candidates`` present under ``raw_dir``.

        The search covers ``raw_dir`` itself and, recursively, any directory
        an archive may have created inside it.

        Args:
            raw_dir: Directory to search.
            *candidates: File names or glob patterns, tried in order.

        Returns:
            Path to the first match.

        Raises:
            PreprocessingError: If none of the candidates exists.
        """
        for candidate in candidates:
            direct = raw_dir / candidate
            if direct.is_file():
                return direct
            match = next(iter(sorted(raw_dir.rglob(candidate))), None)
            if match is not None and match.is_file():
                return match
        available = sorted(path.name for path in raw_dir.rglob("*") if path.is_file())
        raise PreprocessingError(
            f"None of {list(candidates)} found in {raw_dir}. Available files: {available}"
        )

    def write(self, frame: pd.DataFrame, output_path: Path) -> Path:
        """Validate and serialize a canonical frame to parquet.

        Args:
            frame: Canonical frame (float64 features plus ``target``).
            output_path: Destination parquet file.

        Returns:
            Path to the written file.

        Raises:
            PreprocessingError: If the frame violates the canonical contract.
        """
        if self.validate:
            self.check_canonical(frame)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, compression=PARQUET_COMPRESSION, index=False)
        minority = int(frame[TARGET_COLUMN].sum())
        majority = len(frame) - minority
        ratio = majority / minority if minority else float("inf")
        logger.info(
            f"Wrote {output_path.name}: N={len(frame)}, d={frame.shape[1] - 1}, "
            f"IR={ratio:.1f}:1"
        )
        return output_path

    @staticmethod
    def check_canonical(frame: pd.DataFrame) -> None:
        """Assert that a frame satisfies the canonical format contract.

        Args:
            frame: Frame about to be serialized.

        Raises:
            PreprocessingError: If the target is absent, not binary, or does not
                put the minority class in label ``1``; if any feature is
                non-numeric; or if any cell is missing or infinite.
        """
        if TARGET_COLUMN not in frame.columns:
            raise PreprocessingError(f"Canonical frame must contain a '{TARGET_COLUMN}' column.")

        features = frame.drop(columns=[TARGET_COLUMN])
        target = frame[TARGET_COLUMN]

        observed = set(pd.unique(target))
        if not observed <= {0, 1}:
            raise PreprocessingError(
                f"Target must be binary 0/1, found values {sorted(observed)}."
            )
        if observed != {0, 1}:
            raise PreprocessingError(f"Target must contain both classes, found only {observed}.")

        # Label 1 must be the minority. Consumers rely on this to pass a single
        # minority_label across every dataset instead of special-casing some of
        # them; an inverted dataset would compute the minority-centred measures
        # on the wrong class without ever raising. Equal classes are allowed.
        n_minority = int(target.sum())
        n_majority = len(target) - n_minority
        if n_minority > n_majority:
            raise PreprocessingError(
                f"Class 1 must be the minority: {n_minority} row(s) labelled 1 against "
                f"{n_majority} labelled 0. Invert the binarization so that 0 = majority."
            )

        non_numeric = [
            column for column in features.columns
            if not pd.api.types.is_numeric_dtype(features[column])
        ]
        if non_numeric:
            raise PreprocessingError(f"Non-numeric feature column(s): {non_numeric[:10]}")

        if features.shape[1] == 0:
            raise PreprocessingError("Canonical frame has no feature columns.")

        n_missing = int(features.isna().sum().sum())
        if n_missing:
            columns = features.columns[features.isna().any()].tolist()[:10]
            raise PreprocessingError(f"{n_missing} missing value(s) remain in {columns}")

        n_infinite = int(np.isinf(features.to_numpy(dtype="float64")).sum())
        if n_infinite:
            raise PreprocessingError(f"{n_infinite} infinite value(s) remain in the features.")

    # ══════════════════════════════════════════════════════════════════
    # Pilot datasets
    # ══════════════════════════════════════════════════════════════════

    def _preprocess_breast_cancer_wisconsin(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

        ``wdbc.data`` is headerless: column 0 is the patient id, column 1 the
        diagnosis (``M``/``B``), and columns 2-31 the thirty cell-nucleus
        measurements. The id is dropped and ``M`` (malignant) becomes the
        minority class.

        Args:
            raw_dir: Directory holding ``wdbc.data``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "wdbc.data")
        names = ["id", "diagnosis"] + [f"f{index:02d}" for index in range(1, 31)]
        frame = pd.read_csv(source, header=None, names=names)

        y = binarize_column(frame["diagnosis"], meta.get("minority_value", "M"))
        X = to_numeric_frame(frame.drop(columns=["id", "diagnosis"]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_pima_diabetes(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the Pima Indians Diabetes dataset.

        ``diabetes.csv`` is already tabular and complete; only the ``Outcome``
        column is separated out. The biologically implausible zeros in
        ``Glucose``, ``BloodPressure``, ``SkinThickness``, ``Insulin``, and
        ``BMI`` are deliberately preserved, matching the standard benchmark
        treatment recorded in ``datasets.yaml``.

        Args:
            raw_dir: Directory holding ``diabetes.csv``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "diabetes.csv", "*.csv")
        frame = pd.read_csv(source)
        target_column = meta.get("target_column", "Outcome")

        y = binarize_column(frame[target_column], meta.get("minority_value", 1))
        X = to_numeric_frame(frame.drop(columns=[target_column]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_spambase(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Spambase dataset.

        ``spambase.data`` is headerless with 58 columns: 57 continuous
        word/character frequency and capital-run features, then the spam
        indicator in the last column.

        Args:
            raw_dir: Directory holding ``spambase.data``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "spambase.data")
        frame = pd.read_csv(source, header=None)
        names = [f"f{index:02d}" for index in range(1, frame.shape[1])] + ["spam"]
        frame.columns = names

        y = binarize_column(frame["spam"], meta.get("minority_value", 1))
        X = to_numeric_frame(frame.drop(columns=["spam"]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_svmguide1(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the LIBSVM svmguide1 astroparticle dataset.

        The official distribution ships a train/test split (3,089 + 4,000
        instances) in LIBSVM sparse format. Both files are concatenated to
        recover the full N=7,089 benchmark, and the four dense features are
        materialized in index order.

        Args:
            raw_dir: Directory holding ``svmguide1`` and ``svmguide1.t``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        parts = [self.locate(raw_dir, "svmguide1")]
        test_part = raw_dir / "svmguide1.t"
        if test_part.is_file():
            parts.append(test_part)

        frames = [self._read_libsvm(part) for part in parts]
        frame = pd.concat(frames, ignore_index=True)

        y = binarize_column(frame["label"], meta.get("minority_value", 0))
        X = to_numeric_frame(frame.drop(columns=["label"]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_ecoli_imu(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Ecoli dataset with ``imU`` as the minority class.

        ``ecoli.data`` is whitespace-delimited: an accession-number string,
        seven numeric localization features, and the protein localization site.
        The accession is dropped and ``imU`` (inner membrane, uncleavable
        signal sequence; 35 of 336 instances) becomes the positive class,
        reproducing the KEEL ``ecoli3`` benchmark at IR 8.6:1.

        Args:
            raw_dir: Directory holding ``ecoli.data``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "ecoli.data")
        names = ["sequence_name", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "site"]
        frame = pd.read_csv(source, sep=r"\s+", header=None, names=names)

        y = binarize_column(frame["site"], meta.get("minority_value", "imU"))
        X = to_numeric_frame(frame.drop(columns=["sequence_name", "site"]))
        self.write(assemble_canonical(X, y), output_path)

    # ══════════════════════════════════════════════════════════════════
    # Batch A — UCI / OpenML / HTTP datasets
    # ══════════════════════════════════════════════════════════════════

    def _preprocess_mammography(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the OpenML mammography dataset (OpenML id 310).

        The payload arrives as ARFF from the REST endpoint, or as CSV when the
        optional ``openml`` client materialized it; both are accepted. The six
        image-derived features are continuous and the ``class`` attribute is
        the quoted pair ``'-1'``/``'1'``, with ``'1'`` marking a calcification.

        Args:
            raw_dir: Directory holding ``mammography.arff`` or ``.csv``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "mammography.arff", "mammography.csv", "*.arff")
        frame = (
            self._read_arff(source)
            if source.suffix.lower() == ".arff"
            else pd.read_csv(source)
        )
        target_column = meta.get("target_column", "class")

        y = binarize_column(
            frame[target_column].astype(str).str.strip().str.strip("'\""),
            str(meta.get("minority_value", "1")),
        )
        X = to_numeric_frame(frame.drop(columns=[target_column]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_iranian_churn(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Iranian Churn dataset.

        ``Customer Churn.csv`` is fully numeric with a header row; only the
        trailing ``Churn`` indicator is separated out. Column names carry
        double spaces in the original file and are normalized here.

        Args:
            raw_dir: Directory holding ``Customer Churn.csv``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "Customer Churn.csv", "Customer_Churn.csv", "*.csv")
        frame = pd.read_csv(source)
        frame.columns = [" ".join(str(name).split()) for name in frame.columns]
        target_column = meta.get("target_column", "Churn")

        y = binarize_column(frame[target_column], meta.get("minority_value", 1))
        X = to_numeric_frame(frame.drop(columns=[target_column]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_ozone_level(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Ozone Level Detection dataset.

        The forecasting horizon selects the file: ``filename`` gives the
        default (``onehr.data``, the 1-hour peak standard) and ``variant_files``
        maps each variant name to its own file, so ``variant='eighthr'`` reads
        the 8-hour horizon. Both are headerless with a leading date column, 72
        meteorological features encoding missing values as ``'?'``, and a
        trailing ozone-day flag. The date is dropped and the features are
        median-imputed.

        The two horizons share every predictor and differ only in which days
        count as ozone days, so the pair varies the imbalance ratio (33.7:1
        against 14.8:1) with the feature space held constant.

        Args:
            raw_dir: Directory holding the ozone flat files.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Variant name, e.g. ``'eighthr'``. ``None`` reads the
                default horizon.

        Raises:
            PreprocessingError: If ``variant`` is not declared in
                ``variant_files``.
        """
        default_name = str(meta.get("filename") or "onehr.data")
        if variant:
            variant_files = meta.get("variant_files") or {}
            if variant not in variant_files:
                raise PreprocessingError(
                    f"Unknown ozone_level variant '{variant}'; "
                    f"declared variants: {sorted(variant_files)}"
                )
            filename = str(variant_files[variant])
        else:
            filename = default_name
        source = self.locate(raw_dir, filename)
        frame = pd.read_csv(source, header=None, na_values=list(MISSING_TOKENS))
        frame.columns = ["date"] + [f"f{i:02d}" for i in range(1, frame.shape[1] - 1)] + ["ozone"]

        labels = pd.to_numeric(frame["ozone"], errors="coerce")
        y = binarize_column(labels, float(meta.get("minority_value", 1)))
        X = impute_median(to_numeric_frame(frame.drop(columns=["date", "ozone"])))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_adult_census(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Adult Census Income dataset.

        ``adult.data`` (32,561 rows) and ``adult.test`` (16,281 rows) are
        concatenated to the benchmark N=48,842. The test file carries a
        one-line header artifact and appends a period to its labels, both of
        which are normalized. The ``'?'`` placeholders in the categorical
        columns are mode-imputed before one-hot encoding.

        Args:
            raw_dir: Directory holding ``adult.data`` and ``adult.test``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        names = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "income",
        ]
        parts = [
            pd.read_csv(
                self.locate(raw_dir, "adult.data"),
                header=None,
                names=names,
                skipinitialspace=True,
                na_values=list(MISSING_TOKENS),
            )
        ]
        test_file = raw_dir / "adult.test"
        if test_file.is_file():
            parts.append(
                pd.read_csv(
                    test_file,
                    header=None,
                    names=names,
                    skiprows=1,  # the file opens with a '|1x3 Cross validator' banner
                    skipinitialspace=True,
                    na_values=list(MISSING_TOKENS),
                )
            )
        frame = pd.concat(parts, ignore_index=True)
        frame = frame[frame["income"].notna()]

        target_column = meta.get("target_column", "income")
        labels = frame[target_column].astype(str).str.strip().str.rstrip(".")
        y = binarize_column(labels, str(meta.get("minority_value", ">50K")))

        categorical = self._encoding_columns(meta)
        features = impute_mode(frame.drop(columns=[target_column]), categorical)
        X = onehot_encode(features, categorical)
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_wine_quality_red(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Wine Quality (red) dataset.

        ``winequality-red.csv`` is semicolon-delimited with 11 continuous
        physicochemical features and an integer quality score. Following the
        KEEL convention recorded in ``datasets.yaml``, quality 8 is the
        minority class and every other score is the majority.

        Args:
            raw_dir: Directory holding ``winequality-red.csv``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "winequality-red.csv")
        frame = pd.read_csv(source, sep=";")
        target_column = meta.get("target_column", "quality")

        y = binarize_column(pd.to_numeric(frame[target_column], errors="coerce"), 8.0)
        X = to_numeric_frame(frame.drop(columns=[target_column]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_wine_quality_white(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Wine Quality (white) dataset.

        ``winequality-white.csv`` is semicolon-delimited with the same 11
        continuous physicochemical features as the red wine file and an integer
        quality score. The minority class is the low-quality tail declared in
        ``datasets.yaml`` as ``minority_max``: quality 4 or below, 183 of 4,898
        instances, IR 25.8:1.

        Args:
            raw_dir: Directory holding ``winequality-white.csv``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "winequality-white.csv")
        frame = pd.read_csv(source, sep=";")
        target_column = meta.get("target_column", "quality")

        y = binarize_at_most(frame[target_column], float(meta.get("minority_max", 4)))
        X = to_numeric_frame(frame.drop(columns=[target_column]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_abalone_19(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Abalone dataset with 19 rings as the minority class.

        ``abalone.data`` is headerless with a categorical ``sex`` column
        (``M``/``F``/``I``) and seven continuous morphometric measurements.
        Sex is ordinal-encoded and, per the KEEL convention, exactly 19 rings
        becomes the positive class (32 of 4,177 instances).

        Args:
            raw_dir: Directory holding ``abalone.data``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        names = [
            "sex", "length", "diameter", "height", "whole_weight",
            "shucked_weight", "viscera_weight", "shell_weight", "rings",
        ]
        source = self.locate(raw_dir, "abalone.data")
        frame = pd.read_csv(source, header=None, names=names)

        y = binarize_column(pd.to_numeric(frame["rings"], errors="coerce"), 19.0)
        features = ordinal_encode(frame.drop(columns=["rings"]), self._encoding_columns(meta))
        X = to_numeric_frame(features)
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_secom(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI SECOM semiconductor manufacturing dataset.

        ``secom.data`` holds 590 whitespace-delimited sensor readings per lot
        with ``NaN`` tokens for missing measurements; ``secom_labels.data``
        holds the pass/fail flag (``-1``/``1``) and a timestamp, aligned row by
        row. Features missing in more than ``imputation.drop_threshold`` of the
        lots are dropped and the remainder are median-imputed, so the final
        dimensionality is data-dependent and recorded in the manifest.

        Args:
            raw_dir: Directory holding ``secom.data`` and ``secom_labels.data``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        features_file = self.locate(raw_dir, "secom.data")
        labels_file = self.locate(raw_dir, "secom_labels.data")

        frame = pd.read_csv(
            features_file, sep=r"\s+", header=None, na_values=list(MISSING_TOKENS)
        )
        frame.columns = [f"sensor{i:03d}" for i in range(1, frame.shape[1] + 1)]
        labels = pd.read_csv(labels_file, sep=r"\s+", header=None, usecols=[0], names=["label"])

        if len(labels) != len(frame):
            raise PreprocessingError(
                f"secom feature/label mismatch: {len(frame)} rows vs {len(labels)} labels"
            )

        imputation = meta.get("imputation") or {}
        threshold = float(imputation.get("drop_threshold", 0.5)) if isinstance(
            imputation, dict
        ) else 0.5

        y = binarize_column(pd.to_numeric(labels["label"], errors="coerce"), 1.0)
        X = impute_median(drop_high_missing(to_numeric_frame(frame), threshold))
        logger.info(f"secom retained {X.shape[1]} of {frame.shape[1]} sensors")
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_nsl_kdd(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the NSL-KDD intrusion detection dataset.

        ``KDDTrain+.txt`` (125,973 rows) and ``KDDTest+.txt`` (22,544 rows) are
        concatenated to the benchmark N=148,517. The files are headerless with
        41 features, the attack label, and a difficulty score that is dropped.
        Following the PLAN.md binarization protocol, ``normal`` is the majority
        class and every attack type is merged into the positive class. The
        three symbolic features are one-hot encoded after concatenation, so
        levels present in only one split are still represented.

        Args:
            raw_dir: Directory holding the NSL-KDD flat files.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        names = NSL_KDD_COLUMNS
        parts = [pd.read_csv(self.locate(raw_dir, "KDDTrain+.txt"), header=None, names=names)]
        test_file = raw_dir / "KDDTest+.txt"
        if test_file.is_file():
            parts.append(pd.read_csv(test_file, header=None, names=names))
        frame = pd.concat(parts, ignore_index=True)

        labels = frame["label"].astype(str).str.strip()
        y = (labels != "normal").astype("int64")
        if not y.any():
            raise PreprocessingError("NSL-KDD contains no attack rows; check the source files.")

        dropped = [c for c in self._as_sequence(meta.get("features_drop")) if c in frame.columns]
        features = frame.drop(columns=["label", *dropped])
        X = to_numeric_frame(onehot_encode(features, self._encoding_columns(meta)))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_satimage(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Statlog (Landsat Satellite) dataset.

        ``sat.trn`` (4,435 rows) and ``sat.tst`` (2,000 rows) are concatenated
        to N=6,435. Both are headerless and space-delimited: 36 integer values
        in 0-255, the four spectral bands of each pixel of a 3x3 neighbourhood
        read left-to-right and top-to-bottom, followed by the class of the
        central pixel. Class 4 ("damp grey soil", 626 rows) is the minority.

        Args:
            raw_dir: Directory holding ``sat.trn`` and ``sat.tst``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        names = [f"p{pixel}_b{band}" for pixel in range(1, 10) for band in range(1, 5)]
        names.append("class")
        frame = pd.concat(
            [
                pd.read_csv(self.locate(raw_dir, filename), sep=r"\s+", header=None, names=names)
                for filename in ("sat.trn", "sat.tst")
            ],
            ignore_index=True,
        )

        target_column = meta.get("target_column", "class")
        y = binarize_column(frame[target_column], meta.get("minority_value", 4))
        X = to_numeric_frame(frame.drop(columns=[target_column]))
        self.write(assemble_canonical(X, y), output_path)

    # ══════════════════════════════════════════════════════════════════
    # Batch C — bioinformatics and software engineering
    # ══════════════════════════════════════════════════════════════════

    def _preprocess_yeast_me3(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UCI Yeast dataset with ``ME3`` as the minority class.

        ``yeast.data`` is whitespace-delimited: a sequence accession, eight
        numeric localization features, and the protein localization site.
        Per the PLAN.md binarization protocol, ``ME3`` (membrane protein, no
        N-terminal signal; 163 of 1,484 instances) is the positive class,
        reproducing the KEEL ``yeast3`` benchmark at IR 8.1:1.

        Args:
            raw_dir: Directory holding ``yeast.data``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "yeast.data")
        names = [
            "sequence_name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "site",
        ]
        frame = pd.read_csv(source, sep=r"\s+", header=None, names=names)

        y = binarize_column(frame["site"], meta.get("minority_value", "ME3"))
        X = to_numeric_frame(frame.drop(columns=["sequence_name", "site"]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_nasa_pc1(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the NASA PC1 software defect dataset (OpenML id 1068).

        PC1 holds McCabe and Halstead complexity metrics for 1,109 C modules
        of a flight-software project, with a boolean ``defects`` label; 77
        modules are defective.

        Args:
            raw_dir: Directory holding the PC1 ARFF or CSV payload.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        self._promise_defect_dataset(raw_dir, output_path, meta, "nasa_pc1")

    def _preprocess_nasa_jm1(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the NASA JM1 software defect dataset (OpenML id 1053).

        JM1 holds the same 21 McCabe and Halstead metrics as PC1 for 10,885
        C modules of a real-time predictive ground system, 2,106 of them
        defective. A handful of rows carry ``?`` placeholders, which are
        median-imputed.

        Args:
            raw_dir: Directory holding the JM1 ARFF or CSV payload.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        self._promise_defect_dataset(raw_dir, output_path, meta, "nasa_jm1")

    def _promise_defect_dataset(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        key: str,
    ) -> None:
        """Preprocess a NASA PROMISE defect dataset served from OpenML.

        Shared by :meth:`_preprocess_nasa_pc1` and :meth:`_preprocess_nasa_jm1`.
        Deliberately named outside the ``_preprocess_<key>`` namespace so the
        dispatch in :meth:`method_for` does not mistake it for a dataset.

        PC1 and JM1 share a schema — 21 numeric complexity metrics and a
        boolean ``defects`` label — so they share one routine.

        Args:
            raw_dir: Directory holding the dataset's ARFF or CSV payload.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            key: Dataset key, used to locate the downloaded file.
        """
        source = self.locate(raw_dir, f"{key}.arff", f"{key}.csv", "*.arff", "*.csv")
        frame = (
            self._read_arff(source)
            if source.suffix.lower() == ".arff"
            else pd.read_csv(source)
        )
        target_column = meta.get("target_column", "defects")

        labels = frame[target_column].astype(str).str.strip().str.strip("'\"").str.lower()
        y = binarize_column(labels, str(meta.get("minority_value", True)).lower())
        X = impute_median(to_numeric_frame(frame.drop(columns=[target_column])))
        self.write(assemble_canonical(X, y), output_path)

    # ══════════════════════════════════════════════════════════════════
    # Batch C — high-dimensional genomics
    # ══════════════════════════════════════════════════════════════════

    def _preprocess_tcga_brca(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the TCGA-BRCA gene expression dataset from LinkedOmics.

        Joins the RNA-seq matrix (genes x samples) with the clinical table's
        ``PAM50`` row, keeps the samples carrying a subtype call, and makes
        Basal-like the minority class against the other PAM50 subtypes. The
        LinkedOmics matrix is distributed already log2(RSEM+1)-transformed, so
        no further transform is applied.

        Two variants are produced. The default keeps the ``top_k`` genes by
        variance declared under ``dimensionality_filter`` (5,000), which is
        what the distance-based complexity measures need; ``variant='full'``
        keeps all ~20,000 genes.

        Args:
            raw_dir: Directory holding the ``.cct`` expression matrix and the
                ``.tsi`` clinical table.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: ``'full'`` for the unfiltered matrix; ``None`` for the
                variance-filtered default.

        Raises:
            PreprocessingError: If the clinical table has no ``PAM50`` row, or
                no sample is shared by the two files.
        """
        expression_file = self.locate(raw_dir, "*RSEM_log2.cct", "*.cct")
        clinical_file = self.locate(raw_dir, "*Clinical*.tsi", "*.tsi")

        expression = pd.read_csv(expression_file, sep="\t", index_col=0)
        clinical = pd.read_csv(clinical_file, sep="\t", index_col=0)

        subtype_row = meta.get("target_column", "pam50_subtype")
        if "PAM50" not in clinical.index:
            raise PreprocessingError(
                f"Clinical table {clinical_file.name} has no PAM50 row "
                f"(expected for target '{subtype_row}'); found {list(clinical.index)[:10]}"
            )
        subtypes = clinical.loc["PAM50"]

        samples = expression.columns.intersection(subtypes.index)
        if samples.empty:
            raise PreprocessingError(
                "No sample identifier is shared by the expression matrix and the "
                "clinical table; check that both files come from the same release."
            )

        labels = subtypes.loc[samples].astype(str).str.strip()
        labelled = labels.ne("NA") & labels.ne("nan") & labels.ne("")
        labels = labels[labelled]

        y = binarize_column(labels, str(meta.get("minority_value", "Basal")))
        X = impute_median(to_numeric_frame(expression[labels.index].T))

        if variant != "full":
            top_k = int((meta.get("dimensionality_filter") or {}).get("top_k", 5000))
            X = select_top_variance(X, top_k)
        logger.info(
            f"tcga_brca ({variant or 'default'}): {X.shape[0]} samples, {X.shape[1]} genes"
        )
        self.write(assemble_canonical(X, y), output_path)

    # ══════════════════════════════════════════════════════════════════
    # Batch B — Kaggle financial fraud and insurance datasets
    # ══════════════════════════════════════════════════════════════════

    def _preprocess_credit_card_fraud(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the ULB credit card fraud dataset.

        ``creditcard.csv`` is fully numeric and complete: ``V1``-``V28`` are
        PCA components, ``Time`` and ``Amount`` are the only original features,
        and ``Class`` marks the 492 fraudulent transactions of 284,807.

        Args:
            raw_dir: Directory holding ``creditcard.csv``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "creditcard.csv")
        frame = pd.read_csv(source)
        target_column = meta.get("target_column", "Class")

        y = binarize_column(pd.to_numeric(frame[target_column], errors="coerce"), 1.0)
        X = to_numeric_frame(frame.drop(columns=[target_column]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_paysim(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the PaySim synthetic mobile money dataset.

        The account identifiers ``nameOrig``/``nameDest`` are high-cardinality
        surrogate keys and ``isFlaggedFraud`` is the rule-based system's own
        output, so all three are dropped per ``features_drop``. The five-level
        ``type`` column is one-hot encoded and ``step`` is retained as the
        temporal ordering the drift experiments need.

        Args:
            raw_dir: Directory holding the PaySim log CSV.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "PS_20174392719_1491204439457_log.csv", "PS_*.csv", "*.csv")
        frame = pd.read_csv(source)
        target_column = meta.get("target_column", "isFraud")

        y = binarize_column(pd.to_numeric(frame[target_column], errors="coerce"), 1.0)
        dropped = [c for c in self._as_sequence(meta.get("features_drop")) if c in frame.columns]
        features = frame.drop(columns=[target_column, *dropped])
        X = to_numeric_frame(onehot_encode(features, self._encoding_columns(meta)))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_elliptic_bitcoin(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the Elliptic Bitcoin transaction graph dataset.

        Joins the headerless feature matrix (``txId``, ``time_step``, and 165
        anonymized features) with the class table on ``txId``. Of the 203,769
        nodes only 46,564 carry a label; the ``unknown`` nodes are dropped as
        ``drop_unlabeled`` requires. Class ``2`` (illicit) is the minority and
        class ``1`` (licit) the majority. The edge list is ignored: CIPA
        operates on the tabular view, not the graph.

        Args:
            raw_dir: Directory holding the three ``elliptic_txs_*.csv`` files.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        features_file = self.locate(raw_dir, "elliptic_txs_features.csv")
        classes_file = self.locate(raw_dir, "elliptic_txs_classes.csv")

        features = pd.read_csv(features_file, header=None)
        features.columns = ["txId", "time_step"] + [
            f"f{index:03d}" for index in range(1, features.shape[1] - 1)
        ]
        classes = pd.read_csv(classes_file)

        merged = features.merge(classes, on="txId", how="inner")
        labels = merged[meta.get("target_column", "class")].astype(str).str.strip()

        y = normalize_target(
            labels,
            str(meta.get("minority_value", "2")),
            majority_value=str(meta.get("majority_value", "1")),
        )
        X = to_numeric_frame(merged.drop(columns=["txId", "class"]))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_baf(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the Bank Account Fraud dataset (NeurIPS 2022).

        Uses the ``Base`` variant of the suite: one million applications with
        11,029 fraudulent ones. The five categorical columns are one-hot
        encoded and ``month`` is retained, since the benchmark's temporal drift
        across its eight months is part of what makes it useful here.

        Args:
            raw_dir: Directory holding ``Base.csv``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the suite's other variants are not registered.
        """
        source = self.locate(raw_dir, str(meta.get("filename") or "Base.csv"))
        frame = pd.read_csv(source)
        target_column = meta.get("target_column", "fraud_bool")

        y = binarize_column(pd.to_numeric(frame[target_column], errors="coerce"), 1.0)
        features = frame.drop(columns=[target_column])
        X = to_numeric_frame(onehot_encode(features, self._encoding_columns(meta)))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_vehicle_insurance_fraud(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the vehicle insurance claim fraud dataset.

        ``fraud_oracle.csv`` is written with a UTF-8 BOM and mixes numeric and
        categorical claim attributes. ``PolicyNumber`` is a surrogate key and
        is dropped; the twenty declared categorical columns are one-hot
        encoded, and any categorical the registry does not list is encoded too,
        so no object column can reach the canonical frame.

        Args:
            raw_dir: Directory holding ``fraud_oracle.csv``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        source = self.locate(raw_dir, "fraud_oracle.csv")
        frame = pd.read_csv(source, encoding="utf-8-sig")
        target_column = meta.get("target_column", "FraudFound_P")

        y = binarize_column(pd.to_numeric(frame[target_column], errors="coerce"), 1.0)
        dropped = [c for c in self._as_sequence(meta.get("features_drop")) if c in frame.columns]
        features = frame.drop(columns=[target_column, *dropped])

        declared = self._encoding_columns(meta)
        categorical = declared + [c for c in self._object_columns(features) if c not in declared]
        X = to_numeric_frame(onehot_encode(features, categorical))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_ieee_cis_fraud(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the IEEE-CIS fraud detection dataset.

        Joins ``train_transaction.csv`` with ``train_identity.csv`` on
        ``TransactionID`` (a left join: only about a quarter of transactions
        carry identity records), giving the 434-column view the PLAN.md table
        describes. Categorical columns are ordinal-encoded rather than one-hot
        encoded, because ``DeviceInfo`` and the email-domain columns have
        thousands of levels between them and one-hot encoding would multiply
        the dimensionality by an order of magnitude. The heavy missingness is
        then median-imputed, and columns that are missing everywhere are
        dropped.

        Args:
            raw_dir: Directory holding the IEEE-CIS competition CSVs.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        transactions = pd.read_csv(self.locate(raw_dir, "train_transaction.csv"))
        identities = pd.read_csv(self.locate(raw_dir, "train_identity.csv"))
        merged = transactions.merge(identities, on="TransactionID", how="left")
        del transactions, identities

        target_column = meta.get("target_column", "isFraud")
        y = binarize_column(pd.to_numeric(merged[target_column], errors="coerce"), 1.0)

        dropped = [c for c in self._as_sequence(meta.get("features_drop")) if c in merged.columns]
        features = merged.drop(columns=[target_column, *dropped])
        features = ordinal_encode(features, self._object_columns(features))
        X = impute_median(drop_high_missing(to_numeric_frame(features), 1.0))
        logger.info(f"ieee_cis_fraud: {X.shape[0]} transactions, {X.shape[1]} features")
        self.write(assemble_canonical(X, y), output_path)

    # ══════════════════════════════════════════════════════════════════
    # Batch C — large-scale network intrusion detection
    # ══════════════════════════════════════════════════════════════════

    def _preprocess_unsw_nb15(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the UNSW-NB15 intrusion detection dataset.

        Concatenates the official train and test splits to the benchmark
        N=257,673. ``id`` is a surrogate key and ``attack_cat`` is the
        multi-class label, so both are dropped; ``proto``, ``service``, and
        ``state`` are one-hot encoded after concatenation so that levels
        appearing in only one split are still represented.

        The registry sets ``minority_value: 0``, and the data confirms it:
        normal traffic is 93,000 of the 257,673 flows against 164,673 attacks,
        so normal is the minority class here. That keeps the canonical
        contract (``0`` = majority, ``1`` = minority) intact even though the
        semantic convention elsewhere treats attacks as the positive class.

        Args:
            raw_dir: Directory holding the UNSW-NB15 split CSVs.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.
        """
        parts = []
        for filename in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
            candidate = raw_dir / filename
            if candidate.is_file():
                parts.append(pd.read_csv(candidate, encoding="utf-8-sig"))
        if not parts:
            parts.append(pd.read_csv(self.locate(raw_dir, "UNSW_NB15_*-set.csv")))
        frame = pd.concat(parts, ignore_index=True)
        frame.columns = [str(name).strip() for name in frame.columns]

        target_column = meta.get("target_column", "label")
        y = binarize_column(
            pd.to_numeric(frame[target_column], errors="coerce"),
            float(meta.get("minority_value", 0)),
        )
        dropped = [c for c in self._as_sequence(meta.get("features_drop")) if c in frame.columns]
        features = frame.drop(columns=[target_column, *dropped])
        X = to_numeric_frame(onehot_encode(features, self._encoding_columns(meta)))
        self.write(assemble_canonical(X, y), output_path)

    def _preprocess_cic_ids_2017(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the CIC-IDS-2017 intrusion detection dataset.

        Concatenates the eight daily flow CSVs (~2.8M rows of 78 numeric
        features). The original column names carry leading spaces — including
        the ``' Label'`` target — and are stripped. Several flow-rate features
        contain infinities and NaNs where a flow has zero duration; those rows
        are dropped, as ``imputation: drop_inf`` requires. Per the PLAN.md
        binarization protocol, ``BENIGN`` is the majority and every attack type
        is merged into the positive class.

        Args:
            raw_dir: Directory holding the daily ``*_ISCX.csv`` files.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.

        Raises:
            PreprocessingError: If no daily CSV is present in ``raw_dir``.
        """
        sources = sorted(raw_dir.rglob("*.csv"))
        if not sources:
            raise PreprocessingError(f"No daily CIC-IDS-2017 CSV found in {raw_dir}")

        parts = []
        for source in sources:
            part = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
            part.columns = [str(name).strip() for name in part.columns]
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        del parts

        target_column = str(meta.get("target_column", " Label")).strip()
        labels = frame[target_column].astype(str).str.strip().str.upper()
        y = (labels != "BENIGN").astype("int64")
        if not y.any():
            raise PreprocessingError("CIC-IDS-2017 contains no attack rows; check the sources.")

        features = to_numeric_frame(frame.drop(columns=[target_column]))
        features = features.replace([np.inf, -np.inf], np.nan)

        usable = features.notna().all(axis=1)
        dropped = int((~usable).sum())
        if dropped:
            logger.info(f"cic_ids_2017: dropped {dropped} row(s) with Inf/NaN flow features")
        X = drop_high_missing(features.loc[usable], 0.0)
        self.write(assemble_canonical(X, y.loc[usable]), output_path)

    # ══════════════════════════════════════════════════════════════════
    # Batch C — multivariate time series (space weather)
    # ══════════════════════════════════════════════════════════════════

    def _preprocess_swan_sf(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the SWAN-SF solar flare prediction benchmark.

        Each instance is a tab-separated multivariate time series of 60
        timesteps at 12-minute cadence, stored inside per-partition tarballs
        under ``FL/`` (M- and X-class flares) and ``NF/`` (flare-quiet, B- and
        C-class). Following the PLAN.md binarization protocol, M+X is the
        positive class and FQ+B+C the negative one.

        The 24 magnetic field parameters declared in ``parameter_columns`` are
        summarized by the 8 statistics of ``preprocessing.statistics``, giving
        the 192 tabular features the registry specifies. The tarballs are read
        as streams, so no intermediate 15 GB extraction is needed. Whichever
        partitions are present in ``raw_dir`` are processed, so a partial
        download yields a usable — if smaller — dataset.

        Args:
            raw_dir: Directory holding ``partition*_instances.tar.gz``.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.

        Raises:
            PreprocessingError: If no partition archive is present, or no
                instance could be parsed from those that are.
        """
        archives = sorted(raw_dir.glob("partition*_instances.tar.gz"))
        if not archives:
            raise PreprocessingError(
                f"No SWAN-SF partition archive in {raw_dir}. Expected files named "
                f"'partition<N>_instances.tar.gz' from the Harvard Dataverse record."
            )

        spec = meta.get("preprocessing") or {}
        statistics = self._as_sequence(spec.get("statistics")) or list(MVTS_STATISTICS)
        parameters = self._as_sequence(meta.get("parameter_columns"))
        if not parameters:
            raise PreprocessingError(
                "swan_sf needs 'parameter_columns' in datasets.yaml to know which of the "
                "instance columns are the magnetic field parameters."
            )

        rows: list[np.ndarray] = []
        labels: list[int] = []
        for archive in archives:
            parsed = self._read_swan_partition(archive, parameters, statistics)
            rows.extend(parsed[0])
            labels.extend(parsed[1])
            logger.info(f"swan_sf: {archive.name} contributed {len(parsed[1])} instance(s)")

        if not rows:
            raise PreprocessingError(
                f"No SWAN-SF instance could be parsed from {len(archives)} archive(s)."
            )

        columns = [f"{parameter}_{statistic}" for parameter in parameters
                   for statistic in statistics]
        X = pd.DataFrame(np.vstack(rows), columns=columns)
        y = pd.Series(labels, dtype="int64")
        logger.info(f"swan_sf: {len(y)} instances, {X.shape[1]} features, {int(y.sum())} M+X")
        self.write(assemble_canonical(impute_median(X), y), output_path)

    @staticmethod
    def _read_swan_partition(
        archive: Path,
        parameters: Sequence[str],
        statistics: Sequence[str],
    ) -> tuple[list[np.ndarray], list[int]]:
        """Stream one SWAN-SF partition tarball into feature rows and labels.

        Args:
            archive: A ``partition<N>_instances.tar.gz`` file.
            parameters: Names of the magnetic field parameter columns to keep.
            statistics: Statistics to compute per parameter.

        Returns:
            A pair ``(rows, labels)``: one feature vector and one ``0``/``1``
            label per parsed instance.

        Raises:
            PreprocessingError: If the archive cannot be opened.
        """
        rows: list[np.ndarray] = []
        labels: list[int] = []
        skipped = 0

        try:
            with tarfile.open(archive, "r:gz") as bundle:
                for member in bundle:
                    if not member.isfile() or not member.name.endswith(".csv"):
                        continue
                    handle = bundle.extractfile(member)
                    if handle is None:
                        continue
                    series = DatasetPreprocessor._parse_swan_instance(handle.read(), parameters)
                    if series is None:
                        skipped += 1
                        continue
                    rows.append(extract_mvts_features(series, statistics))
                    labels.append(1 if "/FL/" in f"/{member.name}" else 0)
        except (tarfile.TarError, OSError, EOFError) as exc:
            raise PreprocessingError(f"Cannot read SWAN-SF archive {archive}: {exc}") from exc

        if skipped:
            logger.warning(f"swan_sf: skipped {skipped} unparsable instance(s) in {archive.name}")
        return rows, labels

    @staticmethod
    def _parse_swan_instance(payload: bytes, parameters: Sequence[str]) -> np.ndarray | None:
        """Parse one tab-separated MVTS instance into a numeric array.

        Args:
            payload: Raw bytes of the instance CSV.
            parameters: Parameter column names to extract, in order.

        Returns:
            An array of shape ``(timesteps, len(parameters))`` with missing
            values as NaN, or ``None`` when the instance has no header, no
            timesteps, or none of the requested parameters.
        """
        lines = payload.decode("utf-8", errors="replace").splitlines()
        if len(lines) < 2:
            return None

        header = lines[0].split("\t")
        try:
            indices = [header.index(parameter) for parameter in parameters]
        except ValueError:
            return None

        series = np.full((len(lines) - 1, len(indices)), np.nan, dtype="float64")
        for row, line in enumerate(lines[1:]):
            fields = line.split("\t")
            for column, index in enumerate(indices):
                if index < len(fields):
                    try:
                        series[row, column] = float(fields[index])
                    except ValueError:
                        pass  # leaves NaN; extract_mvts_features ignores it
        return series

    # ══════════════════════════════════════════════════════════════════
    # Batch C — rotating machinery vibration
    # ══════════════════════════════════════════════════════════════════

    def _preprocess_cwru_bearing(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the CWRU bearing fault dataset.

        Each ``.mat`` file is one continuous Drive End accelerometer recording
        of a single operating condition at 48 kHz under 1 HP of motor load. The
        recording is cut into non-overlapping windows of ``window`` samples and
        each window is summarized by the nine time-domain condition indicators
        of :func:`extract_time_domain_features`, giving one row per window.

        The parameters come from ``preprocessing`` in ``datasets.yaml``, which
        records the pipeline recovered from the COMIA project so that both
        projects see the same rows.

        Polarity note: normal operation is the minority here (230 windows
        against 2,070 fault windows), so normal is class ``1``. COMIA labels
        fault as ``1``, but fault is its majority; keeping that encoding would
        break the canonical ``0`` = majority contract. The partition of the
        data is identical — only the label names swap.

        Args:
            raw_dir: Directory holding the per-condition ``.mat`` recordings.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.

        Raises:
            PreprocessingError: If ``conditions`` is absent from the registry,
                if no recording could be read, or if scipy is unavailable.
        """
        conditions = meta.get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            raise PreprocessingError(
                "cwru_bearing needs a 'conditions' mapping (label -> CWRU file number) "
                "in datasets.yaml."
            )

        spec = meta.get("preprocessing") or {}
        window = int(spec.get("window", 2048))
        overlap = int(spec.get("overlap", 0))
        limit = spec.get("windows_per_condition")
        channel = str(spec.get("channel", "DE"))
        statistics = self._as_sequence(spec.get("statistics")) or list(TIME_DOMAIN_STATISTICS)

        rows: list[np.ndarray] = []
        labels: list[str] = []
        for label, number in sorted(conditions.items()):
            source = self.locate(raw_dir, f"{label}_{int(number):03d}.mat", f"*{number}*.mat")
            signal = self._read_mat_channel(source, channel, number=int(number))
            segments = segment_signal(
                signal, window=window, overlap=overlap,
                limit=int(limit) if limit else None,
            )
            rows.append(extract_time_domain_features(segments, statistics))
            labels.extend([label] * len(segments))
            logger.debug(f"cwru_bearing: {label} -> {len(segments)} window(s) from {source.name}")

        if not rows:
            raise PreprocessingError(f"No CWRU recording could be read from {raw_dir}")

        X = pd.DataFrame(np.vstack(rows), columns=list(statistics))
        condition = pd.Series(labels, name="condition")

        # 'normal' is the minority operating condition; every fault type is the majority.
        y = condition.str.lower().str.contains("normal").astype("int64")
        logger.info(
            f"cwru_bearing: {len(condition)} windows, {X.shape[1]} features, "
            f"{int(y.sum())} normal"
        )
        self.write(assemble_canonical(X, y), output_path)

    @staticmethod
    def _read_mat_channel(path: Path, channel: str, number: int | None = None) -> np.ndarray:
        """Read one accelerometer channel out of a CWRU-style ``.mat`` file.

        CWRU names its variables ``X<file number>_<channel>_time``. A few of the
        distributed files bundle a second, unrelated recording — file 175
        (IR014, load 1) also carries ``X217_DE_time`` — so when ``number`` is
        given the channel belonging to that file number is preferred and the
        stray recording is ignored. Without ``number`` the lowest-sorting
        matching key is used.

        Args:
            path: MATLAB file to read.
            channel: Channel tag to look for, e.g. ``'DE'`` (Drive End) or
                ``'FE'`` (Fan End).
            number: CWRU file number whose recording is wanted.

        Returns:
            The channel's samples as a 1-D array.

        Raises:
            PreprocessingError: If scipy is unavailable, the file cannot be
                read, or it holds no variable for ``channel``.
        """
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover - scipy ships with scikit-learn
            raise PreprocessingError(
                f"Reading {path.name} needs scipy: pip install scipy"
            ) from exc

        try:
            payload = loadmat(str(path))
        except (OSError, ValueError, NotImplementedError) as exc:
            raise PreprocessingError(f"Cannot read MATLAB file {path}: {exc}") from exc

        keys = sorted(
            key for key in payload
            if not key.startswith("__") and channel in key and "time" in key.lower()
        )
        if not keys:
            available = sorted(k for k in payload if not k.startswith("__"))
            raise PreprocessingError(
                f"{path.name} has no '{channel}' time channel; available: {available}"
            )

        chosen = keys[0]
        if number is not None:
            preferred = f"X{int(number):03d}_{channel}_time"
            matching = [key for key in keys if key == preferred or key.startswith(f"X{number}_")]
            if matching:
                chosen = matching[0]
            elif len(keys) > 1:
                logger.warning(
                    f"{path.name} has no channel for file number {number}; "
                    f"falling back to '{chosen}' out of {keys}"
                )
        if len(keys) > 1:
            logger.debug(f"{path.name} bundles {len(keys)} recordings {keys}; using '{chosen}'")
        return np.asarray(payload[chosen], dtype="float64").ravel()

    def _preprocess_seu_gearbox(
        self,
        raw_dir: Path,
        output_path: Path,
        meta: dict[str, Any],
        variant: str | None = None,
    ) -> None:
        """Preprocess the Southeast University gearbox dataset.

        Each recording is a tab-separated capture of eight synchronized
        channels, preceded by a DAQ header that ends with a ``Data`` marker.
        The declared ``channel`` is cut into non-overlapping windows, every
        window is turned into its FFT magnitude spectrum by
        :func:`extract_spectral_features`, and ``windows_per_file`` of them are
        drawn without replacement using a seeded generator, so the selection is
        identical on every machine.

        Healthy operation is the minority class: the gearset ships two healthy
        recordings against eight faulted ones (chipped, missing, root and
        surface gear faults, each at two speed-load settings).

        Args:
            raw_dir: Directory holding the gearset CSVs.
            output_path: Destination parquet file.
            meta: Registry metadata for the dataset.
            variant: Unused; the dataset has a single variant.

        Raises:
            PreprocessingError: If ``conditions`` is absent from the registry
                or no recording could be read.
        """
        conditions = meta.get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            raise PreprocessingError(
                "seu_gearbox needs a 'conditions' mapping (file stem -> normal|fault) "
                "in datasets.yaml."
            )

        spec = meta.get("preprocessing") or {}
        window = int(spec.get("window", 256))
        overlap = int(spec.get("overlap", 0))
        per_file = int(spec.get("windows_per_file", 250))
        channel = int(spec.get("channel", 2))
        generator = np.random.default_rng(int(spec.get("random_state", 42)))

        rows: list[np.ndarray] = []
        labels: list[str] = []
        for stem, kind in sorted(conditions.items()):
            source = self.locate(raw_dir, f"{stem}.csv")
            signal = self._read_seu_channel(source, channel)
            spectra = extract_spectral_features(
                segment_signal(signal, window=window, overlap=overlap)
            )
            take = min(per_file, len(spectra))
            chosen = generator.choice(len(spectra), size=take, replace=False)
            rows.append(spectra[chosen])
            labels.extend([kind] * take)
            logger.debug(f"seu_gearbox: {stem} -> {take} of {len(spectra)} windows")

        if not rows:
            raise PreprocessingError(f"No SEU gearset recording could be read from {raw_dir}")

        features = np.vstack(rows)
        columns = [f"fft{index:03d}" for index in range(1, features.shape[1] + 1)]
        X = pd.DataFrame(features, columns=columns)
        condition = pd.Series(labels, name="condition")
        y = binarize_column(condition, str(meta.get("minority_value", "normal")))
        logger.info(
            f"seu_gearbox: {len(condition)} windows, {X.shape[1]} features, "
            f"{int(y.sum())} healthy"
        )
        self.write(assemble_canonical(X, y), output_path)

    @staticmethod
    def _read_seu_channel(path: Path, channel: int) -> np.ndarray:
        """Read one channel out of an SEU gearbox capture.

        The file opens with a DAQ header — title, sampling settings, channel
        legend — terminated by a line whose first field is ``Data``. Rows below
        it hold the eight channels, tab-separated in the distributed files but
        comma-separated in some mirrors, so the delimiter is detected.

        Args:
            path: Recording to read.
            channel: 1-based channel index to extract.

        Returns:
            The channel's samples as a 1-D array.

        Raises:
            PreprocessingError: If the header has no ``Data`` marker, the file
                cannot be read, or the channel is out of range.
        """
        marker = None
        delimiter = "\t"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for index, line in enumerate(handle):
                    first = line.replace(",", "\t").split("\t")[0].strip()
                    if first == "Data":
                        marker = index
                        delimiter = "\t" if "\t" in line else ","
                        break
                    if index > 200:
                        break
        except OSError as exc:
            raise PreprocessingError(f"Cannot read SEU recording {path}: {exc}") from exc

        if marker is None:
            raise PreprocessingError(
                f"{path.name} has no 'Data' marker ending the DAQ header; "
                f"the file may be truncated."
            )

        try:
            column = pd.read_csv(
                path,
                sep=delimiter,
                skiprows=marker + 1,
                header=None,
                usecols=[channel - 1],
                dtype="float64",
                na_values=list(MISSING_TOKENS),
            )
        except (OSError, ValueError) as exc:
            raise PreprocessingError(
                f"Cannot parse channel {channel} of {path.name}: {exc}"
            ) from exc

        values = column.iloc[:, 0].to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise PreprocessingError(f"Channel {channel} of {path.name} holds no finite samples.")
        return values

    # ── Registry helpers ───────────────────────────────────────────────

    @staticmethod
    def _object_columns(df: pd.DataFrame) -> list[str]:
        """Return the columns that still hold non-numeric values.

        Args:
            df: Feature frame.

        Returns:
            Names of the object/categorical columns, in frame order.
        """
        return [
            column for column in df.columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]

    @staticmethod
    def _as_sequence(value: Any) -> list[str]:
        """Normalize a scalar-or-list registry field into a list of strings.

        Args:
            value: A string, a list of strings, or ``None``.

        Returns:
            A list of strings; empty when ``value`` is ``None``.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    @classmethod
    def _encoding_columns(cls, meta: dict[str, Any]) -> list[str]:
        """Return the categorical columns declared under ``encoding.columns``.

        Args:
            meta: Registry metadata for a dataset.

        Returns:
            The declared column names; empty when the entry sets
            ``encoding: none`` or omits the field.
        """
        encoding = meta.get("encoding")
        if not isinstance(encoding, dict):
            return []
        return cls._as_sequence(encoding.get("columns"))

    # ── Format readers ─────────────────────────────────────────────────

    @staticmethod
    def _read_arff(path: Path) -> pd.DataFrame:
        """Read a dense ARFF file into a frame.

        Supports the subset of the format the registry's OpenML sources emit:
        ``%`` comments, ``@relation``/``@attribute``/``@data`` headers, and
        comma-separated dense instances. Sparse ARFF and instance weights are
        not supported, and are reported rather than silently mis-parsed.

        Args:
            path: ARFF file to read.

        Returns:
            A frame whose columns are the declared attribute names, with values
            left as strings for the caller to coerce.

        Raises:
            PreprocessingError: If the file has no ``@data`` section, declares
                no attributes, or uses the sparse instance format.
        """
        attributes: list[str] = []
        rows: list[list[str]] = []
        in_data = False

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("%"):
                        continue
                    if not in_data:
                        lowered = stripped.lower()
                        if lowered.startswith("@attribute"):
                            attributes.append(stripped.split()[1].strip("'\""))
                        elif lowered.startswith("@data"):
                            in_data = True
                        continue
                    if stripped.startswith("{"):
                        raise PreprocessingError(f"Sparse ARFF is not supported: {path}")
                    rows.append([field.strip() for field in stripped.split(",")])
        except OSError as exc:
            raise PreprocessingError(f"Cannot read ARFF file {path}: {exc}") from exc

        if not in_data:
            raise PreprocessingError(f"ARFF file {path} has no @data section.")
        if not attributes:
            raise PreprocessingError(f"ARFF file {path} declares no attributes.")

        width = len(attributes)
        usable = [row for row in rows if len(row) == width]
        if not usable:
            raise PreprocessingError(
                f"ARFF file {path} has no instance matching its {width} declared attributes."
            )
        if len(usable) < len(rows):
            logger.warning(f"Skipped {len(rows) - len(usable)} malformed row(s) in {path.name}")
        return pd.DataFrame(usable, columns=attributes)

    @staticmethod
    def _read_libsvm(path: Path) -> pd.DataFrame:
        """Read a LIBSVM sparse file into a dense frame.

        Args:
            path: File in ``<label> <index>:<value> ...`` format.

        Returns:
            A frame with a ``label`` column followed by ``x1``, ``x2``, …
            feature columns, missing entries filled with ``0.0``.

        Raises:
            PreprocessingError: If the file cannot be parsed.
        """
        labels: list[float] = []
        rows: list[dict[int, float]] = []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.split()
                    if not fields:
                        continue
                    labels.append(float(fields[0]))
                    rows.append(
                        {
                            int(index): float(value)
                            for index, _, value in (f.partition(":") for f in fields[1:])
                            if value
                        }
                    )
        except (OSError, ValueError) as exc:
            raise PreprocessingError(f"Cannot parse LIBSVM file {path}: {exc}") from exc

        if not rows:
            raise PreprocessingError(f"LIBSVM file {path} contains no instances.")

        n_features = max((max(row) for row in rows if row), default=0)
        columns = [f"x{index}" for index in range(1, n_features + 1)]
        dense = np.zeros((len(rows), n_features), dtype="float64")
        for position, row in enumerate(rows):
            for index, value in row.items():
                dense[position, index - 1] = value

        frame = pd.DataFrame(dense, columns=columns)
        frame.insert(0, "label", labels)
        return frame


# ── Functional facade ──────────────────────────────────────────────────

def preprocess_dataset(
    name: str,
    meta: dict[str, Any] | None = None,
    raw_dir: Path | None = None,
    output_path: Path | None = None,
    variant: str | None = None,
) -> Path:
    """Preprocess a dataset using a default :class:`DatasetPreprocessor`.

    Args:
        name: Dataset key.
        meta: Registry metadata. Looked up from the registry when omitted.
        raw_dir: Raw directory. Defaults to ``<store>/raw/<name>``.
        output_path: Destination parquet. Defaults to the store's canonical
            path for ``name`` and ``variant``.
        variant: Optional variant name.

    Returns:
        Path to the written parquet file.

    Raises:
        PreprocessingError: If the dataset has no routine or its output
            violates the canonical contract.
    """
    from imbdata.config import default_config

    config = default_config()
    raw_dir = Path(raw_dir) if raw_dir is not None else config.raw_dir(name)
    if output_path is None:
        output_path = config.processed_path(name, variant)
    return DatasetPreprocessor().preprocess(name, raw_dir, Path(output_path), meta, variant)
