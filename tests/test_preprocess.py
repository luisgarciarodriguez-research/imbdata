"""
tests.test_preprocess — Unit tests for the preprocessing layer.

Covers the pure transformation functions and the canonical-format contract
enforced by :class:`imbdata.preprocess.DatasetPreprocessor`.

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

from imbdata.exceptions import PreprocessingError
from imbdata.preprocess import (
    MVTS_STATISTICS,
    DatasetPreprocessor,
    assemble_canonical,
    binarize_at_most,
    binarize_column,
    drop_constant_columns,
    drop_high_missing,
    extract_mvts_features,
    extract_spectral_features,
    extract_time_domain_features,
    impute_median,
    impute_mode,
    normalize_target,
    onehot_encode,
    ordinal_encode,
    TIME_DOMAIN_STATISTICS,
    segment_signal,
    select_top_variance,
    to_numeric_frame,
)

PILOT_KEYS = [
    "breast_cancer_wisconsin",
    "pima_diabetes",
    "spambase",
    "svmguide1",
    "ecoli_imu",
]


# ── Pure transformations ──────────────────────────────────────────────

def test_binarize_column_marks_the_minority_value() -> None:
    """The declared minority value maps to 1 and everything else to 0."""
    result = binarize_column(pd.Series(["B", "M", "B", "M"]), "M")
    assert result.tolist() == [0, 1, 0, 1]
    assert result.dtype == "int64"


def test_binarize_column_falls_back_to_string_comparison() -> None:
    """Padded and quoted flat-file labels still match the declared value."""
    assert binarize_column(pd.Series([" positive ", "negative"]), "positive").tolist() == [1, 0]


def test_binarize_column_without_matches_raises() -> None:
    """A minority value that never occurs is a preprocessing error."""
    with pytest.raises(PreprocessingError, match="never occurs"):
        binarize_column(pd.Series(["A", "B"]), "Z")


def test_binarize_at_most_marks_the_low_tail() -> None:
    """Values at or below the threshold map to 1 and the rest to 0."""
    assert binarize_at_most(pd.Series([3, 5, 8, 4]), 4).tolist() == [1, 0, 0, 1]


def test_binarize_at_most_without_matches_raises() -> None:
    """A threshold below the observed range is a preprocessing error."""
    with pytest.raises(PreprocessingError, match="at or below"):
        binarize_at_most(pd.Series([5, 6, 7]), 2)


def test_normalize_target_marks_rows_outside_the_two_labels() -> None:
    """Rows belonging to neither declared label become NA for later dropping."""
    result = normalize_target(pd.Series(["1", "2", "unknown"]), "2", majority_value="1")
    assert result.tolist()[:2] == [0, 1]
    assert pd.isna(result.iloc[2])


def test_normalize_target_without_majority_keeps_every_row() -> None:
    """With no explicit majority value, non-minority rows are simply zeros."""
    assert normalize_target(pd.Series([1, 2, 3]), 2).tolist() == [0, 1, 0]


def test_onehot_encode_expands_categoricals_to_float_indicators() -> None:
    """Each level becomes its own float64 '<column>=<level>' indicator."""
    encoded = onehot_encode(pd.DataFrame({"t": ["a", "b", "a"]}), ["t"])
    assert encoded.columns.tolist() == ["t=a", "t=b"]
    assert set(map(str, encoded.dtypes)) == {"float64"}


def test_onehot_encode_ignores_absent_columns() -> None:
    """Listing a column the frame does not have is a no-op, not an error."""
    frame = pd.DataFrame({"a": [1, 2]})
    assert onehot_encode(frame, ["missing"]).equals(frame)


def test_ordinal_encode_numbers_levels_deterministically() -> None:
    """Levels are numbered by sorted order so runs are reproducible."""
    encoded = ordinal_encode(pd.DataFrame({"s": ["M", "F", "I"]}), ["s"])
    assert encoded["s"].tolist() == [2.0, 0.0, 1.0]


def test_impute_median_fills_gaps_with_the_column_median() -> None:
    """Missing numeric entries take their column's median."""
    assert impute_median(pd.DataFrame({"a": [1.0, None, 3.0]}))["a"].tolist() == [1.0, 2.0, 3.0]


def test_impute_median_handles_fully_missing_columns() -> None:
    """A column with no observed value falls back to zero, preserving the contract."""
    assert impute_median(pd.DataFrame({"a": [None, None]}))["a"].tolist() == [0.0, 0.0]


def test_impute_mode_fills_categoricals_with_the_most_frequent_level() -> None:
    """Categorical gaps take the column's modal level."""
    assert impute_mode(pd.DataFrame({"c": ["a", None, "a"]}))["c"].tolist() == ["a", "a", "a"]


def test_drop_high_missing_removes_over_sparse_columns() -> None:
    """Columns above the missing-value threshold are dropped."""
    frame = pd.DataFrame({"keep": [1.0, 2.0], "drop": [None, None]})
    assert drop_high_missing(frame, 0.5).columns.tolist() == ["keep"]


def test_drop_high_missing_rejects_out_of_range_thresholds() -> None:
    """A threshold outside [0, 1] is a programming error."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        drop_high_missing(pd.DataFrame({"a": [1.0]}), 1.5)


def test_drop_constant_columns_removes_zero_variance_features() -> None:
    """Single-valued columns carry no information and are removed."""
    frame = pd.DataFrame({"varies": [1, 2], "constant": [7, 7]})
    assert drop_constant_columns(frame).columns.tolist() == ["varies"]


def test_to_numeric_frame_maps_missing_tokens_to_nan() -> None:
    """UCI's '?' placeholder becomes NaN and the column becomes float64."""
    converted = to_numeric_frame(pd.DataFrame({"a": ["1.5", "?", " 3 "]}))
    assert converted["a"].dtype == "float64"
    assert converted["a"].tolist()[0] == 1.5 and np.isnan(converted["a"].iloc[1])


def test_assemble_canonical_produces_float_features_and_int_target() -> None:
    """The assembled frame satisfies the dtype half of the format contract."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    frame = assemble_canonical(X, pd.Series([0, 1, 0]))
    assert frame["a"].dtype == "float64" and frame["target"].dtype == "int64"


def test_assemble_canonical_drops_unlabeled_rows() -> None:
    """Rows whose target is NA are removed together with their features."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, None], dtype="Int64")
    frame = assemble_canonical(X, y)
    assert len(frame) == 2 and frame.index.tolist() == [0, 1]


def test_assemble_canonical_renames_a_feature_colliding_with_target() -> None:
    """A raw feature literally named 'target' is renamed, not overwritten."""
    X = pd.DataFrame({"target": [1.0, 2.0]})
    frame = assemble_canonical(X, pd.Series([0, 1]))
    assert frame.columns.tolist() == ["target_feature", "target"]


def test_assemble_canonical_without_labeled_rows_raises() -> None:
    """An entirely unlabeled dataset is a preprocessing error."""
    with pytest.raises(PreprocessingError, match="No labelled rows"):
        assemble_canonical(pd.DataFrame({"a": [1.0]}), pd.Series([None], dtype="Int64"))


# ── Dimensionality reduction and MVTS summarization ───────────────────

def test_select_top_variance_keeps_the_most_variable_columns() -> None:
    """Low-variance columns are dropped first."""
    frame = pd.DataFrame({"flat": [1.0, 1.0, 1.0], "wide": [0.0, 5.0, 10.0]})
    assert select_top_variance(frame, 1).columns.tolist() == ["wide"]


def test_select_top_variance_preserves_original_column_order() -> None:
    """The surviving columns keep their position in the frame."""
    frame = pd.DataFrame({"a": [0.0, 9.0], "b": [0.0, 1.0], "c": [0.0, 8.0]})
    assert select_top_variance(frame, 2).columns.tolist() == ["a", "c"]


def test_select_top_variance_is_a_no_op_when_k_exceeds_the_width() -> None:
    """Asking for more columns than exist returns the frame unchanged."""
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    assert select_top_variance(frame, 99).equals(frame)


def test_select_top_variance_rejects_non_positive_k() -> None:
    """A non-positive top_k is a programming error."""
    with pytest.raises(ValueError, match="must be positive"):
        select_top_variance(pd.DataFrame({"a": [1.0]}), 0)


def test_extract_mvts_features_produces_one_value_per_parameter_statistic() -> None:
    """SWAN-SF's 24 parameters and 8 statistics give the declared 192 features."""
    series = np.random.default_rng(0).normal(size=(60, 24))
    assert extract_mvts_features(series).shape == (24 * len(MVTS_STATISTICS),)


def test_extract_mvts_features_is_parameter_major() -> None:
    """All statistics of one parameter precede the next parameter's."""
    series = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    values = extract_mvts_features(series, ["mean", "min"])
    assert values.tolist() == [2.0, 0.0, 20.0, 10.0]


def test_extract_mvts_features_computes_a_least_squares_slope() -> None:
    """A perfectly linear parameter recovers its slope exactly."""
    series = np.array([[0.0], [3.0], [6.0], [9.0]])
    assert extract_mvts_features(series, ["slope"])[0] == pytest.approx(3.0)


def test_extract_mvts_features_ignores_missing_timesteps() -> None:
    """NaN timesteps are dropped per parameter rather than poisoning the mean."""
    series = np.array([[1.0], [np.nan], [3.0]])
    assert extract_mvts_features(series, ["mean"])[0] == pytest.approx(2.0)


def test_extract_mvts_features_returns_zeros_for_sparse_parameters() -> None:
    """A parameter with fewer than two observations yields zeros, never NaN."""
    series = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    values = extract_mvts_features(series, ["mean", "std"])
    assert values[:2].tolist() == [0.0, 0.0]
    assert np.isfinite(values).all()


def test_extract_mvts_features_output_is_always_finite() -> None:
    """A constant parameter has no defined skew or autocorrelation; both are 0."""
    series = np.full((10, 1), 7.0)
    assert np.isfinite(extract_mvts_features(series)).all()


def test_extract_mvts_features_rejects_a_one_dimensional_input() -> None:
    """The input must be (timesteps, parameters)."""
    with pytest.raises(ValueError, match="must be 2-D"):
        extract_mvts_features(np.arange(5.0))


def test_extract_mvts_features_rejects_an_unknown_statistic() -> None:
    """An unsupported statistic names the ones that are supported."""
    with pytest.raises(ValueError, match="Unsupported statistic"):
        extract_mvts_features(np.zeros((3, 1)), ["median"])


# ── Vibration segmentation and condition indicators ───────────────────

def test_segment_signal_cuts_non_overlapping_windows() -> None:
    """A signal splits into floor(n / window) windows when nothing overlaps."""
    assert segment_signal(np.arange(10.0), window=4).shape == (2, 4)


def test_segment_signal_honours_overlap() -> None:
    """Overlapping windows advance by window minus overlap."""
    segments = segment_signal(np.arange(10.0), window=4, overlap=2)
    assert segments.shape == (4, 4)
    assert segments[1].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_segment_signal_respects_the_limit() -> None:
    """`limit` caps how many leading windows are kept."""
    assert segment_signal(np.arange(100.0), window=4, limit=3).shape == (3, 4)


def test_segment_signal_rejects_a_signal_shorter_than_one_window() -> None:
    """A recording too short to fill one window is an error, not an empty array."""
    with pytest.raises(ValueError, match="shorter than one window"):
        segment_signal(np.arange(3.0), window=4)


def test_segment_signal_rejects_overlap_at_or_above_the_window() -> None:
    """An overlap equal to the window would never advance."""
    with pytest.raises(ValueError, match=r"\[0, window\)"):
        segment_signal(np.arange(10.0), window=4, overlap=4)


def test_time_domain_features_produce_one_column_per_statistic() -> None:
    """The nine COMIA condition indicators give nine columns."""
    windows = np.random.default_rng(0).normal(size=(20, 2048))
    assert extract_time_domain_features(windows).shape == (20, len(TIME_DOMAIN_STATISTICS))


def test_time_domain_features_use_the_sample_standard_deviation() -> None:
    """`sd` is the ddof=1 estimator, which is what reproduces COMIA's CSV."""
    window = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    got = extract_time_domain_features(window, ["sd"])[0, 0]
    assert got == pytest.approx(np.std(window[0], ddof=1))


def test_time_domain_kurtosis_is_excess_over_the_sample_sd() -> None:
    """Kurtosis subtracts three and standardizes by the sample sd."""
    rng = np.random.default_rng(1)
    window = rng.normal(size=(1, 4096))
    values = window[0]
    expected = ((values - values.mean()) ** 4).mean() / values.std(ddof=1) ** 4 - 3.0
    assert extract_time_domain_features(window, ["kurtosis"])[0, 0] == pytest.approx(expected)


def test_time_domain_crest_and_form_follow_comia_definitions() -> None:
    """crest = max / rms and form = rms / mean, both on the raw window."""
    window = np.array([[1.0, 2.0, 3.0, 4.0]])
    rms = np.sqrt((window[0] ** 2).mean())
    got = extract_time_domain_features(window, ["crest", "form"])[0]
    assert got[0] == pytest.approx(4.0 / rms)
    assert got[1] == pytest.approx(rms / window[0].mean())


def test_time_domain_features_reject_an_unknown_statistic() -> None:
    """An unsupported indicator names the supported ones."""
    with pytest.raises(ValueError, match="Unsupported statistic"):
        extract_time_domain_features(np.zeros((2, 8)), ["entropy"])


def test_spectral_features_halve_the_window_length() -> None:
    """A 256-sample window becomes 128 magnitude bins."""
    assert extract_spectral_features(np.ones((3, 256))).shape == (3, 128)


def test_spectral_features_are_non_negative_magnitudes() -> None:
    """FFT magnitudes are absolute values, so never negative."""
    windows = np.random.default_rng(0).normal(size=(5, 64))
    assert (extract_spectral_features(windows) >= 0).all()


def test_spectral_features_reject_a_one_dimensional_input() -> None:
    """The input must be (n_windows, window)."""
    with pytest.raises(ValueError, match="must be 2-D"):
        extract_spectral_features(np.ones(16))


# ── Canonical contract ────────────────────────────────────────────────

def test_check_canonical_accepts_a_valid_frame(canonical_frame: pd.DataFrame) -> None:
    """A conforming frame passes validation silently."""
    assert DatasetPreprocessor.check_canonical(canonical_frame) is None


def test_check_canonical_rejects_a_missing_target(canonical_frame: pd.DataFrame) -> None:
    """A frame without a target column is rejected."""
    with pytest.raises(PreprocessingError, match="must contain a 'target'"):
        DatasetPreprocessor.check_canonical(canonical_frame.drop(columns=["target"]))


def test_check_canonical_rejects_a_non_binary_target(canonical_frame: pd.DataFrame) -> None:
    """Multi-class targets violate the binary contract."""
    frame = canonical_frame.copy()
    frame.loc[0, "target"] = 2
    with pytest.raises(PreprocessingError, match="binary 0/1"):
        DatasetPreprocessor.check_canonical(frame)


def test_check_canonical_rejects_a_single_class_target(canonical_frame: pd.DataFrame) -> None:
    """A target with only one class present is rejected."""
    frame = canonical_frame.copy()
    frame["target"] = 0
    with pytest.raises(PreprocessingError, match="both classes"):
        DatasetPreprocessor.check_canonical(frame)


def test_check_canonical_rejects_an_inverted_polarity(canonical_frame: pd.DataFrame) -> None:
    """Label 1 must be the minority, or minority-centred measures read backwards."""
    frame = canonical_frame.copy()
    frame["target"] = [1, 1, 1, 1, 0, 0]
    with pytest.raises(PreprocessingError, match="must be the minority"):
        DatasetPreprocessor.check_canonical(frame)


def test_check_canonical_allows_perfectly_balanced_classes(
    canonical_frame: pd.DataFrame,
) -> None:
    """A 50/50 split has no minority, so it is accepted rather than rejected."""
    frame = canonical_frame.copy()
    frame["target"] = [0, 0, 0, 1, 1, 1]
    assert DatasetPreprocessor.check_canonical(frame) is None


def test_check_canonical_rejects_categorical_features(canonical_frame: pd.DataFrame) -> None:
    """Un-encoded categorical columns violate the contract."""
    frame = canonical_frame.copy()
    frame["category"] = "a"
    with pytest.raises(PreprocessingError, match="Non-numeric"):
        DatasetPreprocessor.check_canonical(frame)


def test_check_canonical_rejects_missing_values(canonical_frame: pd.DataFrame) -> None:
    """Any residual NaN is rejected before serialization."""
    frame = canonical_frame.copy()
    frame.loc[0, "f1"] = np.nan
    with pytest.raises(PreprocessingError, match="missing value"):
        DatasetPreprocessor.check_canonical(frame)


def test_check_canonical_rejects_infinities(canonical_frame: pd.DataFrame) -> None:
    """Infinities break downstream complexity measures and are rejected."""
    frame = canonical_frame.copy()
    frame.loc[0, "f1"] = np.inf
    with pytest.raises(PreprocessingError, match="infinite value"):
        DatasetPreprocessor.check_canonical(frame)


def test_write_serializes_a_readable_parquet(
    tmp_path: Path, canonical_frame: pd.DataFrame
) -> None:
    """write() produces a parquet file that round-trips unchanged."""
    path = DatasetPreprocessor().write(canonical_frame, tmp_path / "alpha.parquet")
    assert pd.read_parquet(path).equals(canonical_frame)


def test_write_refuses_an_invalid_frame(tmp_path: Path, canonical_frame: pd.DataFrame) -> None:
    """Validation runs before serialization, so no file is left behind."""
    path = tmp_path / "alpha.parquet"
    with pytest.raises(PreprocessingError):
        DatasetPreprocessor().write(canonical_frame.drop(columns=["target"]), path)
    assert not path.exists()


# ── Dispatch ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", PILOT_KEYS)
def test_pilot_datasets_have_a_preprocessing_routine(name: str) -> None:
    """Every pilot dataset is implemented."""
    assert DatasetPreprocessor().supports(name)


def test_every_implemented_routine_names_a_registered_dataset() -> None:
    """No `_preprocess_*` helper leaks into dispatch as a phantom dataset.

    A shared helper named `_preprocess_promise` once made `implemented()`
    report a dataset key that the registry does not contain.
    """
    from imbdata.registry import DatasetRegistry

    registered = set(DatasetRegistry().list_all())
    assert set(DatasetPreprocessor().implemented()) <= registered


def test_unknown_ozone_variant_names_the_declared_ones(tmp_path: Path) -> None:
    """Asking for a variant the registry does not declare fails loudly."""
    meta = {"filename": "onehr.data", "variant_files": {"eighthr": "eighthr.data"}}
    with pytest.raises(PreprocessingError, match="Unknown ozone_level variant"):
        DatasetPreprocessor().preprocess(
            "ozone_level", tmp_path, tmp_path / "out.parquet", meta, variant="twelvehr"
        )


def test_preprocess_unimplemented_dataset_raises(tmp_path: Path) -> None:
    """Dispatching to a dataset with no routine reports how to add one."""
    with pytest.raises(PreprocessingError, match="No preprocessing routine"):
        DatasetPreprocessor().preprocess(
            "not_a_dataset", tmp_path, tmp_path / "out.parquet", meta={}
        )


def test_locate_reports_the_available_files_when_nothing_matches(tmp_path: Path) -> None:
    """A missing raw file error names what the directory actually holds."""
    (tmp_path / "present.txt").write_text("x", encoding="utf-8")
    with pytest.raises(PreprocessingError, match="present.txt"):
        DatasetPreprocessor.locate(tmp_path, "absent.data")


def test_locate_finds_files_nested_by_archive_extraction(tmp_path: Path) -> None:
    """Files an archive placed in a subdirectory are still found."""
    nested = tmp_path / "bundle"
    nested.mkdir()
    (nested / "wdbc.data").write_text("x", encoding="utf-8")
    assert DatasetPreprocessor.locate(tmp_path, "wdbc.data").name == "wdbc.data"


def test_read_libsvm_densifies_sparse_rows(tmp_path: Path) -> None:
    """Absent LIBSVM indices are materialized as zeros."""
    path = tmp_path / "sparse"
    path.write_text("1 1:2.0 3:4.0\n0 2:1.0\n", encoding="utf-8")
    frame = DatasetPreprocessor._read_libsvm(path)
    assert frame.columns.tolist() == ["label", "x1", "x2", "x3"]
    assert frame.iloc[1].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_read_libsvm_on_an_empty_file_raises(tmp_path: Path) -> None:
    """A LIBSVM file with no instances is a preprocessing error."""
    path = tmp_path / "empty"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(PreprocessingError, match="no instances"):
        DatasetPreprocessor._read_libsvm(path)
