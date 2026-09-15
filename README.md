# imbdata

**Centralized dataset repository for imbalanced classification benchmarks.**

`imbdata` manages a single local data store (`~/.imbdata/`) that downloads,
preprocesses, caches, versions, and verifies the benchmark datasets shared
across several doctoral projects. Consumer projects stop carrying `data/`
directories, download scripts, and duplicated preprocessing code — they call
two functions instead.

```python
import imbdata

X, y = imbdata.load("credit_card_fraud")
```

- **Author:** Luis García Rodríguez · DCIC, IIMAS-UNAM · CVU 905206 · ORCID [0009-0004-9514-5508](https://orcid.org/0009-0004-9514-5508)
- **Advisor:** Dr. José Antonio Neme Castillo · Anomalocaris, IIMAS-UNAM
- **Version:** 0.3.0 · **License:** MIT

---

## Why

Several projects (CIPA Extended, HEAD-Fraud, SynthDrift, TXAI-Audit, climate
anomaly detection) need overlapping datasets. Without a shared repository,
PaySim (493 MB), CIC-IDS-2017 (~1.3 GB), and IEEE-CIS (~1.2 GB) get duplicated
per project, preprocessing silently diverges between them, and download plus
SHA-256 verification logic is reimplemented every time.

`imbdata` provides one installable package that owns all of it.

## Install

```bash
pip install -e .                  # from a local clone
pip install -e '.[kaggle,dev]'    # with the Kaggle CLI and the test tooling
```

Requires Python ≥ 3.10 and `pandas<3.0`. Both numpy 1.x and 2.x are supported
(verified under numpy 2.5.3); the one caveat is that `seu_gearbox` rebuilt from
raw under a different numpy major version gets a different SHA-256 (see
`CHANGELOG.md`, 0.2.0). If your default interpreter carries pandas 3.x, or a
pandas built against another numpy ABI, install into a dedicated virtual
environment:

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

## Public API

```python
import imbdata

# Load a dataset — downloads and preprocesses on first call, cached after
X, y = imbdata.load("credit_card_fraud")

# Load with metadata
X, y, meta = imbdata.load("credit_card_fraud", return_meta=True)

# List datasets, optionally by domain
imbdata.list_datasets()
imbdata.list_datasets(domain="financial_fraud")
# → ['baf', 'credit_card_fraud', 'elliptic_bitcoin', 'ieee_cis_fraud', 'paysim']

# Inspect metadata without loading the features
imbdata.info("credit_card_fraud")

# Pre-download in bulk
imbdata.ensure(["credit_card_fraud", "paysim"])
imbdata.ensure(domain="financial_fraud")

# Verify integrity of everything cached, and locate the store
imbdata.verify()      # → {'spambase': 'OK', 'paysim': 'MISSING', ...}
imbdata.store_path()  # → PosixPath('/home/luis/.imbdata')
```

## Canonical data format

Every processed dataset is a Parquet file with:

- all feature columns as **`float64`**, plus a **`target`** column of `int64`
  where **`0` = majority** and **`1` = minority**;
- **no missing values** (imputed or dropped during preprocessing);
- **no categorical columns** (one-hot or ordinal encoded);
- a default `RangeIndex`, no index column;
- filename `{dataset_key}.parquet`.

This contract is the interface between `imbdata` and every consumer project,
and it is enforced by `DatasetPreprocessor.check_canonical()` before any file
is written.

## Variants

Some datasets are served in more than one preprocessed form. Variants live
beside the default as `{key}__{variant}.parquet` and carry their own manifest
entry and SHA-256:

```python
X, y = imbdata.load("tcga_brca")                  # top-5,000 genes by variance
X, y = imbdata.load("tcga_brca", variant="full")  # all ~20,000 genes

X, y = imbdata.load("ozone_level")                    # 1-hour peak horizon, IR 33.7:1
X, y = imbdata.load("ozone_level", variant="eighthr") # 8-hour peak horizon, IR 14.8:1
```

The two `ozone_level` horizons share every feature and every day and differ
only in which days count as ozone days, so the pair varies the imbalance ratio
with the feature space held constant.

## Command line

```bash
imbdata list                          # every registered dataset
imbdata list --domain medicine        # filtered by domain
imbdata list --cached                 # only what is already materialized
imbdata info credit_card_fraud        # metadata, statistics, SHA-256
imbdata download spambase ecoli_imu   # download + preprocess
imbdata download --domain medicine    # a whole domain
imbdata download --all                # everything
imbdata verify                        # check cached files against the manifest
imbdata status                        # store path, size, and counters
```

## Data store layout

```
~/.imbdata/
├── config.json      # user configuration (store path override)
├── manifest.json    # per-dataset {sha256, size, rows, features, IR, timestamp}
├── raw/             # original downloaded files, one directory per dataset
│   ├── spambase/
│   └── ecoli_imu/
└── processed/       # canonical parquet files, ready to load
    ├── spambase.parquet
    └── ecoli_imu.parquet
```

Relocate the store with the `IMBDATA_STORE` environment variable, or with a
`store_path` key in `~/.imbdata/config.json`.

## Architecture

| Class | Module | Responsibility |
|-------|--------|----------------|
| `StoreConfig` | `config.py` | Store path resolution, directory tree, atomic JSON I/O |
| `DatasetRegistry` | `registry.py` | Parses `datasets.yaml`; lookup, filtering, validation |
| `DownloadManager` | `download.py` | Selects a download engine per dataset source |
| `HttpDownloader` etc. | `download.py` | HTTP, GitHub, Dataverse, OpenML, and Kaggle engines |
| `DatasetPreprocessor` | `preprocess.py` | One `_preprocess_<key>()` routine per dataset |
| `ManifestManager` | `verify.py` | SHA-256 recording and integrity verification |
| `DatasetService` | `api.py` | Orchestrates download → preprocess → record → load |
| `CLI` | `cli.py` | Argument parsing and subcommand dispatch |

Stateless transformations are plain functions: `compute_sha256`,
`binarize_column`, `binarize_at_most`, `normalize_target`, `onehot_encode`, `ordinal_encode`,
`impute_median`, `impute_mode`, `drop_high_missing`, `drop_constant_columns`,
`select_top_variance`, `to_numeric_frame`, `assemble_canonical`, and the
signal-summarizing trio `segment_signal`, `extract_time_domain_features`,
`extract_spectral_features` alongside `extract_mvts_features`.

Each module also exposes a thin function facade over a lazily-created default
instance, so the public API stays a two-call surface while remaining fully
injectable in tests.

## Registered datasets

30 datasets across 15 domains. `datasets.yaml` holds the metadata, the direct
file URLs, and the binarization, encoding, and imputation rules for each.

| Domain | Datasets |
|--------|----------|
| financial_fraud | `baf`, `credit_card_fraud`, `elliptic_bitcoin`, `ieee_cis_fraud`, `paysim` |
| medicine | `breast_cancer_wisconsin`, `mammography`, `pima_diabetes`, `tcga_brca` |
| cybersecurity | `cic_ids_2017`, `nsl_kdd`, `unsw_nb15` |
| manufacturing | `cwru_bearing`, `secom`, `seu_gearbox` |
| bioinformatics | `ecoli_imu`, `svmguide1`, `yeast_me3` |
| software_engineering | `nasa_jm1`, `nasa_pc1` |
| space_weather | `swan_sf` |
| remote_sensing | `satimage` |
| others | `abalone_19`, `adult_census`, `iranian_churn`, `ozone_level`, `spambase`, `vehicle_insurance_fraud`, `wine_quality_red`, `wine_quality_white` |

All 30 are implemented and cached; `imbdata verify` reports 30 OK.

Sources that need credentials or a manual step are handled as follows:

- **Kaggle** datasets use the `kaggle` CLI with `~/.kaggle/kaggle.json`. An
  entry may declare a `download.mirror`, which is used both when credentials
  are missing and when the API rejects them.
- **Registration-gated or decommissioned hosts** (CIC-IDS-2017's UNB portal,
  UNSW-NB15's retired CloudStor) are served from mirrors declared in the
  registry. For any dataset with no reachable source, placing the expected raw
  files in `~/.imbdata/raw/<key>/` by hand makes `load()` skip the download and
  go straight to preprocessing.

See [`STATUS.md`](STATUS.md) for which datasets are implemented, their measured
N/d/IR, and every discrepancy found against the reference table in `PLAN.md`.

## Adding a dataset

1. Add an entry to `src/imbdata/datasets.yaml` with its metadata and a
   `download:` block holding direct file URLs:

   ```yaml
   my_dataset:
     domain: medicine
     source: uci
     target_column: label
     minority_value: 1
     download:
       engine: http
       files:
         - url: https://example.org/my_dataset.zip
           extract: true
           expect: [my_dataset.data]
   ```

2. Add a `_preprocess_my_dataset()` method to `DatasetPreprocessor`.

   Keep every parameter that changes what the data *is* — window length,
   feature list, class polarity, dimensionality filters — in `datasets.yaml`
   rather than in the method, so a reader can see it and a reviewer can diff
   it. `cwru_bearing`, `seu_gearbox`, `swan_sf` and `tcga_brca` all follow this.
3. Run `imbdata download my_dataset` to verify both steps.
4. Commit and bump the version.

No consumer project needs any change.

## Testing

```bash
pytest tests/          # fast run, a few seconds
pytest tests/ -m ""    # everything, including the multi-gigabyte datasets
```

Unit tests run offline against temporary stores and synthetic fixtures.
Integration tests that need real data skip themselves when the dataset is not
yet cached, so the suite is green on a fresh checkout.

Cases that read or hash the datasets in the millions of rows (PaySim,
CIC-IDS-2017, IEEE-CIS, SWAN-SF, BAF, …) carry the `slow` marker and are
deselected by default through `addopts = "-m 'not slow'"` in `pyproject.toml`.

## Coding standards

PEP 8 (primary) with the Google Python Style Guide as a complement: Google-style
docstrings on every module, class, and public function; type annotations on
every public interface; a 99-character line limit; f-strings; specific exception
types derived from `ImbdataError`; and `logging` rather than `print()` for
operational messages. See `PLAN.md` for the full statement.

## Status

Current progress, checkpoints, and known discrepancies against the reference
dataset table are tracked in [`STATUS.md`](STATUS.md).

## License

MIT — see [`LICENSE`](LICENSE).
