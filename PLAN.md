# imbdata — Imbalanced Classification Dataset Repository

**Purpose:** Centralized local repository for downloading, versioning, preprocessing,
and serving imbalanced classification benchmark datasets across multiple research projects.
**Author:** Luis García Rodríguez · IIMAS-UNAM · CVU 905206
**Version:** 0.1.0
**License:** MIT

---

## Problem Solved

Multiple doctoral projects (CIPA Extended, HEAD-Fraud, SynthDrift, TXAI-Audit, climate
anomaly detection) require overlapping datasets. Without a shared repository:
- PaySim (493 MB), CIC-IDS-2017 (~1.3 GB), IEEE-CIS (~1.2 GB) get duplicated per project
- Different preprocessing in different projects → silent version divergence
- Download scripts are reimplemented per project
- SHA-256 verification logic is duplicated

`imbdata` solves this by providing a single installable Python package that manages a
central data store (`~/.imbdata/`) with download, preprocessing, caching, and versioning.

---

## Design Principles

1. **Single store, multiple consumers.** Datasets live in `~/.imbdata/`, not in project directories.
2. **Immutable processed files.** Once a dataset version is preprocessed and cached, it is never
   silently re-preprocessed. Version changes require explicit `imbdata update`.
3. **Lazy download.** Datasets are downloaded on first access, not on install.
4. **Deterministic preprocessing.** Every preprocessing step is logged in `manifest.json`
   with SHA-256 hashes, so any consumer can verify they have the exact same data.
5. **Minimal API.** A consumer project needs exactly two calls: `imbdata.load(name)` and `imbdata.list_datasets()`.
6. **Offline-friendly.** Once downloaded, no network access is needed.

---

## Coding Standards & Conventions

These standards apply to `imbdata` and to all consumer projects (CIPA Extended,
HEAD-Fraud, SynthDrift, TXAI-Audit). They are non-negotiable for every file committed.

### Normative References

| Priority | Standard | Scope |
|----------|----------|-------|
| 1 (primary) | PEP 8 — Style Guide for Python Code | Formatting, naming, layout |
| 2 (complement) | Google Python Style Guide | Docstrings, imports, exceptions, type annotations |
| 3 (reference) | PSF — Python Software Foundation guidelines | Packaging, distribution, versioning |

When PEP 8 and Google conflict, PEP 8 wins. Google Style Guide extends PEP 8 on
docstring format (Google-style docstrings), import ordering, and exception handling.
PSF guidelines govern `pyproject.toml`, versioning (SemVer), and distribution.

### Programming Paradigm

**Primary: Object-Oriented Programming (OOP).**
All modules that manage state, encapsulate behavior, or represent domain entities
use classes. This includes download engines, preprocessors, the dataset registry,
and the configuration manager.

**Complementary: Functional Programming (FP).**
Pure functions are preferred for stateless transformations: feature extraction,
statistical computations, data validation, and utility operations. FP is justified
when the function has no side effects, takes inputs and returns outputs without
mutating state, and would gain nothing from being wrapped in a class.

**Decision rule:** If it manages state or resources (files, connections, caches) → class.
If it transforms data without side effects → function. If uncertain → class (OOP is primary).

### Naming Conventions (PEP 8)

| Element | Convention | Example |
|---------|-----------|---------|
| Package / module | lowercase, underscores | `imbdata`, `download.py` |
| Class | CamelCase | `DatasetRegistry`, `KaggleDownloader` |
| Method / function | snake_case | `load_dataset()`, `compute_sha256()` |
| Constant | UPPER_SNAKE | `DEFAULT_STORE_PATH`, `MAX_RETRIES` |
| Private attribute | leading underscore | `self._cache`, `_parse_config()` |
| Type variable | CamelCase | `T`, `DataFrameT` |
| File / dataset key | snake_case | `credit_card_fraud.parquet` |

### Import Order (Google Style, enforced by isort)

```python
# 1. Standard library
import json
import logging
from pathlib import Path

# 2. Third-party packages
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 3. Local/project imports
from imbdata.config import store_path
from imbdata.registry import DatasetRegistry
```

### Type Annotations (PEP 484 + PEP 604)

All public functions and methods carry type annotations. Use `from __future__ import annotations`
at the top of every module for PEP 604 union syntax (`str | None` instead of `Optional[str]`).

```python
from __future__ import annotations

def load(name: str, variant: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    ...
```

### Docstring Standard

**Format: Google-style docstrings** (per Google Python Style Guide §3.8).
Every module, class, method, and public function must have a docstring.

#### Module-Level Docstring (mandatory in every .py file)

```python
\"\"\"
imbdata.download — Dataset download engines.

Provides download engines for Kaggle, UCI, OpenML, GitHub, and generic HTTP
sources. Each engine implements the DownloadEngine protocol and manages
retry logic, progress reporting, and integrity verification.

Author:
    Luis García Rodríguez
    Doctorado en Ciencia e Ingeniería de la Computación (DCIC)
    IIMAS — Universidad Nacional Autónoma de México (UNAM)
    CVU: 905206 · ORCID: 0009-0004-9514-5508

Project:
    imbdata v0.1.0 — Imbalanced Classification Dataset Repository
    Advisor: Dr. José Antonio Neme Castillo
    Research Group: Anomalocaris
\"\"\"
```

#### Class Docstring

```python
class KaggleDownloader:
    \"\"\"Download engine for Kaggle-hosted datasets.

    Manages authentication via ~/.kaggle/kaggle.json, handles rate
    limiting, and supports both dataset and competition downloads.

    Attributes:
        credentials_path: Path to the Kaggle API credentials file.
        max_retries: Maximum number of download retry attempts.
        timeout: Request timeout in seconds.

    Example:
        >>> downloader = KaggleDownloader()
        >>> downloader.download("mlg-ulb/creditcardfraud", target_dir)
    \"\"\"
```

#### Method / Function Docstring

```python
def download(self, dataset_slug: str, target_dir: Path) -> Path:
    \"\"\"Download a dataset from Kaggle to the specified directory.

    Downloads the dataset identified by its Kaggle slug, extracts
    compressed files, and returns the path to the downloaded content.

    Args:
        dataset_slug: Kaggle dataset identifier in the format
            'owner/dataset-name' (e.g., 'mlg-ulb/creditcardfraud').
        target_dir: Local directory where files will be saved.
            Created if it does not exist.

    Returns:
        Path to the directory containing the downloaded files.

    Raises:
        FileNotFoundError: If Kaggle credentials are not configured.
        ConnectionError: If the download fails after max_retries attempts.
        ValueError: If dataset_slug format is invalid.

    Example:
        >>> downloader = KaggleDownloader(max_retries=3)
        >>> path = downloader.download("mlg-ulb/creditcardfraud", Path("/tmp/data"))
        >>> list(path.glob("*.csv"))
        [PosixPath('/tmp/data/creditcard.csv')]
    \"\"\"
```

### Error Handling (Google Style §2.4)

- Use specific exception types, never bare `except:`.
- Document raised exceptions in docstrings.
- Use custom exception classes for domain-specific errors:

```python
class ImbdataError(Exception):
    \"\"\"Base exception for all imbdata errors.\"\"\"

class DatasetNotFoundError(ImbdataError):
    \"\"\"Raised when a dataset key is not in the registry.\"\"\"

class DownloadError(ImbdataError):
    \"\"\"Raised when a dataset download fails.\"\"\"

class IntegrityError(ImbdataError):
    \"\"\"Raised when SHA-256 verification fails.\"\"\"
```

### Line Length & Formatting

- Maximum line length: 99 characters (PEP 8 relaxation for readability).
- Indentation: 4 spaces (never tabs).
- Trailing commas in multi-line structures (Google Style §3.4).
- f-strings preferred over .format() or % formatting.

### Testing (pytest)

- Test file naming: `test_{module}.py`
- Test function naming: `test_{method}_{scenario}_{expected}`
- Use fixtures for shared setup.
- Minimum: one test per public method, covering happy path + one error case.

### Logging

- Use `logging` module, never `print()` for operational messages.
- Logger per module: `logger = logging.getLogger(__name__)`
- Levels: DEBUG for internals, INFO for user-visible operations, WARNING for recoverable issues, ERROR for failures.

---

## OOP Architecture

### Core Classes

```
DatasetRegistry          # Reads datasets.yaml, provides metadata lookup
├── get(name) → dict
├── list_all() → list[str]
├── filter(domain=...) → list[str]
└── validate() → bool

DownloadEngine (Protocol) # Abstract interface for download engines
├── download(meta, target_dir) → Path
└── supports(source_type) → bool

KaggleDownloader(DownloadEngine)
HttpDownloader(DownloadEngine)
OpenMLDownloader(DownloadEngine)
GitHubDownloader(DownloadEngine)

DownloadManager           # Selects engine based on dataset source type
├── download(name) → Path
└── _select_engine(source) → DownloadEngine

DatasetPreprocessor       # Per-dataset preprocessing dispatch
├── preprocess(name, raw_dir, output_path) → None
├── _preprocess_credit_card_fraud(raw_dir, output_path) → None
├── _preprocess_paysim(...) → None
└── ... (one method per dataset)

ManifestManager           # SHA-256 verification and manifest I/O
├── update(name, path) → None
├── verify(name) → str
├── verify_all() → dict[str, str]
└── read() → dict

StoreConfig               # Data store path resolution and config management
├── store_path() → Path
├── raw_dir(name) → Path
├── processed_path(name, variant?) → Path
└── ensure_dirs() → dict[str, Path]
```

### Functional Components (justified)

These are pure functions, stateless, with no side effects — FP is the natural fit:

```python
# verify.py — pure computation
def compute_sha256(path: Path) -> str: ...

# preprocess.py — pure data transformations (called by DatasetPreprocessor methods)
def binarize_column(series: pd.Series, minority_value) -> pd.Series: ...
def onehot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame: ...
def impute_median(df: pd.DataFrame) -> pd.DataFrame: ...
def extract_mvts_features(mvts: np.ndarray, statistics: list[str]) -> np.ndarray: ...
def normalize_target(y: pd.Series, minority_value, majority_value=None) -> pd.Series: ...
```

```
imbdata/
├── src/
│   └── imbdata/
│       ├── __init__.py           # Public API: load(), list_datasets(), info(), ensure()
│       ├── api.py                # Core API implementation
│       ├── registry.py           # Dataset registry (reads datasets.yaml)
│       ├── download.py           # Download engines (Kaggle, UCI, OpenML, HTTP, GitHub)
│       ├── preprocess.py         # Per-dataset preprocessing pipelines
│       ├── config.py             # Store path, versioning, configuration
│       ├── verify.py             # SHA-256 verification + manifest management
│       └── datasets.yaml         # Dataset registry (bundled with package)
├── tests/
│   ├── test_api.py
│   ├── test_registry.py
│   ├── test_preprocess.py
│   └── test_verify.py
├── pyproject.toml
├── README.md
├── PLAN.md                       # This file
├── LICENSE                       # MIT
└── .gitignore
```

### Data Store Layout (`~/.imbdata/`)

```
~/.imbdata/
├── config.json                   # User configuration (store path override, Kaggle creds path)
├── manifest.json                 # Global manifest: per-dataset {version, sha256_raw, sha256_processed, date}
├── raw/
│   ├── credit_card_fraud/
│   │   └── creditcard.csv
│   ├── paysim/
│   │   └── PS_20174392719_1491204439457_log.csv
│   └── ...
└── processed/
    ├── credit_card_fraud.parquet
    ├── paysim.parquet
    └── ...
```

---

## Public API

```python
import imbdata

# List all available datasets
names = imbdata.list_datasets()
# → ['abalone_19', 'adult_census', 'baf', 'breast_cancer_wisconsin', ...]

# Load a dataset (downloads + preprocesses on first call, then cached)
X, y = imbdata.load("credit_card_fraud")
# X: pd.DataFrame (float64 features), y: pd.Series (int: 0=majority, 1=minority)

# Load with metadata
X, y, meta = imbdata.load("credit_card_fraud", return_meta=True)
# meta: dict with {domain, ir, n_total, n_minority, d, source_url, sha256, version}

# Get dataset info without loading
info = imbdata.info("credit_card_fraud")
# → {'name': 'credit_card_fraud', 'domain': 'financial_fraud', 'N': 284807, ...}

# Ensure datasets are downloaded (bulk pre-download)
imbdata.ensure(["credit_card_fraud", "paysim", "ieee_cis_fraud"])

# Ensure all datasets in a domain
imbdata.ensure(domain="financial_fraud")

# Get store path
imbdata.store_path()
# → PosixPath('/home/luis/.imbdata')

# Filter datasets
financial = imbdata.list_datasets(domain="financial_fraud")
# → ['baf', 'credit_card_fraud', 'elliptic_bitcoin', 'ieee_cis_fraud', 'paysim']

# Verify integrity of all cached datasets
report = imbdata.verify()
# → {'credit_card_fraud': 'OK', 'paysim': 'OK', ...}
```

---

## Canonical Data Format

Every processed dataset is a Parquet file with:
- **Columns:** all numeric features (float64) + `target` (int: 0=majority, 1=minority)
- **No missing values** (imputed or dropped during preprocessing)
- **No categorical columns** (one-hot or ordinal encoded during preprocessing)
- **No index column** (default RangeIndex)
- **Filename:** `{dataset_key}.parquet`

This format contract is the interface between `imbdata` and all consumer projects.

---

## Dataset Registry (28 datasets, 10 domains)

| # | Key | Domain | N | d | IR | Source |
|---|-----|--------|---|---|-----|--------|
| 1 | credit_card_fraud | financial_fraud | 284,807 | 30 | 577:1 | Kaggle |
| 2 | paysim | financial_fraud | 6,354,407 | 11 | 744:1 | Kaggle |
| 3 | ieee_cis_fraud | financial_fraud | 590,540 | 434 | 28.6:1 | Kaggle |
| 4 | elliptic_bitcoin | financial_fraud | 46,564 | 165 | ~10:1 | Kaggle |
| 5 | baf | financial_fraud | 1,000,000 | 30 | ~90:1 | Kaggle |
| 6 | breast_cancer_wisconsin | medicine | 569 | 30 | 1.68:1 | UCI |
| 7 | pima_diabetes | medicine | 768 | 8 | 1.87:1 | Kaggle |
| 8 | tcga_brca | medicine | ~1,215 | ~20,500 | ~5.5:1 | LinkedOmics |
| 9 | mammography | medicine | 11,183 | 6 | 42:1 | OpenML |
| 10 | nsl_kdd | cybersecurity | 148,517 | 41 | Variable | UNB |
| 11 | cic_ids_2017 | cybersecurity | ~2,830,743 | 78 | Variable | UNB |
| 12 | unsw_nb15 | cybersecurity | 257,673 | 49 | ~30:1 | UNSW |
| 13 | cwru_bearing | manufacturing | Variable | Variable | Configurable | CWRU |
| 14 | seu_gearbox | manufacturing | ~105,600 | Variable | Configurable | GitHub |
| 15 | secom | manufacturing | 1,567 | 590 | 14:1 | UCI |
| 16 | svmguide1 | bioinformatics | 7,089 | 4 | 3.07:1 | LIBSVM |
| 17 | yeast_me3 | bioinformatics | 1,484 | 8 | 28:1 | KEEL |
| 18 | ecoli_imu | bioinformatics | 336 | 7 | ~8.6:1 | KEEL |
| 19 | vehicle_insurance_fraud | insurance | 15,420 | 33 | ~16:1 | Kaggle |
| 20 | iranian_churn | telecommunications | 3,150 | 13 | 5.39:1 | UCI |
| 21 | nasa_pc1 | software_engineering | 1,109 | 21 | ~13:1 | PROMISE |
| 22 | nasa_jm1 | software_engineering | 10,885 | 21 | ~4.2:1 | PROMISE |
| 23 | swan_sf | space_weather | ~4,075 | 192 | ~60:1 | Harvard Dataverse |
| 24 | spambase | digital_communications | 4,601 | 57 | 1.54:1 | UCI |
| 25 | ozone_level | environmental | 2,536 | 72 | 34:1 | UCI |
| 26 | adult_census | social_sciences | 48,842 | 14 | 3.17:1 | UCI |
| 27 | wine_quality_red | food_agriculture | 1,599 | 11 | ~26:1 | UCI |
| 28 | abalone_19 | marine_biology | 4,177 | 8 | 130:1 | UCI |

Domains: financial_fraud, medicine, cybersecurity, manufacturing, bioinformatics,
insurance, telecommunications, software_engineering, space_weather,
digital_communications, environmental, social_sciences, food_agriculture, marine_biology

---

## Preprocessing Specifications

All preprocessing logic lives in `src/imbdata/preprocess.py`. Each dataset has a
dedicated function `_preprocess_{key}()` that reads from `raw/` and writes to `processed/`.

### Binarization Protocol

| Dataset | Majority (0) | Minority (1) |
|---------|-------------|--------------|
| unsw_nb15 | Normal | Attack (9 types merged) |
| nsl_kdd | Normal | Attack |
| cic_ids_2017 | BENIGN | Attack (all types merged) |
| wine_quality_red | Quality ≤ 7 | Quality = 8 |
| abalone_19 | Rings ≠ 19 | Rings = 19 |
| swan_sf | FQ + B + C | M + X (major flare) |
| elliptic_bitcoin | Licit | Illicit (drop unlabeled) |
| cwru_bearing / seu_gearbox | Normal | Fault |
| ecoli_imu | All other | iMU |
| yeast_me3 | All other | ME3 |
| tcga_brca | Other PAM50 | Basal-like |

### Special Preprocessing

| Dataset | Procedure |
|---------|-----------|
| swan_sf | MVTS → 192 statistical features (8 stats × 24 params) |
| tcga_brca | log₂(RSEM+1); variance filter top-5K genes for full version; full d=20,500 also stored |
| secom | Drop features >50% missing; median imputation; record final d |
| ieee_cis_fraud | Join Transaction + Identity; onehot/ordinal encode categoricals; median impute |
| cic_ids_2017 | Concatenate daily CSVs; remove Inf/NaN rows |
| adult_census | One-hot encode 8 categorical columns; mode impute '?' |

### Dataset Variants

Some datasets support multiple preprocessed versions:

```python
# Default: top-5000 genes by variance
X, y = imbdata.load("tcga_brca")

# Full dimensionality (d=20,500)
X, y = imbdata.load("tcga_brca", variant="full")
```

Variants are stored as separate parquet files: `tcga_brca.parquet`, `tcga_brca__full.parquet`.

---

## Progress Tracking Protocol

**Regla obligatoria para Claude Code:** al completar cada módulo, tarea, batch,
o iteración significativa, se debe emitir un reporte de estatus y actualizar
el archivo `STATUS.md` en la raíz del proyecto. Este protocolo aplica tanto a
imbdata como a todos los proyectos consumidores (CIPA Extended, HEAD-Fraud, etc.).

### Cuándo emitir un reporte de estatus

1. **Al completar un módulo** (archivo .py funcional con tests pasando).
2. **Al completar un batch** (grupo de tareas relacionadas).
3. **Al alcanzar un checkpoint de verificación** (acceptance criteria pasando).
4. **Al encontrar un bloqueante** (error no resuelto, decisión pendiente, dependencia faltante).
5. **Al terminar una sesión de trabajo** en Claude Code (resumen de cierre).

### Formato del reporte (emitir en consola)

```
════════════════════════════════════════════════════════
📊 STATUS REPORT — {proyecto} · {fecha/hora}
════════════════════════════════════════════════════════

✅ COMPLETADO:
   - {tarea 1.1}: {descripción breve del resultado}
   - {tarea 1.2}: {descripción breve del resultado}

⚠️  EN PROGRESO:
   - {tarea 1.3}: {estado parcial, qué falta}

❌ BLOQUEADO:
   - {tarea 1.4}: {razón del bloqueo, qué se necesita}

📈 MÉTRICAS:
   - Módulos implementados: X / Y
   - Tests pasando: X / Y
   - Datasets funcionales: X / 28

🔜 SIGUIENTES PASOS:
   1. {próxima tarea inmediata}
   2. {tarea después de la anterior}
   3. {tarea siguiente}

⏱️  ESTIMACIÓN: {horas restantes para el siguiente checkpoint}
════════════════════════════════════════════════════════
```

### Archivo STATUS.md (mantenido automáticamente)

Claude Code debe crear y actualizar `STATUS.md` en la raíz del proyecto después
de cada reporte. El archivo es acumulativo (log de reportes, más reciente arriba):

```markdown
# STATUS — imbdata

## Último reporte: {fecha}

### Estado actual
- Fase: {fase actual}
- Módulos: {X}/{Y} implementados
- Tests: {X}/{Y} pasando
- Datasets: {X}/28 funcionales
- Bloqueantes: {lista o "ninguno"}

### Siguientes pasos
1. {paso 1}
2. {paso 2}
3. {paso 3}

---

## Historial

### {fecha anterior}
{reporte anterior resumido}
```

---

## Implementation Tasks

### Phase 1 — Core Package (Sep 1–4)

- [ ] **1.1** Initialize repository: `pyproject.toml`, README, LICENSE, .gitignore
- [ ] **1.2** Implement `exceptions.py`: `ImbdataError`, `DatasetNotFoundError`, `DownloadError`, `IntegrityError`
- [ ] **1.3** Implement `config.py` → clase `StoreConfig`: store path resolution, config.json management

> **🔖 Checkpoint 1.3:** `python -c "from imbdata.config import StoreConfig; s = StoreConfig(); print(s.store_path())"` → imprime `~/.imbdata`
> **📊 Emitir STATUS REPORT** — módulos: 2/8, tests: 0, datasets: 0/28

- [ ] **1.4** Implement `registry.py` → clase `DatasetRegistry`: parse `datasets.yaml`, expose metadata
- [ ] **1.5** Implement `verify.py` → clase `ManifestManager` + función `compute_sha256()`

> **🔖 Checkpoint 1.5:** `python -c "import imbdata; print(imbdata.list_datasets())"` → lista 28 nombres
> **📊 Emitir STATUS REPORT** — módulos: 4/8, tests: 0, datasets: 0/28

- [ ] **1.6** Implement `download.py` → clases `DownloadManager`, `HttpDownloader`, `KaggleDownloader`, `OpenMLDownloader`, `GitHubDownloader`
- [ ] **1.7** Implement `preprocess.py` → clase `DatasetPreprocessor` con los 5 datasets piloto (BCW, PIMA, SpamBase, SVMGUIDE1, Ecoli)
- [ ] **1.8** Refactor `api.py` → integrar `StoreConfig`, `DatasetRegistry`, `DownloadManager`, `DatasetPreprocessor`, `ManifestManager`
- [ ] **1.9** Update `__init__.py`: re-export public API

> **🔖 Checkpoint 1.9 — Pilot validation:**
> ```bash
> python -c "
> import imbdata
> for name in ['breast_cancer_wisconsin', 'pima_diabetes', 'spambase', 'svmguide1', 'ecoli_imu']:
>     X, y = imbdata.load(name)
>     ir = y.value_counts()[0] / y.value_counts()[1]
>     print(f'  ✓ {name}: N={len(y)}, d={X.shape[1]}, IR={ir:.1f}:1')
> print('✅ Pilot datasets OK')
> "
> ```
> **📊 Emitir STATUS REPORT** — módulos: 8/8, tests: 0, datasets: 5/28, siguiente: tests

### Phase 2 — Testing (Sep 4–5)

- [ ] **2.1** Unit tests for `DatasetRegistry` (parse YAML, filter by domain, unknown key raises `DatasetNotFoundError`)
- [ ] **2.2** Unit tests for `ManifestManager` (SHA-256, manifest CRUD)
- [ ] **2.3** Unit tests for `StoreConfig` (path resolution, dir creation)
- [ ] **2.4** Integration tests for API (load, list, info on pilot datasets)
- [ ] **2.5** End-to-end: download + preprocess + load + verify for pilot datasets

> **🔖 Checkpoint 2.5:** `pytest tests/ -v` → all green
> **📊 Emitir STATUS REPORT** — módulos: 8/8, tests: all passing, datasets: 5/28, siguiente: bulk download

### Phase 3 — Bulk Download & Validation (Sep 5–7)

#### Batch A — UCI / OpenML / HTTP datasets
- [ ] **3.1** Implement preprocessing for: mammography, iranian_churn, ozone_level, adult_census, wine_quality_red, abalone_19, secom, nsl_kdd

> **🔖 Checkpoint 3.1:** 13/28 datasets funcionales (5 pilot + 8 batch A)
> **📊 Emitir STATUS REPORT**

#### Batch B — Kaggle datasets
- [ ] **3.2** Implement preprocessing for: credit_card_fraud, paysim, ieee_cis_fraud, elliptic_bitcoin, baf, pima_diabetes (Kaggle), vehicle_insurance_fraud

> **🔖 Checkpoint 3.2:** 20/28 datasets funcionales
> **📊 Emitir STATUS REPORT**

#### Batch C — Special preprocessing datasets
- [ ] **3.3** Implement preprocessing for: swan_sf, tcga_brca, cic_ids_2017, unsw_nb15, cwru_bearing, seu_gearbox, nasa_pc1, nasa_jm1, yeast_me3

> **🔖 Checkpoint 3.3:** 28/28 datasets funcionales
> **📊 Emitir STATUS REPORT**

- [ ] **3.4** Freeze manifest.json with SHA-256 hashes
- [ ] **3.5** Run `imbdata.verify()` → all OK
- [ ] **3.6** Implement `cli.py` → clase `CLI` con subcomandos list, download, verify, info, status

> **🔖 Checkpoint FINAL:**
> ```bash
> python -c "
> import imbdata
> ds = imbdata.list_datasets()
> assert len(ds) == 28, f'Expected 28, got {len(ds)}'
> for name in ds:
>     X, y = imbdata.load(name)
>     assert set(y.unique()) == {0, 1}, f'{name}: target not binary'
>     assert X.isnull().sum().sum() == 0, f'{name}: has nulls'
>     ir = y.value_counts()[0] / y.value_counts()[1]
>     print(f'  ✓ {name}: N={len(y)}, d={X.shape[1]}, IR={ir:.1f}:1')
> print(f'✅ All {len(ds)} datasets loaded from {imbdata.store_path()}')
> "
> ```
> **📊 Emitir STATUS REPORT FINAL** — módulos: completo, tests: all passing, datasets: 28/28
> **📊 Actualizar STATUS.md** con reporte final y preparar handoff a CIPA Extended

---

## CLI (optional, nice-to-have)

```bash
# List all datasets
imbdata list

# List financial fraud datasets
imbdata list --domain financial_fraud

# Download specific datasets
imbdata download credit_card_fraud paysim

# Download all
imbdata download --all

# Verify integrity
imbdata verify

# Show info
imbdata info credit_card_fraud

# Show store path and size
imbdata status
```

Implemented via entry point in `pyproject.toml`:
```toml
[project.scripts]
imbdata = "imbdata.cli:main"
```

---

## Consumer Integration Pattern

In any research project (CIPA Extended, HEAD-Fraud, etc.):

```python
# requirements.txt (or pyproject.toml dependency)
# imbdata @ file:///home/luis/repos/imbdata  (editable local install)
# or: imbdata @ git+https://github.com/luisgarciarodriguez-research/imbdata.git

import imbdata

# Works immediately — no data/ directory, no download scripts, no preprocessing code
X, y = imbdata.load("credit_card_fraud")

# Bulk pre-download for a project
imbdata.ensure(domain="financial_fraud")
imbdata.ensure(["mammography", "secom", "swan_sf"])
```

---

## Extensibility

Adding a new dataset requires only:
1. Add entry to `datasets.yaml` (metadata, URL, preprocessing params)
2. Add `_preprocess_{key}()` function to `preprocess.py`
3. Run `imbdata download {key}` to verify download + preprocessing
4. Commit → version bump

No consumer project needs any change.

---

## Risk Registry

| Risk | P | I | Mitigation |
|------|---|---|------------|
| Kaggle API changes authentication | B | M | Fallback: manual download to raw/ + auto-detect |
| UCI/source URL changes or breaks | M | M | manifest.json records last-known-good hash; detect stale download |
| Large downloads fail mid-transfer | M | B | Use requests with retry + resume; partial download detection |
| Preprocessing divergence across machines | B | A | SHA-256 of processed files in manifest; verify() on each load (optional, configurable) |
| Package grows too large for pip install | B | B | datasets.yaml is <50 KB; actual data never in package, always lazy-downloaded |
