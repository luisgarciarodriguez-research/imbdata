# Changelog

All notable changes to `imbdata` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, a change to the data a dataset key serves is
treated as breaking and bumps the minor version: a consumer that pins a version
and upgrades would otherwise receive different rows without notice.

`STATUS.md` carries the reasoning behind each decision; this file records what
changed and which datasets it moves.

## [0.4.0] — 2026-09-17

Adds one dataset key and one new API surface. No existing dataset key changes
the data it serves, and no existing SHA-256 changes.

### Added

- **`saml_d`** (SAML-D, synthetic AML transaction monitoring; Kaggle,
  `berkanoztas/synthetic-transaction-monitoring-dataset-aml`), in the
  `financial_fraud` domain: 9,504,852 transactions, 71 features, 9,873
  laundering rows (IR 961.7:1). `Date` and `Time` fold into the numeric
  `timestamp` column (Unix seconds, UTC, via the new
  `combine_date_time_epoch`); the surrogate account keys and the
  `Laundering_type` typology are dropped, the latter because it filters the
  target; the five remaining categorical columns are one-hot encoded. Cited as
  "Oztas et al. IEEE ICEBE 2023", licence CC BY-NC-SA 4.0. **No native concept
  drift: any drift used with SAML-D is researcher-constructed.**
- `ONEHOT_MAX_LEVELS` (50) and the cardinality split behind it: a declared
  categorical above that many levels is ordinal-encoded instead, the criterion
  that already justifies `ieee_cis_fraud`. No published SAML-D column reaches
  it.

- **`imbdata.fraud`**, an endpoint serving the per-row context the canonical
  format cannot carry: `fraud.list_datasets()`, `fraud.info()`, `fraud.load()`,
  `fraud.ensure()` and `fraud.verify()`, returning a frozen `FraudDataset`
  (`name`, `X`, `y`, `context`, `nodes`, `edges`, `meta`). `X` and `y` are
  exactly what `imbdata.load()` returns; `context` has one row per row of `X`,
  in the same order, with a fixed ten-column schema (`t`, `t_seconds`,
  `event_time`, `amount`, `currency`, `src_id`, `dst_id`, `entity_id`,
  `node_id`, `typology`). A concept a dataset does not publish is an all-null
  column, declared as such in its registry block, so one pipeline covers every
  dataset.
  Alignment is by construction: each served dataset's raw read step is now a
  shared `_read_<key>` reader (`DatasetPreprocessor.read_raw`) feeding both the
  canonical parquet and the context.
- **An optional `fraud:` block in the registry**, declared for the five
  datasets of the fraud study (`credit_card_fraud`, `paysim`,
  `ieee_cis_fraud`, `elliptic_bitcoin`, `saml_d`): the native time column and
  its unit (`second`, `hour` or `step`), whether that time is a calendar
  instant, the amount and its currency, the graph (`account` or `transaction`),
  the entity, `synthetic`, and optionally `typology` and `drift_provenance`.
  `DatasetRegistry.validate()` checks its schema and enumerations;
  `REQUIRED_FIELDS` is unchanged, so user registries without the block still
  validate. New `DatasetRegistry.filter(fraud=True)` and `fraud_block()`.
- **Elliptic's graph**, served whole: `nodes` holds all 203,769 nodes with
  their label and their row in `X` (`<NA>` for the 157,205 unlabelled ones),
  and `edges` all 234,355 edges, including those reaching unlabelled nodes.
  Dropping them would cut the graph a consumer measures, while `X`, `y` and
  `context` keep the 46,564 labelled rows.
- **`NotAFraudDatasetError`**, raised when a registered dataset declares no
  `fraud` block. An unknown key still raises `DatasetNotFoundError`.
- **`imbdata fraud list|info|download`** on the command line, and
  `StoreConfig.fraud_dir()`/`fraud_path()` for the artefact locations.
- `combine_date_time_epoch`, `to_seconds` and `encode_accounts` as pure
  functions, the last factorizing payer and payee over the sorted union of
  their values so an account keeps one code on both sides, independent of row
  order.

### Changed

- The registry grows from 30 to 31 entries, in the same 15 domains, so
  `list_datasets()` returns 31 keys and
  `list_datasets(domain="financial_fraud")` returns 6 instead of 5.
- `DatasetPreprocessor` gained the shared readers (`read_raw`, `reader_for`,
  `has_reader`, `read_elliptic_edges`) and the `_preprocess_*` routines of the
  five fraud datasets now build on them. **The canonical parquet of all five is
  byte-identical to 0.3.1**: rebuilt in a temporary store, the SHA-256 of
  `credit_card_fraud`, `paysim`, `ieee_cis_fraud` and `elliptic_bitcoin`
  matches the manifest exactly.

### Storage

- Context artefacts live in `<store>/processed/fraud/<key>.<part>.parquet` and
  are recorded in the store's single `manifest.json` under the `fraud/`
  namespace (`fraud/<key>.<part>`), with their digest, size, row count and
  path. `imbdata.verify()` and `imbdata verify` therefore report exactly the
  31 canonical datasets, before and after the artefacts exist; `fraud.verify()`
  reports the artefacts. Like the canonical files they are immutable once
  written and are only rebuilt on an explicit `force=True`.

### Notes

- **No existing dataset key changes the data it serves, and no existing
  SHA-256 changes.** 0.4.0 is a minor bump because the set a key listing
  returns changes, not the data.
- `context['typology']` is derived from the label and must never be used as a
  feature; the warning is in `FraudDataset`'s docstring and in PLAN.md.
- What `imbdata` deliberately does not decide, because they are the consumer's
  method: a surrogate entity for IEEE-CIS, currency conversion for SAML-D, the
  temporal windows of PaySim and the ULB dataset, how to treat Elliptic's
  unknown nodes, and any calendar anchor for the datasets whose origin was
  never published.

---

## [0.3.1] — 2026-09-16

Metadata only: no dataset key changes the data it serves, and no SHA-256
changes.

### Added

- **A `license` block in all 30 registry entries**, with `name`, `spdx`,
  `status`, `url`, `restrictions`, `cite`, `checked` and, where needed, `notes`.
  `status` is `declared` (a standard licence from the owner: 17 datasets),
  `owner_terms` (the owner's own terms, with no standard licence: 7) or
  `none_declared` (no licence or terms from the owner: 6). The block records
  the owner's terms, never a mirror's: the CC0 labels on the Kaggle copies of
  `cic_ids_2017`, `vehicle_insurance_fraud` and the former `pima_diabetes`
  upload were applied by uploaders who do not own the data. It records facts,
  not whether a given use is allowed; each consumer decides that.
- `info()` returns the block, and `imbdata info` prints it as dotted keys
  (`license.spdx`, `license.restrictions`, …).
- A test requiring a valid `license.status` in every bundled entry. The field
  is deliberately not in `REQUIRED_FIELDS`, so user registries without it still
  validate.

## [0.3.0] — 2026-09-14

No existing dataset key changes the data it serves.

### Added

- **`wine_quality_white`** (N=4,898, d=11, 183 minority, IR 25.8:1, CC BY 4.0).
  The minority class is the low-quality tail (`quality <= 4`), the binarization
  the imbalanced-learn benchmark uses, not the single grade the red wine file
  takes from KEEL. Both wines come from the same UCI archive and the same study.
- **`satimage`** (N=6,435, d=36, 626 minority, IR 9.3:1, CC BY 4.0), the Statlog
  Landsat Satellite dataset with class 4 ("damp grey soil") as the minority. It
  brings a new domain, `remote_sensing`, the fifteenth in the registry.
- `binarize_at_most()`: a pure transformation for ordinal targets whose minority
  class is a tail rather than a single level. `binarize_column()` tests equality
  and cannot express that.

### Changed

- The registry grows from 28 datasets in 14 domains to 30 in 15.

## [0.2.0] — 2026-09-10

### Changed

- **`ozone_level` now serves the 1-hour peak horizon** (N=2,536, 73 ozone days,
  IR 33.7:1). The 8-hour horizon it served before is kept as the `eighthr`
  variant (N=2,534, 160 ozone days, IR 14.8:1). Both label the same 72
  predictors over the same days, so the pair varies the imbalance ratio with the
  feature space held constant. **Breaking:** the default target changes.
- **`seu_gearbox` grows from N=2,500 to N=10,000**, raising `windows_per_file`
  from 250 to 1,000. `d` stays at 128 and IR at 4:1. **Breaking:** row count and
  content change.
- `numpy` and `scikit-learn` lose their `<2.0` ceilings; the package now
  installs against numpy 2.x. Verified under numpy 2.5.3 with the full suite and
  by rebuilding six datasets from raw.
- The version now has a single source of truth in `imbdata.__version__`.
  Hatchling reads it, the CLI reports it and the HTTP `User-Agent` derives from
  it, instead of each hardcoding the string separately.

### Added

- `variant_files` in the registry: a variant may point at a different source
  file, not only at a different filter. `ozone_level` uses it for its horizons.
- `check_canonical()` now enforces that class `1` is the minority. That half of
  the canonical contract was previously exercised only by a test, so a new
  preprocessing routine could invert the polarity and still pass validation.
  Perfectly balanced classes remain allowed.

### Fixed

- `fetch_files()` honours an explicit `extract: false`. It previously could not
  distinguish "unspecified" from "explicitly false", so an archive a
  preprocessor wants to stream could not be left packed.
- `implemented()` no longer reports a phantom dataset. A shared helper named
  `_preprocess_promise` was being discovered by the `_preprocess_<key>`
  dispatch; it is now `_promise_defect_dataset`, with a regression test
  asserting that every implemented routine names a registered dataset.

### Known limitation

- **`seu_gearbox` is not bit-reproducible across numpy major versions.** It is
  the only dataset whose features come from `numpy.fft.rfft`, and numpy 1.x and
  2.x differ in the last bit: rebuilding moves 62.3% of its cells, by at most
  4.4e-16 absolute and 5.6e-13 relative, with an identical `target`. The
  SHA-256 changes completely, so a store rebuilt under a different numpy major
  version reports `HASH_MISMATCH` for that one dataset. It is a version
  difference, not corruption — confirm by comparing values, not digests.

## [0.1.0] — 2026-09-08

Initial release. An installable package managing a local store (`~/.imbdata/`)
of 28 imbalanced classification benchmark datasets across 14 domains, with lazy
download, deterministic preprocessing, SHA-256 verification and a single
canonical format: `float64` features plus an `int64` `target` where `0` is the
majority and `1` the minority, free of missing values and categorical columns.

[0.4.0]: https://github.com/luisgarciarodriguez-research/imbdata/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/luisgarciarodriguez-research/imbdata/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/luisgarciarodriguez-research/imbdata/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/luisgarciarodriguez-research/imbdata/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/luisgarciarodriguez-research/imbdata/releases/tag/v0.1.0
