# Changelog

All notable changes to `imbdata` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, a change to the data a dataset key serves is
treated as breaking and bumps the minor version: a consumer that pins a version
and upgrades would otherwise receive different rows without notice.

`STATUS.md` carries the reasoning behind each decision; this file records what
changed and which datasets it moves.

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

[0.2.0]: https://github.com/luisgarciarodriguez-research/imbdata/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/luisgarciarodriguez-research/imbdata/releases/tag/v0.1.0
