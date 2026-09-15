"""
tests.test_registry — Unit tests for :class:`imbdata.registry.DatasetRegistry`.

Covers YAML parsing, metadata lookup, domain/source filtering, structural
validation, and the error raised for unknown dataset keys.

Author:
    Luis García Rodríguez
    Doctorado en Ciencia e Ingeniería de la Computación (DCIC)
    IIMAS — Universidad Nacional Autónoma de México (UNAM)
    CVU: 905206 · ORCID: 0009-0004-9514-5508

Project:
    imbdata v0.3.0 — Imbalanced Classification Dataset Repository
    Advisor: Dr. José Antonio Neme Castillo
    Research Group: Anomalocaris
"""

from __future__ import annotations

from pathlib import Path

import pytest

from imbdata.exceptions import DatasetNotFoundError, RegistryError
from imbdata.registry import DatasetRegistry, get_dataset_meta, get_registry

EXPECTED_DATASET_COUNT = 30


def test_bundled_registry_declares_all_datasets() -> None:
    """The registry shipped with the package holds the 30 registered datasets."""
    registry = DatasetRegistry()
    assert len(registry) == EXPECTED_DATASET_COUNT
    assert len(registry.list_all()) == EXPECTED_DATASET_COUNT


def test_validate_bundled_registry_passes() -> None:
    """Every bundled entry declares the mandatory metadata fields."""
    assert DatasetRegistry().validate() is True


def test_get_known_key_returns_metadata(synthetic_registry_file: Path) -> None:
    """Looking up a registered key returns its metadata plus the key itself."""
    registry = DatasetRegistry(synthetic_registry_file)
    meta = registry.get("alpha_set")
    assert meta["domain"] == "medicine"
    assert meta["key"] == "alpha_set"


def test_get_unknown_key_raises_dataset_not_found(synthetic_registry_file: Path) -> None:
    """An unregistered key raises DatasetNotFoundError, not KeyError."""
    registry = DatasetRegistry(synthetic_registry_file)
    with pytest.raises(DatasetNotFoundError, match="Unknown dataset"):
        registry.get("delta_set")


def test_get_near_miss_key_suggests_the_intended_dataset(
    synthetic_registry_file: Path,
) -> None:
    """A typo close to a real key is answered with a suggestion."""
    registry = DatasetRegistry(synthetic_registry_file)
    with pytest.raises(DatasetNotFoundError, match="Did you mean 'alpha_set'"):
        registry.get("alpha_sett")


def test_get_returns_a_copy_so_callers_cannot_mutate_the_registry(
    synthetic_registry_file: Path,
) -> None:
    """Mutating a returned mapping does not corrupt the cached registry."""
    registry = DatasetRegistry(synthetic_registry_file)
    registry.get("alpha_set")["domain"] = "tampered"
    assert registry.get("alpha_set")["domain"] == "medicine"


def test_filter_by_domain_returns_only_that_domain(synthetic_registry_file: Path) -> None:
    """Domain filtering returns the sorted matching keys."""
    registry = DatasetRegistry(synthetic_registry_file)
    assert registry.filter(domain="medicine") == ["alpha_set", "beta_set"]


def test_filter_by_several_criteria_intersects_them(synthetic_registry_file: Path) -> None:
    """Multiple criteria are combined conjunctively."""
    registry = DatasetRegistry(synthetic_registry_file)
    assert registry.filter(domain="medicine", source="kaggle") == ["beta_set"]


def test_filter_without_matches_returns_empty_list(synthetic_registry_file: Path) -> None:
    """A filter nothing satisfies yields an empty list rather than raising."""
    assert DatasetRegistry(synthetic_registry_file).filter(domain="astronomy") == []


def test_domains_and_sources_are_deduplicated(synthetic_registry_file: Path) -> None:
    """Domain and source listings are sorted and distinct."""
    registry = DatasetRegistry(synthetic_registry_file)
    assert registry.domains() == ["cybersecurity", "medicine"]
    assert registry.sources() == ["kaggle", "openml", "uci"]


def test_missing_registry_file_raises_registry_error(tmp_path: Path) -> None:
    """Pointing the registry at a nonexistent file raises RegistryError."""
    registry = DatasetRegistry(tmp_path / "absent.yaml")
    with pytest.raises(RegistryError, match="not found"):
        registry.list_all()


def test_malformed_registry_file_raises_registry_error(tmp_path: Path) -> None:
    """A YAML document that is not a mapping of entries raises RegistryError."""
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="must define a mapping"):
        DatasetRegistry(path).list_all()


def test_validate_reports_entries_missing_required_fields(tmp_path: Path) -> None:
    """An entry without `source` is reported by validate()."""
    path = tmp_path / "incomplete.yaml"
    path.write_text("orphan:\n  domain: medicine\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="orphan"):
        DatasetRegistry(path).validate()


def test_membership_and_iteration_use_dataset_keys(synthetic_registry_file: Path) -> None:
    """The registry behaves like a sorted collection of dataset keys."""
    registry = DatasetRegistry(synthetic_registry_file)
    assert "beta_set" in registry
    assert list(registry) == ["alpha_set", "beta_set", "gamma_set"]


def test_module_facade_matches_the_default_instance() -> None:
    """get_registry() and get_dataset_meta() serve the bundled registry."""
    assert len(get_registry()) == EXPECTED_DATASET_COUNT
    assert get_dataset_meta("spambase")["domain"] == "digital_communications"
