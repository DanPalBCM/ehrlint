"""Shared fixtures.

Every fixture here builds its data from :mod:`ehrlint.synth`. Nothing in the
test suite reads a real record, and nothing in it reads a file that is not
generated inside a `tmp_path` or committed under `examples/`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from ehrlint.checks.base import registry_snapshot, restore_registry
from ehrlint.context import AuditContext
from ehrlint.io.features import build_feature_matrix
from ehrlint.schemas.task import TaskSpec
from ehrlint.synth.generator import SyntheticDataset, generate


def pytest_addoption(parser: pytest.Parser) -> None:
    """Opt in to the measurements that take minutes."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run the full-scale performance measurements (SPEC §4.5)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect `slow` and `live_download` unless they are asked for.

    Implemented rather than only declared: a marker documented as gated but
    not actually gated means the default suite silently takes minutes, and
    that is how a slow test ends up deleted instead of fixed.
    """
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="needs --run-slow")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not os.environ.get("EHRLINT_LIVE_DOWNLOAD"):
        skip_download = pytest.mark.skip(reason="needs EHRLINT_LIVE_DOWNLOAD=1")
        for item in items:
            if "live_download" in item.keywords:
                item.add_marker(skip_download)


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Restore the check registry after every test.

    Autouse because a test that registers a custom check would otherwise add
    it to every later audit, and a battery that grows as the suite runs makes
    the per-check assertions depend on test order.
    """
    snapshot = registry_snapshot()
    try:
        yield
    finally:
        restore_registry(snapshot)


@pytest.fixture(scope="session")
def clean_dataset() -> SyntheticDataset:
    """A small clean dataset, shared because generating it is pure."""
    return generate(120, seed=0)


@pytest.fixture
def dataset(clean_dataset: SyntheticDataset) -> SyntheticDataset:
    """A mutable copy of the clean dataset."""
    return clean_dataset.copy()


def make_context(data: SyntheticDataset) -> AuditContext:
    """Build a context from a dataset, wiring up features when present."""
    return AuditContext.from_frames(
        data.events,
        task_spec=data.task,
        labels=data.labels,
        splits=data.splits_mapping(),
        features=(
            None if data.features is None else build_feature_matrix(data.features, source="synth")
        ),
    )


@pytest.fixture
def clean_context(clean_dataset: SyntheticDataset) -> AuditContext:
    """A context over the clean dataset."""
    return make_context(clean_dataset)


@pytest.fixture
def task() -> TaskSpec:
    """The default synthetic task spec."""
    from ehrlint.synth.generator import default_task

    return default_task()


@pytest.fixture
def meds_dir(tmp_path: Path, clean_dataset: SyntheticDataset) -> Path:
    """The clean dataset written out in real MEDS layout."""
    root = tmp_path / "meds"
    clean_dataset.write(root)
    return root


@pytest.fixture
def tiny_events() -> pl.DataFrame:
    """Three hand-written events, for schema-level tests."""
    from datetime import datetime

    return pl.DataFrame(
        {
            "subject_id": ["A", "A", "B"],
            "time": [datetime(2020, 1, 1), datetime(2020, 2, 1), None],
            "code": ["ed_visit", "E11.65", "GENDER_F"],
            "code_system": ["VISIT", "ICD10CM", "DEMOGRAPHIC"],
            "numeric_value": [None, None, None],
            "text_value": [None, None, None],
            "encounter_id": ["A-1", "A-2", None],
        }
    )
