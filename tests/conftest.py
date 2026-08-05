"""Shared fixtures for the ACIS quality-assurance suite.

The suite never depends on the ~500 MB production extract. Every test runs
against a synthetic dataset that reproduces the schema and the awkward
characteristics of the real file (see ``tests/synthetic_data.py``), so the
suite is runnable on a clean clone and in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.synthetic_data import DATA_SEP, make_dataframe, write_dataset  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_sep() -> str:
    return DATA_SEP


# Large enough that every province clears the >=1000 policy threshold that
# src/hypothesis_testing.py requires before it will run the province tests.
N_ROWS = 12_000


@pytest.fixture(scope="session")
def synthetic_df():
    """In-memory synthetic policy frame."""
    return make_dataframe(n_rows=N_ROWS, seed=42)


@pytest.fixture(scope="session")
def synthetic_data_file(tmp_path_factory) -> Path:
    """Synthetic dataset written to disk in the production wire format."""
    target = tmp_path_factory.mktemp("acis-data") / "MachineLearningRating_v3.txt"
    return write_dataset(target, n_rows=N_ROWS, seed=42)


@pytest.fixture(scope="session")
def modeling_module():
    import src.modeling as modeling

    return modeling


@pytest.fixture(scope="session")
def hypothesis_module():
    import src.hypothesis_testing as hypothesis_testing

    return hypothesis_testing
