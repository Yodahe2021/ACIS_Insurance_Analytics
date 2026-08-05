"""Notebook reproducibility and hygiene checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ABSOLUTE_PATH = re.compile(r"[A-Za-z]:\\{1,2}Users\\{1,2}|/Users/[a-z]|/home/[a-z]")

EDA_NOTEBOOK = "01_EDA_and_Stats.ipynb"
INTERPRETATION_NOTEBOOK = "Model_Interpretation.ipynb"
NOTEBOOKS = [EDA_NOTEBOOK, INTERPRETATION_NOTEBOOK]


def _load(repo_root: Path, name: str) -> dict:
    """Parse a notebook, skipping the test if it is not present.

    The path is deliberately not stored on the returned dict: these tests
    serialise it and scan for absolute paths, so injecting the checkout
    location would make the result depend on where the repo was cloned.
    """
    path = repo_root / "notebooks" / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebooks_exist(repo_root: Path):
    assert sorted(p.name for p in (repo_root / "notebooks").glob("*.ipynb")), "no notebooks found"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_valid_nbformat(repo_root: Path, name: str):
    notebook = _load(repo_root, name)
    assert notebook.get("nbformat") == 4
    assert isinstance(notebook.get("cells"), list)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_no_error_outputs(repo_root: Path, name: str):
    notebook = _load(repo_root, name)
    errors = [
        f"{o.get('ename')}: {o.get('evalue')}"
        for cell in notebook["cells"]
        for o in cell.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"{name} was committed with tracebacks: {errors}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_reads_data_through_a_relative_path(repo_root: Path, name: str):
    notebook = _load(repo_root, name)
    sources = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert not ABSOLUTE_PATH.search(sources), f"{name} hard-codes a machine-specific path"


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            EDA_NOTEBOOK,
            marks=pytest.mark.xfail(
                strict=True,
                reason="QA-011: 01_EDA_and_Stats.ipynb is committed with outputs containing the "
                "author's local Windows paths",
            ),
        ),
        INTERPRETATION_NOTEBOOK,
    ],
)
def test_notebook_outputs_contain_no_absolute_paths(repo_root: Path, name: str):
    notebook = _load(repo_root, name)
    assert not ABSOLUTE_PATH.search(json.dumps(notebook)), "committed outputs leak absolute filesystem paths"


@pytest.mark.xfail(
    strict=True,
    reason="QA-015: execution counts are non-sequential, so the notebooks were not run top-to-bottom "
    "in a fresh kernel before being committed",
)
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_was_run_top_to_bottom(repo_root: Path, name: str):
    notebook = _load(repo_root, name)
    counts = [c.get("execution_count") for c in notebook["cells"] if c["cell_type"] == "code"]
    executed = [c for c in counts if c is not None]
    assert executed == list(range(1, len(executed) + 1)), f"execution counts are {executed}"
