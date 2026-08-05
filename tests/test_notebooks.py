"""Notebook reproducibility and hygiene checks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ABSOLUTE_PATH = re.compile(r"[A-Za-z]:\\{1,2}Users\\{1,2}|/Users/[a-z]|/home/[a-z]")


def _notebooks(repo_root: Path) -> list[Path]:
    return sorted(p for p in (repo_root / "notebooks").glob("*.ipynb"))


@pytest.fixture(params=["01_EDA_and_Stats.ipynb", "Model_Interpretation.ipynb"])
def notebook(request, repo_root: Path) -> dict:
    path = repo_root / "notebooks" / request.param
    if not path.exists():
        pytest.skip(f"{request.param} not present")
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb["__path__"] = str(path)
    return nb


def test_notebooks_exist(repo_root: Path):
    assert _notebooks(repo_root), "no notebooks found"


def test_notebook_is_valid_nbformat(notebook: dict):
    assert notebook.get("nbformat") == 4
    assert isinstance(notebook.get("cells"), list)


def test_notebook_has_no_error_outputs(notebook: dict):
    errors = [
        f"{o.get('ename')}: {o.get('evalue')}"
        for cell in notebook["cells"]
        for o in cell.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"{notebook['__path__']} was committed with tracebacks: {errors}"


def test_notebook_reads_data_through_a_relative_path(notebook: dict):
    sources = "".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert not ABSOLUTE_PATH.search(sources), "notebook source hard-codes a machine-specific path"


@pytest.mark.xfail(
    strict=True,
    reason="QA-011: notebooks are committed with outputs containing the author's local Windows paths",
)
def test_notebook_outputs_contain_no_absolute_paths(notebook: dict):
    text = json.dumps(notebook)
    assert not ABSOLUTE_PATH.search(text), "committed outputs leak absolute filesystem paths"


@pytest.mark.xfail(
    strict=True,
    reason="QA-015: execution counts are non-sequential, so the notebooks were not run top-to-bottom "
    "in a fresh kernel before being committed",
)
def test_notebook_was_run_top_to_bottom(notebook: dict):
    counts = [c.get("execution_count") for c in notebook["cells"] if c["cell_type"] == "code"]
    executed = [c for c in counts if c is not None]
    assert executed == list(range(1, len(executed) + 1)), f"execution counts are {executed}"
