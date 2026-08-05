"""End-to-end runs of the deliverables exactly as a reviewer would invoke them.

Both scripts hard-code ``data/MachineLearningRating_v3.txt`` relative to the
working directory, so each test builds a throwaway project root containing the
real ``src/`` tree and a synthetic extract, then runs the script as a
subprocess and asserts on its exit code and stdout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.synthetic_data import write_dataset

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def project_sandbox(tmp_path_factory, repo_root: Path) -> Path:
    sandbox = tmp_path_factory.mktemp("acis-run")
    shutil.copytree(repo_root / "src", sandbox / "src")
    write_dataset(sandbox / "data" / "MachineLearningRating_v3.txt", n_rows=6000, seed=7)
    return sandbox


def _run(sandbox: Path, script: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, script],
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )


def test_hypothesis_testing_script_runs_clean(project_sandbox: Path):
    result = _run(project_sandbox, "src/hypothesis_testing.py")
    assert result.returncode == 0, f"src/hypothesis_testing.py failed:\n{result.stderr[-2000:]}"
    assert "TASK 3: A/B Hypothesis Testing Results" in result.stdout
    assert "Traceback" not in result.stderr


def test_modeling_script_runs_clean(project_sandbox: Path):
    result = _run(project_sandbox, "src/modeling.py")
    assert result.returncode == 0, f"src/modeling.py failed:\n{result.stderr[-3000:]}"
    for section in (
        "Claim Probability Model AUC-ROC",
        "Claim Severity Model RMSE",
        "Risk-Based Premium Calculation",
        "Most Influential Features",
    ):
        assert section in result.stdout, f"missing pipeline stage in stdout: {section}"


def test_scripts_fail_loudly_when_the_dataset_is_absent(tmp_path: Path, repo_root: Path):
    """A missing dataset must not look like a successful run."""
    sandbox = tmp_path / "empty"
    shutil.copytree(repo_root / "src", sandbox / "src")
    result = _run(sandbox, "src/hypothesis_testing.py")
    assert "Error: Data file not found" in result.stdout


@pytest.mark.parametrize("notebook", ["01_EDA_and_Stats.ipynb", "Model_Interpretation.ipynb"])
def test_notebook_executes_from_a_clean_kernel(tmp_path_factory, repo_root: Path, notebook: str):
    """Every committed notebook must run top-to-bottom against a fresh checkout."""
    pytest.importorskip("nbclient")
    import nbformat
    from nbclient import NotebookClient

    source = repo_root / "notebooks" / notebook
    if not source.exists():
        pytest.skip(f"{notebook} not present")

    sandbox = tmp_path_factory.mktemp("acis-nb")
    (sandbox / "notebooks").mkdir()
    shutil.copy(source, sandbox / "notebooks" / notebook)
    write_dataset(sandbox / "data" / "MachineLearningRating_v3.txt", n_rows=6000, seed=7)

    nb = nbformat.read(sandbox / "notebooks" / notebook, as_version=4)
    os.environ.setdefault("MPLBACKEND", "Agg")
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={
        "metadata": {"path": str(sandbox / "notebooks")}
    })
    client.execute()
