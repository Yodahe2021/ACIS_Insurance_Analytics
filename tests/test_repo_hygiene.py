"""Repository, documentation and dependency hygiene checks.

Tests marked ``xfail`` correspond to open findings in ``docs/QA_REPORT.md``;
the finding ID is quoted in the reason string. They are ``strict`` so that the
suite fails loudly the moment a defect is fixed and the marker becomes stale.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

CONFLICT_MARKER = re.compile(r"^(<{7}|={7}|>{7})(\s|$)")

STDLIB_OR_LOCAL = {
    "os",
    "sys",
    "re",
    "json",
    "pathlib",
    "warnings",
    "datetime",
    "dataclasses",
    "importlib",
    "typing",
    "__future__",
    "src",
    "tests",
}

# Third-party distribution name for each imported top-level module.
IMPORT_TO_DISTRIBUTION = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "shap": "shap",
    "statsmodels": "statsmodels",
    "dvc": "dvc",
}


def _tracked_text_files(repo_root: Path) -> list[Path]:
    skip_dirs = {".git", ".venv", "__pycache__", "data", "reports"}
    out = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".zip"}:
            continue
        out.append(path)
    return out


def _top_level_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_merge_conflict_markers(repo_root: Path):
    """Unresolved conflict markers must never reach a tracked file."""
    offenders = []
    for path in _tracked_text_files(repo_root):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, start=1):
            if CONFLICT_MARKER.match(line):
                offenders.append(f"{path.relative_to(repo_root)}:{lineno}: {line}")
    assert not offenders, "unresolved merge conflict markers:\n" + "\n".join(offenders)


def test_requirements_cover_every_third_party_import(repo_root: Path):
    """``pip install -r requirements.txt`` must be enough to run src/."""
    requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    declared = {
        re.split(r"[<>=!~\[]", line.strip())[0].lower()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    }

    imported: set[str] = set()
    for path in (repo_root / "src").glob("*.py"):
        imported |= _top_level_imports(path.read_text(encoding="utf-8"))

    missing = sorted(
        IMPORT_TO_DISTRIBUTION.get(name, name)
        for name in imported - STDLIB_OR_LOCAL
        if IMPORT_TO_DISTRIBUTION.get(name, name).lower() not in declared
    )
    assert not missing, f"imported by src/ but absent from requirements.txt: {missing}"


def test_requirements_are_pinned(repo_root: Path):
    """Unpinned requirements silently break the pipeline when upstream ships a major release."""
    unpinned = [
        line.strip()
        for line in (repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#") and not re.search(r"[<>=~]", line)
    ]
    assert not unpinned, f"requirements without a version constraint: {unpinned}"


def test_generated_artifacts_are_ignored(repo_root: Path):
    """Virtualenvs and bytecode caches must not be committable."""
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".venv", "__pycache__"):
        assert pattern in gitignore, f"{pattern} is not ignored by .gitignore"


def test_dvc_pointer_file_is_not_ignored(repo_root: Path):
    """A ``data/`` blanket ignore also hides the ``.dvc`` pointers, defeating DVC."""
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    blanket = [line for line in gitignore if line.strip() in {"data", "data/", "/data", "/data/"}]
    assert not blanket, (
        "'.gitignore' excludes the whole data/ directory, so data/*.dvc pointer files can never "
        f"be committed and 'dvc pull' can never restore the dataset (offending lines: {blanket})"
    )


def test_a_reproducible_pipeline_is_defined(repo_root: Path):
    """``dvc repro`` must be able to rebuild every artefact from the extract."""
    assert (repo_root / "dvc.yaml").exists(), "no DVC pipeline definition"


@pytest.mark.xfail(
    strict=True,
    reason="QA-001: BLOCKED on the client - no .dvc pointer can be tracked until the real extract "
    "and a shared remote are provided, so 'dvc pull' still restores nothing",
)
def test_dataset_is_tracked_by_dvc(repo_root: Path):
    pointers = [p for p in repo_root.rglob("*.dvc") if p.is_file()]
    assert pointers, "no DVC pointer file found; the dataset is not reproducible from a clean clone"


def test_license_file_exists(repo_root: Path):
    """QA-010: the README advertises an MIT badge."""
    assert (repo_root / "LICENSE").exists() or (repo_root / "LICENSE.md").exists()


def test_figures_are_stored_once(repo_root: Path):
    """QA-012: figures used to be committed twice, under two directories that drifted apart."""
    assert not (repo_root / "notebooks" / "reports").exists(), (
        "figures are committed twice (reports/figures and notebooks/reports/figures)"
    )


def test_generated_outputs_are_ignored(repo_root: Path):
    """Models, metrics and regenerated figures must never dirty the working tree."""
    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/" in gitignore


def test_notebook_outputs_are_stripped_on_commit(repo_root: Path):
    """QA-011/QA-015 stay fixed only if the hook is wired in."""
    config_file = repo_root / ".pre-commit-config.yaml"
    assert config_file.exists(), "no pre-commit configuration"
    assert "nbstripout" in config_file.read_text(encoding="utf-8")


def test_a_model_card_documents_the_pricing_model(repo_root: Path):
    card = repo_root / "docs" / "MODEL_CARD.md"
    assert card.exists(), "no model card; the pricing model ships undocumented"
    text = card.read_text(encoding="utf-8").lower()
    for section in ("intended use", "limitation", "fairness", "monitoring"):
        assert section in text, f"the model card does not cover {section}"
