---
name: testing-acis-qa
description: How to set up, run and adversarially test the ACIS insurance-analytics pipeline (src/ scripts, notebooks, pytest QA harness) from a clean clone without the real dataset.
---

# Testing the ACIS risk-analytics repo

## Environment (takes ~1 minute, no secrets required)

```bash
git clone https://github.com/Yodahe2021/ACIS_Insurance_Analytics.git
cd ACIS_Insurance_Analytics && git checkout <branch>
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # ~35 s; requirements.txt alone is enough to run src/
```

No credentials, network services, or logins are needed. Everything is CLI — there is no web app, so
only record a screen capture if you deliberately demo the terminal (the deliverable here *is* a set of
terminal scripts, so a maximized Konsole session is the interpretable artifact; enlarge the font via
the View ▸ Enlarge Font menu, since `ctrl+shift+plus` is typed literally into the shell).

## The dataset is not obtainable

`data/MachineLearningRating_v3.txt` (~500 MB) is not in the repo and `dvc pull` is a silent no-op (no
`.dvc` pointer is tracked). Do **not** wait for it. Generate a synthetic stand-in instead:

```bash
python -m tests.synthetic_data     # writes data/MachineLearningRating_v3.txt (~8.7 MB, 20k rows)
```

The pytest suite generates its own fixtures in tmpdirs, so `pytest -m "not slow"` needs no `data/`
directory at all — deleting `data/` is a good adversarial probe and should not change the result.

## Commands and expected results

```bash
ruff check .                # -> "All checks passed!"
pytest                      # -> 82 passed, 1 xfailed  (~18 s)
pytest -m "not slow"        # -> 75 passed, 1 xfailed
pytest -m slow              # -> 7 passed (runs both entry points + both notebooks for real)
python -m src.hypothesis_testing   # exit 0, prints "TASK 3: A/B Hypothesis Testing Results"
MPLBACKEND=Agg python -m src.modeling
# exit 0, ends with the SHAP table and the artefact paths it wrote
```

Run the entry points as modules (`python -m src.modeling`); `src/` is a package and the plain script
path breaks the `from src import config` imports.

Both runs write only into `artifacts/` (git-ignored): `artifacts/models/*.joblib`,
`artifacts/metrics/model_metrics.json`, `artifacts/metrics/hypothesis_tests.{json,csv}`. Check those
rather than scraping stdout.

Notebooks headlessly: `cd notebooks && MPLBACKEND=Agg ../.venv/bin/jupyter nbconvert --to notebook
--execute --output /tmp/out.ipynb <nb>` (cwd must be `notebooks/` — the notebooks resolve the repo
root from the working directory).

## Traps worth knowing

* **`xgboost` must stay `<3.0`.** shap ≤0.49 cannot parse xgboost 3.x boosters and the SHAP step dies
  with `ValueError: could not convert string to float: '[...]'`. To prove a pin is load-bearing, make a
  throwaway venv, `pip install "xgboost>=3"` and re-run `python -m src.modeling` — it should exit 1.
* **Exactly one `strict=True` xfail remains**: `test_dataset_is_tracked_by_dvc` (QA-001, blocked on the
  client's extract and a shared remote). Every other QA-ID in `docs/QA_REPORT.md` is now pinned by a
  *passing* test. An `XPASS` is a hard failure — check the short summary, not just the pass count.
* **The model quality gates are statistical.** `pytest` trains real models on a seeded synthetic
  fixture and asserts AUC > 0.6, a Brier score no worse than the base rate, positive severity
  predictions and severity R² ≥ the mean baseline. If one fails, look at the metrics before assuming
  flakiness — the seeds are fixed.
* **Always re-run the suite from a non-`/home` checkout.** `tests/test_notebooks.py` scans notebook JSON
  for absolute paths with a regex that matches `/home/[a-z]`, so anything leaking the checkout location
  into that scan makes the result depend on where the clone lives — and CI on GitHub Actions
  (`/home/runner/...`) hides it. Copy the tree to e.g. `/tmp/acis-pathtest` and confirm the counts are
  identical; a strict xfail flipping to `XPASS(strict)` is that bug returning.
* **A full run must leave the tree clean.** Notebook and pipeline output now goes to `artifacts/`;
  `git status --short` after `pytest -m slow` must be empty, and CI enforces it. If the nine committed
  `reports/figures/*` change, that regression is back. Still prefer a throwaway clone — a warm `.venv`
  masks the fresh-install failures worth testing.
* **Every number the repo prints comes from synthetic data** (QA-027). Treat metrics as evidence that
  the pipeline works, never as evidence about the real book.

## Devin Secrets Needed

None.
