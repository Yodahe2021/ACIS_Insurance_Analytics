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
do not screen-record the run; capture terminal screenshots for the report instead. For legible
screenshots, launch Konsole with a big-font profile rather than fighting the font shortcut
(`ctrl+shift+plus` is typed literally into the shell):

```bash
mkdir -p ~/.local/share/konsole
printf '[Appearance]\nFont=Monospace,15,-1,5,50,0,0,0,0,0\n\n[General]\nName=Big\nParent=FALLBACK/\n' \
  > ~/.local/share/konsole/Big.profile
DISPLAY=:0 konsole --hide-menubar --profile Big &
wmctrl -r Konsole -b add,maximized_vert,maximized_horz
```

Clone **outside `/home`** to keep the checkout-path probe honest. `/srv` needs
`sudo mkdir -p /srv/<dir> && sudo chown $USER /srv/<dir>` first; `/tmp` works without sudo.

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
pytest                      # -> 87 passed, 1 xfailed  (~19 s)
pytest -m "not slow"        # -> 80 passed, 1 xfailed
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

## Adversarial probes that have actually found bugs

Drive corrupt inputs through `ACIS_DATA_PATH=<file> ACIS_OUTPUT_DIR=<tmp> python -m src.modeling`
(and `-m src.hypothesis_testing`), asserting **exit code ≠ 0, no numbers on stdout, no files in the
output dir**. Build harnesses in `/tmp`, never inside the repo, or the tree-cleanliness check lies.

* **Truncation was the gap in the data contract, and is the probe that found it.** A file cut with
  `head -c $((size/2))` used to **exit 0 and price the book off half the exposure**: pandas pads the
  severed last row with `NaN` and `dropna(subset=["TotalClaims"])` hides it. `_truncation_errors` now
  rejects a short final record (QA-031), and `ACIS_MIN_ROWS=<expected>` catches a cut that lands on a
  line boundary. Keep both in the corrupt-input matrix alongside the cases that were always handled:
  empty file, header-only, missing `TotalClaims`, all-null target, non-numeric `TotalPremium`.
* **Check the exposure gates in both directions.** `Province` (`MIN_SEGMENT_POLICIES`=1000),
  `PostalCode` (`MIN_ZIP_POLICIES`=500) and — since QA-032 — the gender frequency z-test are all gated,
  and a partial family downgrades any surviving rejection to `PROVISIONAL ONLY`. A ~40-policy book must
  print no `may justify a rating change` block; a book with a large real province effect must print
  one. Test both, or an always-off gate looks like a pass. `src/modeling.py` separately refuses a book
  under 500 policies / 50 claims rather than failing inside `train_test_split`.
* **Attack the pricing guards directly in `calculate_risk_based_premium`.** Injecting a stub severity
  model returning −50 000 / −1e12 / −inf still yields exactly `PREMIUM_FLOOR`
  (`400/0.92 = 434.78260869565213`), while the unguarded formula reaches −R1e12. Only `NaN` propagates
  (`np.clip`/`np.maximum` pass NaN through), and no real fitted pipeline produced NaN even with all-NaN
  features, ±1e308 numerics or unseen categories. This stub-injection probe is much stronger evidence
  than re-running the happy-path fixture, where the floor binds only trivially.
* **Expect the baseline path.** On degenerate/low-signal books, 5-fold CV legitimately picks
  `baseline_mean` and SHAP prints *"SHAP analysis skipped: the severity model is the mean-claim
  baseline"* — correct behaviour, not a crash. It also means the **xgboost≥3 negative control only
  reproduces when the XGBoost severity model wins**, so run that control against the standard synthetic
  extract, not an adversarial one.

## Devin Secrets Needed

None.
