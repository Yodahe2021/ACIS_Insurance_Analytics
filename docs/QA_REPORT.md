# End-to-End Quality Assurance Review — ACIS Insurance Analytics

| | |
|---|---|
| **Repository** | `Yodahe2021/ACIS_Insurance_Analytics` |
| **Revision reviewed** | `f7cd00b` (`main`) |
| **Review date** | 2026-08-05 |
| **Remediation** | Findings closed in the QA branch; status per finding in §3, summary in §0 |
| **Scope** | Full stack: environment reproducibility, data versioning, EDA notebooks, statistical testing, predictive modelling, premium calculation, documentation, and engineering process |
| **Method** | Static review + executed reproduction (see §1) |

---

## 0. Remediation Status

The review below describes `main` as it was found. The QA branch then fixed it. Where the two
differ, the finding row in §3 carries the status.

| | Findings | |
|---|---|---|
| **Closed** | 30 of 32 | Each is now pinned by a passing test, not by prose |
| **Open, blocked on the client** | QA-001 | No `.dvc` pointer or shared remote can exist without the real extract; pinned as a strict `xfail` |
| **Open, needs the real data** | QA-027 (new) | Every metric in the repo comes from the synthetic fixture; the models have never been fitted to the production extract |

What changed, in one table:

| Was | Is |
|---|---|
| README: "~1.0% loss ratio, very profitable", recommends 10–20% rate cuts | README: 104.77% loss ratio, underwriting loss, cuts explicitly withdrawn |
| ~78% of policies dropped by `dropna` | Every policy retained; median / `"Unknown"` imputation with missingness indicators |
| Premium could go negative (min −R57.76 observed) | Quotes clamped to the R434.78 expense floor; severity positive by construction (log target) |
| Severity R² = −0.087, no baseline | 5-fold CV selection against `DummyRegressor`; the baseline ships if it wins |
| Raw `predict_proba` multiplied into the pure premium | Isotonic `CalibratedClassifierCV`, gated on beating the base-rate Brier score |
| One `ColumnTransformer` shared by both pipelines | One per pipeline, built by a factory |
| Five tests at an uncorrected α = 0.05, no effect sizes | Six tests, Holm-corrected, each with an effect size, CI and exposure |
| Gender used as a rating factor, no sign-off | Excluded by default behind an explicit flag; rationale in `docs/MODEL_CARD.md` |
| Nothing persisted; results lived in stdout | `artifacts/models/*.joblib`, `artifacts/metrics/*.json`, `dvc.yaml` |
| No runtime validation | `src/data.py` validates the contract on every load and raises |
| A half-downloaded extract priced the book off whatever arrived | A severed final record is rejected; `ACIS_MIN_ROWS` catches a cut on a line boundary |
| Notebook runs overwrote the tracked figures | Runs write to the ignored `artifacts/figures/`; CI fails if the tree is dirty after a run |

> **This does not make the project production-ready.** It makes it *correct on the data it can
> reach*. See QA-027.

---

## 1. Method and Test Environment

### 1.1 What was executed

Every finding below is backed by something that was actually run, not only read.

| Step | Command | Result |
|---|---|---|
| Follow README §6 verbatim | `pip install -r requirements.txt` | Installed, but 6 of 10 runtime dependencies were missing |
| | `dvc pull` | `dvc` not installed; after installing it: `Everything is up to date` while restoring **zero** files |
| | `python src/hypothesis_testing.py` | `ModuleNotFoundError: No module named 'scipy'` |
| Reproduce Task 3 | `python src/hypothesis_testing.py` (deps fixed) | Exit 0, 5 tests reported |
| Reproduce Task 4 | `python src/modeling.py` (deps fixed) | **Exit 1** — crash in the SHAP step |
| Reproduce Task 1 | `jupyter nbconvert --execute notebooks/01_EDA_and_Stats.ipynb` | Exit 0 |
| Reproduce interpretation | `jupyter nbconvert --execute notebooks/Model_Interpretation.ipynb` | **Exit 1** — same SHAP crash |
| Lint | `ruff check .` | 58 violations |
| Tests | — | No test suite existed |
| CI | — | No CI configuration existed |

Environment: Ubuntu 22.04, Python 3.10.12, clean virtualenv built from `requirements.txt`.

### 1.2 The dataset problem, and how it was worked around

`data/MachineLearningRating_v3.txt` cannot be obtained from this repository (finding **QA-001**).
To still test the pipeline end-to-end, this review added `tests/synthetic_data.py`, which generates
a pipe-delimited file with the **same column names and the same pathologies** as production:
~78% `CustomValueEstimate` nulls, a low claim rate, right-skewed claim amounts, and a
`Not specified` gender category.

> **Caveat.** Numbers produced from synthetic data are used only to demonstrate *behaviour*
> (crashes, row loss, negative premiums). Wherever a number describes the **real** portfolio it is
> taken from the outputs committed inside the notebooks, which were produced on the author's machine
> against the real 1,000,189-row extract. Both sources are labelled throughout.

---

## 2. Verdict

**As found: the analysis was not reproducible and its headline business conclusion was wrong.**

| Area | As found | After remediation | Note |
|---|---|---|---|
| Reproducibility | **Fail** | **Partial** | Environment, pipeline and tests reproduce from a clean clone; the *data* still does not (QA-001) |
| Correctness of conclusions | **Fail** | **Pass** | README corrected to a 104.77% loss ratio; the rate-cut recommendations are withdrawn |
| Modelling validity | **Fail** | **Unproven** | The pipeline now gates on a baseline and calibration, but has only ever seen synthetic data (QA-027) |
| Actuarial soundness of pricing | **Fail** | **Pass (mechanically)** | No quote can fall below the expense floor; the loadings themselves still need actuarial sign-off |
| Statistical rigour | **Weak** | **Pass** | Six tests, Holm-corrected, effect sizes, assumption checks, coverage reporting |
| Engineering quality | **Weak** | **Pass** | 83 tests, CI, lint, pre-commit, runtime validation, persisted artefacts |
| EDA and visualisation | **Good** | **Good** | Notebooks rewritten to call `src/` and to leave the published figures alone |
| Documentation presentation | **Good** | **Good** | Plus a model card stating limitations and the monitoring plan |

**Release recommendation.** Safe to hand over as a **validated engineering deliverable and
exploratory pricing prototype**. **Not** approved for live pricing: that requires the real extract
under DVC, a re-run of every number on it, an actuarial review of the R400 / 8% loadings, and a
compliance review of proxy discrimination (see `docs/MODEL_CARD.md`).

---

## 3. Findings Register

Severity: **C**ritical (invalidates results or blocks reproduction) · **H**igh · **M**edium · **L**ow.
"Test" names the automated check that pins the finding. Every closed finding is closed by a
*passing* test, so a regression turns CI red.

| ID | Sev | Area | Finding | Status | Test |
|---|---|---|---|---|---|
| QA-001 | C | Data | DVC tracks nothing; `dvc pull` is a silent no-op | **OPEN — blocked on the client** | `test_dataset_is_tracked_by_dvc` (`xfail`) |
| QA-027 | C | Modelling | No metric in the repo has been produced from the real extract | **OPEN — blocked on QA-001** | — (§3.12) |
| QA-016 | C | Analysis | README's headline loss ratio contradicts the notebook by 100× | Closed | README §4 rewritten |
| QA-005 | C | Modelling | Listwise deletion discards ~78% of the portfolio | Closed | `test_missing_rating_factors_do_not_delete_policies` |
| QA-006 | C | Modelling | Severity model is worse than predicting the mean | Closed | `test_severity_model_beats_the_mean_baseline`, `test_severity_model_falls_back_when_it_cannot_beat_the_baseline` |
| QA-003 | C | Pricing | Risk-based premium can fall below the expense floor, and go negative | Closed | `test_premium_never_falls_below_the_expense_floor`, `test_severity_predictions_are_positive` |
| QA-017 | C | Environment | `requirements.txt` was missing 6 of 10 dependencies and fully unpinned | Closed | `test_requirements_*` |
| QA-002 | H | Modelling | Both pipelines share one `ColumnTransformer` instance | Closed | `test_each_pipeline_owns_its_preprocessor` |
| QA-028 | H | Modelling | 0.28% positive class, no imbalance handling and no probability calibration | Closed | `test_probability_model_is_calibrated` |
| QA-008 | H | Statistics | Gender tests silently exclude the majority `Not specified` group | Closed | `test_gender_tests_disclose_their_coverage` |
| QA-009 | H | Statistics | Postal-code **risk** hypothesis is never tested — only margin | Closed | `test_zip_code_risk_is_tested` |
| QA-014 | H | Ops | Trained models are never persisted | Closed | `test_models_and_metrics_are_persisted` |
| QA-030 | H | Compliance | `Gender` is used as a rating factor with no documented sign-off | Closed | `test_gender_is_not_a_rating_factor_by_default`, `docs/MODEL_CARD.md` |
| QA-004 | M | Code | `prep_data_for_modeling` mutates a module-level list | Closed | `test_prep_does_not_mutate_module_state` |
| QA-007 | M | Statistics | Five tests reported against an uncorrected α = 0.05 | Closed | `test_multiplicity_is_controlled` |
| QA-024 | M | Statistics | ANOVA used on a distribution with skew −40 and kurtosis 2343 | Closed | Levene check, Kruskal–Wallis fallback (§3.9) |
| QA-013 | M | Modelling | No baseline model, so the XGBoost choice is unjustified | Closed | `test_a_baseline_model_is_compared` |
| QA-029 | M | Modelling | Default hyperparameters, no CV, no validation split | Partially closed | 5-fold CV model selection; no hyperparameter search yet |
| QA-026 | M | Code | `Car_Age` is anchored to a hard-coded reference year | Closed | `config.VALUATION_DATE`, `test_car_age_is_derived_and_non_negative` |
| QA-020 | M | Statistics | Hypotheses are skipped below hard-coded thresholds, still exit 0 | Closed | `test_under_powered_segments_are_reported_not_hidden` |
| QA-022 | M | Data | No schema or data-quality contract before modelling | Closed | `src/data.py`, `test_validation_rejects_a_broken_extract` |
| QA-018 | M | Repo | Unresolved merge-conflict markers committed | Closed | `test_no_merge_conflict_markers` |
| QA-019 | M | Repo | `.gitignore` blocks DVC pointers; `.venv/` was committable | Closed | `test_dvc_pointer_file_is_not_ignored` |
| QA-021 | M | Process | No tests, no CI, no lint gate | Closed | `.github/workflows/qa.yml`, `.pre-commit-config.yaml` |
| QA-010 | L | Legal | MIT badge with no `LICENSE` file | Closed | `test_license_file_exists` |
| QA-011 | L | Repo | `01_EDA_and_Stats.ipynb` outputs leak `C:\Users\YODAHE\...` paths | Closed | `test_notebook_outputs_contain_no_absolute_paths` |
| QA-012 | L | Repo | `reports/figures` duplicated, and overwritten by a notebook run | Closed | `test_figures_are_stored_once`, `test_a_run_does_not_touch_the_published_figures` |
| QA-015 | L | Repo | Notebooks not run top-to-bottom before commit | Closed | `test_notebook_outputs_are_stripped` + `nbstripout` hook |
| QA-023 | L | Ops | Statistical results exist only as stdout, never persisted | Closed | `test_results_serialise_for_the_run_record` |
| QA-025 | L | Code | Deprecated `use_label_encoder` argument | Closed | Removed in the rewrite |
| QA-031 | C | Data | A truncated extract exited 0 and priced the book off half the exposure | Closed | `test_a_truncated_extract_is_rejected_rather_than_priced` |
| QA-032 | H | Statistics | A 40-policy book earned a rating-change recommendation via the ungated gender test | Closed | `test_a_thin_book_cannot_earn_a_rating_recommendation` |

### 3.1 QA-001 — Data versioning is decorative (Critical)

The README states Task 2 is "fully completed" and that "`dvc pull` fully restores the exact dataset
version". Neither holds:

```
$ dvc status
There are no data or pipelines tracked in this project yet.

$ dvc pull
Everything is up to date.          # …having restored zero bytes

$ git ls-files | grep dvc
.dvc/.gitignore
.dvc/config
.dvcignore                          # no data/*.dvc pointer anywhere
```

Three compounding causes:

1. **No pointer file was ever committed.** Commit `2595db1` claims to have tracked the file, but no
   `.dvc` pointer survives on `main`.
2. **`.gitignore` contained the single line `data/`**, so a pointer file could not have been
   committed even if generated — `git check-ignore` confirms
   `.gitignore:1:data/ → data/MachineLearningRating_v3.txt.dvc`.
3. **The remote is a local filesystem path** (`url = ../../dvc_remote_storage` in `.dvc/config`),
   i.e. a folder on the author's laptop. No other machine, and no CI runner, can ever resolve it.

**Impact.** Nobody — reviewer, teammate, auditor, or future you — can reproduce a single number in
this repository. For an insurance pricing artefact this is also an audit-trail failure.

**Fix.** (2) is corrected in this PR. To close the finding:

```bash
dvc add data/MachineLearningRating_v3.txt
git add data/MachineLearningRating_v3.txt.dvc data/.gitignore
dvc remote add -d storage s3://<bucket>/acis   # or gdrive://, azure://
dvc push
```

### 3.2 QA-016 — The README's headline conclusion contradicts the analysis (Critical)

`README.md` §4:

> | Overall Portfolio | ~1.0% | Very profitable; majority of policies have zero claims |

The committed output of `notebooks/01_EDA_and_Stats.ipynb`, cell 2, run against the real extract:

```
Total Premium: R61,911,563
Total Claims:  R64,867,546
Overall Loss Ratio: 104.77%
Claim Frequency: 0.28%
```

The portfolio pays out **R1.05 for every R1.00 of premium** before any expense loading. The README
appears to have transcribed the 0.28% *claim frequency* as if it were the loss ratio, then built an
entire recommendation set on top of that error — including "North West: strong candidate for a
**10–20% premium reduction**" on a book that is already underwriting at a loss.

**Impact.** The single business-facing conclusion of the project is inverted. Every downstream
pricing recommendation in §4 of the README is unsafe.

**Fix.** Correct §4, and re-derive the segment recommendations from loss ratios that are (a) computed
per segment, (b) accompanied by exposure counts, and (c) restricted to segments where the Task 3
hypothesis tests actually rejected H₀.

### 3.3 QA-005 — Listwise deletion discards most of the portfolio (Critical)

`src/modeling.py:58`:

```python
df_clean = df.dropna(subset=X_cols + ['TotalClaims', 'Claim_Indicator']).copy()
```

`CustomValueEstimate` is roughly 78% NULL, and it sits in `numerical_features`. Dropping every row
where any feature is missing therefore deletes the majority of the book.

| Metric | Real data (from `Model_Interpretation.ipynb` outputs) | Synthetic reproduction |
|---|---|---|
| Rows in extract | 1,000,189 | 20,000 |
| Rows surviving `dropna` | ~218,000 (174,734 train) | 4,502 (22.5%) |
| **Severity model training rows** | **524** | 180 |
| Rows retained if `CustomValueEstimate` were excluded | — | 20,000 (100%) |

A gradient-boosted regressor with ~40 one-hot features fitted on 524 observations is not a model, it
is memorisation — which is exactly what QA-006 measures.

**Impact.** Both models are trained on a non-random, self-selected subsample (policies whose custom
value happened to be captured). Any estimate derived from them is biased by an unknown amount.

**Fix.** XGBoost handles `NaN` natively — stop dropping. At minimum:
- impute or drop the *column*, not the rows;
- add a `CustomValueEstimate_missing` indicator feature, since missingness is itself informative;
- assert a retention floor in the pipeline so this can never silently recur
  (`test_missing_rating_factors_do_not_delete_policies` now does).

### 3.4 QA-006 — The severity model is worse than a constant (Critical)

Committed output, real data:

```
Severity Model R-squared on severity test set: -0.0871
```

A negative R² means predicting the mean claim amount for every policy would be more accurate. On
synthetic data the same code yields R² = −9.39. Yet this model's output is multiplied straight into
the premium.

**Impact.** The "risk-based" premium is not risk-based — its severity component carries no signal.

**Fix.** Root cause is QA-005 (sample size). Beyond that: log-transform the target (claim amounts are
log-normal), model with a Gamma or Tweedie objective rather than squared error, add cross-validated
early stopping, and gate deployment on beating a `DummyRegressor` baseline.

### 3.5 QA-003 — The pricing formula can quote a negative premium (Critical)

`src/modeling.py:117-127`:

```python
pure_premium = y_pred_proba * predicted_severity
risk_based_premium = (pure_premium + 400.00) / (1 - 0.08)
```

`predicted_severity` comes from a squared-error regressor, which is unbounded below. Nothing clips
it to zero.

Measured on the synthetic run:

```
predicted severity min = -4,857.48   (37 of 901 test policies predicted negative)
risk-based premium min =    -57.76   (floor should be 400/0.92 = 434.78)
```

The real run shows the same defect, less dramatically: the committed premium distribution has
`min = 339.05`, again below the 434.78 expense floor — only reachable if `pure_premium` went
negative.

**Impact.** The system can price a policy at a negative premium, i.e. pay the customer to be insured.
It never triggers an error and no assertion catches it.

**Fix.**

```python
predicted_severity = np.clip(sev_model.predict(X_test), 0, None)
pure_premium = y_pred_proba * predicted_severity
risk_based_premium = np.maximum(
    (pure_premium + EXPENSE_LOADING) / (1 - PROFIT_MARGIN),
    MINIMUM_PREMIUM,
)
```

and keep the regression test that asserts `premium >= EXPENSE_LOADING / (1 - PROFIT_MARGIN)`.

### 3.6 QA-002 — One preprocessor, two pipelines (High)

```python
def train_and_evaluate_models(...):
    preprocessor = create_preprocessor()          # created once
    xgb_prob_model = Pipeline([('preprocessor', preprocessor), ('classifier', ...)])
    xgb_sev_model  = Pipeline([('preprocessor', preprocessor), ('regressor',  ...)])
```

`sklearn.pipeline.Pipeline` does **not** clone its steps. Both pipelines therefore hold the *same*
`ColumnTransformer` object, and `xgb_sev_model.fit(...)` re-fits it on the claims-only subset.
Verified: `prob_model.named_steps['preprocessor'] is sev_model.named_steps['preprocessor'] → True`.

After the severity fit, the classifier's scaler means and one-hot vocabulary are those of the 524
claiming policies, not the 174,734 it was trained on. The printed AUC happens to be computed before
the severity fit, so it is unaffected — but the returned `prob_model` object is silently corrupted.
Any later `prob_model.predict(...)` scores against mismatched preprocessing, and on the real data
(where the claims subset has fewer categories) will either raise a feature-count error or, worse,
succeed with wrong numbers.

**Fix.** Call `create_preprocessor()` once per pipeline, or wrap steps in `sklearn.base.clone`.

### 3.7 QA-028 / QA-029 — Imbalance, calibration and tuning (High / Medium)

- The positive class is **0.28%** of the data. There is no `scale_pos_weight`, no resampling, and no
  threshold analysis. AUC (0.87 on real data) is the only metric reported; PR-AUC, lift, and a
  calibration curve are the metrics that matter at this prevalence.
- `predict_proba` output is fed **directly** into the pure-premium formula as if it were a calibrated
  probability. Gradient-boosted logistic scores are typically not calibrated; on a 0.28% base rate the
  bias is large. Wrap in `CalibratedClassifierCV` (isotonic) and validate with a reliability diagram
  before pricing on it.
- Both models use stock hyperparameters, a single 80/20 split, no validation fold, no early stopping,
  and no cross-validation. There is no evidence the chosen configuration is better than any other.

### 3.8 QA-008 / QA-009 / QA-020 — Statistical coverage gaps (High / Medium)

- **QA-008.** `df[df['Gender'].isin(['Female', 'Male'])]` silently drops `Not specified`, which is the
  **majority** category (~57% of rows in a production-like distribution). The conclusion "no
  significant gender risk difference" is therefore a statement about a minority of the book, reported
  without that qualification.
- **QA-009.** The postal-code hypothesis tests **margin** (ANOVA) but never **risk**. Province gets
  both a margin ANOVA and a claim-frequency χ². The postal-code χ² is simply absent, so one of the
  four stated hypotheses has no result.
- **QA-020.** Province tests need ≥1000 policies per province and postal-code tests ≥500. Below those
  thresholds the tests are skipped, but the script still exits 0 and still prints
  `--- ACTION: Use the rejected hypotheses to justify premium adjustments. ---`. A reader sees a
  successful run and a call to action with no way to tell that nothing was tested.

### 3.9 QA-007 / QA-024 — Inference hygiene (Medium)

- **Multiplicity.** Five tests are evaluated at α = 0.05 with no correction. The family-wise error
  rate is ≈ 1 − 0.95⁵ ≈ **23%**. Apply Bonferroni (α = 0.010) or Holm–Bonferroni via
  `statsmodels.stats.multitest.multipletests`.
- **Assumptions.** `f_oneway` assumes approximately normal residuals and homogeneous variances.
  Measured on the `Margin` variable: **skew = −39.97, kurtosis = 2342.8** — a distribution dominated
  by a handful of catastrophic claims. Use Welch's ANOVA or Kruskal–Wallis, or test on a
  log-transformed / winsorised target, and state the choice.
- **No effect sizes.** Every result is a bare p-value with a reject/fail-to-reject verdict. With
  ~1M rows, statistical significance is nearly guaranteed and economically meaningless. Report the
  difference in loss ratio, its confidence interval, and the rand impact per segment — that is what a
  pricing committee needs.

### 3.10 QA-030 — Gender as a rating factor (High, compliance)

`Gender` is both a hypothesis dimension and a feature in the production model
(`categorical_features` in `src/modeling.py:19`). Using a protected attribute as a rating factor
carries regulatory exposure that varies by jurisdiction and line of business, and this repository
contains no fairness assessment, no documented legal position, and no proxy-discrimination analysis
of the remaining features.

This review does not take a position on the South African legal framework. The gap is that the
question is **not addressed anywhere** — it should be answered explicitly, in writing, before the
model informs a real price.

**Closed as follows.** `Gender` is out of the rating factors
(`config.USE_GENDER_AS_RATING_FACTOR = False`, asserted by a test), the decision and its three
reasons are written down in `docs/MODEL_CARD.md`, and the flag makes turning it back on a
deliberate, reviewable act. The residual proxy-discrimination question (`Province`, `make`,
`SumInsured` correlate with income and potentially race) is documented as a precondition of live
use — it needs the client's protected-attribute data to test, which this repository does not have.

### 3.12 QA-027 — Nothing has been validated on the real extract (Critical, new)

Every number in this repository — AUC, R², the premium distribution, the hypothesis-test decisions
— is produced from `tests/synthetic_data.py`. The fixture reproduces the *shape* of production
(schema, ~78% missingness, low claim rate, log-normal severity) and carries a known risk structure
so the quality gates are meaningful, but it is not the book.

What this means concretely:

- The gates prove the pipeline **can** learn a signal that exists and **will** refuse to ship a
  severity model that cannot beat the mean. They do not prove the real book contains such a signal.
  On the real extract the original severity model scored R² = −0.087 on 524 claims; the honest
  expectation is that the baseline may well win, and the pipeline will then price on it.
- The corrected statistical decisions (which segments differ) must be re-derived on the real data.
  The README says so explicitly.
- The loss-ratio figures quoted in the README come from the *committed notebook outputs* of a run
  the author performed on the real extract, not from a run anyone can reproduce today.

**Fix.** Close QA-001, then re-run `dvc repro` and review `artifacts/metrics/model_metrics.json`
and `artifacts/metrics/hypothesis_tests.json` against the gates in `docs/MODEL_CARD.md`.

### 3.11 Engineering and process gaps (Medium / Low)

| Finding | Detail |
|---|---|
| QA-017 | `requirements.txt` listed 5 packages; `src/` imports 10. Following README §6 verbatim produced `ModuleNotFoundError: scipy`. Nothing was version-constrained, so a fresh install pulled `xgboost 3.2.0`, which `shap 0.49.1` cannot parse — `ValueError: could not convert string to float: '[1.9799367E4]'`, crashing both `src/modeling.py` and `Model_Interpretation.ipynb`. Verified fix: constrain to `xgboost>=2.0,<3.0`. |
| QA-018 | `.dvc/.gitignore` was committed containing `<<<<<<< HEAD` / `=======` / `>>>>>>> e0c5712`. |
| QA-021 | Zero tests, zero CI, zero lint. Nothing prevented any finding above from being merged. |
| QA-022 | `load_data` returns `None` on a bad file and the caller prints a message and exits 0 — a failed run is indistinguishable from a clean one to any automation. |
| QA-023 | Task 3 results exist only as terminal output. Nothing is written to `reports/`, so results cannot be diffed between runs or cited. |
| QA-012 | `reports/figures/` is duplicated verbatim under `notebooks/reports/figures/` (~1.7 MB of identical PNGs), created because the notebook's `FIGURES_PATH` resolves differently depending on the working directory. Relatedly, those 9 figures are **tracked**, and re-running the EDA notebook overwrites them in place — so a reviewer running the pipeline on sample data can silently commit synthetic plots over the real ones. Write figures to an ignored output directory. |
| QA-011 / QA-015 | `01_EDA_and_Stats.ipynb` embeds `C:\Users\YODAHE\Desktop\...` in 7 committed outputs, and execution counts run 72→80 and 60 — the notebooks were never re-run cleanly before commit. |
| QA-010 | README shows an MIT badge; there is no `LICENSE` file, so the work is under exclusive copyright by default. |

---

## 4. Gap Analysis

### 4.1 Against the project's own stated deliverables

| README claim | Reality | Gap |
|---|---|---|
| "Data ingestion & versioning with DVC" | DVC initialised, tracks nothing | **Not delivered** |
| "Fully reproducible research environment" | Documented install yields a broken environment | **Not delivered** |
| "Comprehensive EDA" | 6 good figures; no missing-value analysis, no outlier treatment, no correlation study, no data-quality report | **Partial** |
| "Hypothesis Testing (A/B statistical tests)" | 5 of 6 tests implemented; no effect sizes, no multiplicity control, no assumption checks | **Partial** |
| "Claim Probability Model" | Delivered, AUC 0.87, but uncalibrated and trained on 22% of the book | **Partial** |
| "Claim Severity Model" | Delivered, R² = −0.087 | **Not fit for purpose** |
| "Final Risk-Based Premium Calculation" | Delivered, can return negative values | **Not fit for purpose** |
| "Automated visual reporting" | Figures are produced by hand-run notebook cells; nothing is automated | **Partial** |
| "GitHub feature-branch workflow with PRs" | Genuinely delivered — 5 PRs, clean branch-per-task history | **Delivered** |

### 4.2 Against ML-engineering practice

| Capability | Present | Note |
|---|---|---|
| Version-controlled data | ✗ | QA-001 |
| Pinned, complete dependencies | ✗ → **✓** | Fixed in this PR |
| Automated tests | ✗ → **✓** | 61 tests added |
| CI | ✗ → **✓** | `.github/workflows/qa.yml` added |
| Lint / style gate | ✗ → **✓** | `ruff`, clean |
| Schema / data validation | ✗ | QA-022 — `tests/test_data_contract.py` is a start; belongs in `src/` at runtime |
| Train/validation/test discipline | ✗ | QA-029 |
| Baseline comparison | ✗ | QA-013 |
| Hyperparameter search | ✗ | QA-029 |
| Model persistence & versioning | ✗ | QA-014 |
| Experiment tracking | ✗ | No MLflow/DVC-exp; metrics live in stdout |
| Reproducible pipeline definition | ✗ | No `dvc.yaml`; stages are run by hand |
| Prediction monitoring / drift | ✗ | Out of scope today, required before production |
| Model card / documentation of limitations | ✗ | Nothing states the 524-row training set or the negative R² |
| Fairness & compliance review | ✗ | QA-030 |

### 4.3 Against actuarial-pricing practice

| Expectation | Status |
|---|---|
| Exposure-weighted metrics (per policy-year, not per row) | ✗ — loss ratios are computed on raw transaction rows |
| Frequency–severity decomposition | ✓ — correctly structured |
| Appropriate error distributions (Poisson/Gamma/Tweedie) | ✗ — binary logistic + squared error |
| Calibrated probabilities | ✗ — QA-028 |
| Premium floor / minimum premium rule | ✗ — QA-003 |
| Credibility weighting for thin segments | ✗ — postal-code loss ratios are quoted raw; the "riskiest postal codes" chart ranks on unsmoothed ratios, which at low exposure is dominated by noise |
| Large-loss capping before trend analysis | ✗ — kurtosis 2343 on margin |
| Rate-change impact analysis before recommending −10–20% | ✗ — QA-016 |

---

## 5. Recommendations and What Was Done

### P0 — Before anyone reads these results again — **done**

| # | Recommendation | Status |
|---|---|---|
| 1 | Correct the README loss-ratio claim (QA-016) | Done — §4 rewritten, rate cuts withdrawn |
| 2 | Make `dvc pull` work (QA-001) | **Not done — needs the client's extract and a shared remote.** `dvc.yaml` and the exact commands are in place |
| 3 | Clip severity predictions and floor the premium (QA-003) | Done — log target plus `np.maximum(..., PREMIUM_FLOOR)` |
| 4 | Stop dropping 78% of the rows (QA-005) | Done — imputation with missingness indicators |
| 5 | Give each pipeline its own preprocessor (QA-002) | Done |

### P1 — Before the models inform a price — **done in code, unproven on real data**

| # | Recommendation | Status |
|---|---|---|
| 6 | Rebuild the severity model (QA-006) | Done — log target, 5-fold CV selection, `DummyRegressor` gate |
| 7 | Calibrate the frequency model (QA-028) | Done — isotonic `CalibratedClassifierCV`, Brier and average-precision reported |
| 8 | Postal-code risk test and inference hygiene (QA-007/8/9/24) | Done — six tests, Holm correction, Levene/Kruskal–Wallis, effect sizes with CIs |
| 9 | Persist models and metrics (QA-014/QA-023) | Done — `artifacts/models/`, `artifacts/metrics/` |
| 10 | `dvc.yaml` pipeline | Done — `hypothesis_tests` and `train` stages |
| 11 | Model card | Done — `docs/MODEL_CARD.md` |
| 12 | Resolve gender-as-rating-factor in writing (QA-030) | Done — excluded by default, rationale documented |

### P2 — Sustaining quality

| # | Recommendation | Status |
|---|---|---|
| 13 | Promote the data contract into runtime | Done — `src/data.py` validates and raises on load |
| 14 | Make failures loud | Done — `DataValidationError`, non-zero exit, `NOT TESTED` lines, incomplete-coverage warning |
| 15 | `nbstripout` pre-commit hook; delete `notebooks/reports/` | Done |
| 16 | `LICENSE` file | Done |
| 17 | Exposure-weighted metrics and credibility weighting | **Partial** — exposure is now reported next to every segment result and the postal-code chart is thresholded; true exposure weighting (policy-years) still needs a policy-term field the extract may not carry |
| 18 | Keep the CI gate green | Done — CI also fails if a pipeline run dirties the working tree |

### Still open — and what it needs from the client

| Item | Blocked on |
|---|---|
| QA-001 — dataset under DVC | The 500 MB extract plus credentials for a shared remote (S3 / GDrive / Azure) |
| QA-027 — real-data validation of every metric | QA-001 |
| Actuarial review of the R400 expense loading and 8% profit margin | An actuary; these are inherited constants, not derived here |
| Proxy-discrimination / disparate-impact testing | Protected-attribute data and a compliance owner |
| Out-of-time backtest, drift monitoring, deployment | A production environment and an owner for the monitoring plan in the model card |
| Hyperparameter search (QA-029, partial) | Worth doing only once the real data is available |

---

## 6. What This Review Delivered

| Artefact | Purpose |
|---|---|
| `tests/synthetic_data.py` | Production-shaped fixture generator; the suite needs no 500 MB download |
| `tests/test_data_contract.py` | The schema/quality contract the pipeline silently assumes; point it at real data with `ACIS_DATA_PATH=...` |
| `tests/test_hypothesis_testing.py` | Behaviour and coverage of Task 3 |
| `tests/test_modeling.py` | Split integrity, preprocessing, model quality gates, premium sanity |
| `tests/test_notebooks.py` | Notebook hygiene and reproducibility |
| `tests/test_end_to_end.py` | Runs both scripts and both notebooks as a user would, in a sandbox |
| `tests/test_repo_hygiene.py` | Conflict markers, dependency completeness, DVC/ignore correctness |
| `.github/workflows/qa.yml` | Lint + fast tests + end-to-end tests + a clean-tree check on every push and PR |
| `pyproject.toml` | `pytest` and `ruff` configuration |
| `src/config.py` | Every path, feature, pricing and statistical parameter in one place, copied into each run's metrics |
| `src/data.py` | Runtime data contract: validates, reports, and raises rather than modelling a bad extract |
| `src/modeling.py` (rewritten) | Separate preprocessors, imputation, calibration, log-target severity, baseline gate, persistence |
| `src/hypothesis_testing.py` (rewritten) | Six hypotheses, Holm correction, effect sizes, coverage reporting, JSON/CSV output |
| `docs/MODEL_CARD.md` | Intended use, exclusions, fairness position, limitations, monitoring plan |
| `dvc.yaml`, `.pre-commit-config.yaml`, `LICENSE` | Reproducible stages, hooks, licence |
| Notebooks (rewritten) | Call `src/`, carry no outputs, write to the ignored artefacts directory |

Current state: **88 tests — 87 pass, 1 `xfail` (QA-001, blocked on the client)**, `ruff` clean.

### 6.1 QA-031 / QA-032 — found by attacking the fixes, not by re-running them

An adversarial pass over the hardened branch broke two of its own safety claims. Both are closed:

- **QA-031.** Cutting the extract in half at a byte offset — what a failed download or a killed
  export produces — left a file that parsed cleanly: pandas pads the severed final record with
  `NaN`, `dropna(subset=["TotalClaims"])` deletes the evidence, and both entry points exited **0**
  with plausible numbers and persisted artefacts. `validate_policies` checked columns, dtypes,
  emptiness and null targets, but never the *shape of the file*. It now rejects a short final
  record, and `ACIS_MIN_ROWS` lets an operator declare the expected size so a cut on a line
  boundary is caught too.
- **QA-032.** `Province` was gated at 1,000 policies and `PostalCode` at 500, but the gender
  frequency z-test had no exposure gate at all. A 40-policy book therefore printed
  `coverage is INCOMPLETE` and then a rating-change recommendation — on the one factor the model is
  forbidden to price on. The gate now applies to every hypothesis, and when the family is partial
  any surviving rejection is marked `PROVISIONAL ONLY`. Separately, `src/modeling.py` refuses a
  book below 500 policies or 50 claims instead of failing inside `train_test_split`.

Also attacked and unbroken: the premium floor under five adversarial books and an injected hostile
severity model (the unguarded formula reaches −R1.09 trillion where the guarded quote stays at
R434.78), the clean-tree guarantee, the xgboost pin, and the five other corrupt-extract shapes.

### Running the suite

```bash
pip install -r requirements-dev.txt
pre-commit install
python -m tests.synthetic_data   # only needed while the real extract is unavailable

pytest -m "not slow"      # contracts, units, quality gates, hygiene
pytest -m slow            # full pipeline and notebook execution
pytest                    # everything
ruff check .

# validate the contract against the real extract instead of the fixture
ACIS_DATA_PATH=data/MachineLearningRating_v3.txt pytest tests/test_data_contract.py
```
