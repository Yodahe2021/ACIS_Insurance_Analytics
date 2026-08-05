# Model Card — ACIS Risk-Based Pricing

| Item | Value |
|---|---|
| Models | Claim probability (frequency) + claim severity, combined into a risk-based premium |
| Owner | Yodahe Tsegaye |
| Version | Produced by `python -m src.modeling`; the exact configuration and library versions are written into `artifacts/metrics/model_metrics.json` on every run |
| Status | **Not approved for live pricing.** Validated only against a synthetic extract; see Limitations |
| Code | `src/modeling.py`, `src/config.py`, `src/data.py` |

## Intended use

**In scope.** Producing an indicative technical premium for a South African motor policy, and
ranking policies by expected loss for portfolio analysis, reinsurance discussion and rate-review
input.

**Out of scope.** Binding quotes to customers, declining risks, individual underwriting decisions,
any use outside South African motor, and any use on a book whose mix differs materially from the
Feb 2014 – Aug 2015 training period. The model produces a *technical* premium; it contains no
competitive, elasticity or regulatory-loading view.

## The model

```text
pure premium   = P(claim) x E(claim cost | claim)
quoted premium = max( (pure premium + 400) / (1 - 0.08), 434.78 )
```

| Stage | Estimator | Why |
|---|---|---|
| Frequency | XGBoost classifier wrapped in `CalibratedClassifierCV` (isotonic, 3-fold) | The pure premium multiplies by this number, so it must be a calibrated probability and not a ranking score |
| Severity | XGBoost regressor on `log(claim amount)` via `TransformedTargetRegressor` | Claim costs are right-skewed and strictly positive; the log target makes predictions positive by construction |
| Selection | 5-fold CV R² against `DummyRegressor(strategy="mean")` | A severity model that cannot beat the mean claim is not shipped — the baseline is used instead |
| Missing data | Median / `"Unknown"` imputation with missingness indicators | ~78% of policies lack `CustomValueEstimate`; dropping them discarded most of the book |
| Floor | `PREMIUM_FLOOR = 400 / 0.92 = R434.78` | No policy may be quoted below the cost of writing it |

## Rating factors

Vehicle: `Cylinders`, `cubiccapacity`, `kilowatts`, `CustomValueEstimate`, `SumInsured`, `Car_Age`,
`make`, `VehicleType`, `bodytype`. Geographic: `Province`. Security: `AlarmImmobiliser`,
`TrackingDevice`.

## Fairness and protected attributes

**Gender is deliberately excluded from the rating factors** (`USE_GENDER_AS_RATING_FACTOR = False`).
Three reasons:

1. Pricing on a protected attribute requires explicit legal and compliance sign-off, which this
   project does not have.
2. About 57% of the book records gender as "Not specified", so any gender-based rate would be
   applied to a minority of policies and proxied for the rest.
3. The Task 3 hypothesis test does not reject equality of claim frequency or severity between women
   and men once the family of tests is Holm-corrected.

The flag exists so that the decision is explicit and reversible, not so that it is convenient to
turn on: flipping it without that sign-off is a compliance breach.

**Proxy risk remains.** `Province`, `make` and `SumInsured` correlate with income and, in the South
African context, potentially with race. This model has **not** been tested for disparate impact.
A fairness review against the client's protected-attribute data is a precondition of live use.

## Performance

Metrics are written to `artifacts/metrics/model_metrics.json` on every run — that file, not this
page, is the record for a given model.

Gates enforced by the test suite (`tests/test_modeling.py`):

- frequency AUC > 0.6 and a Brier score no worse than always predicting the portfolio claim rate;
- mean predicted probability within 2 points of the observed claim rate;
- every severity prediction strictly positive;
- severity R² at or above the mean baseline on held-out data;
- no quoted premium below the expense floor.

> **The only numbers produced so far come from a synthetic extract.** On the real data the original
> severity model scored R² = −0.087 on 524 training claims. Nothing here supersedes that until the
> pipeline has been run on the real extract and the metrics file reviewed.

## Limitations

1. **Not validated on real data.** The 500 MB extract is not retrievable from this repository
   (QA-001), so every metric quoted anywhere in the repo is from synthetic data.
2. **Severity is trained on very few claims.** At a 0.28% claim rate, ~1M policy-months yield only a
   few thousand claims, and the original run had 524. Expect wide error bars and expect the
   baseline to win on some refits — by design, that is what will then be shipped.
3. **No temporal validation.** The split is random, not out-of-time, so it does not measure how the
   model degrades on future business.
4. **No large-loss treatment.** Claims are not capped or spread; a single catastrophic claim can
   dominate the severity fit.
5. **No exposure weighting.** Rows are policy-months treated as equally weighted observations.
6. **The technical premium is not a market premium.** No expense allocation beyond a flat R400, no
   commission, no reinsurance cost, no IPT, no competitive adjustment.
7. **Distribution shift is unmonitored.** The model has no drift detection in this repository.

## Monitoring and operation

Before live use:

- an actuarial review of the pure-premium build and the R400 / 8% loadings;
- a compliance review covering proxy discrimination and the gender exclusion;
- an out-of-time backtest on a holdout period the model has never seen.

Once live, at minimum monthly:

- **calibration**: predicted vs actual claim frequency, overall and by province;
- **loss ratio** by rating cell against the pricing assumption;
- **feature drift**: missingness rates (especially `CustomValueEstimate`) and category frequencies
  against training;
- **floor rate**: the share of quotes clamped to R434.78 — a rising share means the risk model is
  producing implausibly low expected losses;
- **stability**: severity R² and frequency AUC on each refresh, with a rollback to the mean
  baseline if the severity model stops beating it.

Retrain quarterly, or immediately after a rate change, a mix change or a data-contract change.
`src/data.py` validates the contract on every load and raises rather than scoring a bad extract.

**Set `ACIS_MIN_ROWS` to the expected extract size in any scheduled run.** A severed final record is
always rejected, but an export cut on a line boundary is indistinguishable from a small book
without a declared expectation — and pricing off half the exposure is silent by nature.
