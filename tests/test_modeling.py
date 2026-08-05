"""Behavioural tests for ``src/modeling.py`` (Task 4).

Covers the data split, the preprocessing pipeline, the model quality gates and
the actuarial sanity of the risk-based premium. The gates are meaningful because
the synthetic fixture carries a known risk structure (see
``tests/synthetic_data.py``): a model that cannot recover it is broken.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import brier_score_loss, r2_score, roc_auc_score

from src import config
from tests.synthetic_data import write_dataset

EXPENSE_LOADING = config.EXPENSE_LOADING
PROFIT_MARGIN = config.PROFIT_MARGIN
PREMIUM_FLOOR = config.PREMIUM_FLOOR


@pytest.fixture
def modeling(modeling_module):
    return modeling_module


@pytest.fixture(scope="module")
def split(modeling_module, synthetic_data_file: Path):
    return modeling_module.prep_data_for_modeling(str(synthetic_data_file), "|")


@pytest.fixture(scope="module")
def trained(modeling_module, split):
    return modeling_module.train_and_evaluate_models(*split)


@pytest.fixture(scope="module")
def premiums(modeling_module, trained):
    return modeling_module.calculate_risk_based_premium(
        trained.probability_model, trained.severity_model, trained.x_test, trained.predicted_probability, verbose=False
    )


def test_prep_returns_eight_datasets(split):
    assert len(split) == 8
    assert all(part is not None for part in split), "data preparation failed to load the extract"


def test_prep_returns_nones_for_an_unreadable_file(modeling, tmp_path: Path):
    result = modeling.prep_data_for_modeling(str(tmp_path / "missing.txt"), "|")
    assert len(result) == 8 and all(part is None for part in result)


def test_a_book_too_thin_to_model_is_refused_not_fitted(modeling, tmp_path: Path, capsys):
    """A handful of policies cannot support a split, a calibration fold or a rate."""
    path = write_dataset(tmp_path / "thin.txt", n_rows=40, seed=11)
    result = modeling.prep_data_for_modeling(str(path), "|")

    assert all(part is None for part in result)
    assert "below the" in capsys.readouterr().out


def test_train_and_test_sets_do_not_overlap(split):
    assert not set(split[0].index) & set(split[1].index)


def test_class_balance_is_preserved_by_stratification(split):
    assert split[2].mean() == pytest.approx(split[3].mean(), abs=0.01)


def test_severity_set_contains_only_claiming_policies(split):
    assert (split[6] > 0).all() and (split[7] > 0).all()


def test_split_is_deterministic(modeling, synthetic_data_file: Path):
    first = modeling.prep_data_for_modeling(str(synthetic_data_file), "|")
    second = modeling.prep_data_for_modeling(str(synthetic_data_file), "|")
    assert list(first[0].index) == list(second[0].index), "the split is not reproducible across runs"


def test_car_age_is_derived_and_non_negative(split):
    x_train = split[0]
    assert "Car_Age" in x_train.columns
    assert (x_train["Car_Age"].dropna() >= 0).all()


def test_missing_rating_factors_do_not_delete_policies(modeling, synthetic_data_file: Path, split, data_sep):
    """QA-005: the book is ~78% missing on CustomValueEstimate; imputation keeps it whole."""
    extract = pd.read_csv(synthetic_data_file, sep=data_sep, low_memory=False)
    modelled = len(split[0]) + len(split[1])
    assert modelled == len(extract), (
        f"{modelled:,} of {len(extract):,} policies survived preparation; missing rating factors "
        "must be imputed, not dropped"
    )


def test_preprocessor_imputes_instead_of_failing_on_nulls(modeling, split):
    """A policy with every numeric factor missing must still be priceable."""
    preprocessor = modeling.create_preprocessor()
    x_train, x_test = split[0], split[1].copy()
    preprocessor.fit(x_train)

    x_test.loc[x_test.index[0], ["CustomValueEstimate", "SumInsured", "kilowatts"]] = np.nan
    x_test.loc[x_test.index[0], "make"] = None
    transformed = preprocessor.transform(x_test)
    assert transformed.shape[0] == len(x_test)
    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unseen_categories(modeling, split):
    preprocessor = modeling.create_preprocessor()
    x_train, x_test = split[0], split[1].copy()
    preprocessor.fit(x_train)
    x_test.loc[x_test.index[0], "make"] = "A_MAKE_NEVER_SEEN_BEFORE"
    assert preprocessor.transform(x_test).shape[0] == len(x_test)


def test_each_pipeline_owns_its_preprocessor(modeling):
    """QA-002: a shared ColumnTransformer let the severity fit refit the frequency model's."""
    first = modeling.build_probability_pipeline()
    second = modeling.build_severity_pipeline()
    assert first.named_steps["preprocessor"] is not second.named_steps["preprocessor"]
    assert modeling.create_preprocessor() is not modeling.create_preprocessor()


def test_prep_does_not_mutate_module_state(modeling, synthetic_data_file: Path):
    """QA-004: engineered features must not be appended to a global list."""
    before = list(modeling.numerical_features)
    modeling.prep_data_for_modeling(str(synthetic_data_file), "|")
    modeling.prep_data_for_modeling(str(synthetic_data_file), "|")
    assert modeling.numerical_features == before


def test_probability_model_is_a_usable_probability(trained, split):
    y_pred_proba = trained.predicted_probability
    y_test = split[3]
    assert np.isfinite(y_pred_proba).all()
    assert ((y_pred_proba >= 0) & (y_pred_proba <= 1)).all(), "predicted probabilities outside [0, 1]"
    assert roc_auc_score(y_test, y_pred_proba) > 0.6, "the frequency model does not rank risk"


def test_probability_model_is_calibrated(trained, split):
    """A pure premium multiplies by this number, so it must be a probability, not a score."""
    y_test = split[3]
    y_pred_proba = trained.predicted_probability
    base_rate = np.full_like(y_pred_proba, float(np.mean(y_test)))
    assert brier_score_loss(y_test, y_pred_proba) <= brier_score_loss(y_test, base_rate), (
        "the model's Brier score is worse than always predicting the portfolio claim rate"
    )
    assert y_pred_proba.mean() == pytest.approx(float(np.mean(y_test)), abs=0.02), (
        "predicted claim frequency does not match the observed claim rate"
    )


def test_severity_predictions_are_positive(trained, split):
    """QA-003 at source: a negative predicted claim cost is what produced negative premiums."""
    predictions = trained.severity_model.predict(split[5])
    assert (predictions > 0).all(), f"{int((predictions <= 0).sum())} predicted claim costs are not positive"


def test_severity_model_beats_the_mean_baseline(trained, split):
    """QA-006: an unjustified model is worse than an honest constant."""
    x_train_sev, x_test_sev, y_train_sev, y_test_sev = split[4], split[5], split[6], split[7]
    baseline = DummyRegressor(strategy="mean").fit(x_train_sev, y_train_sev)

    model_r2 = r2_score(y_test_sev, trained.severity_model.predict(x_test_sev))
    baseline_r2 = r2_score(y_test_sev, baseline.predict(x_test_sev))
    assert model_r2 >= baseline_r2, (
        f"severity R^2 is {model_r2:.4f} against {baseline_r2:.4f} for a constant mean predictor"
    )


def test_severity_model_falls_back_when_it_cannot_beat_the_baseline(modeling, split):
    """Fitting on noise must select the baseline rather than price policies on it."""
    rng = np.random.default_rng(0)
    x_train = split[4]
    noise = pd.Series(rng.lognormal(8.0, 1.0, size=len(x_train)), index=x_train.index)
    _, choice, scores = modeling.select_severity_model(x_train, noise)
    assert choice == max(scores, key=lambda name: scores[name])
    assert scores["baseline_mean"] <= max(scores.values())


def test_risk_based_premium_formula(premiums, trained):
    expected = (premiums["Predicted_Prob"] * premiums["Predicted_Severity"] + EXPENSE_LOADING) / (1 - PROFIT_MARGIN)
    assert np.isfinite(premiums["Risk_Based_Premium"]).all()
    assert len(premiums) == len(trained.x_test)
    assert (premiums["Risk_Based_Premium"] >= expected - 1e-9).all(), "the quote undercuts the loaded pure premium"


def test_premium_never_falls_below_the_expense_floor(premiums):
    """QA-003: selling below the expense floor is a guaranteed loss on every policy."""
    minimum = premiums["Risk_Based_Premium"].min()
    assert minimum >= PREMIUM_FLOOR - 1e-9, (
        f"minimum quoted premium is {minimum:,.2f}, below the {PREMIUM_FLOOR:,.2f} expense floor"
    )


def test_premium_rises_with_risk(premiums):
    correlation = premiums["Pure_Premium"].corr(premiums["Risk_Based_Premium"])
    assert correlation > 0.9, f"premium barely tracks expected loss (r={correlation:.2f})"


def test_a_baseline_model_is_compared(modeling):
    """QA-013: the choice of a gradient booster has to be earned against a baseline."""
    source = Path(modeling.__file__).read_text(encoding="utf-8")
    assert "DummyRegressor" in source and "cross_val_score" in source


def test_models_and_metrics_are_persisted(modeling, trained, split, tmp_path: Path):
    """QA-014/QA-023: scoring must not require a retrain, and metrics must survive the run."""
    metrics = modeling.evaluate_models(trained, split)
    paths = modeling.persist(trained, metrics, model_dir=tmp_path / "models", metrics_dir=tmp_path / "metrics")
    assert all(path.exists() for path in paths.values())

    probability_model, severity_model = modeling.load_models(tmp_path / "models")
    reloaded = probability_model.predict_proba(trained.x_test)[:, 1]
    np.testing.assert_allclose(reloaded, trained.predicted_probability, rtol=1e-6)
    assert (severity_model.predict(split[5]) > 0).all()


def test_metrics_payload_is_complete(modeling, trained, split):
    metrics = modeling.evaluate_models(trained, split)
    assert metrics["severity"]["r2"] >= metrics["severity"]["r2_baseline_mean"]
    assert metrics["pricing"]["min_premium"] >= PREMIUM_FLOOR - 1e-9
    assert metrics["config"]["premium_floor"] == PREMIUM_FLOOR
    assert metrics["versions"]["xgboost"]


def test_gender_is_not_a_rating_factor_by_default():
    """Pricing on a protected attribute needs sign-off; the default must be off."""
    assert config.USE_GENDER_AS_RATING_FACTOR is False
    numerical, categorical = config.feature_lists()
    assert "Gender" not in numerical and "Gender" not in categorical


def test_shap_ranks_the_features_that_drive_severity(modeling, trained, split):
    importance = modeling.run_shap_analysis(trained.severity_model, split[4])
    assert not importance.empty
    assert importance.index[0].startswith(("num__", "cat__"))
    # SumInsured is the strongest severity driver in the fixture's ground truth.
    assert "SumInsured" in "".join(importance.head(3).index)
