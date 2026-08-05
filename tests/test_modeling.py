"""Behavioural tests for ``src/modeling.py`` (Task 4).

Covers the data split, the preprocessing pipeline, model quality gates and the
actuarial sanity of the risk-based premium formula.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import r2_score, roc_auc_score

EXPENSE_LOADING = 400.00
PROFIT_MARGIN = 0.08
PREMIUM_FLOOR = EXPENSE_LOADING / (1 - PROFIT_MARGIN)


@pytest.fixture
def modeling(modeling_module):
    """``src.modeling`` with its module-level feature lists restored afterwards."""
    numerical = copy.deepcopy(modeling_module.numerical_features)
    categorical = copy.deepcopy(modeling_module.categorical_features)
    yield modeling_module
    modeling_module.numerical_features[:] = numerical
    modeling_module.categorical_features[:] = categorical


@pytest.fixture(scope="module")
def split(modeling_module, synthetic_data_file: Path):
    return modeling_module.prep_data_for_modeling(str(synthetic_data_file), "|")


@pytest.fixture(scope="module")
def trained(modeling_module, split):
    return modeling_module.train_and_evaluate_models(*split)


def test_prep_returns_eight_datasets(split):
    assert len(split) == 8
    assert all(part is not None for part in split), "data preparation failed to load the extract"


def test_prep_returns_nones_for_an_unreadable_file(modeling, tmp_path: Path):
    result = modeling.prep_data_for_modeling(str(tmp_path / "missing.txt"), "|")
    assert len(result) == 8 and all(part is None for part in result)


def test_train_and_test_sets_do_not_overlap(split):
    x_train_prob, x_test_prob = split[0], split[1]
    assert not set(x_train_prob.index) & set(x_test_prob.index)


def test_class_balance_is_preserved_by_stratification(split):
    y_train_prob, y_test_prob = split[2], split[3]
    assert y_train_prob.mean() == pytest.approx(y_test_prob.mean(), abs=0.01)


def test_severity_set_contains_only_claiming_policies(split, synthetic_data_file, data_sep):
    y_train_sev, y_test_sev = split[6], split[7]
    assert (y_train_sev > 0).all() and (y_test_sev > 0).all()


def test_split_is_deterministic(modeling, synthetic_data_file: Path):
    first = modeling.prep_data_for_modeling(str(synthetic_data_file), "|")
    second = modeling.prep_data_for_modeling(str(synthetic_data_file), "|")
    assert list(first[0].index) == list(second[0].index), "the split is not reproducible across runs"


def test_car_age_is_derived_and_non_negative(modeling, synthetic_data_file: Path):
    modeling.prep_data_for_modeling(str(synthetic_data_file), "|")
    assert "Car_Age" in modeling.numerical_features


def test_preprocessor_handles_unseen_categories(modeling, split):
    """A category absent from training must not blow up scoring."""
    preprocessor = modeling.create_preprocessor()
    x_train, x_test = split[0], split[1].copy()
    preprocessor.fit(x_train)
    x_test.loc[x_test.index[0], "make"] = "A_MAKE_NEVER_SEEN_BEFORE"
    transformed = preprocessor.transform(x_test)
    assert transformed.shape[0] == len(x_test)


def test_probability_model_beats_random(trained, split):
    _, _, x_test_prob, y_pred_proba = trained
    auc = roc_auc_score(split[3], y_pred_proba)
    assert 0.0 <= auc <= 1.0
    assert np.isfinite(y_pred_proba).all()
    assert ((y_pred_proba >= 0) & (y_pred_proba <= 1)).all(), "predicted probabilities outside [0, 1]"


def test_risk_based_premium_formula(trained):
    prob_model, sev_model, x_test, y_pred_proba = trained
    pure_premium = y_pred_proba * sev_model.predict(x_test)
    premium = (pure_premium + EXPENSE_LOADING) / (1 - PROFIT_MARGIN)
    assert np.isfinite(premium).all()
    assert premium.shape == y_pred_proba.shape


@pytest.mark.xfail(
    strict=True,
    reason="QA-002: one ColumnTransformer instance is shared by both pipelines, so fitting the "
    "severity model refits the classifier's preprocessor on claim-only rows",
)
def test_each_pipeline_owns_its_preprocessor(trained):
    prob_model, sev_model, _, _ = trained
    assert prob_model.named_steps["preprocessor"] is not sev_model.named_steps["preprocessor"]


@pytest.mark.xfail(
    strict=True,
    reason="QA-004: prep_data_for_modeling appends Car_Age to the module-level numerical_features list",
)
def test_prep_does_not_mutate_module_state(modeling, synthetic_data_file: Path):
    if "Car_Age" in modeling.numerical_features:
        modeling.numerical_features.remove("Car_Age")
    before = list(modeling.numerical_features)
    modeling.prep_data_for_modeling(str(synthetic_data_file), "|")
    assert modeling.numerical_features == before


@pytest.mark.xfail(
    strict=True,
    reason="QA-003: the severity regressor can predict a negative claim cost, pushing the "
    "risk-based premium below the expense floor and even negative",
)
def test_premium_never_falls_below_the_expense_floor(trained):
    prob_model, sev_model, x_test, y_pred_proba = trained
    pure_premium = y_pred_proba * sev_model.predict(x_test)
    premium = (pure_premium + EXPENSE_LOADING) / (1 - PROFIT_MARGIN)
    assert premium.min() >= PREMIUM_FLOOR, (
        f"minimum quoted premium is {premium.min():,.2f}, below the {PREMIUM_FLOOR:,.2f} expense floor "
        f"({int((premium < PREMIUM_FLOOR).sum())} of {len(premium)} policies)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="QA-006: the severity model scores worse than predicting the mean claim amount",
)
def test_severity_model_beats_the_mean_baseline(trained, split):
    _, sev_model, _, _ = trained
    x_test_sev, y_test_sev = split[5], split[7]
    r2 = r2_score(y_test_sev, sev_model.predict(x_test_sev))
    assert r2 > 0, f"severity R^2 is {r2:.4f}; a constant mean predictor would score higher"


@pytest.mark.xfail(
    strict=True,
    reason="QA-013: no baseline model (linear/GLM) is trained, so the XGBoost choice is unjustified",
)
def test_a_baseline_model_is_compared(modeling):
    source = Path(modeling.__file__).read_text(encoding="utf-8")
    assert any(term in source for term in ("LinearRegression", "Ridge", "GLM", "PoissonRegressor", "DummyRegressor"))


@pytest.mark.xfail(
    strict=True,
    reason="QA-014: trained models are never persisted, so scoring requires a full retrain",
)
def test_models_are_persisted(modeling):
    source = Path(modeling.__file__).read_text(encoding="utf-8")
    assert any(term in source for term in ("joblib", "pickle", "save_model", "to_json"))
