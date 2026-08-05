"""Task 4 - frequency/severity modelling and risk-based pricing.

The pipeline is a classical actuarial decomposition:

    pure premium = P(claim) x E[claim cost | claim]
    quoted premium = max(pure premium + expenses, expense floor) / (1 - margin)

Two design points are load-bearing and were the source of real defects:

* Missing rating factors are **imputed**, never used to delete a policy. The
  book is ~78% missing on ``CustomValueEstimate``; listwise deletion threw away
  four fifths of the exposure and left the severity model with a few hundred
  rows.
* Severity is modelled on the log scale, so a predicted claim cost is
  positive by construction and a quoted premium can never fall below the cost
  of writing the policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import DataValidationError, load_policies, write_json

DATA_PATH = config.DATA_PATH
DATA_SEP = config.DATA_SEP

#: Kept as module-level names for readability and for notebook use. They are
#: never mutated - call ``config.feature_lists()`` for a private copy.
numerical_features, categorical_features = config.feature_lists()

EXPENSE_LOADING = config.EXPENSE_LOADING
PROFIT_MARGIN = config.PROFIT_MARGIN
PREMIUM_FLOOR = config.PREMIUM_FLOOR

EMPTY_SPLIT: tuple[None, ...] = (None,) * 8

#: Below this the split, the calibration folds and the CV selection cannot run,
#: and a book this thin could not support a rate in any case.
MIN_POLICIES_TO_MODEL = 500
MIN_CLAIMS_TO_MODEL = 50


@dataclass
class TrainedModels:
    """Fitted estimators plus the scoring frame used to price the test book."""

    probability_model: object
    severity_model: Pipeline
    x_test: pd.DataFrame
    predicted_probability: np.ndarray

    def __iter__(self):
        yield from (self.probability_model, self.severity_model, self.x_test, self.predicted_probability)


def create_preprocessor(numerical: list[str] | None = None, categorical: list[str] | None = None) -> ColumnTransformer:
    """Build a fresh preprocessor.

    A new instance every call: sharing one ``ColumnTransformer`` between the
    frequency and severity pipelines meant fitting the second silently refitted
    the first on claim-only rows.
    """
    if numerical is None or categorical is None:
        default_numerical, default_categorical = config.feature_lists()
        numerical = list(default_numerical if numerical is None else numerical)
        categorical = list(default_categorical if categorical is None else categorical)

    numeric_pipeline = Pipeline(
        steps=[
            # Missingness in a rating factor is itself predictive, so it is
            # flagged rather than smoothed away.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numerical)),
            ("cat", categorical_pipeline, list(categorical)),
        ],
        remainder="drop",
    )


def build_probability_pipeline(numerical=None, categorical=None) -> Pipeline:
    """Frequency model: P(at least one claim)."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(numerical, categorical)),
            (
                "classifier",
                xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_lambda=1.0,
                    tree_method="hist",
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_severity_pipeline(numerical=None, categorical=None, baseline: bool = False) -> Pipeline:
    """Severity model: E[claim cost | claim], fitted on the log scale.

    ``TransformedTargetRegressor`` with log/exp keeps every prediction strictly
    positive and stops the long right tail of claim amounts from dominating the
    squared-error objective.
    """
    regressor: object
    if baseline:
        regressor = DummyRegressor(strategy="mean")
    else:
        regressor = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(numerical, categorical)),
            ("regressor", TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)),
        ]
    )


def base_pipeline(model) -> Pipeline:
    """Unwrap the fitted ``Pipeline`` inside a calibrated classifier."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def prep_data_for_modeling(file_path=DATA_PATH, sep: str = DATA_SEP):
    """Load the extract and produce the frequency and severity train/test splits.

    Returns eight objects (``X``/``y``, train/test, for both models), or eight
    ``None`` values if the extract cannot be loaded or fails validation.
    """
    try:
        df, _ = load_policies(file_path, sep=sep, verbose=False)
    except DataValidationError as exc:
        print(f"Error loading data for modeling: {exc}")
        return EMPTY_SPLIT

    numerical, categorical = config.feature_lists()
    missing = [column for column in [*numerical, *categorical] if column not in df.columns]
    if missing:
        print(f"Warning: rating factors absent from the extract and skipped: {missing}")
        numerical = [column for column in numerical if column in df.columns]
        categorical = [column for column in categorical if column in df.columns]

    feature_columns = [*numerical, *categorical]
    if not feature_columns:
        print("Error: none of the configured rating factors are present in the extract.")
        return EMPTY_SPLIT

    # Only the target may force a row out; missing features are imputed.
    modelled = df.dropna(subset=["TotalClaims"]).copy()
    x = modelled[feature_columns]
    y_probability = modelled["Claim_Indicator"]

    if y_probability.nunique() < 2:
        print("Error: the extract contains a single claim class; a frequency model cannot be fitted.")
        return EMPTY_SPLIT

    claim_count = int(y_probability.sum())
    if claim_count < MIN_CLAIMS_TO_MODEL or len(modelled) < MIN_POLICIES_TO_MODEL:
        print(
            f"Error: the extract holds {len(modelled):,} policies and {claim_count:,} claims, below the "
            f"{MIN_POLICIES_TO_MODEL:,} / {MIN_CLAIMS_TO_MODEL:,} needed to split, fit and evaluate two models."
        )
        return EMPTY_SPLIT

    x_train_prob, x_test_prob, y_train_prob, y_test_prob = train_test_split(
        x,
        y_probability,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_probability,
    )

    claims = modelled[modelled["Claim_Indicator"] == 1]
    x_train_sev, x_test_sev, y_train_sev, y_test_sev = train_test_split(
        claims[feature_columns],
        claims["TotalClaims"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    return (
        x_train_prob,
        x_test_prob,
        y_train_prob,
        y_test_prob,
        x_train_sev,
        x_test_sev,
        y_train_sev,
        y_test_sev,
    )


def train_and_evaluate_models(
    x_train_prob,
    x_test_prob,
    y_train_prob,
    y_test_prob,
    x_train_sev,
    x_test_sev,
    y_train_sev,
    y_test_sev,
) -> TrainedModels:
    """Fit both models, cross-validate them and print the headline metrics."""
    print("--- TASK 4: Predictive Modeling ---")

    numerical = [column for column in config.feature_lists()[0] if column in x_train_prob.columns]
    categorical = [column for column in config.feature_lists()[1] if column in x_train_prob.columns]

    # --- 1. Claim probability -------------------------------------------------
    # Gradient-boosted scores are not probabilities; isotonic calibration makes
    # them usable as the frequency term of a pure premium.
    probability_model = CalibratedClassifierCV(
        build_probability_pipeline(numerical, categorical),
        method="isotonic",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
    )
    probability_model.fit(x_train_prob, y_train_prob)
    y_pred_proba = probability_model.predict_proba(x_test_prob)[:, 1]

    auc = roc_auc_score(y_test_prob, y_pred_proba)
    print(f"\n1. Claim Probability Model AUC-ROC: {auc:.4f}")
    print(f"   Average precision: {average_precision_score(y_test_prob, y_pred_proba):.4f} "
          f"| Brier: {brier_score_loss(y_test_prob, y_pred_proba):.6f} "
          f"| base rate: {y_train_prob.mean():.4%}")

    # --- 2. Claim severity ----------------------------------------------------
    severity_model, severity_choice, severity_cv = select_severity_model(x_train_sev, y_train_sev, numerical, categorical)
    y_pred_sev = severity_model.predict(x_test_sev)
    rmse = float(np.sqrt(mean_squared_error(y_test_sev, y_pred_sev)))
    print(
        f"2. Claim Severity Model ({severity_choice}) RMSE: {rmse:,.2f} "
        f"| R-squared: {r2_score(y_test_sev, y_pred_sev):.4f} "
        f"| CV R-squared: {severity_cv[severity_choice]:.4f}"
    )

    return TrainedModels(probability_model, severity_model, x_test_prob, y_pred_proba)


def select_severity_model(x_train, y_train, numerical=None, categorical=None) -> tuple[Pipeline, str, dict[str, float]]:
    """Pick the severity model by cross-validation, with a mean-claim fallback.

    A gradient booster that cannot beat "predict the average claim" out of fold
    is not shipped: an unjustified model is worse than an honest constant,
    because it prices individual policies on noise.
    """
    candidates = {
        "baseline_mean": build_severity_pipeline(numerical, categorical, baseline=True),
        "xgboost_log_target": build_severity_pipeline(numerical, categorical),
    }
    folds = KFold(n_splits=min(config.CV_FOLDS, max(2, len(y_train) // 30)), shuffle=True, random_state=config.RANDOM_STATE)

    scores = {
        name: float(np.mean(cross_val_score(model, x_train, y_train, cv=folds, scoring="r2")))
        for name, model in candidates.items()
    }
    choice = max(scores, key=lambda name: scores[name])
    model = candidates[choice]
    model.fit(x_train, y_train)
    return model, choice, scores


def evaluate_models(models: TrainedModels, split, extra: dict | None = None) -> dict:
    """Collect every metric worth tracking between runs into one payload."""
    (_, _, _, y_test_prob, x_train_sev, x_test_sev, y_train_sev, y_test_sev) = split
    y_pred_proba = models.predicted_probability
    y_pred_sev = models.severity_model.predict(x_test_sev)

    baseline_severity = build_severity_pipeline(baseline=True).fit(x_train_sev, y_train_sev)
    baseline_pred = baseline_severity.predict(x_test_sev)

    pricing = calculate_risk_based_premium(
        models.probability_model, models.severity_model, models.x_test, y_pred_proba, verbose=False
    )

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
        },
        "config": {
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "expense_loading": config.EXPENSE_LOADING,
            "profit_margin": config.PROFIT_MARGIN,
            "premium_floor": config.PREMIUM_FLOOR,
            "gender_used_as_rating_factor": config.USE_GENDER_AS_RATING_FACTOR,
        },
        "frequency": {
            "n_test": int(len(y_test_prob)),
            "claim_rate_test": float(np.mean(y_test_prob)),
            "auc_roc": float(roc_auc_score(y_test_prob, y_pred_proba)),
            "average_precision": float(average_precision_score(y_test_prob, y_pred_proba)),
            "brier": float(brier_score_loss(y_test_prob, y_pred_proba)),
            "brier_of_base_rate": float(brier_score_loss(y_test_prob, np.full_like(y_pred_proba, np.mean(y_test_prob)))),
            "mean_predicted_probability": float(np.mean(y_pred_proba)),
        },
        "severity": {
            "n_train": int(len(y_train_sev)),
            "n_test": int(len(y_test_sev)),
            "r2": float(r2_score(y_test_sev, y_pred_sev)),
            "r2_baseline_mean": float(r2_score(y_test_sev, baseline_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test_sev, y_pred_sev))),
            "mae": float(mean_absolute_error(y_test_sev, y_pred_sev)),
            "min_prediction": float(np.min(y_pred_sev)),
        },
        "pricing": {
            "premium_floor": config.PREMIUM_FLOOR,
            "min_premium": float(pricing["Risk_Based_Premium"].min()),
            "median_premium": float(pricing["Risk_Based_Premium"].median()),
            "max_premium": float(pricing["Risk_Based_Premium"].max()),
            "share_at_floor": float((pricing["Risk_Based_Premium"] <= config.PREMIUM_FLOOR + 1e-9).mean()),
        },
    }
    if extra:
        metrics.update(extra)
    return metrics


def calculate_risk_based_premium(prob_model, sev_model, x_test, y_pred_proba=None, verbose: bool = True) -> pd.DataFrame:
    """Price the book: expected loss, loaded for expenses and profit.

    The quote is floored at the expense-only premium. Selling below it is
    guaranteed loss-making no matter how good the risk model is.
    """
    if y_pred_proba is None:
        y_pred_proba = prob_model.predict_proba(x_test)[:, 1]

    predicted_severity = np.clip(sev_model.predict(x_test), 0.0, None)
    pure_premium = y_pred_proba * predicted_severity
    risk_based_premium = np.maximum(
        (pure_premium + config.EXPENSE_LOADING) / (1 - config.PROFIT_MARGIN),
        config.PREMIUM_FLOOR,
    )

    results = pd.DataFrame(
        {
            "Predicted_Prob": y_pred_proba,
            "Predicted_Severity": predicted_severity,
            "Pure_Premium": pure_premium,
            "Risk_Based_Premium": risk_based_premium,
        },
        index=x_test.index,
    )

    if verbose:
        print("\n3. Risk-Based Premium Calculation (Sample):")
        print(results[["Pure_Premium", "Risk_Based_Premium"]].head())
        print(
            f"   Floor R{config.PREMIUM_FLOOR:,.2f} | quoted min R{results['Risk_Based_Premium'].min():,.2f} "
            f"| median R{results['Risk_Based_Premium'].median():,.2f} "
            f"| max R{results['Risk_Based_Premium'].max():,.2f}"
        )

    return results


def run_shap_analysis(model: Pipeline, x_train: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Rank rating factors by mean absolute SHAP value on the severity model."""
    preprocessor = model.named_steps["preprocessor"]
    x_transformed = preprocessor.transform(x_train)
    feature_names = list(preprocessor.get_feature_names_out())

    regressor = model.named_steps["regressor"]
    if isinstance(regressor, TransformedTargetRegressor):
        regressor = regressor.regressor_

    if isinstance(regressor, DummyRegressor):
        print("\n4. SHAP analysis skipped: the severity model is the mean-claim baseline, which has no feature effects.")
        return pd.Series(dtype=float)

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(x_transformed)
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)

    print(f"\n4. Top {top_n} Most Influential Features (Claim Severity Model - SHAP):")
    print(importance.head(top_n))
    return importance


def persist(models: TrainedModels, metrics: dict, model_dir: Path = config.MODEL_DIR, metrics_dir: Path = config.METRICS_DIR) -> dict[str, Path]:
    """Write the fitted models and the run metrics so scoring never retrains."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "probability_model": model_dir / "claim_probability_model.joblib",
        "severity_model": model_dir / "claim_severity_model.joblib",
        "metrics": Path(metrics_dir) / "model_metrics.json",
    }
    joblib.dump(models.probability_model, paths["probability_model"])
    joblib.dump(models.severity_model, paths["severity_model"])
    write_json(metrics, paths["metrics"])
    return paths


def load_models(model_dir: Path = config.MODEL_DIR) -> tuple[object, Pipeline]:
    """Load the persisted models for scoring."""
    model_dir = Path(model_dir)
    return (
        joblib.load(model_dir / "claim_probability_model.joblib"),
        joblib.load(model_dir / "claim_severity_model.joblib"),
    )


def main(file_path=DATA_PATH, sep: str = DATA_SEP) -> int:
    split = prep_data_for_modeling(file_path, sep)
    if split[0] is None:
        return 1

    models = train_and_evaluate_models(*split)
    calculate_risk_based_premium(models.probability_model, models.severity_model, models.x_test, models.predicted_probability)
    run_shap_analysis(models.severity_model, split[4])

    metrics = evaluate_models(models, split)
    paths = persist(models, metrics)
    print("\n5. Artefacts written:")
    for name, path in paths.items():
        print(f"   {name}: {path}")
    print(json.dumps(metrics["severity"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
