"""Contract checks for the policy-transaction extract.

The pipeline reads a pipe-delimited file with no schema guard of any kind. These
tests define the contract the downstream code silently assumes, and can be
pointed at the real extract by setting ``ACIS_DATA_PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import DataValidationError, load_policies, validate_policies
from tests.synthetic_data import write_dataset

REQUIRED_COLUMNS = {
    "PolicyID": "integer",
    "TransactionMonth": "date-like",
    "Province": "string",
    "PostalCode": "integer",
    "Gender": "string",
    "VehicleType": "string",
    "make": "string",
    "bodytype": "string",
    "RegistrationYear": "integer",
    "Cylinders": "numeric",
    "cubiccapacity": "numeric",
    "kilowatts": "numeric",
    "CustomValueEstimate": "numeric",
    "SumInsured": "numeric",
    "AlarmImmobiliser": "string",
    "TrackingDevice": "string",
    "TotalPremium": "numeric",
    "TotalClaims": "numeric",
}

NUMERIC_COLUMNS = [name for name, kind in REQUIRED_COLUMNS.items() if kind in {"numeric", "integer"}]


@pytest.fixture(scope="module")
def dataset(synthetic_data_file: Path, data_sep: str) -> pd.DataFrame:
    """The real extract when ACIS_DATA_PATH is set, otherwise the synthetic stand-in."""
    path = Path(os.environ.get("ACIS_DATA_PATH", synthetic_data_file))
    return pd.read_csv(path, sep=data_sep, low_memory=False)


def test_required_columns_present(dataset: pd.DataFrame):
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataset.columns))
    assert not missing, f"columns consumed by src/ are absent from the extract: {missing}"


def test_numeric_columns_are_numeric(dataset: pd.DataFrame):
    non_numeric = [c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(dataset[c])]
    assert not non_numeric, f"columns parsed as object instead of numeric: {non_numeric}"


def test_transaction_month_is_parseable(dataset: pd.DataFrame):
    parsed = pd.to_datetime(dataset["TransactionMonth"], errors="coerce")
    assert parsed.notna().all(), "TransactionMonth contains unparseable values"


def test_monetary_columns_are_non_negative(dataset: pd.DataFrame):
    for column in ("TotalPremium", "TotalClaims", "SumInsured"):
        negatives = int((dataset[column] < 0).sum())
        assert negatives == 0, f"{column} has {negatives} negative values"


def test_premium_is_never_zero_where_a_loss_ratio_is_computed(dataset: pd.DataFrame):
    """The EDA divides claims by premium per segment; a zero denominator yields inf."""
    grouped = dataset.groupby("Province")[["TotalClaims", "TotalPremium"]].sum()
    assert (grouped["TotalPremium"] > 0).all(), "a province aggregates to zero premium -> infinite loss ratio"


def test_claim_indicator_matches_positive_claims(dataset: pd.DataFrame):
    indicator = np.where(dataset["TotalClaims"] > 0, 1, 0)
    assert indicator.sum() == int((dataset["TotalClaims"] > 0).sum())
    assert set(np.unique(indicator)) <= {0, 1}


def test_gender_categories_are_known(dataset: pd.DataFrame):
    unexpected = set(dataset["Gender"].dropna().unique()) - {"Male", "Female", "Not specified"}
    assert not unexpected, f"unmodelled Gender categories: {sorted(unexpected)}"


def test_registration_year_is_plausible(dataset: pd.DataFrame):
    years = dataset["RegistrationYear"].dropna()
    assert years.between(1900, 2030).all(), "RegistrationYear outside a plausible range -> negative Car_Age"


def test_rating_factors_are_incomplete_and_must_be_imputed(dataset: pd.DataFrame):
    """Pins the property that makes listwise deletion unacceptable (QA-005).

    If a future extract ever arrives complete this fails, and the imputation
    strategy should be revisited rather than silently carried forward.
    """
    retention = len(dataset.dropna(subset=list(REQUIRED_COLUMNS))) / len(dataset)
    assert retention < 0.9, (
        f"{retention:.1%} of rows are complete; the extract no longer needs imputation to stay whole"
    )


def test_validation_accepts_the_extract(dataset: pd.DataFrame):
    report = validate_policies(dataset)
    assert report.ok, f"the extract breaches the data contract: {report.errors}"


def test_validation_rejects_a_broken_extract(dataset: pd.DataFrame):
    broken = dataset.drop(columns=["TotalClaims"])
    assert not validate_policies(broken).ok, "a missing target column was accepted"

    empty = dataset.iloc[:0]
    assert not validate_policies(empty).ok, "an empty extract was accepted"


def test_a_truncated_extract_is_rejected_rather_than_priced(tmp_path: Path):
    """Half a download parses cleanly; pandas pads the severed record with nulls."""
    full = write_dataset(tmp_path / "full.txt", n_rows=400, seed=3)
    payload = full.read_bytes()

    truncated = tmp_path / "truncated.txt"
    truncated.write_bytes(payload[: len(payload) // 2])

    with pytest.raises(DataValidationError, match="truncated"):
        load_policies(truncated, strict=True, verbose=False)

    assert load_policies(full, strict=True, verbose=False)[1].ok, "the intact extract was rejected"


def test_a_short_extract_is_rejected_when_the_expected_size_is_declared(tmp_path: Path, monkeypatch):
    """A download severed on a line boundary is only visible against an expectation."""
    path = write_dataset(tmp_path / "short.txt", n_rows=200, seed=4)
    assert load_policies(path, strict=True, verbose=False)[1].ok

    monkeypatch.setattr(config, "MIN_EXTRACT_ROWS", 1000)
    with pytest.raises(DataValidationError, match="incomplete"):
        load_policies(path, strict=True, verbose=False)
