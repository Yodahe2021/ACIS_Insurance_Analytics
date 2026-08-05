"""Loading, validation and feature derivation for the policy extract.

The data-quality contract that the analysis silently assumed is enforced here,
at runtime, on every load: a malformed extract fails loudly at the door instead
of quietly producing a plausible-looking price.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

REQUIRED_COLUMNS: tuple[str, ...] = (
    "PolicyID",
    "TransactionMonth",
    "Gender",
    "Province",
    "PostalCode",
    "TotalPremium",
    "TotalClaims",
)

MONETARY_COLUMNS: tuple[str, ...] = ("TotalPremium", "TotalClaims", "SumInsured")

#: Above this share of nulls a rating factor is reported as unreliable.
HIGH_MISSINGNESS = 0.5


class DataValidationError(RuntimeError):
    """Raised when the extract violates the data contract."""


@dataclass
class ValidationReport:
    """Outcome of validating an extract against the data contract."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"Data validation: {self.row_count:,} rows, {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        lines += [f"  ERROR   {message}" for message in self.errors]
        lines += [f"  WARNING {message}" for message in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"row_count": self.row_count, "errors": self.errors, "warnings": self.warnings}


def validate_policies(df: pd.DataFrame, required_columns: tuple[str, ...] = REQUIRED_COLUMNS) -> ValidationReport:
    """Check an extract against the contract the pipeline depends on."""
    report = ValidationReport(row_count=len(df))

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        report.errors.append(f"missing required columns: {missing}")
        return report

    if df.empty:
        report.errors.append("extract contains no rows")
        return report

    for column in ("TotalPremium", "TotalClaims"):
        if not pd.api.types.is_numeric_dtype(df[column]):
            report.errors.append(f"{column} is {df[column].dtype}, expected a numeric dtype")
        elif df[column].isna().all():
            report.errors.append(f"{column} is entirely null")

    if pd.api.types.is_numeric_dtype(df["TotalPremium"]) and df["TotalPremium"].fillna(0).sum() <= 0:
        report.errors.append("total premium across the book is not positive")

    if pd.to_datetime(df["TransactionMonth"], errors="coerce").isna().all():
        report.errors.append("TransactionMonth cannot be parsed as a date")

    for column in MONETARY_COLUMNS:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            negatives = int((df[column] < 0).sum())
            if negatives:
                report.warnings.append(f"{column} is negative on {negatives:,} rows ({negatives / len(df):.2%})")

    numerical, categorical = config.feature_lists()
    for column in [*numerical, *categorical]:
        if column in df.columns:
            null_rate = float(df[column].isna().mean())
            if null_rate > HIGH_MISSINGNESS:
                report.warnings.append(f"rating factor {column} is {null_rate:.1%} null; it is imputed, not dropped")

    if "RegistrationYear" in df.columns:
        years = pd.to_numeric(df["RegistrationYear"], errors="coerce")
        implausible = int(((years < 1950) | (years > pd.Timestamp(config.VALUATION_DATE).year + 1)).sum())
        if implausible:
            report.warnings.append(f"RegistrationYear is implausible on {implausible:,} rows")

    return report


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns every downstream task needs, without mutating the input."""
    out = df.copy()
    out["Claim_Indicator"] = np.where(out["TotalClaims"].fillna(0) > 0, 1, 0)
    out["Margin"] = out["TotalPremium"] - out["TotalClaims"]

    if "RegistrationYear" in out.columns:
        valuation_year = pd.Timestamp(config.VALUATION_DATE).year
        car_age = valuation_year - pd.to_numeric(out["RegistrationYear"], errors="coerce")
        out["Car_Age"] = car_age.clip(lower=0)

    return out


def load_policies(
    file_path: str | Path = config.DATA_PATH,
    sep: str = config.DATA_SEP,
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS,
    strict: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Read the extract, validate it and derive the shared analysis columns.

    Raises ``DataValidationError`` on a contract breach when ``strict``; the
    caller gets the report either way so it can log or persist it.
    """
    path = Path(file_path)
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False)
    except FileNotFoundError as exc:
        raise DataValidationError(
            f"extract not found at {path}. The dataset is DVC-tracked: run `dvc pull`, "
            f"or point ACIS_DATA_PATH at a local copy."
        ) from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"{path} could not be parsed with separator {sep!r}: {exc}") from exc

    report = validate_policies(df, required_columns=required_columns)
    if verbose:
        print(report.render())
    if strict and not report.ok:
        raise DataValidationError("; ".join(report.errors))

    return add_derived_columns(df), report


def write_json(payload: dict, path: str | Path) -> Path:
    """Persist a run artefact, creating the directory tree if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
