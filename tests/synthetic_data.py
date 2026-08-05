"""Synthetic stand-in for ``data/MachineLearningRating_v3.txt``.

The real dataset (~500 MB) is not distributable, so the QA suite generates a
small pipe-delimited file with the same column names, dtypes and pathological
characteristics as production data: a very low claim rate, a heavily missing
``CustomValueEstimate`` column, skewed claim amounts and a ``Not specified``
gender category.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_SEP = "|"

PROVINCES = [
    "Gauteng",
    "Western Cape",
    "KwaZulu-Natal",
    "Eastern Cape",
    "North West",
    "Mpumalanga",
    "Limpopo",
    "Free State",
    "Northern Cape",
]
GENDERS = ["Male", "Female", "Not specified"]
GENDER_WEIGHTS = [0.28, 0.15, 0.57]
VEHICLE_TYPES = ["Passenger Vehicle", "Light Commercial", "Medium Commercial", "Heavy Commercial"]
BODY_TYPES = ["Hatch back", "S/D", "D/C", "Panel van", "S/W"]
MAKES = ["TOYOTA", "VOLKSWAGEN", "NISSAN", "MERCEDES-BENZ", "FORD", "HYUNDAI"]
YES_NO = ["Yes", "No"]

# Column order mirrors the header of the production extract for the fields the
# pipeline actually consumes.
COLUMNS = [
    "UnderwrittenCoverID",
    "PolicyID",
    "TransactionMonth",
    "IsVATRegistered",
    "Citizenship",
    "LegalType",
    "Title",
    "Language",
    "Bank",
    "AccountType",
    "MaritalStatus",
    "Gender",
    "Country",
    "Province",
    "PostalCode",
    "MainCrestaZone",
    "SubCrestaZone",
    "ItemType",
    "mmcode",
    "VehicleType",
    "RegistrationYear",
    "make",
    "Model",
    "Cylinders",
    "cubiccapacity",
    "kilowatts",
    "bodytype",
    "NumberOfDoors",
    "VehicleIntroDate",
    "CustomValueEstimate",
    "AlarmImmobiliser",
    "TrackingDevice",
    "CapitalOutstanding",
    "NewVehicle",
    "SumInsured",
    "TermFrequency",
    "CalculatedPremiumPerTerm",
    "ExcessSelected",
    "CoverCategory",
    "CoverType",
    "CoverGroup",
    "Section",
    "Product",
    "StatutoryClass",
    "StatutoryRiskType",
    "TotalPremium",
    "TotalClaims",
]

CLAIM_RATE = 0.05
CUSTOM_VALUE_MISSING_RATE = 0.78


def make_dataframe(n_rows: int = 4000, seed: int = 42, claim_rate: float = CLAIM_RATE) -> pd.DataFrame:
    """Build a synthetic policy-transaction frame with production-like quirks."""
    rng = np.random.default_rng(seed)

    n_policies = max(1, n_rows // 4)
    policy_ids = rng.integers(1, n_policies + 1, size=n_rows)
    months = pd.to_datetime("2014-02-01") + pd.to_timedelta(
        rng.integers(0, 18, size=n_rows) * 30, unit="D"
    )

    claim_indicator = rng.random(n_rows) < claim_rate
    # Claim amounts are right skewed; a handful of policies dominate total loss.
    total_claims = np.where(claim_indicator, rng.lognormal(mean=8.5, sigma=1.6, size=n_rows), 0.0)

    df = pd.DataFrame(
        {
            "UnderwrittenCoverID": np.arange(1, n_rows + 1),
            "PolicyID": policy_ids,
            "TransactionMonth": months.strftime("%Y-%m-%d %H:%M:%S"),
            "IsVATRegistered": rng.choice([True, False], size=n_rows),
            "Citizenship": "",
            "LegalType": rng.choice(["Close Corporation", "Individual", "Private company"], size=n_rows),
            "Title": rng.choice(["Mr", "Mrs", "Ms"], size=n_rows),
            "Language": "English",
            "Bank": rng.choice(["First National Bank", "Standard Bank", "ABSA"], size=n_rows),
            "AccountType": rng.choice(["Current account", "Savings account"], size=n_rows),
            "MaritalStatus": rng.choice(["Married", "Single", "Not specified"], size=n_rows),
            "Gender": rng.choice(GENDERS, size=n_rows, p=GENDER_WEIGHTS),
            "Country": "South Africa",
            "Province": rng.choice(PROVINCES, size=n_rows),
            "PostalCode": rng.choice([1459, 2000, 7784, 122, 299, 4001, 8001, 6001], size=n_rows),
            "MainCrestaZone": rng.choice(["Rand", "Cape Town", "Durban"], size=n_rows),
            "SubCrestaZone": rng.choice(["Johannesburg", "Pretoria", "Bellville"], size=n_rows),
            "ItemType": "Mobility - Motor",
            "mmcode": rng.integers(10000, 99999, size=n_rows),
            "VehicleType": rng.choice(VEHICLE_TYPES, size=n_rows, p=[0.7, 0.2, 0.07, 0.03]),
            "RegistrationYear": rng.integers(1995, 2016, size=n_rows),
            "make": rng.choice(MAKES, size=n_rows),
            "Model": rng.choice(["COROLLA", "POLO", "NP200", "RANGER"], size=n_rows),
            "Cylinders": rng.choice([3, 4, 6, 8], size=n_rows, p=[0.1, 0.75, 0.12, 0.03]),
            "cubiccapacity": rng.choice([1200, 1400, 1600, 2000, 2500, 3000], size=n_rows),
            "kilowatts": rng.integers(40, 190, size=n_rows),
            "bodytype": rng.choice(BODY_TYPES, size=n_rows),
            "NumberOfDoors": rng.choice([2, 4, 5], size=n_rows),
            "VehicleIntroDate": "6/2002",
            "CustomValueEstimate": rng.normal(220_000, 90_000, size=n_rows).round(2),
            "AlarmImmobiliser": rng.choice(YES_NO, size=n_rows),
            "TrackingDevice": rng.choice(YES_NO, size=n_rows),
            "CapitalOutstanding": rng.normal(100_000, 40_000, size=n_rows).round(2),
            "NewVehicle": rng.choice(["More than 6 months", "New"], size=n_rows),
            "SumInsured": rng.normal(250_000, 120_000, size=n_rows).round(2).clip(1000),
            "TermFrequency": "Monthly",
            "CalculatedPremiumPerTerm": rng.gamma(2.0, 60.0, size=n_rows).round(2),
            "ExcessSelected": rng.choice(["Mobility - Windscreen", "No excess"], size=n_rows),
            "CoverCategory": rng.choice(["Windscreen", "Own damage", "Third Party"], size=n_rows),
            "CoverType": rng.choice(["Windscreen", "Own Damage", "Passenger Liability"], size=n_rows),
            "CoverGroup": "Comprehensive - Taxi",
            "Section": "Motor Comprehensive",
            "Product": "Mobility Metered Taxis: Monthly",
            "StatutoryClass": "Commercial",
            "StatutoryRiskType": "IFRS Constant",
            "TotalPremium": rng.gamma(2.0, 35.0, size=n_rows).round(4),
            "TotalClaims": np.round(total_claims, 4),
        }
    )

    # CustomValueEstimate is ~78% NULL in production - the single biggest driver
    # of row loss in the modelling pipeline.
    missing = rng.random(n_rows) < CUSTOM_VALUE_MISSING_RATE
    df.loc[missing, "CustomValueEstimate"] = np.nan

    return df[COLUMNS]


def write_dataset(path: str | Path, n_rows: int = 4000, seed: int = 42, **kwargs) -> Path:
    """Write the synthetic dataset to ``path`` as a pipe-delimited text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_dataframe(n_rows=n_rows, seed=seed, **kwargs).to_csv(path, sep=DATA_SEP, index=False)
    return path


if __name__ == "__main__":
    written = write_dataset("data/MachineLearningRating_v3.txt", n_rows=20_000)
    print(f"Wrote synthetic dataset to {written.resolve()}")
