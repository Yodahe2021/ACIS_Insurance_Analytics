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
from scipy.optimize import brentq

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

#: Ground-truth risk structure. The fixture is not pure noise: a model that
#: works must be able to recover these effects, and the quality gates in the
#: test suite (AUC, severity R^2) only mean something because they exist.
PROVINCE_RISK = {
    "Gauteng": 0.9,
    "Western Cape": -0.4,
    "KwaZulu-Natal": 0.6,
    "Eastern Cape": 0.1,
    "North West": -0.7,
    "Mpumalanga": 0.3,
    "Limpopo": -0.2,
    "Free State": -0.5,
    "Northern Cape": -0.8,
}
VEHICLE_FREQUENCY_RISK = {
    "Passenger Vehicle": -0.2,
    "Light Commercial": 0.4,
    "Medium Commercial": 0.9,
    "Heavy Commercial": 1.4,
}
VEHICLE_SEVERITY_EFFECT = {
    "Passenger Vehicle": 0.0,
    "Light Commercial": 0.45,
    "Medium Commercial": 0.9,
    "Heavy Commercial": 1.5,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_dataframe(n_rows: int = 4000, seed: int = 42, claim_rate: float = CLAIM_RATE) -> pd.DataFrame:
    """Build a synthetic policy-transaction frame with production-like quirks."""
    rng = np.random.default_rng(seed)

    n_policies = max(1, n_rows // 4)
    policy_ids = rng.integers(1, n_policies + 1, size=n_rows)
    months = pd.to_datetime("2014-02-01") + pd.to_timedelta(
        rng.integers(0, 18, size=n_rows) * 30, unit="D"
    )

    provinces = rng.choice(PROVINCES, size=n_rows)
    vehicle_types = rng.choice(VEHICLE_TYPES, size=n_rows, p=[0.7, 0.2, 0.07, 0.03])
    registration_years = rng.integers(1995, 2016, size=n_rows)
    car_age = 2015 - registration_years
    tracking = rng.choice(YES_NO, size=n_rows)
    cylinders = rng.choice([3, 4, 6, 8], size=n_rows, p=[0.1, 0.75, 0.12, 0.03])
    sum_insured = rng.lognormal(mean=12.3, sigma=0.45, size=n_rows).round(2).clip(1000)

    # Claim frequency: province, vehicle class, vehicle age and anti-theft
    # devices move the odds; the intercept is solved for the target claim rate.
    risk_score = (
        np.vectorize(PROVINCE_RISK.get)(provinces)
        + np.vectorize(VEHICLE_FREQUENCY_RISK.get)(vehicle_types)
        + 0.06 * car_age
        - 0.60 * (tracking == "Yes")
    )
    intercept = brentq(lambda c: _sigmoid(c + risk_score).mean() - claim_rate, -25.0, 25.0)
    claim_indicator = rng.random(n_rows) < _sigmoid(intercept + risk_score)

    # Claim severity: driven by the value at risk and the vehicle class, with a
    # heavy lognormal tail so a handful of claims dominate total loss.
    severity_mu = (
        6.4
        + 0.90 * (np.log(sum_insured) - np.log(sum_insured).mean())
        + np.vectorize(VEHICLE_SEVERITY_EFFECT.get)(vehicle_types)
        + 0.10 * cylinders
    )
    total_claims = np.where(claim_indicator, rng.lognormal(mean=severity_mu, sigma=0.70, size=n_rows), 0.0)

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
            "Province": provinces,
            "PostalCode": rng.choice([1459, 2000, 7784, 122, 299, 4001, 8001, 6001], size=n_rows),
            "MainCrestaZone": rng.choice(["Rand", "Cape Town", "Durban"], size=n_rows),
            "SubCrestaZone": rng.choice(["Johannesburg", "Pretoria", "Bellville"], size=n_rows),
            "ItemType": "Mobility - Motor",
            "mmcode": rng.integers(10000, 99999, size=n_rows),
            "VehicleType": vehicle_types,
            "RegistrationYear": registration_years,
            "make": rng.choice(MAKES, size=n_rows),
            "Model": rng.choice(["COROLLA", "POLO", "NP200", "RANGER"], size=n_rows),
            "Cylinders": cylinders,
            "cubiccapacity": rng.choice([1200, 1400, 1600, 2000, 2500, 3000], size=n_rows),
            "kilowatts": rng.integers(40, 190, size=n_rows),
            "bodytype": rng.choice(BODY_TYPES, size=n_rows),
            "NumberOfDoors": rng.choice([2, 4, 5], size=n_rows),
            "VehicleIntroDate": "6/2002",
            # Where it is recorded at all, the custom value tracks the sum insured.
            "CustomValueEstimate": (sum_insured * rng.normal(1.0, 0.15, size=n_rows)).round(2),
            "AlarmImmobiliser": rng.choice(YES_NO, size=n_rows),
            "TrackingDevice": tracking,
            "CapitalOutstanding": rng.normal(100_000, 40_000, size=n_rows).round(2),
            "NewVehicle": rng.choice(["More than 6 months", "New"], size=n_rows),
            "SumInsured": sum_insured,
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
