"""Single source of truth for paths, features and pricing parameters.

Everything that used to be a magic number scattered across the scripts lives
here so that a change to, say, the profit margin is one edit and is picked up
by the pipeline, the tests and the persisted run metadata alike.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Data -------------------------------------------------------------------

#: Override with ``ACIS_DATA_PATH`` to score a different extract without editing code.
DATA_PATH = Path(os.environ.get("ACIS_DATA_PATH", REPO_ROOT / "data" / "MachineLearningRating_v3.txt"))
DATA_SEP = "|"

#: Reference date the book is valued at; drives the ``Car_Age`` feature.
VALUATION_DATE = "2015-08-01"

# --- Outputs ----------------------------------------------------------------

#: Generated artefacts. Overridable so a run can be sandboxed (tests, CI, ad-hoc
#: experiments) instead of overwriting the committed deliverables.
OUTPUT_ROOT = Path(os.environ.get("ACIS_OUTPUT_DIR", REPO_ROOT / "artifacts"))
MODEL_DIR = OUTPUT_ROOT / "models"
METRICS_DIR = OUTPUT_ROOT / "metrics"
FIGURE_DIR = OUTPUT_ROOT / "figures"

# --- Modelling --------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

NUMERICAL_FEATURES: tuple[str, ...] = (
    "Cylinders",
    "cubiccapacity",
    "kilowatts",
    "CustomValueEstimate",
    "SumInsured",
    "Car_Age",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Province",
    "make",
    "VehicleType",
    "bodytype",
    "AlarmImmobiliser",
    "TrackingDevice",
)

#: Gender is analysed in Task 3 but is deliberately kept out of the rating
#: factors: pricing on a protected attribute needs an explicit legal sign-off,
#: and 57% of the book records it as "Not specified" anyway. See
#: ``docs/MODEL_CARD.md``. Flip to True only with that sign-off in hand.
USE_GENDER_AS_RATING_FACTOR = os.environ.get("ACIS_USE_GENDER", "0") == "1"

# --- Pricing ----------------------------------------------------------------

EXPENSE_LOADING = 400.00
PROFIT_MARGIN = 0.08

#: No policy may be quoted below the cost of writing it. This is the premium
#: that carries the expense loading alone, i.e. a pure premium of zero.
PREMIUM_FLOOR = EXPENSE_LOADING / (1 - PROFIT_MARGIN)

# --- Statistics -------------------------------------------------------------

SIGNIFICANCE_LEVEL = 0.05

#: Family-wise error control across the Task 3 hypothesis family.
MULTIPLICITY_METHOD = "holm"

#: A segment below this many policies is reported as under-powered rather than
#: silently dropped.
MIN_SEGMENT_POLICIES = 1000
MIN_ZIP_POLICIES = 500


def feature_lists() -> tuple[list[str], list[str]]:
    """Return copies of the configured feature lists.

    Copies, so that a caller appending an engineered feature can never mutate
    global state (the defect that made two consecutive pipeline runs in one
    process disagree about their feature set).
    """
    categorical = list(CATEGORICAL_FEATURES)
    if USE_GENDER_AS_RATING_FACTOR:
        categorical.append("Gender")
    return list(NUMERICAL_FEATURES), categorical
