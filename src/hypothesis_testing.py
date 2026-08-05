"""Task 3 - A/B hypothesis testing on risk and margin drivers.

Every test reports a p-value, a family-wise corrected p-value, an effect size
and the exposure it was computed on. Three things this deliberately does that
the first version did not:

* **Corrects for multiplicity.** Six tests at alpha=0.05 carry a ~26% chance of
  at least one false rejection; decisions are taken on Holm-adjusted p-values.
* **Reports coverage.** A segment that is too small to test is listed as
  untested, and the gender tests state how much of the book they exclude.
* **Checks assumptions.** Margin distributions are heavy tailed, so equality of
  variance is tested and the analysis falls back to Kruskal-Wallis when ANOVA's
  assumptions do not hold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal, levene, ttest_ind
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from src import config
from src.data import DataValidationError, add_derived_columns, load_policies, write_json

DATA_PATH = config.DATA_PATH
DATA_SEP = config.DATA_SEP
SIGNIFICANCE_LEVEL = config.SIGNIFICANCE_LEVEL

GENDER_GROUPS = ("Female", "Male")
MIN_CLAIMS_FOR_TTEST = 30


@dataclass
class TestResult:
    """One hypothesis test, with everything needed to act on it."""

    name: str
    hypothesis: str
    method: str
    statistic: float
    p_value: float
    effect: str
    n: int
    p_adjusted: float | None = None
    rejected: bool | None = None

    @property
    def decision(self) -> str:
        if self.rejected is None:
            return "Pending correction"
        return "Reject H0 (Significant)" if self.rejected else "Fail to Reject H0 (Not Significant)"


@dataclass
class TestSuiteReport:
    """The whole Task 3 family: what ran, what could not, and what it means."""

    results: list[TestResult] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    alpha: float = SIGNIFICANCE_LEVEL
    correction: str = config.MULTIPLICITY_METHOD

    @property
    def complete(self) -> bool:
        return not self.skipped

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "correction": self.correction,
            "complete": self.complete,
            "coverage": self.coverage,
            "results": [asdict(result) | {"decision": result.decision} for result in self.results],
            "skipped": self.skipped,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(result) | {"decision": result.decision} for result in self.results])


def load_data(file_path=DATA_PATH, sep: str = DATA_SEP) -> pd.DataFrame | None:
    """Load and validate the extract, returning ``None`` if it is unusable."""
    required = ("PolicyID", "TotalClaims", "TotalPremium", "Gender", "Province", "PostalCode")
    try:
        df, _ = load_policies(file_path, sep=sep, required_columns=required, verbose=False)
    except DataValidationError as exc:
        print(f"Error: {exc}")
        return None
    return df


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def _cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if n else 0.0


def _eta_squared(groups: list[pd.Series]) -> float:
    values = pd.concat(groups)
    grand_mean = values.mean()
    between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    total = float(((values - grand_mean) ** 2).sum())
    return float(between / total) if total else 0.0


def _frequency_test(df: pd.DataFrame, column: str, groups: list, label: str) -> TestResult:
    """Chi-squared test of independence between a segment and claim occurrence."""
    table = pd.crosstab(df[column], df["Claim_Indicator"]).loc[groups]
    chi2, p_value, _, _ = chi2_contingency(table)
    rates = (table[1] / table.sum(axis=1)).sort_values()
    return TestResult(
        name=f"[{label}] Claim Frequency",
        hypothesis=f"H0: claim frequency is independent of {column}",
        method=f"Chi-squared on {len(groups)} segments",
        statistic=float(chi2),
        p_value=float(p_value),
        effect=(
            f"Cramer's V = {_cramers_v(table, chi2):.3f}; "
            f"lowest {rates.index[0]} {rates.iloc[0]:.2%} vs highest {rates.index[-1]} {rates.iloc[-1]:.2%}"
        ),
        n=int(table.to_numpy().sum()),
    )


def _margin_test(df: pd.DataFrame, column: str, groups: list, label: str) -> TestResult:
    """Compare margins across segments, choosing the test by variance homogeneity."""
    samples = [df.loc[df[column] == group, "Margin"].dropna() for group in groups]
    homogeneous = float(levene(*samples).pvalue) >= SIGNIFICANCE_LEVEL

    if homogeneous:
        statistic, p_value = f_oneway(*samples)
        method = f"One-way ANOVA on {len(groups)} segments"
    else:
        statistic, p_value = kruskal(*samples)
        method = f"Kruskal-Wallis on {len(groups)} segments (Levene rejected equal variances)"

    means = pd.Series({group: sample.mean() for group, sample in zip(groups, samples, strict=True)}).sort_values()
    return TestResult(
        name=f"[{label}] Margin Difference",
        hypothesis=f"H0: mean margin is equal across {column}",
        method=method,
        statistic=float(statistic),
        p_value=float(p_value),
        effect=(
            f"eta^2 = {_eta_squared(samples):.4f}; "
            f"worst {means.index[0]} R{means.iloc[0]:,.2f} vs best {means.index[-1]} R{means.iloc[-1]:,.2f}"
        ),
        n=int(sum(len(sample) for sample in samples)),
    )


def _gender_tests(df: pd.DataFrame, report: TestSuiteReport) -> None:
    counts = df["Gender"].value_counts(dropna=False)
    tested = df[df["Gender"].isin(GENDER_GROUPS)]
    excluded = len(df) - len(tested)
    report.coverage["gender"] = {
        "categories": {str(key): int(value) for key, value in counts.items()},
        "policies_tested": int(len(tested)),
        "policies_excluded": int(excluded),
        "excluded_share": float(excluded / len(df)) if len(df) else 0.0,
    }
    print(
        f"\n[Gender] Coverage: {len(tested):,} of {len(df):,} policies are Female/Male; "
        f"{excluded:,} ({excluded / max(len(df), 1):.1%}) are excluded from both gender tests "
        f"because Gender is recorded as {sorted(set(df['Gender'].dropna()) - set(GENDER_GROUPS))}."
    )

    grouped = tested.groupby("Gender").agg(policies=("PolicyID", "count"), claims=("Claim_Indicator", "sum"))
    thin = [
        f"{group} ({int(grouped.loc[group, 'policies']):,} policies)"
        for group in GENDER_GROUPS
        if group in grouped.index and grouped.loc[group, "policies"] < config.MIN_SEGMENT_POLICIES
    ]
    if thin:
        reason = f"below the {config.MIN_SEGMENT_POLICIES:,}-policy credibility threshold: {', '.join(thin)}"
        report.skipped.append({"test": "[Gender] Claim Frequency", "reason": reason})
        print(f"\n[Gender] Claim Frequency: NOT TESTED - {reason}.")
    elif set(GENDER_GROUPS) <= set(grouped.index):
        counts_ = grouped.loc[list(GENDER_GROUPS), "claims"].tolist()
        nobs = grouped.loc[list(GENDER_GROUPS), "policies"].tolist()
        statistic, p_value = proportions_ztest(counts_, nobs)
        low, high = confint_proportions_2indep(counts_[0], nobs[0], counts_[1], nobs[1], method="wald")
        report.results.append(
            TestResult(
                name="[Gender] Claim Frequency",
                hypothesis="H0: claim frequency is equal for women and men",
                method="Two-proportion z-test (Female vs Male)",
                statistic=float(statistic),
                p_value=float(p_value),
                effect=(
                    f"rate difference {counts_[0] / nobs[0] - counts_[1] / nobs[1]:+.3%} "
                    f"(95% CI {low:+.3%} to {high:+.3%})"
                ),
                n=int(sum(nobs)),
            )
        )
    else:
        report.skipped.append(
            {"test": "[Gender] Claim Frequency", "reason": "Female and Male are not both present in the extract"}
        )
        print("\n[Gender] Claim Frequency: NOT TESTED - Female and Male are not both present.")

    claims = tested[tested["Claim_Indicator"] == 1]
    women = claims.loc[claims["Gender"] == "Female", "TotalClaims"].dropna()
    men = claims.loc[claims["Gender"] == "Male", "TotalClaims"].dropna()
    if len(women) > MIN_CLAIMS_FOR_TTEST and len(men) > MIN_CLAIMS_FOR_TTEST:
        statistic, p_value = ttest_ind(women, men, equal_var=False)
        report.results.append(
            TestResult(
                name="[Gender] Claim Severity",
                hypothesis="H0: mean claim cost is equal for women and men",
                method="Welch two-sample t-test on claiming policies",
                statistic=float(statistic),
                p_value=float(p_value),
                effect=f"Cohen's d = {_cohens_d(women, men):.3f}; means R{women.mean():,.0f} vs R{men.mean():,.0f}",
                n=int(len(women) + len(men)),
            )
        )
    else:
        report.skipped.append(
            {
                "test": "[Gender] Claim Severity",
                "reason": f"Insufficient data for T-Test: {len(women)} female and {len(men)} male claims, "
                f"need more than {MIN_CLAIMS_FOR_TTEST} each",
            }
        )
        print(
            f"\n[Gender] Claim Severity: NOT TESTED - Insufficient data for T-Test "
            f"({len(women)} female, {len(men)} male claims)."
        )


def _segment_tests(df: pd.DataFrame, column: str, label: str, minimum: int, report: TestSuiteReport) -> None:
    """Run the margin and frequency hypotheses for one segmentation."""
    counts = df[column].value_counts()
    testable = counts[counts >= minimum].index.tolist()
    under_powered = counts[counts < minimum]
    report.coverage[label.lower()] = {
        "segments_total": int(counts.size),
        "segments_tested": len(testable),
        "min_policies": minimum,
        "under_powered": {str(key): int(value) for key, value in under_powered.items()},
    }

    if len(testable) < 2:
        for suffix in ("Margin Difference", "Claim Frequency"):
            reason = (
                f"Insufficient number of major {label.lower()}s: "
                f"{len(testable)} segment(s) reach the {minimum}-policy minimum"
            )
            report.skipped.append({"test": f"[{label}] {suffix}", "reason": reason})
            print(f"\n[{label}] {suffix}: NOT TESTED - {reason}.")
        return

    if under_powered.size:
        print(
            f"\n[{label}] Coverage: {len(testable)} of {counts.size} segments carry at least {minimum} policies; "
            f"{under_powered.size} under-powered segment(s) are excluded and remain untested."
        )

    subset = df[df[column].isin(testable)]
    report.results.append(_margin_test(subset, column, testable, label))
    report.results.append(_frequency_test(subset, column, testable, label))


def run_hypothesis_tests(df: pd.DataFrame | None, alpha: float = SIGNIFICANCE_LEVEL) -> TestSuiteReport:
    """Execute the Task 3 family and report corrected, effect-sized decisions."""
    report = TestSuiteReport(alpha=alpha)
    if df is None:
        print("No data supplied; no hypothesis was tested.")
        return report

    print("--- TASK 3: A/B Hypothesis Testing Results ---")
    if not {"Claim_Indicator", "Margin"} <= set(df.columns):
        df = add_derived_columns(df)

    _gender_tests(df, report)
    _segment_tests(df, "Province", "Province", config.MIN_SEGMENT_POLICIES, report)
    _segment_tests(df, "PostalCode", "ZipCode", config.MIN_ZIP_POLICIES, report)

    if report.results:
        rejected, adjusted, _, _ = multipletests(
            [result.p_value for result in report.results], alpha=alpha, method=report.correction
        )
        for result, is_rejected, p_adjusted in zip(report.results, rejected, adjusted, strict=True):
            result.rejected = bool(is_rejected)
            result.p_adjusted = float(p_adjusted)

    print(f"\n--- Results ({len(report.results)} tests, alpha={alpha}, {report.correction} corrected) ---")
    for result in report.results:
        print(f"\n{result.name} ({result.method}): P-value = {result.p_value:.4f} "
              f"[{report.correction}-adjusted {result.p_adjusted:.4f}]")
        print(f"Decision: {result.decision} | n = {result.n:,} | {result.effect}")

    _print_summary(report)
    return report


def _print_summary(report: TestSuiteReport) -> None:
    significant = [result for result in report.results if result.rejected]

    print("\n--- Summary ---")
    print(f"Tested: {len(report.results)} hypotheses. Not tested: {len(report.skipped)}.")
    for entry in report.skipped:
        print(f"  UNTESTED {entry['test']}: {entry['reason']}")

    if not report.complete:
        print(
            "WARNING: coverage is INCOMPLETE. The untested segments above carry real exposure; "
            "no pricing decision may be taken for them on the basis of this run."
        )

    if significant and not report.complete:
        print("\nACTION: PROVISIONAL ONLY - the correction above was applied to a partial family, so these")
        print("rejections cannot carry a rating change until the untested segments are closed:")
        for result in significant:
            print(f"  - {result.name}: {result.effect}")
    elif significant:
        print("\nACTION: the following rejections survive family-wise correction and may justify a rating change,")
        print("subject to exposure, credibility weighting and a loss-ratio view of the same segment:")
        for result in significant:
            print(f"  - {result.name}: {result.effect}")
    else:
        print("\nACTION: no hypothesis survives family-wise correction. Do not change rates on this evidence.")


def main(file_path=DATA_PATH, sep: str = DATA_SEP, output_dir: Path = config.METRICS_DIR) -> int:
    df = load_data(file_path, sep)
    if df is None:
        return 1

    report = run_hypothesis_tests(df)
    json_path = write_json(report.to_dict(), Path(output_dir) / "hypothesis_tests.json")
    csv_path = Path(output_dir) / "hypothesis_tests.csv"
    report.to_frame().to_csv(csv_path, index=False)
    print(f"\nResults written to {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
