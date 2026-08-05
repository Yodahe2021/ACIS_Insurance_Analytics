"""Behavioural tests for ``src/hypothesis_testing.py`` (Task 3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def loaded(hypothesis_module, synthetic_data_file: Path) -> pd.DataFrame:
    df = hypothesis_module.load_data(str(synthetic_data_file))
    assert df is not None, "load_data returned None for a well-formed extract"
    return df


def _run_and_capture(hypothesis_module, df, capsys) -> str:
    hypothesis_module.run_hypothesis_tests(df)
    return capsys.readouterr().out


def test_load_data_derives_metrics(loaded: pd.DataFrame):
    assert {"Claim_Indicator", "Margin"} <= set(loaded.columns)
    assert loaded["Claim_Indicator"].isin([0, 1]).all()
    pd.testing.assert_series_equal(
        loaded["Margin"],
        loaded["TotalPremium"] - loaded["TotalClaims"],
        check_names=False,
    )


def test_load_data_returns_none_for_a_missing_file(hypothesis_module, tmp_path: Path):
    assert hypothesis_module.load_data(str(tmp_path / "nope.txt")) is None


def test_load_data_rejects_an_extract_with_missing_columns(hypothesis_module, tmp_path: Path, synthetic_df):
    truncated = tmp_path / "truncated.txt"
    synthetic_df.drop(columns=["PostalCode"]).to_csv(truncated, sep="|", index=False)
    assert hypothesis_module.load_data(str(truncated)) is None


def test_wrong_separator_is_detected(hypothesis_module, tmp_path: Path, synthetic_df):
    """A comma-separated file must not be silently accepted as a single-column frame."""
    comma_file = tmp_path / "comma.txt"
    synthetic_df.to_csv(comma_file, sep=",", index=False)
    assert hypothesis_module.load_data(str(comma_file), sep="|") is None


def test_run_hypothesis_tests_tolerates_none(hypothesis_module):
    hypothesis_module.run_hypothesis_tests(None)


def test_all_four_hypotheses_are_reported(hypothesis_module, loaded, capsys):
    output = _run_and_capture(hypothesis_module, loaded, capsys)
    for section in ("[Gender] Claim Frequency", "[Gender] Claim Severity", "[Province] Margin Difference",
                    "[Province] Claim Frequency"):
        assert section in output, f"missing hypothesis result: {section}"


def test_every_decision_line_is_backed_by_a_p_value(hypothesis_module, loaded, capsys):
    output = _run_and_capture(hypothesis_module, loaded, capsys)
    p_values = output.count("P-value =")
    decisions = output.count("Decision:")
    assert p_values == decisions, f"{decisions} decisions reported for {p_values} p-values"


def test_gender_test_runs_on_a_small_but_sufficient_sample(hypothesis_module, capsys, tmp_path, synthetic_df):
    """The severity t-test guard needs >30 claims per gender; verify it degrades gracefully."""
    tiny = synthetic_df.head(200).copy()
    tiny["Claim_Indicator"] = (tiny["TotalClaims"] > 0).astype(int)
    tiny["Margin"] = tiny["TotalPremium"] - tiny["TotalClaims"]
    hypothesis_module.run_hypothesis_tests(tiny)
    output = capsys.readouterr().out
    assert "Insufficient data for T-Test" in output or "[Gender] Claim Severity (T-Test)" in output


def test_small_portfolio_does_not_silently_drop_hypotheses(hypothesis_module, capsys, tmp_path, synthetic_df):
    """Below the hard-coded 1000-policy / 500-policy thresholds the tests are skipped.

    The run still exits 0 and still prints the "use the rejected hypotheses" call to
    action, so a reader cannot tell that no province or postal-code test was executed.
    """
    small = synthetic_df.head(900).copy()
    small["Claim_Indicator"] = (small["TotalClaims"] > 0).astype(int)
    small["Margin"] = small["TotalPremium"] - small["TotalClaims"]
    hypothesis_module.run_hypothesis_tests(small)
    output = capsys.readouterr().out
    assert "Insufficient number of major provinces" in output
    assert "ACTION: Use the rejected hypotheses" in output, (
        "the summary line is printed unconditionally even when no test ran"
    )


@pytest.mark.xfail(
    strict=True,
    reason="QA-009: the postal-code hypothesis only tests margin; claim-frequency risk is never tested",
)
def test_zip_code_risk_is_tested(hypothesis_module, loaded, capsys):
    output = _run_and_capture(hypothesis_module, loaded, capsys)
    assert "[ZipCode] Claim Frequency" in output


@pytest.mark.xfail(
    strict=True,
    reason="QA-007: five tests are reported against an uncorrected alpha=0.05",
)
def test_multiplicity_is_controlled(hypothesis_module):
    source = Path(hypothesis_module.__file__).read_text(encoding="utf-8")
    assert any(term in source.lower() for term in ("bonferroni", "holm", "multipletests", "fdr")), (
        "no family-wise error-rate correction is applied across the hypothesis family"
    )


@pytest.mark.xfail(
    strict=True,
    reason="QA-008: 'Not specified' is the majority Gender value and is dropped without being reported",
)
def test_gender_test_reports_its_coverage(hypothesis_module, loaded, capsys):
    output = _run_and_capture(hypothesis_module, loaded, capsys)
    excluded_share = float((~loaded["Gender"].isin(["Female", "Male"])).mean())
    assert excluded_share < 0.5 or "excluded" in output.lower(), (
        f"{excluded_share:.0%} of policies are excluded from the gender hypothesis without disclosure"
    )
