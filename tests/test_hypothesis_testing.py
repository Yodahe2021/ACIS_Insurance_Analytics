"""Behavioural tests for ``src/hypothesis_testing.py`` (Task 3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import config


@pytest.fixture(scope="module")
def loaded(hypothesis_module, synthetic_data_file: Path) -> pd.DataFrame:
    df = hypothesis_module.load_data(str(synthetic_data_file))
    assert df is not None, "load_data returned None for a well-formed extract"
    return df


@pytest.fixture(scope="module")
def suite(hypothesis_module, loaded):
    return hypothesis_module.run_hypothesis_tests(loaded)


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
    report = hypothesis_module.run_hypothesis_tests(None)
    assert report.results == []


def test_all_hypotheses_in_the_family_are_reported(hypothesis_module, loaded, capsys):
    output = _run_and_capture(hypothesis_module, loaded, capsys)
    for section in (
        "[Gender] Claim Frequency",
        "[Gender] Claim Severity",
        "[Province] Margin Difference",
        "[Province] Claim Frequency",
        "[ZipCode] Margin Difference",
        "[ZipCode] Claim Frequency",
    ):
        assert section in output, f"missing hypothesis result: {section}"


def test_zip_code_risk_is_tested(suite):
    """QA-009: the postal-code hypothesis used to test margin only, never risk."""
    assert "[ZipCode] Claim Frequency" in {result.name for result in suite.results}


def test_every_decision_line_is_backed_by_a_p_value(hypothesis_module, loaded, capsys):
    output = _run_and_capture(hypothesis_module, loaded, capsys)
    assert output.count("P-value =") == output.count("Decision:")


def test_multiplicity_is_controlled(suite):
    """QA-007: six tests at alpha=0.05 carry a ~26% family-wise false-positive rate."""
    assert suite.correction in {"holm", "bonferroni", "fdr_bh"}
    for result in suite.results:
        assert result.p_adjusted is not None
        assert result.p_adjusted >= result.p_value - 1e-12, "adjusted p-value is smaller than the raw one"
        assert result.rejected == (result.p_adjusted < suite.alpha)


def test_every_result_carries_an_effect_size_and_exposure(suite):
    for result in suite.results:
        assert result.effect, f"{result.name} reports no effect size"
        assert result.n > 0, f"{result.name} reports no exposure"


def test_gender_tests_disclose_their_coverage(hypothesis_module, loaded, capsys, suite):
    """QA-008: 'Not specified' is the majority Gender value and was dropped in silence."""
    output = _run_and_capture(hypothesis_module, loaded, capsys)
    excluded_share = float((~loaded["Gender"].isin(["Female", "Male"])).mean())
    assert "excluded" in output.lower()
    assert suite.coverage["gender"]["excluded_share"] == pytest.approx(excluded_share)


def test_under_powered_segments_are_reported_not_hidden(hypothesis_module, capsys, synthetic_df):
    """QA-020: skipped hypotheses used to leave no trace in the output."""
    small = synthetic_df.head(900).copy()
    report = hypothesis_module.run_hypothesis_tests(small)
    output = capsys.readouterr().out

    assert not report.complete
    assert {entry["test"] for entry in report.skipped} >= {"[Province] Margin Difference", "[Province] Claim Frequency"}
    assert "NOT TESTED" in output
    assert "coverage is INCOMPLETE" in output
    assert "Use the rejected hypotheses" not in output, "a blanket call to action was printed for an untested book"


def test_a_thin_book_cannot_earn_a_rating_recommendation(hypothesis_module, capsys, synthetic_df):
    """Every hypothesis carries an exposure gate, and a partial family is provisional.

    A 40-policy book used to clear the gender frequency test - the one factor the
    model is forbidden to price on - and print the same call to action as a
    credible run.
    """
    tiny = synthetic_df.head(40).copy()
    report = hypothesis_module.run_hypothesis_tests(tiny)
    output = capsys.readouterr().out

    assert not report.results, f"a 40-policy book produced testable hypotheses: {report.results}"
    assert len(report.skipped) == 6
    assert "coverage is INCOMPLETE" in output
    assert "may justify a rating change" not in output


def test_a_partial_family_marks_its_rejections_provisional(hypothesis_module, capsys, synthetic_df):
    partial = pd.concat(
        [
            synthetic_df[synthetic_df["Province"] == "Gauteng"].head(4000),
            synthetic_df[synthetic_df["Province"] == "North West"].head(4000),
        ]
    )
    report = hypothesis_module.run_hypothesis_tests(partial)
    output = capsys.readouterr().out

    assert not report.complete, "this book was expected to leave some segment untested"
    if any(result.rejected for result in report.results):
        assert "PROVISIONAL ONLY" in output
        assert "may justify a rating change" not in output


def test_small_gender_sample_degrades_gracefully(hypothesis_module, capsys, synthetic_df):
    tiny = synthetic_df.head(200).copy()
    report = hypothesis_module.run_hypothesis_tests(tiny)
    output = capsys.readouterr().out
    assert "Insufficient data for T-Test" in output
    assert any(entry["test"] == "[Gender] Claim Severity" for entry in report.skipped)


def test_no_rate_change_is_recommended_without_a_surviving_rejection(hypothesis_module, capsys, synthetic_df):
    noise = synthetic_df.head(4000).copy()
    noise["TotalClaims"] = 0.0
    noise.loc[noise.index[:100], "TotalClaims"] = 5000.0
    hypothesis_module.run_hypothesis_tests(noise)
    output = capsys.readouterr().out
    if "no hypothesis survives" in output:
        assert "Do not change rates on this evidence" in output


def test_results_serialise_for_the_run_record(suite):
    """QA-023: statistical findings must outlive the terminal they were printed in."""
    payload = suite.to_dict()
    assert payload["correction"] == config.MULTIPLICITY_METHOD
    assert len(payload["results"]) == len(suite.results)
    assert {"decision", "p_adjusted", "effect", "n"} <= set(payload["results"][0])
    assert not suite.to_frame().empty
