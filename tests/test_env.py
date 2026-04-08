"""
Test suite for the Financial Statement Anomaly Detection environment.
Covers: data generation, anomaly injection, grading, environment API,
and determinism.
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    Action, ActionType, AnomalyFlag, AnomalyType, GroundTruthAnomaly,
    Observation, QuarterData, State,
)
from src.generator import generate_company_data
from src.anomaly_injector import inject_anomalies
from src.grader import grade
from src.environment import FinancialAnomalyEnv
from src.utils import (
    balance_sheet_balanced, cash_flow_reconciles,
    current_assets_sum_check, income_statement_check,
    benfords_chi_squared, leading_digit,
)


def test_generator_produces_valid_data():
    """Generated data should have internally consistent financial statements."""
    metadata, quarters = generate_company_data(seed=42, num_quarters=6)
    assert metadata.name != ""
    assert len(quarters) == 6

    for q in quarters:
        # Balance sheet must balance
        assert q.total_assets == q.total_liabilities + q.equity, \
            f"{q.quarter_label}: assets={q.total_assets} != liab+eq={q.total_liabilities + q.equity}"
        # Cash flow must reconcile
        assert q.net_cash_change == q.cf_operating + q.cf_investing + q.cf_financing, \
            f"{q.quarter_label}: net_cash_change mismatch"
        # Revenue should be positive
        assert q.revenue > 0, f"{q.quarter_label}: revenue should be positive"

    print("PASS: test_generator_produces_valid_data")


def test_generator_deterministic():
    """Same seed should produce identical data."""
    _, q1 = generate_company_data(seed=99, num_quarters=4)
    _, q2 = generate_company_data(seed=99, num_quarters=4)
    for a, b in zip(q1, q2):
        assert a.revenue == b.revenue
        assert a.total_assets == b.total_assets
        assert a.net_income == b.net_income
    print("PASS: test_generator_deterministic")


def test_anomaly_injection_easy():
    """Easy injection should produce 3+ anomalies."""
    _, quarters = generate_company_data(seed=42, num_quarters=4)
    _, anomalies = inject_anomalies(quarters, difficulty="easy", seed=1042)
    assert len(anomalies) >= 2, f"Expected >=2 anomalies, got {len(anomalies)}"
    for a in anomalies:
        assert a.anomaly_id != ""
        assert a.severity >= 1
        assert a.severity <= 5
    print("PASS: test_anomaly_injection_easy")


def test_anomaly_injection_medium():
    """Medium injection should produce more anomalies than easy."""
    _, quarters = generate_company_data(seed=137, num_quarters=6)
    _, anomalies = inject_anomalies(quarters, difficulty="medium", seed=1137)
    assert len(anomalies) >= 3, f"Expected >=3 anomalies, got {len(anomalies)}"
    print("PASS: test_anomaly_injection_medium")


def test_anomaly_injection_hard():
    """Hard injection should produce 5+ anomalies."""
    _, quarters = generate_company_data(seed=256, num_quarters=8)
    _, anomalies = inject_anomalies(quarters, difficulty="hard", seed=1256)
    assert len(anomalies) >= 4, f"Expected >=4 anomalies, got {len(anomalies)}"
    # Should have some high-severity anomalies
    max_sev = max(a.severity for a in anomalies)
    assert max_sev >= 4, f"Hard task should have severity >= 4, got max {max_sev}"
    print("PASS: test_anomaly_injection_hard")


def test_grader_perfect_score():
    """Perfect flags should score close to 1.0."""
    truths = [
        GroundTruthAnomaly(
            anomaly_id="a1", line_item="revenue", quarter="Q1 2023",
            severity=3, anomaly_type=AnomalyType.ARITHMETIC_ERROR,
        ),
        GroundTruthAnomaly(
            anomaly_id="a2", line_item="inventory", quarter="Q2 2023",
            severity=4, anomaly_type=AnomalyType.INVENTORY_TURNOVER,
        ),
    ]
    flags = [
        AnomalyFlag(
            line_item="revenue", quarter="Q1 2023",
            severity=3, anomaly_type=AnomalyType.ARITHMETIC_ERROR,
            explanation="Totals don't add up",
        ),
        AnomalyFlag(
            line_item="inventory", quarter="Q2 2023",
            severity=4, anomaly_type=AnomalyType.INVENTORY_TURNOVER,
            explanation="Inventory spike",
        ),
    ]
    result = grade(flags, truths)
    assert result["score"] >= 0.9, f"Perfect flags should score >= 0.9, got {result['score']}"
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["false_positives"] == 0
    print("PASS: test_grader_perfect_score")


def test_grader_partial_credit():
    """Right location, wrong type should get partial credit."""
    truths = [
        GroundTruthAnomaly(
            anomaly_id="a1", line_item="revenue", quarter="Q1 2023",
            severity=3, anomaly_type=AnomalyType.CHANNEL_STUFFING,
        ),
    ]
    flags = [
        AnomalyFlag(
            line_item="revenue", quarter="Q1 2023",
            severity=3, anomaly_type=AnomalyType.IMPOSSIBLE_CHANGE,  # wrong type
            explanation="Revenue jumped too much",
        ),
    ]
    result = grade(flags, truths)
    assert 0.3 < result["score"] < 0.9, f"Partial credit score unexpected: {result['score']}"
    assert result["precision"] == 0.5  # partial match
    print("PASS: test_grader_partial_credit")


def test_grader_false_positives():
    """All wrong flags should score low."""
    truths = [
        GroundTruthAnomaly(
            anomaly_id="a1", line_item="revenue", quarter="Q1 2023",
            severity=3, anomaly_type=AnomalyType.ARITHMETIC_ERROR,
        ),
    ]
    flags = [
        AnomalyFlag(
            line_item="cash", quarter="Q3 2023",
            severity=1, anomaly_type=AnomalyType.NEGATIVE_VALUE,
            explanation="Wrong guess",
        ),
        AnomalyFlag(
            line_item="inventory", quarter="Q2 2023",
            severity=2, anomaly_type=AnomalyType.DUPLICATE_ENTRY,
            explanation="Another wrong guess",
        ),
    ]
    result = grade(flags, truths)
    assert result["score"] < 0.2, f"All wrong flags should score < 0.2, got {result['score']}"
    assert result["false_positives"] == 2
    print("PASS: test_grader_false_positives")


def test_grader_empty_flags():
    """No flags submitted should score 0."""
    truths = [
        GroundTruthAnomaly(
            anomaly_id="a1", line_item="revenue", quarter="Q1 2023",
            severity=3, anomaly_type=AnomalyType.ARITHMETIC_ERROR,
        ),
    ]
    result = grade([], truths)
    assert result["score"] == 0.0, f"Empty flags should score 0, got {result['score']}"
    print("PASS: test_grader_empty_flags")


def test_grader_determinism():
    """Grader should produce identical scores across 100 runs."""
    truths = [
        GroundTruthAnomaly(
            anomaly_id="a1", line_item="revenue", quarter="Q1 2023",
            severity=3, anomaly_type=AnomalyType.ARITHMETIC_ERROR,
        ),
        GroundTruthAnomaly(
            anomaly_id="a2", line_item="inventory", quarter="Q2 2023",
            severity=4, anomaly_type=AnomalyType.INVENTORY_TURNOVER,
        ),
    ]
    flags = [
        AnomalyFlag(
            line_item="revenue", quarter="Q1 2023",
            severity=3, anomaly_type=AnomalyType.ARITHMETIC_ERROR,
            explanation="test",
        ),
        AnomalyFlag(
            line_item="cash", quarter="Q1 2023",
            severity=1, anomaly_type=AnomalyType.NEGATIVE_VALUE,
            explanation="false positive",
        ),
    ]
    first_score = grade(flags, truths)["score"]
    for i in range(100):
        score = grade(flags, truths)["score"]
        assert score == first_score, f"Run {i}: score {score} != {first_score}"
    print("PASS: test_grader_determinism (100 runs)")


def test_env_reset():
    """reset() should return a valid observation."""
    env = FinancialAnomalyEnv(task_id="easy")
    obs = env.reset()
    assert obs.company is not None
    assert len(obs.quarters) >= 4
    assert obs.step_number == 0
    assert len(obs.prior_flags) == 0
    print("PASS: test_env_reset")


def test_env_step_flag():
    """step() with flag_anomaly should update state."""
    env = FinancialAnomalyEnv(task_id="easy")
    env.reset()

    action = Action(
        action_type=ActionType.FLAG_ANOMALY,
        flag=AnomalyFlag(
            line_item="revenue",
            quarter="Q1 2022",
            severity=2,
            anomaly_type=AnomalyType.ARITHMETIC_ERROR,
            explanation="Test flag",
        ),
    )
    obs, reward, done, info = env.step(action)
    assert info.flags_so_far == 1
    assert obs.step_number == 1
    assert not done
    print("PASS: test_env_step_flag")


def test_env_step_submit():
    """step() with submit_report should end the episode."""
    env = FinancialAnomalyEnv(task_id="easy")
    env.reset()

    action = Action(action_type=ActionType.SUBMIT_REPORT)
    obs, reward, done, info = env.step(action)
    assert done is True
    assert info.done_reason == "submitted"
    print("PASS: test_env_step_submit")


def test_env_step_request_detail():
    """step() with request_detail should return detail and cost -0.1."""
    env = FinancialAnomalyEnv(task_id="easy")
    obs = env.reset()
    quarter_label = obs.quarters[0].quarter_label

    action = Action(
        action_type=ActionType.REQUEST_DETAIL,
        detail_line_item="revenue",
        detail_quarter=quarter_label,
    )
    obs, reward, done, info = env.step(action)
    assert reward.step_reward == -0.1
    assert obs.detail_response is not None
    assert "components" in obs.detail_response
    print("PASS: test_env_step_request_detail")


def test_env_state():
    """state() should return accurate episode info."""
    env = FinancialAnomalyEnv(task_id="easy")
    env.reset()
    state = env.state()
    assert state.step_number == 0
    assert state.done is False
    assert state.task_id == "easy"
    print("PASS: test_env_state")


def test_env_max_steps():
    """Episode should end with penalty at max steps."""
    env = FinancialAnomalyEnv(task_id="easy", max_steps=3)
    env.reset()

    for i in range(3):
        action = Action(
            action_type=ActionType.FLAG_ANOMALY,
            flag=AnomalyFlag(
                line_item="cash", quarter="Q1 2022",
                severity=1, anomaly_type=AnomalyType.NEGATIVE_VALUE,
                explanation="test",
            ),
        )
        obs, reward, done, info = env.step(action)

    assert done is True
    assert info.done_reason == "max_steps"
    print("PASS: test_env_max_steps")


def test_env_full_episode():
    """Run a complete episode: reset -> flags -> submit -> score."""
    env = FinancialAnomalyEnv(task_id="easy")
    obs = env.reset()
    gt = env.get_ground_truth()

    # Flag all known anomalies
    for anomaly in gt:
        action = Action(
            action_type=ActionType.FLAG_ANOMALY,
            flag=AnomalyFlag(
                line_item=anomaly.line_item,
                quarter=anomaly.quarter,
                severity=anomaly.severity,
                anomaly_type=anomaly.anomaly_type,
                explanation="Detected anomaly",
            ),
        )
        obs, reward, done, info = env.step(action)

    # Submit
    action = Action(action_type=ActionType.SUBMIT_REPORT)
    obs, reward, done, info = env.step(action)
    assert done is True

    score = env.get_final_score()
    assert score["score"] >= 0.9, f"Perfect detection should score >= 0.9, got {score['score']}"
    assert score["recall"] == 1.0
    print(f"PASS: test_env_full_episode (score={score['score']})")


def test_utils_benfords():
    """Benford's chi-squared should detect non-natural distributions."""
    # Natural-ish distribution
    natural = [123, 234, 345, 156, 267, 189, 112, 298, 143, 178,
               210, 320, 145, 256, 190, 115, 225, 335, 142, 260]
    chi2_natural = benfords_chi_squared(natural)

    # Fabricated: all start with 5 or 6
    fabricated = [500, 600, 550, 650, 520, 620, 580, 640, 510, 690,
                  560, 630, 570, 610, 540, 660, 530, 670, 590, 680]
    chi2_fabricated = benfords_chi_squared(fabricated)

    assert chi2_fabricated > chi2_natural, \
        f"Fabricated data should have higher chi2: {chi2_fabricated} vs {chi2_natural}"
    print("PASS: test_utils_benfords")


def test_utils_leading_digit():
    """Leading digit extraction should work correctly."""
    assert leading_digit(123) == 1
    assert leading_digit(9876) == 9
    assert leading_digit(-456) == 4
    assert leading_digit(0) == 0
    assert leading_digit(7) == 7
    print("PASS: test_utils_leading_digit")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_generator_produces_valid_data,
        test_generator_deterministic,
        test_anomaly_injection_easy,
        test_anomaly_injection_medium,
        test_anomaly_injection_hard,
        test_grader_perfect_score,
        test_grader_partial_credit,
        test_grader_false_positives,
        test_grader_empty_flags,
        test_grader_determinism,
        test_env_reset,
        test_env_step_flag,
        test_env_step_submit,
        test_env_step_request_detail,
        test_env_state,
        test_env_max_steps,
        test_env_full_episode,
        test_utils_benfords,
        test_utils_leading_digit,
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("Financial Anomaly Detection - Test Suite")
    print("=" * 60)
    print()

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"FAIL: {test_fn.__name__}: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
