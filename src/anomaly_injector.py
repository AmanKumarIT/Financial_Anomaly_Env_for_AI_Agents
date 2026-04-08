"""
Anomaly injection module.  Takes clean financial data and injects specific
anomaly types at configurable difficulty levels.  Records ground truth for
the grader.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .models import (
    AnomalyType,
    GroundTruthAnomaly,
    QuarterData,
)


# ---------------------------------------------------------------------------
# Easy tier injectors
# ---------------------------------------------------------------------------

def _inject_arithmetic_error(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Make total_current_assets != sum of components."""
    idx = rng.randint(0, len(quarters) - 1)
    q = quarters[idx]
    offset = rng.choice([1, -1]) * rng.randint(50000, 500000)
    q.total_current_assets += offset
    q.total_assets += offset  # propagate so balance sheet still "looks" valid
    q.equity += offset
    return [GroundTruthAnomaly(
        anomaly_id=f"arith_{idx}",
        line_item="total_current_assets",
        quarter=q.quarter_label,
        severity=2,
        anomaly_type=AnomalyType.ARITHMETIC_ERROR,
        description="total_current_assets does not equal sum of cash + receivables + inventory + other_current_assets",
    )]


def _inject_duplicate_entry(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Make two consecutive quarters have identical revenue (statistically improbable)."""
    if len(quarters) < 2:
        return []
    idx = rng.randint(0, len(quarters) - 2)
    quarters[idx + 1].revenue = quarters[idx].revenue
    quarters[idx + 1].gross_profit = quarters[idx + 1].revenue - quarters[idx + 1].cogs
    return [GroundTruthAnomaly(
        anomaly_id=f"dup_{idx}",
        line_item="revenue",
        quarter=quarters[idx + 1].quarter_label,
        severity=2,
        anomaly_type=AnomalyType.DUPLICATE_ENTRY,
        description="Revenue is identical to prior quarter (statistically improbable)",
    )]


def _inject_negative_value(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Make inventory or receivables negative."""
    idx = rng.randint(0, len(quarters) - 1)
    q = quarters[idx]
    field = rng.choice(["inventory", "receivables"])
    current_val = getattr(q, field)
    setattr(q, field, -abs(current_val))
    return [GroundTruthAnomaly(
        anomaly_id=f"neg_{idx}",
        line_item=field,
        quarter=q.quarter_label,
        severity=1,
        anomaly_type=AnomalyType.NEGATIVE_VALUE,
        description=f"{field} has a negative value which is not possible",
    )]


def _inject_impossible_change(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Revenue jumps 500%+ in one quarter."""
    if len(quarters) < 2:
        return []
    idx = rng.randint(1, len(quarters) - 1)
    q = quarters[idx]
    q.revenue = q.revenue * rng.randint(5, 10)
    q.gross_profit = q.revenue - q.cogs
    q.operating_income = q.gross_profit - q.operating_expenses
    return [GroundTruthAnomaly(
        anomaly_id=f"imp_{idx}",
        line_item="revenue",
        quarter=q.quarter_label,
        severity=3,
        anomaly_type=AnomalyType.IMPOSSIBLE_CHANGE,
        description="Revenue increased by 500%+ in a single quarter without explanation",
    )]


def _inject_percentage_error(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """gross_profit != revenue - cogs."""
    idx = rng.randint(0, len(quarters) - 1)
    q = quarters[idx]
    error = rng.choice([1, -1]) * rng.randint(100000, 800000)
    q.gross_profit += error
    q.operating_income = q.gross_profit - q.operating_expenses
    return [GroundTruthAnomaly(
        anomaly_id=f"pct_{idx}",
        line_item="gross_profit",
        quarter=q.quarter_label,
        severity=2,
        anomaly_type=AnomalyType.PERCENTAGE_ERROR,
        description="gross_profit does not equal revenue minus cogs",
    )]


# ---------------------------------------------------------------------------
# Medium tier injectors
# ---------------------------------------------------------------------------

def _inject_receivables_divergence(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Receivables grow 3x faster than revenue over several quarters."""
    if len(quarters) < 3:
        return []
    anomalies = []
    start = rng.randint(0, max(0, len(quarters) - 3))
    for i in range(start + 1, min(start + 3, len(quarters))):
        q = quarters[i]
        q.receivables = int(q.receivables * rng.uniform(2.5, 3.5))
        q.total_current_assets = q.cash + q.receivables + q.inventory + q.other_current_assets
        q.total_assets = q.total_current_assets + q.fixed_assets
        q.equity = q.total_assets - q.total_liabilities
        anomalies.append(GroundTruthAnomaly(
            anomaly_id=f"recv_{i}",
            line_item="receivables",
            quarter=q.quarter_label,
            severity=3,
            anomaly_type=AnomalyType.RECEIVABLES_DIVERGENCE,
            description="Receivables growing 3x faster than revenue trend",
        ))
    return anomalies


def _inject_inventory_turnover(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Inventory spikes without corresponding COGS change."""
    idx = rng.randint(1, len(quarters) - 1)
    q = quarters[idx]
    q.inventory = int(q.inventory * rng.uniform(3.0, 5.0))
    q.total_current_assets = q.cash + q.receivables + q.inventory + q.other_current_assets
    q.total_assets = q.total_current_assets + q.fixed_assets
    q.equity = q.total_assets - q.total_liabilities
    return [GroundTruthAnomaly(
        anomaly_id=f"invt_{idx}",
        line_item="inventory",
        quarter=q.quarter_label,
        severity=3,
        anomaly_type=AnomalyType.INVENTORY_TURNOVER,
        description="Inventory spiked without matching COGS movement",
    )]


def _inject_cashflow_mismatch(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Operating cash flow moves opposite to net income for 2+ quarters."""
    anomalies = []
    start = rng.randint(0, max(0, len(quarters) - 2))
    for i in range(start, min(start + 2, len(quarters))):
        q = quarters[i]
        if q.net_income > 0:
            q.cf_operating = -abs(q.cf_operating)
        else:
            q.cf_operating = abs(q.cf_operating) if q.cf_operating < 0 else q.cf_operating * 2
        q.net_cash_change = q.cf_operating + q.cf_investing + q.cf_financing
        anomalies.append(GroundTruthAnomaly(
            anomaly_id=f"cfm_{i}",
            line_item="cf_operating",
            quarter=q.quarter_label,
            severity=4,
            anomaly_type=AnomalyType.CASHFLOW_MISMATCH,
            description="Operating cash flow direction contradicts net income",
        ))
    return anomalies


def _inject_margin_shift(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Gross margin shifts >10% in one quarter without footnote."""
    if len(quarters) < 2:
        return []
    idx = rng.randint(1, len(quarters) - 1)
    q = quarters[idx]
    # Slash COGS to inflate margin dramatically
    q.cogs = int(q.cogs * rng.uniform(0.4, 0.6))
    q.gross_profit = q.revenue - q.cogs
    q.operating_income = q.gross_profit - q.operating_expenses
    return [GroundTruthAnomaly(
        anomaly_id=f"marg_{idx}",
        line_item="gross_profit",
        quarter=q.quarter_label,
        severity=3,
        anomaly_type=AnomalyType.MARGIN_SHIFT,
        description="Gross margin shifted >10% from prior quarter without explanation",
    )]


def _inject_dso_spike(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """DSO increases 40%+ while revenue stays flat."""
    if len(quarters) < 2:
        return []
    idx = rng.randint(1, len(quarters) - 1)
    q = quarters[idx]
    q.receivables = int(q.receivables * rng.uniform(1.8, 2.5))
    q.total_current_assets = q.cash + q.receivables + q.inventory + q.other_current_assets
    q.total_assets = q.total_current_assets + q.fixed_assets
    q.equity = q.total_assets - q.total_liabilities
    return [GroundTruthAnomaly(
        anomaly_id=f"dso_{idx}",
        line_item="receivables",
        quarter=q.quarter_label,
        severity=3,
        anomaly_type=AnomalyType.DSO_SPIKE,
        description="DSO spiked 40%+ while revenue remained flat",
    )]


# ---------------------------------------------------------------------------
# Hard tier injectors
# ---------------------------------------------------------------------------

def _inject_channel_stuffing(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Q4 revenue spike + receivables balloon + next Q1 implied returns."""
    if len(quarters) < 4:
        return []
    # Find a Q4 or use last-2 quarters
    q4_idx = len(quarters) - 2
    q1_idx = len(quarters) - 1
    q4 = quarters[q4_idx]
    q1 = quarters[q1_idx]

    # Inflate Q4 revenue
    q4.revenue = int(q4.revenue * rng.uniform(1.6, 2.0))
    q4.gross_profit = q4.revenue - q4.cogs
    q4.operating_income = q4.gross_profit - q4.operating_expenses
    q4.receivables = int(q4.receivables * rng.uniform(2.5, 3.5))
    q4.total_current_assets = q4.cash + q4.receivables + q4.inventory + q4.other_current_assets
    q4.total_assets = q4.total_current_assets + q4.fixed_assets
    q4.equity = q4.total_assets - q4.total_liabilities

    # Q1 revenue drops + receivables stay elevated
    q1.revenue = int(q1.revenue * rng.uniform(0.5, 0.7))
    q1.gross_profit = q1.revenue - q1.cogs
    q1.operating_income = q1.gross_profit - q1.operating_expenses

    return [
        GroundTruthAnomaly(
            anomaly_id=f"chstf_rev_{q4_idx}",
            line_item="revenue",
            quarter=q4.quarter_label,
            severity=5,
            anomaly_type=AnomalyType.CHANNEL_STUFFING,
            description="Revenue spike with matching receivables balloon (channel stuffing pattern)",
        ),
        GroundTruthAnomaly(
            anomaly_id=f"chstf_recv_{q4_idx}",
            line_item="receivables",
            quarter=q4.quarter_label,
            severity=5,
            anomaly_type=AnomalyType.CHANNEL_STUFFING,
            description="Receivables ballooned alongside revenue spike",
            depends_on=f"chstf_rev_{q4_idx}",
        ),
        GroundTruthAnomaly(
            anomaly_id=f"chstf_drop_{q1_idx}",
            line_item="revenue",
            quarter=q1.quarter_label,
            severity=4,
            anomaly_type=AnomalyType.CHANNEL_STUFFING,
            description="Revenue collapse following prior quarter spike (returns/reversal)",
            depends_on=f"chstf_rev_{q4_idx}",
        ),
    ]


def _inject_cookie_jar(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Over-accrue expenses in good quarters, release in bad quarters."""
    if len(quarters) < 4:
        return []
    anomalies = []
    # Find highest and lowest NI quarters
    sorted_idx = sorted(range(len(quarters)), key=lambda i: quarters[i].net_income, reverse=True)
    good_q = sorted_idx[0]
    bad_q = sorted_idx[-1]

    # Over-accrue in good quarter
    quarters[good_q].operating_expenses = int(quarters[good_q].operating_expenses * 1.4)
    quarters[good_q].operating_income = quarters[good_q].gross_profit - quarters[good_q].operating_expenses
    quarters[good_q].net_income = int(quarters[good_q].net_income * 0.7)
    anomalies.append(GroundTruthAnomaly(
        anomaly_id=f"cj_over_{good_q}",
        line_item="operating_expenses",
        quarter=quarters[good_q].quarter_label,
        severity=5,
        anomaly_type=AnomalyType.COOKIE_JAR,
        description="Operating expenses over-accrued in strong quarter (cookie jar reserve)",
    ))

    # Release in bad quarter
    quarters[bad_q].operating_expenses = int(quarters[bad_q].operating_expenses * 0.5)
    quarters[bad_q].operating_income = quarters[bad_q].gross_profit - quarters[bad_q].operating_expenses
    quarters[bad_q].net_income = int(quarters[bad_q].net_income * 1.5)
    anomalies.append(GroundTruthAnomaly(
        anomaly_id=f"cj_release_{bad_q}",
        line_item="operating_expenses",
        quarter=quarters[bad_q].quarter_label,
        severity=5,
        anomaly_type=AnomalyType.COOKIE_JAR,
        description="Operating expenses released in weak quarter to smooth earnings",
        depends_on=f"cj_over_{good_q}",
    ))
    return anomalies


def _inject_round_tripping(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Cash outflow in investing + matching inflow in operating."""
    idx = rng.randint(0, len(quarters) - 1)
    q = quarters[idx]
    amount = abs(q.cf_operating) // 2
    q.cf_investing -= amount
    q.cf_operating += amount
    q.net_cash_change = q.cf_operating + q.cf_investing + q.cf_financing
    return [GroundTruthAnomaly(
        anomaly_id=f"rt_{idx}",
        line_item="cf_operating",
        quarter=q.quarter_label,
        severity=5,
        anomaly_type=AnomalyType.ROUND_TRIPPING,
        description="Suspicious matching cash flows: outflow in investing mirrors inflow in operating",
    )]


def _inject_early_revenue(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Revenue recognized upfront from multi-year contract."""
    if len(quarters) < 3:
        return []
    idx = rng.randint(0, len(quarters) - 3)
    q = quarters[idx]
    # Triple revenue in one quarter
    boost = int(q.revenue * rng.uniform(1.5, 2.5))
    q.revenue += boost
    q.gross_profit = q.revenue - q.cogs
    q.operating_income = q.gross_profit - q.operating_expenses
    # But next quarters drop
    for j in range(idx + 1, min(idx + 3, len(quarters))):
        quarters[j].revenue = int(quarters[j].revenue * rng.uniform(0.5, 0.7))
        quarters[j].gross_profit = quarters[j].revenue - quarters[j].cogs
        quarters[j].operating_income = quarters[j].gross_profit - quarters[j].operating_expenses
    return [GroundTruthAnomaly(
        anomaly_id=f"erev_{idx}",
        line_item="revenue",
        quarter=q.quarter_label,
        severity=5,
        anomaly_type=AnomalyType.EARLY_REVENUE,
        description="Revenue spike followed by multi-quarter decline (possible early recognition)",
    )]


def _inject_benfords_violation(
    quarters: List[QuarterData], rng: random.Random
) -> List[GroundTruthAnomaly]:
    """Make expense line items violate Benford's Law by forcing leading digits."""
    anomalies = []
    for idx in range(len(quarters)):
        q = quarters[idx]
        # Force all expenses to start with 5 or 6 (unlikely under Benford's)
        for field in ["operating_expenses", "cogs"]:
            val = getattr(q, field)
            if val > 0:
                s = str(val)
                forced_digit = str(rng.choice([5, 6]))
                new_val = int(forced_digit + s[1:])
                setattr(q, field, new_val)
        q.gross_profit = q.revenue - q.cogs
        q.operating_income = q.gross_profit - q.operating_expenses
    # Single anomaly covering the pattern
    anomalies.append(GroundTruthAnomaly(
        anomaly_id="benford_all",
        line_item="operating_expenses",
        quarter="ALL",
        severity=4,
        anomaly_type=AnomalyType.BENFORDS_LAW,
        description="Expense line items violate Benford's Law distribution across all quarters",
    ))
    return anomalies


# ---------------------------------------------------------------------------
# Injection dispatcher
# ---------------------------------------------------------------------------

EASY_INJECTORS = [
    _inject_arithmetic_error,
    _inject_duplicate_entry,
    _inject_negative_value,
    _inject_impossible_change,
    _inject_percentage_error,
]

MEDIUM_INJECTORS = [
    _inject_receivables_divergence,
    _inject_inventory_turnover,
    _inject_cashflow_mismatch,
    _inject_margin_shift,
    _inject_dso_spike,
]

HARD_INJECTORS = [
    _inject_channel_stuffing,
    _inject_cookie_jar,
    _inject_round_tripping,
    _inject_early_revenue,
    _inject_benfords_violation,
]


def inject_anomalies(
    quarters: List[QuarterData],
    difficulty: str,
    seed: int,
    count: int | None = None,
) -> Tuple[List[QuarterData], List[GroundTruthAnomaly]]:
    """
    Inject anomalies into a copy of the quarterly data.

    Args:
        quarters: List of clean QuarterData.
        difficulty: One of 'easy', 'medium', 'hard'.
        seed: RNG seed for reproducibility.
        count: Override number of anomaly types to inject.

    Returns:
        (modified quarters, list of ground truth anomalies)
    """
    rng = random.Random(seed)
    all_anomalies: List[GroundTruthAnomaly] = []

    if difficulty == "easy":
        pool = EASY_INJECTORS
        n = count or rng.randint(3, 4)
    elif difficulty == "medium":
        pool = EASY_INJECTORS + MEDIUM_INJECTORS
        n = count or rng.randint(4, 6)
    elif difficulty == "hard":
        pool = EASY_INJECTORS + MEDIUM_INJECTORS + HARD_INJECTORS
        n = count or rng.randint(5, 8)
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    chosen = rng.sample(pool, min(n, len(pool)))
    for injector in chosen:
        result = injector(quarters, rng)
        all_anomalies.extend(result)

    return quarters, all_anomalies
