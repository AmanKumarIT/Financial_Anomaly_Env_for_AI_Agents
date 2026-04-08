"""
Utility functions for financial ratio calculations, Benford's Law analysis,
and cross-statement validation.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .models import QuarterData


# ---------------------------------------------------------------------------
# Benford's Law
# ---------------------------------------------------------------------------

BENFORD_EXPECTED = {
    1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
    5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046,
}


def leading_digit(value: int) -> int:
    """Return the leading digit of an integer (ignoring sign and zero)."""
    v = abs(value)
    if v == 0:
        return 0
    while v >= 10:
        v //= 10
    return v


def benfords_chi_squared(values: List[int]) -> float:
    """
    Compute Benford's Law chi-squared statistic for a list of integer values.
    Higher values indicate more deviation from expected distribution.
    """
    counts: Dict[int, int] = {d: 0 for d in range(1, 10)}
    total = 0
    for v in values:
        d = leading_digit(v)
        if d >= 1:
            counts[d] += 1
            total += 1
    if total == 0:
        return 0.0
    chi2 = 0.0
    for digit in range(1, 10):
        observed = counts[digit]
        expected = BENFORD_EXPECTED[digit] * total
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected
    return chi2


# ---------------------------------------------------------------------------
# Financial ratios
# ---------------------------------------------------------------------------

def gross_margin(q: QuarterData) -> float:
    """Gross profit / Revenue."""
    if q.revenue == 0:
        return 0.0
    return q.gross_profit / q.revenue


def current_ratio(q: QuarterData) -> float:
    """Total current assets / Total current liabilities."""
    if q.total_current_liabilities == 0:
        return 0.0
    return q.total_current_assets / q.total_current_liabilities


def days_sales_outstanding(q: QuarterData) -> float:
    """DSO = (Receivables / Revenue) * 90 (for a quarter)."""
    if q.revenue == 0:
        return 0.0
    return (q.receivables / q.revenue) * 90


def ocf_to_ni_ratio(q: QuarterData) -> float:
    """Operating cash flow / Net income."""
    if q.net_income == 0:
        return 0.0
    return q.cf_operating / q.net_income


def inventory_turnover(q: QuarterData) -> float:
    """COGS / Inventory (annualized from quarterly = multiply by 4)."""
    if q.inventory == 0:
        return 0.0
    return (q.cogs * 4) / q.inventory


def debt_to_equity(q: QuarterData) -> float:
    """Total liabilities / Equity."""
    if q.equity == 0:
        return 0.0
    return q.total_liabilities / q.equity


# ---------------------------------------------------------------------------
# Cross-statement validation helpers
# ---------------------------------------------------------------------------

def balance_sheet_balanced(q: QuarterData) -> bool:
    """Check assets == liabilities + equity."""
    return q.total_assets == q.total_liabilities + q.equity


def cash_flow_reconciles(q: QuarterData) -> bool:
    """Check net cash change == sum of CF components."""
    return q.net_cash_change == q.cf_operating + q.cf_investing + q.cf_financing


def current_assets_sum_check(q: QuarterData) -> bool:
    """Check total_current_assets == sum of components."""
    expected = q.cash + q.receivables + q.inventory + q.other_current_assets
    return q.total_current_assets == expected


def income_statement_check(q: QuarterData) -> bool:
    """Check gross_profit == revenue - cogs."""
    return q.gross_profit == q.revenue - q.cogs


def cents_to_dollars(cents: int) -> float:
    """Convert integer cents to dollar float for display."""
    return cents / 100.0


def dollars_to_cents(dollars: float) -> int:
    """Convert dollar float to integer cents for storage."""
    return int(round(dollars * 100))
