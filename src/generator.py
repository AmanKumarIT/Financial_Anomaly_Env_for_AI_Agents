"""
Generates synthetic but realistic financial statements for fictional companies.
Uses industry baselines from public financial databases and applies controlled
random variation to produce 4-8 quarters of clean data.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .models import (
    CompanyMetadata,
    CompanySize,
    Industry,
    QuarterData,
)
from .utils import dollars_to_cents

# ---------------------------------------------------------------------------
# Industry baseline profiles (ratios sourced from Damodaran Online aggregates)
# ---------------------------------------------------------------------------

INDUSTRY_PROFILES = {
    Industry.TECHNOLOGY: {
        "gross_margin": (0.65, 0.75),
        "opex_ratio": (0.35, 0.45),       # opex / revenue
        "current_ratio": (2.5, 4.0),
        "dso_days": (45, 75),
        "inventory_ratio": (0.02, 0.08),   # inventory / revenue
        "debt_equity": (0.2, 0.6),
        "ocf_ni": (1.2, 1.8),
        "capex_ratio": (0.05, 0.12),
        "seasonal_q4_boost": 0.08,
    },
    Industry.MANUFACTURING: {
        "gross_margin": (0.25, 0.35),
        "opex_ratio": (0.12, 0.20),
        "current_ratio": (1.5, 2.5),
        "dso_days": (35, 55),
        "inventory_ratio": (0.15, 0.30),
        "debt_equity": (0.5, 1.2),
        "ocf_ni": (1.0, 1.5),
        "capex_ratio": (0.08, 0.18),
        "seasonal_q4_boost": 0.04,
    },
    Industry.RETAIL: {
        "gross_margin": (0.30, 0.45),
        "opex_ratio": (0.20, 0.30),
        "current_ratio": (1.0, 1.8),
        "dso_days": (5, 15),
        "inventory_ratio": (0.20, 0.35),
        "debt_equity": (0.6, 1.5),
        "ocf_ni": (1.1, 1.6),
        "capex_ratio": (0.04, 0.10),
        "seasonal_q4_boost": 0.18,  # holiday season
    },
    Industry.HEALTHCARE: {
        "gross_margin": (0.55, 0.70),
        "opex_ratio": (0.30, 0.40),
        "current_ratio": (1.5, 3.0),
        "dso_days": (40, 65),
        "inventory_ratio": (0.05, 0.12),
        "debt_equity": (0.3, 0.8),
        "ocf_ni": (1.0, 1.4),
        "capex_ratio": (0.06, 0.14),
        "seasonal_q4_boost": 0.03,
    },
    Industry.FINANCIAL_SERVICES: {
        "gross_margin": (0.50, 0.65),
        "opex_ratio": (0.25, 0.38),
        "current_ratio": (1.2, 2.0),
        "dso_days": (30, 50),
        "inventory_ratio": (0.0, 0.02),
        "debt_equity": (1.0, 3.0),
        "ocf_ni": (0.9, 1.3),
        "capex_ratio": (0.03, 0.08),
        "seasonal_q4_boost": 0.05,
    },
}

REVENUE_RANGES = {
    CompanySize.SMALL: (10_000_000, 50_000_000),
    CompanySize.MEDIUM: (50_000_000, 500_000_000),
    CompanySize.LARGE: (500_000_000, 2_000_000_000),
}

COMPANY_NAMES = [
    "Apex Dynamics Corp", "Veriton Industries", "Crestline Holdings",
    "NovaEdge Solutions", "Pinnacle Systems Inc", "Meridian Analytics",
    "Stratos Group Ltd", "Caliber Technologies", "Summit Financial Corp",
    "Prism Data Services", "BluePeak Manufacturing", "Orion Healthcare Inc",
    "Trident Retail Group", "Cypher Software Ltd", "Atlas Infrastructure",
]


def _rand_in(low: float, high: float, rng: random.Random) -> float:
    return low + rng.random() * (high - low)


def generate_company_data(
    seed: int,
    num_quarters: int = 6,
    industry: Industry | None = None,
    size: CompanySize | None = None,
) -> Tuple[CompanyMetadata, List[QuarterData]]:
    """
    Generate a complete set of clean financial statements for a fictional company.

    Args:
        seed: RNG seed for reproducibility.
        num_quarters: Number of quarters to generate (4-8).
        industry: Force a specific industry, or random if None.
        size: Force a specific size, or random if None.

    Returns:
        Tuple of (CompanyMetadata, list of QuarterData).
    """
    rng = random.Random(seed)

    # Pick industry & size
    if industry is None:
        industry = rng.choice(list(Industry))
    if size is None:
        size = rng.choice(list(CompanySize))

    profile = INDUSTRY_PROFILES[industry]
    company_name = rng.choice(COMPANY_NAMES)

    # Base annual revenue
    rev_low, rev_high = REVENUE_RANGES[size]
    annual_revenue = rng.uniform(rev_low, rev_high)
    quarterly_revenue_base = annual_revenue / 4

    # Pick ratio targets
    gm = _rand_in(*profile["gross_margin"], rng)
    opex_r = _rand_in(*profile["opex_ratio"], rng)
    cr = _rand_in(*profile["current_ratio"], rng)
    dso = _rand_in(*profile["dso_days"], rng)
    inv_r = _rand_in(*profile["inventory_ratio"], rng)
    de = _rand_in(*profile["debt_equity"], rng)
    ocf_ni_r = _rand_in(*profile["ocf_ni"], rng)
    capex_r = _rand_in(*profile["capex_ratio"], rng)
    seasonal_boost = profile["seasonal_q4_boost"]

    # Growth rate per quarter
    qoq_growth = rng.uniform(-0.02, 0.06)

    fiscal_year_start = rng.choice([2021, 2022, 2023])
    metadata = CompanyMetadata(
        name=company_name,
        industry=industry,
        size=size,
        num_quarters=num_quarters,
        fiscal_year_start=fiscal_year_start,
    )

    quarters: List[QuarterData] = []
    prev_cash = dollars_to_cents(rng.uniform(0.05, 0.15) * annual_revenue)

    for i in range(num_quarters):
        year = fiscal_year_start + i // 4
        q_num = (i % 4) + 1
        label = f"Q{q_num} {year}"

        # Revenue with growth + seasonal variation
        growth_factor = (1 + qoq_growth) ** i
        seasonal = 1.0 + (seasonal_boost if q_num == 4 else -seasonal_boost / 3)
        variation = rng.uniform(0.95, 1.05)
        revenue = quarterly_revenue_base * growth_factor * seasonal * variation
        revenue_c = dollars_to_cents(revenue)

        # Income statement
        cogs_c = dollars_to_cents(revenue * (1 - gm) * rng.uniform(0.97, 1.03))
        gross_profit_c = revenue_c - cogs_c
        opex_c = dollars_to_cents(revenue * opex_r * rng.uniform(0.95, 1.05))
        operating_income_c = gross_profit_c - opex_c
        interest_c = dollars_to_cents(revenue * rng.uniform(0.005, 0.02))
        pretax = operating_income_c - interest_c
        tax_c = dollars_to_cents(max(0, pretax * 0.01) * rng.uniform(0.20, 0.28))
        net_income_c = pretax - tax_c

        # Balance sheet
        receivables_c = dollars_to_cents(revenue * (dso / 90) * rng.uniform(0.92, 1.08))
        inventory_c = dollars_to_cents(revenue * inv_r * rng.uniform(0.90, 1.10))
        other_ca_c = dollars_to_cents(revenue * rng.uniform(0.02, 0.06))

        # Cash flow
        depreciation = dollars_to_cents(revenue * rng.uniform(0.02, 0.05))
        wc_change = dollars_to_cents(revenue * rng.uniform(-0.03, 0.03))
        cf_operating_c = dollars_to_cents(
            (net_income_c + depreciation - wc_change) * 0.01
            * ocf_ni_r * rng.uniform(0.90, 1.10)
        )
        # Simpler: just derive from net income * ocf ratio
        cf_operating_c = int(net_income_c * ocf_ni_r * rng.uniform(0.90, 1.10))

        capex = dollars_to_cents(revenue * capex_r * rng.uniform(0.85, 1.15))
        cf_investing_c = -capex
        cf_financing_c = dollars_to_cents(revenue * rng.uniform(-0.05, 0.03))

        net_cash_change_c = cf_operating_c + cf_investing_c + cf_financing_c
        cash_c = prev_cash + net_cash_change_c
        if cash_c < 0:
            # Inject financing to keep cash positive
            injection = abs(cash_c) + dollars_to_cents(revenue * 0.02)
            cf_financing_c += injection
            net_cash_change_c = cf_operating_c + cf_investing_c + cf_financing_c
            cash_c = prev_cash + net_cash_change_c

        total_ca_c = cash_c + receivables_c + inventory_c + other_ca_c
        fixed_assets_c = dollars_to_cents(annual_revenue * rng.uniform(0.3, 0.8))
        total_assets_c = total_ca_c + fixed_assets_c

        # Liabilities
        payables_c = dollars_to_cents(revenue * rng.uniform(0.05, 0.12))
        short_debt_c = dollars_to_cents(revenue * rng.uniform(0.02, 0.08))
        total_cl_c = payables_c + short_debt_c
        long_debt_c = dollars_to_cents(total_assets_c * rng.uniform(0.10, 0.30))
        total_liab_c = total_cl_c + long_debt_c

        # Equity = assets - liabilities (forced balance)
        equity_c = total_assets_c - total_liab_c

        q = QuarterData(
            quarter_label=label,
            cash=cash_c,
            receivables=receivables_c,
            inventory=inventory_c,
            other_current_assets=other_ca_c,
            total_current_assets=total_ca_c,
            fixed_assets=fixed_assets_c,
            total_assets=total_assets_c,
            payables=payables_c,
            short_term_debt=short_debt_c,
            total_current_liabilities=total_cl_c,
            long_term_debt=long_debt_c,
            total_liabilities=total_liab_c,
            equity=equity_c,
            revenue=revenue_c,
            cogs=cogs_c,
            gross_profit=gross_profit_c,
            operating_expenses=opex_c,
            operating_income=operating_income_c,
            interest_expense=interest_c,
            tax_expense=tax_c,
            net_income=net_income_c,
            cf_operating=cf_operating_c,
            cf_investing=cf_investing_c,
            cf_financing=cf_financing_c,
            net_cash_change=net_cash_change_c,
        )
        quarters.append(q)
        prev_cash = cash_c

    return metadata, quarters
