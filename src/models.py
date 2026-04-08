"""
Typed Pydantic models for the Financial Statement Anomaly Detection environment.
Defines Observation, Action, Reward, and State schemas used across the OpenEnv API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnomalyType(str, Enum):
    """All anomaly categories the agent can flag."""
    ARITHMETIC_ERROR = "arithmetic_error"
    DUPLICATE_ENTRY = "duplicate_entry"
    NEGATIVE_VALUE = "negative_value"
    IMPOSSIBLE_CHANGE = "impossible_change"
    PERCENTAGE_ERROR = "percentage_error"
    RECEIVABLES_DIVERGENCE = "receivables_divergence"
    INVENTORY_TURNOVER = "inventory_turnover"
    CASHFLOW_MISMATCH = "cashflow_mismatch"
    MARGIN_SHIFT = "margin_shift"
    DSO_SPIKE = "dso_spike"
    CHANNEL_STUFFING = "channel_stuffing"
    COOKIE_JAR = "cookie_jar"
    ROUND_TRIPPING = "round_tripping"
    EARLY_REVENUE = "early_revenue"
    BENFORDS_LAW = "benfords_law"


class ActionType(str, Enum):
    """Possible agent actions per step."""
    FLAG_ANOMALY = "flag_anomaly"
    REQUEST_DETAIL = "request_detail"
    SUBMIT_REPORT = "submit_report"


class Industry(str, Enum):
    TECHNOLOGY = "technology"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"


class CompanySize(str, Enum):
    SMALL = "small"       # $10M-50M revenue
    MEDIUM = "medium"     # $50M-500M
    LARGE = "large"       # $500M+


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class CompanyMetadata(BaseModel):
    """Static company information for an episode."""
    name: str
    industry: Industry
    size: CompanySize
    num_quarters: int
    fiscal_year_start: int = Field(description="Starting fiscal year, e.g. 2022")


class QuarterData(BaseModel):
    """Financial data for a single quarter.  All amounts in integer cents."""
    quarter_label: str = Field(description="e.g. 'Q1 2023'")
    # Balance sheet
    cash: int = 0
    receivables: int = 0
    inventory: int = 0
    other_current_assets: int = 0
    total_current_assets: int = 0
    fixed_assets: int = 0
    total_assets: int = 0
    payables: int = 0
    short_term_debt: int = 0
    total_current_liabilities: int = 0
    long_term_debt: int = 0
    total_liabilities: int = 0
    equity: int = 0
    # Income statement
    revenue: int = 0
    cogs: int = 0
    gross_profit: int = 0
    operating_expenses: int = 0
    operating_income: int = 0
    interest_expense: int = 0
    tax_expense: int = 0
    net_income: int = 0
    # Cash flow statement
    cf_operating: int = 0
    cf_investing: int = 0
    cf_financing: int = 0
    net_cash_change: int = 0


class AnomalyFlag(BaseModel):
    """An anomaly flagged by the agent."""
    line_item: str
    quarter: str
    severity: int = Field(ge=1, le=5)
    anomaly_type: AnomalyType
    explanation: str = ""


class GroundTruthAnomaly(BaseModel):
    """A known injected anomaly (used internally by the grader)."""
    line_item: str
    quarter: str
    severity: int = Field(ge=1, le=5)
    anomaly_type: AnomalyType
    description: str = ""
    depends_on: Optional[str] = Field(
        default=None,
        description="ID of parent anomaly this one depends on (for ordering bonus)",
    )
    anomaly_id: str = ""


# ---------------------------------------------------------------------------
# Top-level OpenEnv models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees each step."""
    company: CompanyMetadata
    quarters: List[QuarterData]
    footnotes: List[str] = Field(default_factory=list)
    prior_flags: List[AnomalyFlag] = Field(default_factory=list)
    detail_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sub-ledger detail returned after a request_detail action",
    )
    step_number: int = 0
    max_steps: int = 20


class Action(BaseModel):
    """What the agent does each step."""
    action_type: ActionType
    # For flag_anomaly
    flag: Optional[AnomalyFlag] = None
    # For request_detail
    detail_line_item: Optional[str] = None
    detail_quarter: Optional[str] = None


class Reward(BaseModel):
    """Reward signal returned after each step."""
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)


class Info(BaseModel):
    """Extra info dict returned alongside reward."""
    message: str = ""
    flags_so_far: int = 0
    steps_remaining: int = 0
    done_reason: Optional[str] = None


class State(BaseModel):
    """Full internal state returned by state()."""
    step_number: int = 0
    max_steps: int = 20
    flags_submitted: List[AnomalyFlag] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False
    task_id: str = ""
