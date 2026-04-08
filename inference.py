"""
Baseline inference script for the Financial Anomaly Detection environment.
Uses the OpenAI API to run a model against all 3 tasks and produce
reproducible baseline scores.

Usage:
    export OPENAI_API_KEY=your_key_here
    python -m baseline.inference --url http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

try:
    import openai
    import requests
except ImportError:
    print("Install dependencies: pip install openai requests")
    sys.exit(1)


SYSTEM_PROMPT = """You are a forensic financial auditor. You will receive quarterly financial statements (balance sheet, income statement, cash flow statement) for a company.

Your task:
1. Analyze the financial data for anomalies, errors, and suspicious patterns.
2. Flag each anomaly you find with: line_item, quarter, severity (1-5), anomaly_type, and explanation.
3. When done analyzing, submit your report.

Anomaly types you can flag:
- arithmetic_error: Column totals don't add up
- duplicate_entry: Identical values across quarters (statistically improbable)
- negative_value: Values that should be positive are negative
- impossible_change: Changes exceeding physically possible bounds
- percentage_error: Computed ratios don't match component values
- receivables_divergence: Receivables growing much faster than revenue
- inventory_turnover: Inventory changes without matching COGS movement
- cashflow_mismatch: Operating cash flow contradicts net income direction
- margin_shift: Gross margin shifts dramatically without explanation
- dso_spike: Days Sales Outstanding spikes while revenue stays flat
- channel_stuffing: Revenue spike + receivables balloon + next quarter drop
- cookie_jar: Expenses over-accrued in good quarters, released in bad ones
- round_tripping: Matching cash flows between investing and operating
- early_revenue: Revenue spike followed by multi-quarter decline
- benfords_law: Leading digit distribution of expenses violates Benford's Law

Respond ONLY with valid JSON. Each response should be one action:
- To flag an anomaly: {"action_type": "flag_anomaly", "flag": {"line_item": "...", "quarter": "...", "severity": N, "anomaly_type": "...", "explanation": "..."}}
- To request detail: {"action_type": "request_detail", "detail_line_item": "...", "detail_quarter": "..."}
- To submit report: {"action_type": "submit_report"}

Flag all anomalies you find, then submit your report."""


def format_financial_data(obs: Dict) -> str:
    """Format observation data into readable text for the LLM."""
    company = obs["company"]
    lines = [
        f"Company: {company['name']}",
        f"Industry: {company['industry']}",
        f"Size: {company['size']}",
        f"Quarters: {company['num_quarters']}",
        "",
    ]

    for q in obs["quarters"]:
        lines.append(f"=== {q['quarter_label']} ===")
        lines.append("BALANCE SHEET:")
        lines.append(f"  Cash: ${q['cash']/100:,.2f}")
        lines.append(f"  Receivables: ${q['receivables']/100:,.2f}")
        lines.append(f"  Inventory: ${q['inventory']/100:,.2f}")
        lines.append(f"  Other Current Assets: ${q['other_current_assets']/100:,.2f}")
        lines.append(f"  Total Current Assets: ${q['total_current_assets']/100:,.2f}")
        lines.append(f"  Fixed Assets: ${q['fixed_assets']/100:,.2f}")
        lines.append(f"  Total Assets: ${q['total_assets']/100:,.2f}")
        lines.append(f"  Payables: ${q['payables']/100:,.2f}")
        lines.append(f"  Short-term Debt: ${q['short_term_debt']/100:,.2f}")
        lines.append(f"  Total Current Liabilities: ${q['total_current_liabilities']/100:,.2f}")
        lines.append(f"  Long-term Debt: ${q['long_term_debt']/100:,.2f}")
        lines.append(f"  Total Liabilities: ${q['total_liabilities']/100:,.2f}")
        lines.append(f"  Equity: ${q['equity']/100:,.2f}")
        lines.append("INCOME STATEMENT:")
        lines.append(f"  Revenue: ${q['revenue']/100:,.2f}")
        lines.append(f"  COGS: ${q['cogs']/100:,.2f}")
        lines.append(f"  Gross Profit: ${q['gross_profit']/100:,.2f}")
        lines.append(f"  Operating Expenses: ${q['operating_expenses']/100:,.2f}")
        lines.append(f"  Operating Income: ${q['operating_income']/100:,.2f}")
        lines.append(f"  Interest Expense: ${q['interest_expense']/100:,.2f}")
        lines.append(f"  Tax Expense: ${q['tax_expense']/100:,.2f}")
        lines.append(f"  Net Income: ${q['net_income']/100:,.2f}")
        lines.append("CASH FLOW:")
        lines.append(f"  Operating: ${q['cf_operating']/100:,.2f}")
        lines.append(f"  Investing: ${q['cf_investing']/100:,.2f}")
        lines.append(f"  Financing: ${q['cf_financing']/100:,.2f}")
        lines.append(f"  Net Cash Change: ${q['net_cash_change']/100:,.2f}")
        lines.append("")

    if obs.get("footnotes"):
        lines.append("FOOTNOTES:")
        for fn in obs["footnotes"]:
            lines.append(f"  - {fn}")

    return "\n".join(lines)


def run_baseline(
    base_url: str,
    task_id: str,
    model: str = "gpt-4o",
    verbose: bool = False,
) -> Dict:
    """
    Run the baseline agent against one task.

    Args:
        base_url: URL of the environment API.
        task_id: One of 'easy', 'medium', 'hard'.
        model: OpenAI model to use.
        verbose: Print step-by-step output.

    Returns:
        Dict with task_id, score, steps, flags.
    """
    client = openai.OpenAI()

    # Reset environment
    reset_resp = requests.post(
        f"{base_url}/reset",
        json={"task_id": task_id, "session_id": f"baseline_{task_id}"},
    )
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()
    obs = reset_data["observation"]

    # Format initial data
    financial_text = format_financial_data(obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze these financial statements and flag all anomalies:\n\n{financial_text}"},
    ]

    step_count = 0
    max_steps = obs.get("max_steps", 20)
    all_flags = []

    while step_count < max_steps:
        # Call OpenAI
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,  # deterministic
                max_tokens=500,
            )
        except Exception as e:
            print(f"  OpenAI API error: {e}")
            break

        assistant_msg = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": assistant_msg})

        # Parse action
        try:
            # Handle markdown code blocks
            clean = assistant_msg
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
            action_data = json.loads(clean.strip())
        except json.JSONDecodeError:
            if verbose:
                print(f"  Step {step_count}: Failed to parse JSON, submitting report")
            action_data = {"action_type": "submit_report"}

        # Send to environment
        step_resp = requests.post(
            f"{base_url}/step",
            json={"session_id": f"baseline_{task_id}", "action": action_data},
        )
        step_resp.raise_for_status()
        step_data = step_resp.json()

        step_count += 1
        done = step_data["done"]

        if verbose:
            reward = step_data["reward"]["step_reward"]
            info_msg = step_data["info"]["message"]
            print(f"  Step {step_count}: {action_data.get('action_type')} | reward={reward} | {info_msg}")

        if action_data.get("action_type") == "flag_anomaly" and action_data.get("flag"):
            all_flags.append(action_data["flag"])

        if done:
            break

        # Add environment feedback to messages
        feedback = f"Step reward: {step_data['reward']['step_reward']}. {step_data['info']['message']}. Steps remaining: {step_data['info']['steps_remaining']}."
        if step_data["observation"].get("detail_response"):
            feedback += f"\nDetail: {json.dumps(step_data['observation']['detail_response'])}"
        messages.append({"role": "user", "content": feedback + "\n\nContinue analyzing or submit your report if done."})

    # Get final score
    score_resp = requests.post(
        f"{base_url}/score",
        json={"session_id": f"baseline_{task_id}"},
    )
    score_resp.raise_for_status()
    result = score_resp.json()["result"]

    return {
        "task_id": task_id,
        "steps": step_count,
        "flags_submitted": len(all_flags),
        "score": result["score"],
        "precision": result["precision"],
        "recall": result["recall"],
        "severity_accuracy": result.get("severity_accuracy", 0),
        "total_anomalies": result.get("total_anomalies", 0),
        "false_positives": result.get("false_positives", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for Financial Anomaly Detection")
    parser.add_argument("--url", default="http://localhost:7860", help="Environment API URL")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"], help="Tasks to run")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step output")
    parser.add_argument("--output", default="baseline_results.json", help="Output file for results")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    results = []
    for task_id in args.tasks:
        print(f"\nRunning baseline on task: {task_id}")
        print("-" * 40)
        result = run_baseline(
            base_url=args.url,
            task_id=task_id,
            model=args.model,
            verbose=args.verbose,
        )
        results.append(result)
        print(f"  Score: {result['score']}")
        print(f"  Precision: {result['precision']}")
        print(f"  Recall: {result['recall']}")
        print(f"  Flags: {result['flags_submitted']} | Anomalies: {result['total_anomalies']} | FP: {result['false_positives']}")

    # Save results
    output = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
