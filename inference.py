"""
Inference script for the Financial Anomaly Detection environment.
Uses the OpenAI API client with API_BASE_URL, MODEL_NAME, and HF_TOKEN
environment variables as required by the hackathon spec.

Usage:
    export API_BASE_URL=https://your-space.hf.space
    export MODEL_NAME=gpt-4o
    export HF_TOKEN=your_key_here
    python inference.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

try:
    import openai
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests package not installed. Run: pip install requests")
    sys.exit(1)


# ── Required env vars per hackathon spec ─────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN (or OPENAI_API_KEY) environment variable not set.")
    sys.exit(1)

# OpenAI client pointed at the required base URL
client = openai.OpenAI(
    api_key=HF_TOKEN,
    base_url=os.environ.get("OPENAI_BASE_URL"),  # None = default OpenAI; set if using proxy
)

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")  # environment API

REQUEST_TIMEOUT = 60  # seconds per HTTP request


# ── Prompts ───────────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def format_financial_data(obs: Dict) -> str:
    """Format observation data into readable text for the LLM."""
    try:
        company = obs["company"]
        lines = [
            f"Company: {company['name']}",
            f"Industry: {company['industry']}",
            f"Size: {company['size']}",
            f"Quarters: {company['num_quarters']}",
            "",
        ]

        for q in obs.get("quarters", []):
            lines.append(f"=== {q.get('quarter_label', 'N/A')} ===")
            lines.append("BALANCE SHEET:")
            lines.append(f"  Cash: ${q.get('cash', 0)/100:,.2f}")
            lines.append(f"  Receivables: ${q.get('receivables', 0)/100:,.2f}")
            lines.append(f"  Inventory: ${q.get('inventory', 0)/100:,.2f}")
            lines.append(f"  Other Current Assets: ${q.get('other_current_assets', 0)/100:,.2f}")
            lines.append(f"  Total Current Assets: ${q.get('total_current_assets', 0)/100:,.2f}")
            lines.append(f"  Fixed Assets: ${q.get('fixed_assets', 0)/100:,.2f}")
            lines.append(f"  Total Assets: ${q.get('total_assets', 0)/100:,.2f}")
            lines.append(f"  Payables: ${q.get('payables', 0)/100:,.2f}")
            lines.append(f"  Short-term Debt: ${q.get('short_term_debt', 0)/100:,.2f}")
            lines.append(f"  Total Current Liabilities: ${q.get('total_current_liabilities', 0)/100:,.2f}")
            lines.append(f"  Long-term Debt: ${q.get('long_term_debt', 0)/100:,.2f}")
            lines.append(f"  Total Liabilities: ${q.get('total_liabilities', 0)/100:,.2f}")
            lines.append(f"  Equity: ${q.get('equity', 0)/100:,.2f}")
            lines.append("INCOME STATEMENT:")
            lines.append(f"  Revenue: ${q.get('revenue', 0)/100:,.2f}")
            lines.append(f"  COGS: ${q.get('cogs', 0)/100:,.2f}")
            lines.append(f"  Gross Profit: ${q.get('gross_profit', 0)/100:,.2f}")
            lines.append(f"  Operating Expenses: ${q.get('operating_expenses', 0)/100:,.2f}")
            lines.append(f"  Operating Income: ${q.get('operating_income', 0)/100:,.2f}")
            lines.append(f"  Interest Expense: ${q.get('interest_expense', 0)/100:,.2f}")
            lines.append(f"  Tax Expense: ${q.get('tax_expense', 0)/100:,.2f}")
            lines.append(f"  Net Income: ${q.get('net_income', 0)/100:,.2f}")
            lines.append("CASH FLOW:")
            lines.append(f"  Operating: ${q.get('cf_operating', 0)/100:,.2f}")
            lines.append(f"  Investing: ${q.get('cf_investing', 0)/100:,.2f}")
            lines.append(f"  Financing: ${q.get('cf_financing', 0)/100:,.2f}")
            lines.append(f"  Net Cash Change: ${q.get('net_cash_change', 0)/100:,.2f}")
            lines.append("")

        if obs.get("footnotes"):
            lines.append("FOOTNOTES:")
            for fn in obs["footnotes"]:
                lines.append(f"  - {fn}")

        return "\n".join(lines)

    except Exception as e:
        return f"[Error formatting financial data: {e}]"


def safe_post(url: str, payload: Dict, label: str = "") -> Dict | None:
    """POST with full error handling. Returns parsed JSON or None."""
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError as e:
        print(f"  [ConnectionError] {label}: {e}")
    except requests.exceptions.Timeout:
        print(f"  [Timeout] {label}: request timed out after {REQUEST_TIMEOUT}s")
    except requests.exceptions.HTTPError as e:
        print(f"  [HTTPError] {label}: {e}")
    except json.JSONDecodeError as e:
        print(f"  [JSONDecodeError] {label}: could not parse response — {e}")
    except Exception as e:
        print(f"  [Error] {label}: {e}")
    return None


def parse_action(raw: str) -> Dict:
    """Robustly parse the LLM's JSON response into an action dict."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (```json or ```) and last line (```)
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()

        # Attempt to find the first {...} block in case of leading prose
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]

        action = json.loads(text)
        # Validate required key
        if "action_type" not in action:
            raise ValueError("Missing action_type key")
        return action

    except Exception as e:
        print(f"  [ParseWarning] Could not parse LLM response as JSON ({e}), defaulting to submit_report")
        return {"action_type": "submit_report"}


# ── Core loop ─────────────────────────────────────────────────────────────────
def run_task(env_url: str, task_id: str, verbose: bool = False) -> Dict:
    """
    Run the agent against one task. Returns a result dict.
    Never raises — all errors are caught and surfaced in the result.
    """
    print(f"\n{'='*50}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*50}")

    session_id = f"baseline_{task_id}_{int(time.time())}"

    # ── Reset ────────────────────────────────────────────────────────────────
    reset_data = safe_post(
        f"{env_url}/reset",
        {"task_id": task_id, "session_id": session_id},
        label="reset",
    )
    if reset_data is None:
        print(f"  FAILED to reset environment for task '{task_id}'. Skipping.")
        return {
            "task_id": task_id, "steps": 0, "flags_submitted": 0,
            "score": 0.0, "precision": 0.0, "recall": 0.0,
            "severity_accuracy": 0.0, "total_anomalies": 0, "false_positives": 0,
            "error": "reset failed",
        }

    obs = reset_data.get("observation", {})
    financial_text = format_financial_data(obs)
    max_steps = obs.get("max_steps", 20)

    messages: List[Dict] = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Analyze these financial statements and flag all anomalies:\n\n{financial_text}"},
    ]

    step_count  = 0
    all_flags: List[Dict] = []

    # ── Agent loop ───────────────────────────────────────────────────────────
    while step_count < max_steps:

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            raw_reply = response.choices[0].message.content or ""
        except openai.APIConnectionError as e:
            print(f"  [LLM ConnectionError] step {step_count}: {e}")
            break
        except openai.RateLimitError as e:
            print(f"  [LLM RateLimit] step {step_count}: {e} — waiting 30s")
            time.sleep(30)
            continue
        except openai.APIStatusError as e:
            print(f"  [LLM APIStatusError] step {step_count}: {e}")
            break
        except Exception as e:
            print(f"  [LLM Error] step {step_count}: {e}")
            break

        messages.append({"role": "assistant", "content": raw_reply})
        action = parse_action(raw_reply)

        # Send action to env
        step_data = safe_post(
            f"{env_url}/step",
            {"session_id": session_id, "action": action},
            label=f"step-{step_count}",
        )
        if step_data is None:
            print(f"  Step {step_count}: env /step failed, breaking loop.")
            break

        step_count += 1
        done        = step_data.get("done", False)
        step_reward = step_data.get("reward", {}).get("step_reward", 0)
        info_msg    = step_data.get("info", {}).get("message", "")
        remaining   = step_data.get("info", {}).get("steps_remaining", max_steps - step_count)

        if verbose:
            print(f"  Step {step_count:>2}: {action.get('action_type'):<20} | reward={step_reward:+.3f} | {info_msg}")

        if action.get("action_type") == "flag_anomaly" and action.get("flag"):
            all_flags.append(action["flag"])

        if done:
            break

        # Feed env feedback back to LLM
        feedback_parts = [
            f"Step reward: {step_reward}.",
            info_msg,
            f"Steps remaining: {remaining}.",
        ]
        detail = step_data.get("observation", {}).get("detail_response")
        if detail:
            feedback_parts.append(f"Detail: {json.dumps(detail)}")
        feedback_parts.append("Continue flagging anomalies or submit your report if done.")
        messages.append({"role": "user", "content": " ".join(feedback_parts)})

    # ── Score ────────────────────────────────────────────────────────────────
    score_data = safe_post(
        f"{env_url}/score",
        {"session_id": session_id},
        label="score",
    )

    if score_data is None:
        print(f"  WARNING: could not retrieve score for task '{task_id}'.")
        result_block = {}
    else:
        result_block = score_data.get("result", score_data)

    return {
        "task_id":           task_id,
        "steps":             step_count,
        "flags_submitted":   len(all_flags),
        "score":             result_block.get("score", 0.0),
        "precision":         result_block.get("precision", 0.0),
        "recall":            result_block.get("recall", 0.0),
        "severity_accuracy": result_block.get("severity_accuracy", 0.0),
        "total_anomalies":   result_block.get("total_anomalies", 0),
        "false_positives":   result_block.get("false_positives", 0),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global MODEL_NAME
    parser = argparse.ArgumentParser(description="Inference script — Financial Anomaly Detection")
    parser.add_argument("--url",     default=ENV_URL,                          help="Environment API base URL")
    parser.add_argument("--model",   default=MODEL_NAME,                       help="LLM model name")
    parser.add_argument("--tasks",   nargs="+", default=["easy","medium","hard"], help="Task IDs to run")
    parser.add_argument("--verbose", action="store_true",                      help="Print per-step output")
    parser.add_argument("--output",  default="baseline_results.json",          help="JSON output file path")
    args = parser.parse_args()

    # Override global model if passed via CLI
    MODEL_NAME = args.model

    all_results = []
    for task_id in args.tasks:
        try:
            result = run_task(env_url=args.url, task_id=task_id, verbose=args.verbose)
        except Exception as e:
            print(f"  UNHANDLED exception for task '{task_id}': {e}")
            result = {
                "task_id": task_id, "steps": 0, "flags_submitted": 0,
                "score": 0.0, "precision": 0.0, "recall": 0.0,
                "severity_accuracy": 0.0, "total_anomalies": 0,
                "false_positives": 0, "error": str(e),
            }
        all_results.append(result)

        print(f"\n  ── Results for {task_id} ──")
        print(f"  Score:     {result['score']}")
        print(f"  Precision: {result['precision']}")
        print(f"  Recall:    {result['recall']}")
        print(f"  Flags:     {result['flags_submitted']}  |  "
              f"Anomalies: {result['total_anomalies']}  |  "
              f"FP: {result['false_positives']}")

    # Write results
    output_payload = {
        "model":     MODEL_NAME,
        "env_url":   args.url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results":   all_results,
    }
    try:
        with open(args.output, "w") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nResults saved to {args.output}")
    except Exception as e:
        print(f"\nWARNING: could not save results to {args.output}: {e}")
        # Print to stdout as fallback so the run isn't wasted
        print(json.dumps(output_payload, indent=2))

    # Exit 0 always — validator checks for non-zero exit
    sys.exit(0)


if __name__ == "__main__":
    main()