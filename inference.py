"""
Inference Script — Financial Anomaly Detection Environment
==========================================================
MANDATORY env vars:
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    LOCAL_IMAGE_NAME (optional) local docker image name if using from_docker_image()

Defaults are set only for API_BASE_URL and MODEL_NAME.

STDOUT FORMAT (exactly, in order):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("[DEBUG] requests not installed. Run: pip install requests", flush=True)
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("[DEBUG] openai not installed. Run: pip install openai", flush=True)
    sys.exit(1)


# ── Env vars (per hackathon spec) ─────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional

# ENV_URL = where YOUR financial anomaly environment server is running
ENV_URL          = os.getenv("ENV_URL", "http://localhost:7860")

if not API_KEY:
    print("[DEBUG] HF_TOKEN is not set. Please set it before running.", flush=True)
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARK             = "financial_anomaly"
TASKS                 = ["easy", "medium", "hard"]
MAX_STEPS             = 20
TEMPERATURE           = 0.0
MAX_TOKENS            = 500
SUCCESS_SCORE_THRESHOLD = 0.1
REQUEST_TIMEOUT       = 60  # seconds

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ── Structured log helpers ────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt ────────────────────────────────────────────────────────────────────
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
def format_financial_data(obs: dict) -> str:
    try:
        company = obs.get("company", {})
        lines = [
            f"Company: {company.get('name', 'N/A')}",
            f"Industry: {company.get('industry', 'N/A')}",
            f"Size: {company.get('size', 'N/A')}",
            f"Quarters: {company.get('num_quarters', 'N/A')}",
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


def safe_post(url: str, payload: dict, label: str = "") -> Optional[dict]:
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError as e:
        print(f"[DEBUG] ConnectionError {label}: {e}", flush=True)
    except requests.exceptions.Timeout:
        print(f"[DEBUG] Timeout {label}: timed out after {REQUEST_TIMEOUT}s", flush=True)
    except requests.exceptions.HTTPError as e:
        print(f"[DEBUG] HTTPError {label}: {e}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Error {label}: {e}", flush=True)
    return None


def parse_action(raw: str) -> dict:
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text  = "\n".join(inner).strip()
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        action = json.loads(text)
        if "action_type" not in action:
            raise ValueError("Missing action_type")
        return action
    except Exception as e:
        print(f"[DEBUG] parse_action failed ({e}), defaulting to submit_report", flush=True)
        return {"action_type": "submit_report"}


def action_to_str(action: dict) -> str:
    """Convert action dict to a compact single-line string for [STEP] logging."""
    atype = action.get("action_type", "unknown")
    if atype == "flag_anomaly":
        flag = action.get("flag", {})
        return f"flag_anomaly({flag.get('anomaly_type','?')}:{flag.get('line_item','?')})"
    if atype == "request_detail":
        return f"request_detail({action.get('detail_line_item','?')}:{action.get('detail_quarter','?')})"
    return atype


# ── Single task runner ────────────────────────────────────────────────────────
def run_task(task_id: str) -> None:
    """
    Runs one full episode for task_id.
    Emits [START], [STEP]×n, [END] — [END] always fires via finally.
    """
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    session_id  = f"inference_{task_id}_{int(time.time())}"
    step_count  = 0
    rewards: List[float] = []
    score       = 0.0
    success     = False

    try:
        # ── Reset ────────────────────────────────────────────────────────────
        reset_data = safe_post(
            f"{ENV_URL}/reset",
            {"task_id": task_id, "session_id": session_id},
            label="reset",
        )
        if reset_data is None:
            print(f"[DEBUG] Could not reset env for task '{task_id}'", flush=True)
            return  # finally will still fire and emit [END]

        obs      = reset_data.get("observation", {})
        max_steps = obs.get("max_steps", MAX_STEPS)
        financial_text = format_financial_data(obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Analyze these financial statements and flag all anomalies:\n\n{financial_text}"},
        ]

        # ── Agent loop ───────────────────────────────────────────────────────
        while step_count < max_steps:
            # Call LLM
            llm_error: Optional[str] = None
            raw_reply = ""
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                raw_reply = (completion.choices[0].message.content or "").strip()
            except Exception as exc:
                llm_error = str(exc)
                print(f"[DEBUG] LLM error step {step_count+1}: {exc}", flush=True)
                raw_reply = ""

            messages.append({"role": "assistant", "content": raw_reply})
            action = parse_action(raw_reply) if raw_reply else {"action_type": "submit_report"}

            # Send to environment
            step_data = safe_post(
                f"{ENV_URL}/step",
                {"session_id": session_id, "action": action},
                label=f"step-{step_count+1}",
            )

            step_count += 1

            if step_data is None:
                # Env call failed — log step with reward 0, mark done
                log_step(
                    step=step_count,
                    action=action_to_str(action),
                    reward=0.0,
                    done=True,
                    error="env_step_failed",
                )
                rewards.append(0.0)
                break

            step_reward = float(step_data.get("reward", {}).get("step_reward", 0.0))
            done        = bool(step_data.get("done", False))
            info_msg    = step_data.get("info", {}).get("message", "")
            remaining   = step_data.get("info", {}).get("steps_remaining", max_steps - step_count)

            rewards.append(step_reward)
            log_step(
                step=step_count,
                action=action_to_str(action),
                reward=step_reward,
                done=done,
                error=llm_error,
            )

            if done:
                break

            # Feed env response back to LLM
            feedback_parts = [
                f"Step reward: {step_reward:.2f}.",
                info_msg,
                f"Steps remaining: {remaining}.",
            ]
            detail = step_data.get("observation", {}).get("detail_response")
            if detail:
                feedback_parts.append(f"Detail: {json.dumps(detail)}")
            feedback_parts.append("Continue flagging anomalies or submit your report if done.")
            messages.append({"role": "user", "content": " ".join(feedback_parts)})

        # ── Score ────────────────────────────────────────────────────────────
        score_data = safe_post(
            f"{ENV_URL}/score",
            {"session_id": session_id},
            label="score",
        )
        if score_data is not None:
            result_block = score_data.get("result", score_data)
            score = float(result_block.get("score", 0.0))
        else:
            # Fallback: normalise from rewards
            max_possible = max_steps * 1.0
            score = min(max(sum(rewards) / max_possible, 0.001), 0.999) if max_possible > 0 else 0.001

        score   = min(max(score, 0.001), 0.999)  # clamp to (0, 1) exclusive — validator rejects 0.0 and 1.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Unhandled exception in run_task({task_id}): {e}", flush=True)

    finally:
        score = min(max(score, 0.001), 0.999)
        log_end(success=success, steps=step_count, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    tasks_to_run = os.getenv("TASKS", ",".join(TASKS)).split(",")
    tasks_to_run = [t.strip() for t in tasks_to_run if t.strip()]

    for task_id in tasks_to_run:
        run_task(task_id)

    sys.exit(0)


if __name__ == "__main__":
    main()