# Setup & Deployment Guide

Complete step-by-step instructions for running the Financial Statement Anomaly Detection environment locally, in Docker, and on Hugging Face Spaces.

---

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Docker (for containerized deployment)
- Git (for version control and HF Spaces deployment)
- A Hugging Face account (for cloud deployment)
- An OpenAI API key (for running the baseline agent only)

---

## 1. Local Development Setup

### 1.1 Clone and Install

```bash
git clone <your-repo-url>
cd financial-anomaly-env

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 1.2 Verify Installation

```bash
# Run the test suite (19 tests)
python tests/test_env.py
```

Expected output:
```
============================================================
Financial Anomaly Detection - Test Suite
============================================================

PASS: test_generator_produces_valid_data
PASS: test_generator_deterministic
PASS: test_anomaly_injection_easy
PASS: test_anomaly_injection_medium
PASS: test_anomaly_injection_hard
PASS: test_grader_perfect_score
PASS: test_grader_partial_credit
PASS: test_grader_false_positives
PASS: test_grader_empty_flags
PASS: test_grader_determinism (100 runs)
PASS: test_env_reset
PASS: test_env_step_flag
PASS: test_env_step_submit
PASS: test_env_step_request_detail
PASS: test_env_state
PASS: test_env_max_steps
PASS: test_env_full_episode (score=1.0)
PASS: test_utils_benfords
PASS: test_utils_leading_digit

Results: 19 passed, 0 failed, 19 total
All tests passed!
```

### 1.3 Start the API Server

```bash
python server.py
```

Server runs at `http://localhost:7860`. Verify with:

```bash
curl http://localhost:7860/
# Returns: {"name":"financial-anomaly-detection","version":"1.0.0","status":"running","tasks":["easy","medium","hard"]}

curl http://localhost:7860/health
# Returns: {"status":"ok"}
```

### 1.4 Quick API Test

```bash
# Reset with easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "session_id": "demo"}'

# Flag an anomaly
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "action": {"action_type": "flag_anomaly", "flag": {"line_item": "revenue", "quarter": "Q1 2022", "severity": 3, "anomaly_type": "arithmetic_error", "explanation": "Totals mismatch"}}}'

# Submit report
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "action": {"action_type": "submit_report"}}'

# Get score
curl -X POST http://localhost:7860/score \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo"}'
```

---

## 2. Docker Deployment

### 2.1 Build the Image

```bash
docker build -t financial-anomaly-env .
```

### 2.2 Run the Container

```bash
docker run -p 7860:7860 financial-anomaly-env
```

### 2.3 Verify

```bash
curl http://localhost:7860/health
# Returns: {"status":"ok"}
```

### 2.4 Run Tests Inside Container

```bash
docker run financial-anomaly-env python tests/test_env.py
```

---

## 3. Hugging Face Spaces Deployment

### 3.1 Create a New Space

1. Go to https://huggingface.co/new-space
2. Enter a space name (e.g., `financial-anomaly-detection`)
3. Select **Docker** as the SDK
4. Set visibility to **Public**
5. Click **Create Space**

### 3.2 Push Code to the Space

```bash
# Add the HF Space as a remote
git remote add hf https://huggingface.co/spaces/<your-username>/financial-anomaly-detection

# Push
git push hf main
```

### 3.3 Add the OpenEnv Tag

Edit the Space's `README.md` metadata to include:

```yaml
---
title: Financial Statement Anomaly Detection
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---
```

### 3.4 Verify Deployment

Once the Space shows status **Running**:

```bash
curl https://<your-username>-financial-anomaly-detection.hf.space/health
# Returns: {"status":"ok"}
```

---

## 4. Running the Baseline Agent

### 4.1 Set API Key

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 4.2 Run Against Local Server

```bash
# Start server first
python server.py &

# Run baseline
python -m baseline.inference --url http://localhost:7860 --verbose
```

### 4.3 Run Against HF Space

```bash
python -m baseline.inference \
  --url https://<your-username>-financial-anomaly-detection.hf.space \
  --verbose
```

### 4.4 Output

Results are saved to `baseline_results.json`:

```json
{
  "model": "gpt-4o",
  "timestamp": "2025-01-15T12:00:00Z",
  "results": [
    {"task_id": "easy", "score": 0.92, "precision": 0.85, "recall": 1.0, ...},
    {"task_id": "medium", "score": 0.61, "precision": 0.70, "recall": 0.57, ...},
    {"task_id": "hard", "score": 0.33, "precision": 0.45, "recall": 0.30, ...}
  ]
}
```

### 4.5 Customize the Model

```bash
# Use a different model
python -m baseline.inference --model gpt-4o-mini --url http://localhost:7860

# Run only specific tasks
python -m baseline.inference --tasks easy medium --url http://localhost:7860
```

---

## 5. Using the Environment Programmatically

### 5.1 Direct Python Usage (No Server)

```python
from src import FinancialAnomalyEnv, Action, ActionType, AnomalyFlag, AnomalyType

# Create environment
env = FinancialAnomalyEnv(task_id="easy")

# Reset
obs = env.reset()
print(f"Company: {obs.company.name}")
print(f"Quarters: {len(obs.quarters)}")

# Flag an anomaly
action = Action(
    action_type=ActionType.FLAG_ANOMALY,
    flag=AnomalyFlag(
        line_item="revenue",
        quarter="Q1 2022",
        severity=3,
        anomaly_type=AnomalyType.ARITHMETIC_ERROR,
        explanation="Column totals mismatch",
    ),
)
obs, reward, done, info = env.step(action)
print(f"Reward: {reward.step_reward}")

# Submit report
obs, reward, done, info = env.step(
    Action(action_type=ActionType.SUBMIT_REPORT)
)

# Get final score
score = env.get_final_score()
print(f"Score: {score['score']}")
```

### 5.2 Via HTTP API

```python
import requests

BASE = "http://localhost:7860"

# Reset
r = requests.post(f"{BASE}/reset", json={"task_id": "medium", "session_id": "s1"})
obs = r.json()["observation"]

# Step
r = requests.post(f"{BASE}/step", json={
    "session_id": "s1",
    "action": {
        "action_type": "flag_anomaly",
        "flag": {
            "line_item": "receivables",
            "quarter": "Q3 2022",
            "severity": 3,
            "anomaly_type": "receivables_divergence",
            "explanation": "Receivables growing faster than revenue"
        }
    }
})
print(r.json()["reward"])

# Score
r = requests.post(f"{BASE}/score", json={"session_id": "s1"})
print(r.json()["result"])
```

---

## 6. Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root directory (`financial-anomaly-env/`) |
| Server won't start on port 7860 | Check if another process uses port 7860: `lsof -i :7860` |
| Docker build fails | Verify Docker is running and you have internet access for pip |
| HF Space stuck on "Building" | Check the Space logs for build errors. Verify Dockerfile syntax. |
| Baseline script errors with "OPENAI_API_KEY not set" | Run `export OPENAI_API_KEY=your_key` before running the script |
| Tests fail on `test_generator_produces_valid_data` | Verify Python 3.11+. Run `python --version`. |
| Score always 0.0 | You're flagging the wrong line_item/quarter. Check ground truth with `env.get_ground_truth()` |
