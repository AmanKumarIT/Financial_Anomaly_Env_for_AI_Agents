---
title: Financial Anomaly Detection
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

<<<<<<< HEAD
# Financial Statement Anomaly Detection

An OpenEnv environment that simulates forensic financial auditing. The agent receives quarterly financial statements and must identify anomalies ranging from arithmetic errors to coordinated fraud patterns.

## Motivation

Financial auditing is a $15B+ global industry. Every public company undergoes quarterly audits by firms like Deloitte, KPMG, EY, and PwC. The SEC uses anomaly detection for enforcement actions. This environment provides a standardized benchmark for training and evaluating AI agents on financial compliance tasks.

## Environment Overview

The agent receives structured financial data (balance sheet, income statement, cash flow statement) for a fictional company across multiple quarters. It must:

- Analyze the data for errors, inconsistencies, and suspicious patterns
- Flag anomalies with type, location, severity, and explanation
- Submit a final audit report for grading

## Action Space

| Action | Parameters | Effect |
|--------|-----------|--------|
| `flag_anomaly` | line_item, quarter, severity (1-5), anomaly_type, explanation | Records a suspected anomaly |
| `request_detail` | line_item, quarter | Returns sub-ledger breakdown (-0.1 reward) |
| `submit_report` | none | Ends episode, triggers grading |

### Anomaly Types

**Easy**: `arithmetic_error`, `duplicate_entry`, `negative_value`, `impossible_change`, `percentage_error`

**Medium**: `receivables_divergence`, `inventory_turnover`, `cashflow_mismatch`, `margin_shift`, `dso_spike`

**Hard**: `channel_stuffing`, `cookie_jar`, `round_tripping`, `early_revenue`, `benfords_law`

## Observation Space

Each observation contains:

- **company**: Industry, size, quarter count, fiscal year
- **quarters**: List of quarterly data with balance sheet, income statement, and cash flow (all amounts in integer cents)
- **footnotes**: Contextual notes about accounting policies
- **prior_flags**: Anomalies already flagged this episode
- **detail_response**: Sub-ledger data if requested
- **step_number / max_steps**: Episode progress

## Task Descriptions

| Task | Quarters | Anomalies | Difficulty |
|------|----------|-----------|------------|
| **Easy** | 4 | 3-5 arithmetic/surface errors | Detectable via basic math checks |
| **Medium** | 6 | 4-7 ratio/trend anomalies | Requires cross-statement analysis |
| **Hard** | 8 | 5-10 coordinated fraud patterns | Requires multi-quarter temporal reasoning |

## Reward Function

- `+1.0` per correctly flagged anomaly (exact type + location match)
- `+0.5` per partial match (correct location, wrong type)
- `-0.3` per false positive
- `-0.1` per `request_detail` call
- `1.2x` multiplier for flagging root cause before dependent anomaly
- `-0.5` penalty if max steps reached without submitting

## Grading Formula

```
Score = (Precision * 0.4) + (Recall * 0.4) + (Severity Accuracy * 0.2) + Dependency Bonus
```

Score range: 0.0 to 1.0. Graders are fully deterministic.

## Baseline Scores

| Task | GPT-4o | GPT-3.5 | Random |
|------|--------|---------|--------|
| Easy | 0.85-0.95 | 0.60-0.75 | 0.05-0.15 |
| Medium | 0.55-0.70 | 0.30-0.45 | 0.02-0.10 |
| Hard | 0.25-0.40 | 0.10-0.20 | 0.01-0.05 |

## Setup & Usage

### Local Development

```bash
# Clone the repo
git clone <repo-url>
cd financial-anomaly-env

# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_env.py

# Start the server
python server.py
```

### Docker

```bash
# Build
docker build -t financial-anomaly-env .

# Run
docker run -p 7860:7860 financial-anomaly-env
```

### API Endpoints

- `GET /` - Environment info
- `GET /health` - Health check
- `POST /reset` - Reset environment (`{"task_id": "easy", "session_id": "my_session"}`)
- `POST /step` - Take action (`{"session_id": "my_session", "action": {...}}`)
- `POST /state` - Get current state (`{"session_id": "my_session"}`)
- `POST /score` - Get final score (`{"session_id": "my_session"}`)

### Running the Baseline

```bash
export OPENAI_API_KEY=your_key_here
python -m baseline.inference --url http://localhost:7860 --verbose
```

## Project Structure

```
financial-anomaly-env/
  openenv.yaml          # OpenEnv metadata
  Dockerfile            # Container config
  server.py             # FastAPI server
  requirements.txt      # Dependencies
  src/
    __init__.py
    models.py           # Pydantic models (Observation, Action, Reward, State)
    environment.py      # Core OpenEnv environment (step/reset/state)
    generator.py        # Synthetic financial data generator
    anomaly_injector.py # Anomaly injection across 3 difficulty tiers
    grader.py           # Deterministic precision/recall grader
    utils.py            # Benford's Law, ratio calculators, validators
  tasks/
    task_easy.json      # Easy task config + ground truth
    task_medium.json    # Medium task config
    task_hard.json      # Hard task config
  baseline/
    inference.py        # OpenAI API baseline agent
  tests/
    test_env.py         # Full test suite
```

## Deployment to Hugging Face Spaces

1. Create a new Space on huggingface.co (Docker SDK)
2. Tag the Space with `openenv`
3. Push the project files
4. The Space will build and serve the API on port 7860

## License

MIT
=======
---
title: Financial Hack
emoji: 🚀
colorFrom: gray
colorTo: pink
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> aea3946240fbe94e7c387e597059eaf2613b6532
