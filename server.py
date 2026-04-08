"""
FastAPI server exposing the Financial Anomaly Detection environment
via REST API endpoints matching the OpenEnv specification.
"""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.environment import FinancialAnomalyEnv
from src.models import Action, Observation, Reward, Info, State

app = FastAPI(
    title="Financial Statement Anomaly Detection",
    description="OpenEnv environment for training agents to audit financial statements",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active environment instances keyed by session_id
_envs: Dict[str, FinancialAnomalyEnv] = {}


class ResetRequest(BaseModel):
    task_id: str = "easy"
    session_id: str = "default"
    max_steps: int = 20


class StepRequest(BaseModel):
    session_id: str = "default"
    action: Action


class StateRequest(BaseModel):
    session_id: str = "default"


class ScoreRequest(BaseModel):
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Health & info
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "financial-anomaly-detection",
        "version": "1.0.0",
        "status": "running",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# OpenEnv API
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> Dict:
    """Create or reset an environment instance and return the initial observation."""
    if req.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {req.task_id}")

    env = FinancialAnomalyEnv(task_id=req.task_id, max_steps=req.max_steps)
    obs = env.reset()
    _envs[req.session_id] = env

    return {"observation": obs.model_dump(), "done": False}


@app.post("/step")
def step(req: StepRequest) -> Dict:
    """Process one agent action and return the result."""
    env = _envs.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active session '{req.session_id}'. Call /reset first.",
        )

    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info.model_dump(),
    }


@app.post("/state")
def get_state(req: StateRequest) -> Dict:
    """Return the current internal state of the episode."""
    env = _envs.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active session '{req.session_id}'. Call /reset first.",
        )
    return {"state": env.state().model_dump()}


@app.post("/score")
def get_score(req: ScoreRequest) -> Dict:
    """Get the final grader score for the current episode."""
    env = _envs.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active session '{req.session_id}'. Call /reset first.",
        )
    return {"result": env.get_final_score()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
