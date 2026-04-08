"""
Core OpenEnv environment for Financial Statement Anomaly Detection.
Implements step(), reset(), state() per the OpenEnv specification.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .anomaly_injector import inject_anomalies
from .generator import generate_company_data
from .grader import grade
from .models import (
    Action,
    ActionType,
    AnomalyFlag,
    CompanyMetadata,
    GroundTruthAnomaly,
    Info,
    Observation,
    QuarterData,
    Reward,
    State,
)


TASKS_DIR = Path(__file__).parent.parent / "tasks"


class FinancialAnomalyEnv:
    """
    Financial Statement Anomaly Detection environment.

    The agent receives quarterly financial statements and must identify
    injected anomalies by flagging them with type, location, severity,
    and explanation.
    """

    def __init__(self, task_id: str = "easy", max_steps: int = 20):
        self.task_id = task_id
        self.max_steps = max_steps

        # Episode state
        self._metadata: Optional[CompanyMetadata] = None
        self._quarters: List[QuarterData] = []
        self._ground_truth: List[GroundTruthAnomaly] = []
        self._flags: List[AnomalyFlag] = []
        self._step_number: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._task_config: Dict[str, Any] = {}

        # Load task config
        self._load_task(task_id)

    def _load_task(self, task_id: str) -> None:
        """Load task configuration from JSON file."""
        task_file = TASKS_DIR / f"task_{task_id}.json"
        if task_file.exists():
            with open(task_file, "r") as f:
                self._task_config = json.load(f)
        else:
            # Default configs
            defaults = {
                "easy": {"seed": 42, "num_quarters": 4, "difficulty": "easy"},
                "medium": {"seed": 137, "num_quarters": 6, "difficulty": "medium"},
                "hard": {"seed": 256, "num_quarters": 8, "difficulty": "hard"},
            }
            self._task_config = defaults.get(task_id, defaults["easy"])

    def reset(self) -> Observation:
        """
        Generate a fresh company with anomalies and return the initial observation.
        Clears all episode state.
        """
        seed = self._task_config.get("seed", 42)
        num_q = self._task_config.get("num_quarters", 6)
        difficulty = self._task_config.get("difficulty", "easy")

        # Generate clean data
        self._metadata, clean_quarters = generate_company_data(
            seed=seed,
            num_quarters=num_q,
        )

        # Inject anomalies
        self._quarters, self._ground_truth = inject_anomalies(
            quarters=clean_quarters,
            difficulty=difficulty,
            seed=seed + 1000,  # separate seed for injection
        )

        # Reset episode state
        self._flags = []
        self._step_number = 0
        self._cumulative_reward = 0.0
        self._done = False

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        """
        Process one agent action.

        Args:
            action: The agent's action (flag_anomaly, request_detail, or submit_report).

        Returns:
            Tuple of (observation, reward, done, info).
        """
        if self._done:
            return (
                self._build_observation(),
                Reward(step_reward=0.0, cumulative_reward=self._cumulative_reward),
                True,
                Info(message="Episode already finished.", done_reason="already_done"),
            )

        self._step_number += 1
        step_reward = 0.0
        info_msg = ""
        done_reason = None
        detail_response = None

        if action.action_type == ActionType.FLAG_ANOMALY and action.flag:
            flag = action.flag
            self._flags.append(flag)

            # Compute immediate reward
            matched = False
            matched_gt = None
            for gt in self._ground_truth:
                loc_match = (
                    flag.line_item == gt.line_item
                    and (flag.quarter == gt.quarter or gt.quarter == "ALL")
                )
                if loc_match:
                    if flag.anomaly_type.value == gt.anomaly_type.value:
                        step_reward = 1.0
                        matched = True
                        matched_gt = gt
                    else:
                        step_reward = 0.5
                        matched = True
                        matched_gt = gt
                    break

            if not matched:
                step_reward = -0.3  # false positive

            info_msg = f"Flag recorded: {flag.line_item} @ {flag.quarter}"

        elif action.action_type == ActionType.REQUEST_DETAIL:
            step_reward = -0.1  # exploration cost
            # Return a synthetic sub-ledger breakdown
            if action.detail_line_item and action.detail_quarter:
                detail_response = self._generate_detail(
                    action.detail_line_item, action.detail_quarter
                )
                info_msg = f"Detail returned for {action.detail_line_item} @ {action.detail_quarter}"
            else:
                info_msg = "request_detail requires detail_line_item and detail_quarter"

        elif action.action_type == ActionType.SUBMIT_REPORT:
            self._done = True
            done_reason = "submitted"
            # Final grading
            grade_result = grade(self._flags, self._ground_truth)
            step_reward = 0.0
            info_msg = f"Report submitted. Final score: {grade_result['score']}"

        # Check max steps
        if self._step_number >= self.max_steps and not self._done:
            self._done = True
            done_reason = "max_steps"
            step_reward -= 0.5
            info_msg = "Max steps reached. Episode ended with penalty."

        self._cumulative_reward += step_reward

        obs = self._build_observation(detail_response=detail_response)
        reward = Reward(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            breakdown={"step": round(step_reward, 4)},
        )
        info = Info(
            message=info_msg,
            flags_so_far=len(self._flags),
            steps_remaining=self.max_steps - self._step_number,
            done_reason=done_reason,
        )

        return obs, reward, self._done, info

    def state(self) -> State:
        """Return the current internal state of the episode."""
        return State(
            step_number=self._step_number,
            max_steps=self.max_steps,
            flags_submitted=list(self._flags),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            task_id=self.task_id,
        )

    def get_final_score(self) -> Dict:
        """Run the grader on all submitted flags and return the full result."""
        return grade(self._flags, self._ground_truth)

    def get_ground_truth(self) -> List[GroundTruthAnomaly]:
        """Return ground truth anomalies (for testing/debugging only)."""
        return self._ground_truth

    def _build_observation(
        self, detail_response: Optional[Dict] = None
    ) -> Observation:
        """Build the current observation for the agent."""
        return Observation(
            company=self._metadata,
            quarters=self._quarters,
            footnotes=self._task_config.get("footnotes", []),
            prior_flags=list(self._flags),
            detail_response=detail_response,
            step_number=self._step_number,
            max_steps=self.max_steps,
        )

    def _generate_detail(self, line_item: str, quarter: str) -> Dict:
        """Generate a synthetic sub-ledger breakdown for a line item."""
        rng = random.Random(hash(f"{line_item}_{quarter}"))
        target_q = None
        for q in self._quarters:
            if q.quarter_label == quarter:
                target_q = q
                break
        if target_q is None:
            return {"error": f"Quarter {quarter} not found"}

        total = getattr(target_q, line_item, None)
        if total is None:
            return {"error": f"Line item {line_item} not found"}

        # Generate 3-5 sub-components
        n_subs = rng.randint(3, 5)
        sub_labels = [f"{line_item}_component_{i+1}" for i in range(n_subs)]
        weights = [rng.random() for _ in range(n_subs)]
        weight_sum = sum(weights)
        components = {}
        running = 0
        for i, label in enumerate(sub_labels):
            if i == n_subs - 1:
                components[label] = total - running
            else:
                val = int(total * weights[i] / weight_sum)
                components[label] = val
                running += val

        return {
            "line_item": line_item,
            "quarter": quarter,
            "total": total,
            "components": components,
        }
