"""
Deterministic grader for scoring agent anomaly detection performance.
Uses precision/recall with severity weighting and partial credit.
Score range: 0.0 to 1.0.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .models import AnomalyFlag, GroundTruthAnomaly


def _match_flag_to_truth(
    flag: AnomalyFlag,
    truths: List[GroundTruthAnomaly],
    already_matched: Set[str],
) -> Tuple[str, float]:
    """
    Try to match an agent flag to a ground truth anomaly.

    Returns:
        (matched_anomaly_id, credit)  where credit is 1.0 (exact), 0.5 (partial), or 0.0 (miss).
    """
    for gt in truths:
        if gt.anomaly_id in already_matched:
            continue
        # Location match: same line_item AND (same quarter OR gt.quarter == "ALL")
        location_match = (
            flag.line_item == gt.line_item
            and (flag.quarter == gt.quarter or gt.quarter == "ALL")
        )
        if location_match:
            if flag.anomaly_type.value == gt.anomaly_type.value:
                return gt.anomaly_id, 1.0   # exact match
            else:
                return gt.anomaly_id, 0.5   # partial: right location, wrong type
    return "", 0.0


def _severity_accuracy(
    flags: List[AnomalyFlag],
    truths: List[GroundTruthAnomaly],
    matches: Dict[str, str],  # flag_idx -> anomaly_id
) -> float:
    """
    Compute severity accuracy for matched flags.
    Returns 1.0 - (mean absolute severity error / 4.0).
    """
    if not matches:
        return 0.0
    truth_map = {gt.anomaly_id: gt for gt in truths}
    total_error = 0.0
    count = 0
    for flag_idx_str, anomaly_id in matches.items():
        flag_idx = int(flag_idx_str)
        if anomaly_id and anomaly_id in truth_map:
            pred_sev = flags[flag_idx].severity
            actual_sev = truth_map[anomaly_id].severity
            total_error += abs(pred_sev - actual_sev)
            count += 1
    if count == 0:
        return 0.0
    mae = total_error / count
    return max(0.0, 1.0 - mae / 4.0)


def _dependency_bonus(
    flags: List[AnomalyFlag],
    truths: List[GroundTruthAnomaly],
    matches: Dict[str, str],
) -> float:
    """
    Compute dependency ordering bonus.  If an agent flags a root cause
    BEFORE its dependent anomaly, the root cause gets a 1.2x multiplier.
    Returns a bonus value added to the score.
    """
    truth_map = {gt.anomaly_id: gt for gt in truths}
    # Build reverse map: anomaly_id -> flag index (order)
    anomaly_to_flag_order: Dict[str, int] = {}
    for flag_idx_str, anomaly_id in matches.items():
        if anomaly_id:
            anomaly_to_flag_order[anomaly_id] = int(flag_idx_str)

    bonus = 0.0
    for gt in truths:
        if gt.depends_on and gt.depends_on in anomaly_to_flag_order:
            if gt.anomaly_id in anomaly_to_flag_order:
                parent_order = anomaly_to_flag_order[gt.depends_on]
                child_order = anomaly_to_flag_order[gt.anomaly_id]
                if parent_order < child_order:
                    bonus += 0.02  # small bonus per correct ordering
    return bonus


def grade(
    flags: List[AnomalyFlag],
    ground_truth: List[GroundTruthAnomaly],
) -> Dict:
    """
    Grade agent flags against ground truth anomalies.

    Args:
        flags: Agent-submitted anomaly flags.
        ground_truth: Known injected anomalies.

    Returns:
        Dict with 'score' (0.0-1.0), 'precision', 'recall',
        'severity_accuracy', 'breakdown'.
    """
    if not ground_truth:
        # No anomalies to find; penalize any flags as false positives
        score = 1.0 if not flags else max(0.0, 1.0 - len(flags) * 0.3)
        return {
            "score": round(score, 4),
            "precision": 1.0 if not flags else 0.0,
            "recall": 1.0,
            "severity_accuracy": 1.0,
            "details": "No anomalies to detect.",
        }

    already_matched: Set[str] = set()
    matches: Dict[str, str] = {}  # str(flag_idx) -> anomaly_id
    credits: List[float] = []

    for i, flag in enumerate(flags):
        anomaly_id, credit = _match_flag_to_truth(flag, ground_truth, already_matched)
        if anomaly_id:
            already_matched.add(anomaly_id)
            matches[str(i)] = anomaly_id
        else:
            matches[str(i)] = ""
        credits.append(credit)

    # Precision: weighted correct / total flags
    total_flags = len(flags)
    if total_flags == 0:
        precision = 0.0
    else:
        precision = sum(credits) / total_flags

    # Recall: matched anomalies / total anomalies
    total_anomalies = len(ground_truth)
    matched_count = sum(1 for c in credits if c > 0)
    recall = matched_count / total_anomalies if total_anomalies > 0 else 0.0

    # Severity accuracy
    sev_acc = _severity_accuracy(flags, ground_truth, matches)

    # Dependency bonus
    dep_bonus = _dependency_bonus(flags, ground_truth, matches)

    # Final score: weighted combination
    raw_score = (precision * 0.4) + (recall * 0.4) + (sev_acc * 0.2) + dep_bonus
    score = min(1.0, max(0.0, raw_score))

    return {
        "score": round(score, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "severity_accuracy": round(sev_acc, 4),
        "dependency_bonus": round(dep_bonus, 4),
        "flags_submitted": total_flags,
        "anomalies_found": matched_count,
        "total_anomalies": total_anomalies,
        "false_positives": total_flags - matched_count,
    }
