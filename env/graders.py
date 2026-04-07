from __future__ import annotations


def grade(metrics: dict, phase_switched: bool, action_duration: int, task_cfg: dict) -> float:
    score = 0.0
    cleared = metrics["cleared"]
    wait_total = metrics["wait_total"]
    queue_total = metrics["queue_total"]
    starvation_arms = metrics["starvation_arms"]
    emergencies_yielded = metrics["emergencies_yielded"]
    emergencies_stuck = metrics["emergencies_stuck"]
    target_clear = task_cfg["target_clear_per_step"]
    target_wait = task_cfg["target_avg_wait"]

    score += 0.30 * min(1.0, cleared / max(target_clear, 1))
    avg_wait = wait_total / max(cleared, 1)
    score += 0.30 * max(0.0, 1.0 - (avg_wait / (target_wait * 2)))
    score += 0.20 * max(0.0, 1.0 - (queue_total / task_cfg.get("max_queue_threshold", 40)))
    total_em = emergencies_yielded + emergencies_stuck
    score += 0.10 * (emergencies_yielded / total_em if total_em > 0 else 1.0)
    score += 0.05 * (0.0 if phase_switched and action_duration < 15 else 0.7 if phase_switched else 1.0)
    score += 0.05 * max(0.0, 1.0 - (starvation_arms / 4))
    return round(min(1.0, max(0.0, score)), 4)
