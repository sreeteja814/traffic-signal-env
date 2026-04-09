"""
inference.py – Baseline inference script for SmartTrafficEnv (OpenEnv spec).

The OpenEnv checker requires this file to exist at the repo root.
It runs 3 baseline agents across all 3 tasks with seed=42 for
fully reproducible scores.

Usage
-----
    python inference.py                    # run all agents x all tasks
    python inference.py --task easy        # one task only
    python inference.py --agent greedy     # one agent only
    python inference.py --verbose          # step-level output
"""
from __future__ import annotations

import argparse
import json
import random

from env import SmartTrafficEnv, Action

SEED = 42


# ── Baseline agents ───────────────────────────────────────────────────────────

def random_agent(obs, rng: random.Random) -> Action:
    """Randomly selects phase and duration every step."""
    return Action(
        phase=rng.randint(0, 3),
        duration=rng.choice([15, 20, 30, 45, 60]),
    )


def fixed_cycle_agent(obs, _rng) -> Action:
    """Rotates phases 0→1→2→3 every 30 s regardless of traffic state."""
    return Action(phase=obs.episode_step % 4, duration=30)


def greedy_agent(obs, _rng) -> Action:
    """Always gives green to the most congested N-S or E-W pair.
    Preempts immediately for emergency vehicles."""
    arms = {a.direction: a for a in obs.arms}

    # emergency preemption
    for arm in obs.arms:
        if arm.has_emergency:
            phase = 0 if arm.direction in ["north", "south"] else 1
            return Action(phase=phase, duration=25)

    ns = arms["north"].queue_length + arms["south"].queue_length
    ew = arms["east"].queue_length  + arms["west"].queue_length
    phase    = 0 if ns >= ew else 1
    duration = max(15, min(75, max(ns, ew) * 3))
    return Action(phase=phase, duration=int(round(duration / 5) * 5))


AGENTS = {
    "random":      random_agent,
    "fixed_cycle": fixed_cycle_agent,
    "greedy":      greedy_agent,
}

TASKS = ["easy", "medium", "hard"]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_episode(task_id: str, agent_name: str, verbose: bool = False) -> dict:
    try:
        env = SmartTrafficEnv(task_id=task_id)
        rng = random.Random(SEED)
        obs = env.reset()
        rewards: list[float] = []

        while True:
            action = AGENTS[agent_name](obs, rng)
            result = env.step(action)
            rewards.append(result.reward)

            if verbose:
                info = result.info
                print(
                    f"  [{task_id}][{agent_name}] "
                    f"step={info['step']:>3}  phase={action.phase}  "
                    f"dur={action.duration:>2}s  "
                    f"cleared={info['cleared']:>3}  "
                    f"queue={info['queue_total']:>3}  "
                    f"reward={result.reward:.4f}"
                )

            if result.done:
                break
            obs = result.observation

        ep = env.episode_result()
        return {
            "task":          task_id,
            "agent":         agent_name,
            "mean_reward":   ep.mean_reward,
            "total_cleared": ep.total_cleared,
            "avg_wait_s":    ep.avg_wait_time,
            "steps":         ep.total_steps,
        }
    except Exception as e:
        print(f"[ERROR] task={task_id} agent={agent_name}: {e}")
        return {
            "task":          task_id,
            "agent":         agent_name,
            "mean_reward":   0.0,
            "total_cleared": 0,
            "avg_wait_s":    0.0,
            "steps":         0,
            "error":         str(e),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenEnv baseline inference script")
    parser.add_argument("--task",    choices=TASKS,          default=None,
                        help="Run a single task only")
    parser.add_argument("--agent",   choices=list(AGENTS),   default=None,
                        help="Run a single agent only")
    parser.add_argument("--verbose", action="store_true",
                        help="Print step-level details")
    args = parser.parse_args()

    tasks_to_run  = [args.task]  if args.task  else TASKS
    agents_to_run = [args.agent] if args.agent else list(AGENTS)

    results = []
    for task in tasks_to_run:
        for agent in agents_to_run:
            print(f"Running  task={task:<8}  agent={agent} ...")
            r = run_episode(task, agent, verbose=args.verbose)
            results.append(r)
            print(
                f"  -> mean_reward={r['mean_reward']:.4f}  "
                f"cleared={r['total_cleared']}  "
                f"avg_wait={r['avg_wait_s']:.1f}s\n"
            )

    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"{'Task':<10} {'Agent':<15} {'Mean Reward':>12} {'Cleared':>8} {'Avg Wait':>10}")
    print("-" * 62)
    for r in results:
        print(
            f"{r['task']:<10} {r['agent']:<15} "
            f"{r['mean_reward']:>12.4f} {r['total_cleared']:>8} "
            f"{r['avg_wait_s']:>9.1f}s"
        )
    print("=" * 62)

    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved -> baseline_results.json")


if __name__ == "__main__":
    main()
