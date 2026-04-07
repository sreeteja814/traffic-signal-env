from __future__ import annotations
from .models import Action, EpisodeResult, Observation, StepResult
from .simulator import IntersectionSimulator
from .tasks import TASKS
from .graders import grade


class SmartTrafficEnv:
    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task '{task_id}'. Choose from {list(TASKS)}")
        self.task_id = task_id
        self.task_cfg = TASKS[task_id]
        self.sim = IntersectionSimulator(
            arrival_rates=self.task_cfg["arrival_rates"],
            seed=self.task_cfg["seed"],
            incident_prob=self.task_cfg["incident_prob"],
            emergency_rate=self.task_cfg["emergency_rate"],
        )
        self._current_phase = 0
        self._phase_elapsed = 0.0
        self._episode_step = 0
        self._rewards: list[float] = []
        self._infos: list[dict] = []
        self._done = False

    def reset(self) -> Observation:
        self.sim.reset(start_time_hours=self.task_cfg["start_time"])
        self._current_phase = 0
        self._phase_elapsed = 0.0
        self._episode_step = 0
        self._rewards = []
        self._infos = []
        self._done = False
        return self._make_obs()

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        prev_phase = self._current_phase
        switched = action.phase != prev_phase
        metrics = self.sim.step(action.phase, action.duration)
        self._current_phase = action.phase
        self._phase_elapsed = float(action.duration)
        self._episode_step += 1
        reward = grade(metrics, switched, action.duration, self.task_cfg)
        self._rewards.append(reward)
        info = {
            "step": self._episode_step,
            "reward": reward,
            "cleared": metrics["cleared"],
            "queue_total": metrics["queue_total"],
            "queue_by_arm": metrics["queue_by_arm"],
            "avg_wait": round(metrics["wait_total"] / max(metrics["cleared"], 1), 2),
            "incidents": metrics["incidents"],
            "emergencies_yielded": metrics["emergencies_yielded"],
            "phase_switched": switched,
        }
        self._infos.append(info)
        self._done = self._episode_step >= self.task_cfg["max_steps"]
        return StepResult(
            observation=None if self._done else self._make_obs(),
            reward=reward, done=self._done, info=info,
        )

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "episode_step": self._episode_step,
            "current_phase": self._current_phase,
            "phase_elapsed": self._phase_elapsed,
            "total_cleared": self.sim.total_cleared,
            "mean_reward_so_far": round(sum(self._rewards) / len(self._rewards), 4) if self._rewards else 0.0,
            "queue_by_arm": {d: len(self.sim.queues[d]) for d in self.sim.queues},
            "active_incidents": list(self.sim.incidents),
            "done": self._done,
        }

    def episode_result(self) -> EpisodeResult:
        cleared = max(self.sim.total_cleared, 1)
        return EpisodeResult(
            task_id=self.task_id,
            total_steps=self._episode_step,
            mean_reward=round(sum(self._rewards) / max(len(self._rewards), 1), 4),
            total_cleared=self.sim.total_cleared,
            avg_wait_time=round(self.sim.total_wait_accumulated / cleared, 2),
            rewards=self._rewards,
        )

    def _make_obs(self) -> Observation:
        return Observation(
            arms=self.sim.get_arm_states(),
            current_phase=self._current_phase,
            phase_elapsed=self._phase_elapsed,
            time_of_day=round(self.sim.sim_time / 3600.0, 4),
            episode_step=self._episode_step,
            total_cleared=self.sim.total_cleared,
        )
