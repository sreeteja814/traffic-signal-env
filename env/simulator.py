from __future__ import annotations
import random, math
from .models import ArmState

DIRECTIONS = ["north", "south", "east", "west"]
PHASE_GREEN = {
    0: ["north", "south"],
    1: ["east", "west"],
    2: ["north"],
    3: ["east"],
}
DISCHARGE_PER_TICK = 2


class Vehicle:
    def __init__(self, arrival_time: float, has_emergency: bool = False):
        self.arrival_time = arrival_time
        self.wait_time = 0.0
        self.has_emergency = has_emergency


class IntersectionSimulator:
    def __init__(self, arrival_rates: dict, seed: int = 42,
                 incident_prob: float = 0.0, emergency_rate: float = 0.002):
        self._seed = seed
        self.arrival_rates = arrival_rates
        self.incident_prob = incident_prob
        self.emergency_rate = emergency_rate
        self.rng = random.Random(seed)
        self.tick_seconds = 5
        self.queues: dict[str, list[Vehicle]] = {d: [] for d in DIRECTIONS}
        self.incidents: set[str] = set()
        self.sim_time = 0.0
        self.total_cleared = 0
        self.total_wait_accumulated = 0.0

    def reset(self, start_time_hours: float = 8.0):
        self.rng = random.Random(self._seed)
        self.sim_time = start_time_hours * 3600.0
        self.total_cleared = 0
        self.total_wait_accumulated = 0.0
        self.incidents = set()
        self.queues = {d: [] for d in DIRECTIONS}
        for d in DIRECTIONS:
            n = int(self.arrival_rates[d] * 0.4)
            for _ in range(n):
                wait = self.rng.uniform(0, 20)
                v = Vehicle(self.sim_time - wait)
                v.wait_time = wait
                self.queues[d].append(v)

    def step(self, phase: int, duration: int) -> dict:
        ticks = max(1, duration // self.tick_seconds)
        green_dirs = PHASE_GREEN[phase]
        step_cleared = 0
        step_wait_total = 0.0
        step_stops = 0
        emergencies_yielded = 0

        for _ in range(ticks):
            self.sim_time += self.tick_seconds
            for d in DIRECTIONS:
                if d not in self.incidents and self.rng.random() < self.incident_prob / max(ticks, 1):
                    self.incidents.add(d)
                elif d in self.incidents and self.rng.random() < 0.3:
                    self.incidents.discard(d)
            for d in DIRECTIONS:
                lam = (self.arrival_rates[d] / 60.0) * self.tick_seconds
                for _ in range(self._poisson(lam)):
                    is_em = self.rng.random() < self.emergency_rate
                    self.queues[d].append(Vehicle(self.sim_time, is_em))
                    step_stops += 1
            for d in DIRECTIONS:
                for v in self.queues[d]:
                    v.wait_time += self.tick_seconds
            for d in green_dirs:
                if d in self.incidents:
                    continue
                n = min(len(self.queues[d]), DISCHARGE_PER_TICK)
                for _ in range(n):
                    v = self.queues[d].pop(0)
                    step_cleared += 1
                    step_wait_total += v.wait_time
                    self.total_wait_accumulated += v.wait_time
                    if v.has_emergency:
                        emergencies_yielded += 1

        self.total_cleared += step_cleared
        emergencies_stuck = sum(1 for d in DIRECTIONS for v in self.queues[d] if v.has_emergency)
        starvation = sum(1 for d in DIRECTIONS if self.queues[d] and self.queues[d][0].wait_time > 120)

        return {
            "cleared": step_cleared,
            "wait_total": step_wait_total,
            "stops": step_stops,
            "starvation_arms": starvation,
            "emergencies_yielded": emergencies_yielded,
            "emergencies_stuck": emergencies_stuck,
            "queue_total": sum(len(q) for q in self.queues.values()),
            "incidents": list(self.incidents),
            "queue_by_arm": {d: len(self.queues[d]) for d in DIRECTIONS},
        }

    def get_arm_states(self) -> list[ArmState]:
        states = []
        for d in DIRECTIONS:
            q = self.queues[d]
            avg_wait = sum(v.wait_time for v in q) / len(q) if q else 0.0
            density = min(1.0, len(q) / 25.0)
            speed = max(5.0, 50.0 * (1 - density))
            has_em = any(v.has_emergency for v in q)
            states.append(ArmState(direction=d, queue_length=len(q),
                avg_wait_time=round(avg_wait, 2), avg_speed=round(speed, 2),
                density=round(density, 3), has_emergency=has_em))
        return states

    def _poisson(self, lam: float) -> int:
        if lam <= 0: return 0
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1
