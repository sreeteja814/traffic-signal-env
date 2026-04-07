from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional


class ArmState(BaseModel):
    direction: Literal["north", "south", "east", "west"]
    queue_length: int = Field(ge=0)
    avg_wait_time: float = Field(ge=0.0)
    avg_speed: float = Field(ge=0.0)
    density: float = Field(ge=0.0, le=1.0)
    has_emergency: bool = False


class Observation(BaseModel):
    arms: list[ArmState]
    current_phase: int = Field(ge=0, le=3)
    phase_elapsed: float = Field(ge=0.0)
    time_of_day: float = Field(ge=0.0, le=24.0)
    episode_step: int = Field(ge=0)
    total_cleared: int = Field(ge=0)


class Action(BaseModel):
    phase: int = Field(ge=0, le=3)
    duration: int = Field(ge=10, le=90)


class StepResult(BaseModel):
    observation: Optional[Observation] = None
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    task_id: str
    total_steps: int
    mean_reward: float
    total_cleared: int
    avg_wait_time: float
    rewards: list[float]
