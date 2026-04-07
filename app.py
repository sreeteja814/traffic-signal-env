"""
app.py – FastAPI app for SmartTrafficEnv (OpenEnv compliant).
Serves the interactive UI at / and the OpenEnv API endpoints.

OpenEnv checker expects:
  POST /reset          → resets with default task (easy)
  POST /reset/{task_id}→ resets with specific task
  POST /step           → step with default task
  POST /step/{task_id} → step with specific task
  GET  /state          → state of default task
  GET  /spec           → openenv.yaml as JSON
"""
from __future__ import annotations
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from env import SmartTrafficEnv, Action, TASKS

app = FastAPI(
    title="Smart Traffic Signal Control — OpenEnv",
    description="RL environment: control a 4-way signalised intersection across 3 difficulty tiers.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

DEFAULT_TASK = "easy"

# In-memory session store: task_id → SmartTrafficEnv
_envs: dict[str, SmartTrafficEnv] = {}


def _make_env(task_id: str) -> SmartTrafficEnv:
    try:
        return SmartTrafficEnv(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _get_env(task_id: str) -> SmartTrafficEnv:
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"No active episode for '{task_id}'. Call POST /reset/{task_id} first.")
    return _envs[task_id]


# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return (STATIC / "index.html").read_text()


# ── INFO ──────────────────────────────────────────────────────────────────────
@app.get("/info")
def info():
    return {
        "name": "smart-traffic-signal-control",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "default_task": DEFAULT_TASK,
        "docs": "/docs",
        "spec": "/spec",
    }


@app.get("/tasks")
def list_tasks():
    return {
        tid: {
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
        }
        for tid, cfg in TASKS.items()
    }


@app.get("/spec")
def get_spec():
    spec_path = Path(__file__).parent / "openenv.yaml"
    with open(spec_path) as f:
        return yaml.safe_load(f)


# ── OPENENV RESET ─────────────────────────────────────────────────────────────
@app.post("/reset")
def reset_default():
    """Reset with default task (easy). OpenEnv checker calls this."""
    env = _make_env(DEFAULT_TASK)
    _envs[DEFAULT_TASK] = env
    obs = env.reset()
    return obs.model_dump()


@app.post("/reset/{task_id}")
def reset(task_id: str):
    """Reset with specific task_id."""
    env = _make_env(task_id)
    _envs[task_id] = env
    obs = env.reset()
    return obs.model_dump()


# ── OPENENV STEP ──────────────────────────────────────────────────────────────
@app.post("/step")
def step_default(action: Action):
    """Step with default task (easy). OpenEnv checker calls this."""
    env = _get_env(DEFAULT_TASK)
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result.model_dump()


@app.post("/step/{task_id}")
def step(task_id: str, action: Action):
    """Step with specific task_id."""
    env = _get_env(task_id)
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result.model_dump()


# ── OPENENV STATE ─────────────────────────────────────────────────────────────
@app.get("/state")
def state_default():
    """State of default task."""
    return _get_env(DEFAULT_TASK).state()


@app.get("/state/{task_id}")
def state(task_id: str):
    """State of specific task."""
    return _get_env(task_id).state()


# ── EPISODE RESULT ────────────────────────────────────────────────────────────
@app.get("/result")
def result_default():
    return _get_env(DEFAULT_TASK).episode_result().model_dump()


@app.get("/result/{task_id}")
def result(task_id: str):
    return _get_env(task_id).episode_result().model_dump()
