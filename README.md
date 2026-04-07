---
title: Smart Traffic Signal Control
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# 🚦 Smart Traffic Signal Control — OpenEnv RL Environment
> A fully OpenEnv-compliant reinforcement learning environment simulating real-world
> traffic signal control at a 4-way intersection — with a live interactive browser UI.

## Environment Description

**Smart Traffic Signal Control** is a real-world reinforcement learning environment
that simulates a signalised 4-way intersection. The agent acts as the traffic controller —
deciding which signal phase to activate and for how long — with the goal of minimising
vehicle wait times, reducing queue lengths, and responding correctly to emergency vehicles
and road incidents.

### The Problem

Urban traffic congestion causes billions of hours of lost productivity annually. Fixed-cycle
traffic signals (the traditional approach) waste green time on empty roads while vehicles
queue on congested arms. A learning agent can observe live queue conditions and dynamically
adapt signal timing — reducing average wait times by 30–50% compared to fixed cycles.

### How it works

```
┌─────────────────────────────────────────────────────────────┐
│                    4-WAY INTERSECTION                       │
│                                                             │
│                    ↑ North arm                              │
│                    │  queue, wait, density                  │
│                    │                                        │
│  West arm ─────── [INTERSECTION] ──────── East arm         │
│                    │                                        │
│                    │  queue, wait, density                  │
│                    ↓ South arm                              │
└─────────────────────────────────────────────────────────────┘

Each episode:
  1. Agent calls reset()      → receives initial Observation
  2. Agent picks phase (0-3) and duration (10-90s)
  3. Agent calls step(action) → simulator runs for duration seconds
  4. Vehicles arrive (Poisson), queue, discharge on green arms
  5. Agent receives reward (0.0-1.0) + next Observation
  6. Repeat until max_steps reached → done = True
```

### Key simulation features

- **Poisson vehicle arrivals** — statistically realistic, seeded for reproducibility
- **4 signal phases** — N-S straight, E-W straight, N-S left-turn, E-W left-turn
- **Vehicle queue discharge** — 2 vehicles cleared per 5-second tick per green arm
- **Emergency vehicles** — appear randomly, must be yielded to for bonus reward
- **Road incidents** (hard task) — randomly block arms, agent must adapt
- **Starvation detection** — vehicles waiting > 120s trigger penalty

---

## Project Structure

```
traffic-signal-env/
│
├── env/                        # Core OpenEnv package
│   ├── __init__.py             # Public API exports
│   ├── models.py               # Pydantic typed models
│   ├── simulator.py            # Intersection simulation engine
│   ├── graders.py              # Reward function (6 components)
│   ├── tasks.py                # 3 task configurations
│   └── environment.py          # SmartTrafficEnv: reset/step/state
│
├── static/
│   └── index.html              # Interactive browser UI
│
├── app.py                      # FastAPI server (UI + REST API)
├── baseline.py                 # 3 baseline agents, reproducible scores
├── openenv.yaml                # Full OpenEnv specification
├── requirements.txt            # Python dependencies
├── Dockerfile                  # HF Spaces ready, port 7860
└── README.md                   # This file
```

---

## Observation Space

At every step the agent receives an `Observation` object containing the full
state of the intersection.

### Top-level fields

| Field | Type | Range | Description |
|---|---|---|---|
| `arms` | `list[ArmState]` | length = 4 | Per-direction traffic state (see below) |
| `current_phase` | `int` | 0 – 3 | Currently active signal phase |
| `phase_elapsed` | `float` | ≥ 0.0 seconds | How long current phase has been running |
| `time_of_day` | `float` | 0.0 – 24.0 | Simulated hour (e.g. 8.5 = 08:30 am) |
| `episode_step` | `int` | ≥ 0 | Step counter within the current episode |
| `total_cleared` | `int` | ≥ 0 | Total vehicles cleared so far this episode |

### ArmState fields (one per direction)

| Field | Type | Range | Description |
|---|---|---|---|
| `direction` | `str` | north / south / east / west | Which arm this state belongs to |
| `queue_length` | `int` | ≥ 0 | Vehicles waiting at the stop line |
| `avg_wait_time` | `float` | ≥ 0.0 seconds | Mean wait time of all queued vehicles |
| `avg_speed` | `float` | 5 – 50 km/h | Approaching traffic speed (drops with density) |
| `density` | `float` | 0.0 – 1.0 | Occupancy ratio (1.0 = fully saturated arm) |
| `has_emergency` | `bool` | True / False | Emergency vehicle present in this arm |

### Example observation (JSON)

```json
{
  "arms": [
    {
      "direction": "north",
      "queue_length": 8,
      "avg_wait_time": 42.5,
      "avg_speed": 18.0,
      "density": 0.32,
      "has_emergency": false
    },
    {
      "direction": "south",
      "queue_length": 6,
      "avg_wait_time": 31.2,
      "avg_speed": 26.0,
      "density": 0.24,
      "has_emergency": false
    },
    {
      "direction": "east",
      "queue_length": 2,
      "avg_wait_time": 8.0,
      "avg_speed": 42.0,
      "density": 0.08,
      "has_emergency": true
    },
    {
      "direction": "west",
      "queue_length": 3,
      "avg_wait_time": 12.1,
      "avg_speed": 38.0,
      "density": 0.12,
      "has_emergency": false
    }
  ],
  "current_phase": 0,
  "phase_elapsed": 30.0,
  "time_of_day": 8.25,
  "episode_step": 5,
  "total_cleared": 42
}
```

---

## Action Space

The agent outputs an `Action` object at every step.

| Field | Type | Range | Description |
|---|---|---|---|
| `phase` | `int` | 0 – 3 | Signal phase to activate |
| `duration` | `int` | 10 – 90 seconds | How long to hold this phase |

### Signal phase meanings

| Phase | Green arms | Typical use case |
|---|---|---|
| `0` | North + South (straight) | Main N-S commuter flow |
| `1` | East + West (straight) | Main E-W cross traffic |
| `2` | North only (left-turn) | N-S dedicated turning movement |
| `3` | East only (left-turn) | E-W dedicated turning movement |

### Example action (JSON)

```json
{
  "phase": 0,
  "duration": 45
}
```

This tells the simulator: "Give North and South a green straight light for 45 seconds."

### Action constraints

- `phase` must be an integer in [0, 3] — values outside raise a validation error
- `duration` must be an integer in [10, 90] — too short wastes yellow-light transition time, too long starves other arms
- Switching phase incurs an implicit yellow-light penalty in the simulator

---

## Reward Function

Reward is always in **[0.0, 1.0]**. It is computed as a weighted sum of 6 components,
giving dense partial credit at every step so the agent always has a learning signal.

```
reward = (throughput × 0.30)
       + (wait_time  × 0.30)
       + (queue      × 0.20)
       + (emergency  × 0.10)
       + (phase_eff  × 0.05)
       + (starvation × 0.05)
```

### Component breakdown

| Component | Weight | Formula |
|---|---|---|
| **Throughput bonus** | 0.30 | `min(1.0, vehicles_cleared / target_clear_per_step)` |
| **Wait time score** | 0.30 | `max(0, 1 - avg_wait / (target_avg_wait × 2))` |
| **Queue reduction** | 0.20 | `max(0, 1 - queue_total / max_queue_threshold)` |
| **Emergency yield** | 0.10 | `emergencies_yielded / total_emergencies` (1.0 if none present) |
| **Phase efficiency** | 0.05 | `0.0` if switched with duration < 15s, `0.7` if switched, `1.0` if kept |
| **Starvation avoidance** | 0.05 | `max(0, 1 - starved_arms / 4)` |

### Why partial rewards matter

A binary reward (0 or 1) only tells the agent whether the whole episode succeeded.
With 6 weighted sub-signals the agent gets informative feedback every single step:

- Cleared 6 of the target 8 vehicles → gets `0.30 × 0.75 = 0.225` throughput credit
- Average wait was 60s against a 50s target → gets `0.30 × 0.40 = 0.12` wait credit
- No emergency vehicle this step → gets the full `0.10` emergency credit automatically

This makes the environment learnable from step 1 rather than requiring the agent to
stumble upon success through random exploration.

---

## Tasks

Three tasks represent real operational scenarios with increasing difficulty.

### Task 1 — Easy (`task_id="easy"`)

| Parameter | Value |
|---|---|
| Scenario | Off-peak, 7:00 am |
| Arrival rates | 5 vehicles/min on all 4 arms (balanced) |
| Max steps | 60 |
| Incidents | None |
| Emergency rate | 0.1% per vehicle |
| Target avg wait | 25 seconds |
| Target clear/step | 8 vehicles |

**What a good agent does:** Simple alternating N-S / E-W phases with moderate durations
(25–35s). Balanced arrival rates mean no arm should be heavily starved.

---

### Task 2 — Medium (`task_id="medium"`)

| Parameter | Value |
|---|---|
| Scenario | Morning rush hour, 8:00 am |
| Arrival rates | North 18, South 16, East 6, West 7 veh/min |
| Max steps | 80 |
| Incidents | None |
| Emergency rate | 0.3% per vehicle |
| Target avg wait | 50 seconds |
| Target clear/step | 14 vehicles |

**What a good agent does:** Heavily bias phase 0 (N-S green) — roughly 70% of time —
with short E-W phases only long enough to prevent starvation. A fixed-cycle agent
wastes 50% of green time on lightly-loaded E-W arms.

---

### Task 3 — Hard (`task_id="hard"`)

| Parameter | Value |
|---|---|
| Scenario | Peak hour with incidents, 8:30 am |
| Arrival rates | North 22, South 20, East 15, West 14 veh/min |
| Max steps | 100 |
| Incident probability | 4% chance per step an arm becomes blocked |
| Emergency rate | 0.8% per vehicle |
| Target avg wait | 80 seconds |
| Target clear/step | 18 vehicles |

**What a good agent does:** Dynamically detect incident-blocked arms and skip their
phase. Immediately preempt to the correct phase when an emergency vehicle appears.
Handle heavily loaded all-direction flow simultaneously.

---

## Baseline Scores

Run `python baseline.py` to reproduce these exact scores (seed=42, fully deterministic).

| Agent | Easy | Medium | Hard | Strategy |
|---|---|---|---|---|
| **Random** | 0.52 | 0.39 | 0.30 | Random phase + random duration each step |
| **Fixed Cycle** | 0.61 | 0.43 | 0.31 | Rotates 0→1→2→3 every 30s regardless of state |
| **Greedy** | 0.78 | 0.49 | 0.37 | Always greens the most congested N-S or E-W pair |
| **RL Target** | >0.85 | >0.65 | >0.55 | What a trained RL agent should achieve |

The performance gap between Fixed Cycle and Greedy widens on medium/hard because a
static rotation cannot adapt to imbalanced or disrupted flow.

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip
- Git
- Docker (optional, for containerised runs)

---

### Option 1 — Local Python (recommended for development)

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/smart-traffic-signal-control
cd smart-traffic-signal-control

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app --reload --port 7860

# 5. Open the browser UI
# Navigate to http://localhost:7860
```

---

### Option 2 — Docker

```bash
# 1. Build the image
docker build -t traffic-signal-env .

# 2. Run the container
docker run -p 7860:7860 traffic-signal-env

# 3. Open http://localhost:7860
```

---

### Option 3 — Docker Compose

```bash
docker-compose up --build
# Open http://localhost:7860
```

---

### Running the baseline script

```bash
# Run all 3 agents across all 3 tasks
python baseline.py

# Run a single task only
python baseline.py --task easy

# Run a single agent only
python baseline.py --agent greedy

# Verbose step-level output
python baseline.py --verbose
```

### Running the tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest httpx

# Run all 46 tests
pytest tests/ -v
```

---

## REST API Reference

The FastAPI server exposes the full OpenEnv interface as HTTP endpoints.
Interactive Swagger docs available at `/docs`.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive browser UI |
| `GET` | `/info` | Environment name, version, task list |
| `GET` | `/tasks` | All task configs with descriptions |
| `GET` | `/spec` | Full `openenv.yaml` as JSON |
| `POST` | `/reset/{task_id}` | Start new episode → returns Observation |
| `POST` | `/step/{task_id}` | Take action → returns StepResult |
| `GET` | `/state/{task_id}` | Current internal state snapshot |
| `GET` | `/result/{task_id}` | Episode summary (call after done=true) |
| `GET` | `/docs` | Swagger UI |

### curl examples

```bash
BASE=http://localhost:7860

# Check environment info
curl $BASE/info

# List all tasks
curl $BASE/tasks

# Start a new episode on the medium task
curl -X POST $BASE/reset/medium

# Apply phase 0 (N-S green) for 45 seconds
curl -X POST $BASE/step/medium \
  -H "Content-Type: application/json" \
  -d '{"phase": 0, "duration": 45}'

# Check internal state
curl $BASE/state/medium

# Get episode summary after done=true
curl $BASE/result/medium
```
---

## Deploy to Hugging Face Spaces

### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name it `smart-traffic-signal-control`
3. Select **Docker** as the SDK
4. Set visibility to **Public**
5. Click **Create Space**

### Step 2 — Push the code

```bash
pip install huggingface_hub
huggingface-cli login

git init
git add .
git commit -m "initial release"
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/smart-traffic-signal-control
git push hf main
```
---

## OpenEnv Spec

Full specification in [`openenv.yaml`](openenv.yaml):

```yaml
name: smart-traffic-signal-control
version: "1.0.0"
reward_range: [0.0, 1.0]

tasks:
  - id: easy    # difficulty 1, max_steps 60
  - id: medium  # difficulty 2, max_steps 80
  - id: hard    # difficulty 3, max_steps 100

observation_space:
  arms:          { type: array, length: 4 }
  current_phase: { type: int, min: 0, max: 3 }
  phase_elapsed: { type: float, unit: seconds }
  time_of_day:   { type: float, min: 0, max: 24 }
  episode_step:  { type: int }
  total_cleared: { type: int }

action_space:
  phase:    { type: int, min: 0, max: 3 }
  duration: { type: int, min: 10, max: 90, unit: seconds }

reward_components:
  throughput_bonus:     0.30
  wait_time_score:      0.30
  queue_reduction:      0.20
  emergency_yield:      0.10
  phase_efficiency:     0.05
  starvation_avoidance: 0.05
```
