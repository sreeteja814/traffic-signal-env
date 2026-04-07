TASKS: dict[str, dict] = {
    "easy": {
        "description": "Off-peak balanced flow – uniform arrivals, no incidents",
        "difficulty": 1,
        "arrival_rates": {"north": 5, "south": 5, "east": 5, "west": 5},
        "start_time": 7.0, "max_steps": 60, "seed": 42,
        "incident_prob": 0.0, "emergency_rate": 0.001,
        "target_clear_per_step": 8, "target_avg_wait": 25.0, "max_queue_threshold": 20,
    },
    "medium": {
        "description": "Rush-hour imbalance – N-S dominant commuter flow",
        "difficulty": 2,
        "arrival_rates": {"north": 18, "south": 16, "east": 6, "west": 7},
        "start_time": 8.0, "max_steps": 80, "seed": 42,
        "incident_prob": 0.0, "emergency_rate": 0.003,
        "target_clear_per_step": 14, "target_avg_wait": 50.0, "max_queue_threshold": 40,
    },
    "hard": {
        "description": "Peak flow + random road incidents + emergency vehicles",
        "difficulty": 3,
        "arrival_rates": {"north": 22, "south": 20, "east": 15, "west": 14},
        "start_time": 8.5, "max_steps": 100, "seed": 42,
        "incident_prob": 0.04, "emergency_rate": 0.008,
        "target_clear_per_step": 18, "target_avg_wait": 80.0, "max_queue_threshold": 60,
    },
}
