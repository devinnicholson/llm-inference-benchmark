from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llmbench.workload import RequestSpec, Workload


PROFILES = {
    "short_chat",
    "long_rag",
    "coding",
    "batch_summary",
    "mixed_bursty",
}


def generate_workload(profile: str, requests: int, seed: int = 0) -> Workload:
    if profile not in PROFILES:
        profiles = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown profile {profile!r}; expected one of: {profiles}")
    if requests <= 0:
        raise ValueError("requests must be positive")

    rng = random.Random(seed)
    arrival_ms = 0.0
    specs: list[RequestSpec] = []

    for index in range(requests):
        prompt_tokens, output_tokens, priority, interarrival_ms = _sample_request(
            profile,
            index,
            rng,
        )
        if index > 0:
            arrival_ms += interarrival_ms

        specs.append(
            RequestSpec(
                id=f"{profile}-{index + 1:04d}",
                arrival_ms=round(arrival_ms, 3),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                priority=priority,
            )
        )

    return Workload(
        name=f"{profile}_{requests}_seed{seed}",
        description=_description(profile, requests, seed),
        requests=tuple(specs),
    )


def write_workload(workload: Workload, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(workload_to_dict(workload), indent=2) + "\n",
        encoding="utf-8",
    )


def workload_to_dict(workload: Workload) -> dict[str, Any]:
    return {
        "name": workload.name,
        "description": workload.description,
        "requests": [asdict(request) for request in workload.requests],
    }


def _sample_request(
    profile: str,
    index: int,
    rng: random.Random,
) -> tuple[int, int, str, float]:
    if profile == "short_chat":
        return rng.randint(64, 256), rng.randint(32, 128), "interactive", rng.uniform(8, 28)

    if profile == "long_rag":
        return rng.randint(1800, 6400), rng.randint(128, 480), "interactive", rng.uniform(35, 130)

    if profile == "coding":
        return rng.randint(700, 2600), rng.randint(256, 900), "interactive", rng.uniform(18, 75)

    if profile == "batch_summary":
        return rng.randint(1600, 5200), rng.randint(80, 320), "batch", rng.uniform(5, 20)

    if profile == "mixed_bursty":
        if index > 0 and index % 12 == 0:
            interarrival_ms = rng.uniform(120, 260)
        else:
            interarrival_ms = rng.uniform(1, 9)
        prompt_tokens, output_tokens, priority = _sample_mixed_shape(rng)
        return prompt_tokens, output_tokens, priority, interarrival_ms

    raise AssertionError(f"unhandled profile {profile}")


def _sample_mixed_shape(rng: random.Random) -> tuple[int, int, str]:
    draw = rng.random()
    if draw < 0.45:
        return rng.randint(64, 320), rng.randint(32, 160), "interactive"
    if draw < 0.70:
        return rng.randint(800, 2400), rng.randint(180, 700), "interactive"
    if draw < 0.90:
        return rng.randint(1800, 6400), rng.randint(128, 520), "interactive"
    return rng.randint(2000, 5600), rng.randint(80, 320), "batch"


def _description(profile: str, requests: int, seed: int) -> str:
    return (
        f"Generated {profile} workload with {requests} requests using seed {seed}. "
        "Token ranges are synthetic and intended for scheduler and KV-cache studies."
    )

