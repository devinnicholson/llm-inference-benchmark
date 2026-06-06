from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestSpec:
    id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int
    priority: str = "normal"
    deadline_ms: float | None = None


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    requests: tuple[RequestSpec, ...]


def load_workload(path: str | Path) -> Workload:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workload file must contain a JSON object")

    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("workload must contain a non-empty requests list")

    requests = tuple(_parse_request(item, index) for index, item in enumerate(raw_requests))
    return Workload(
        name=str(payload.get("name", Path(path).stem)),
        description=str(payload.get("description", "")),
        requests=tuple(sorted(requests, key=lambda request: request.arrival_ms)),
    )


def _parse_request(item: Any, index: int) -> RequestSpec:
    if not isinstance(item, dict):
        raise ValueError(f"request {index} must be an object")

    request_id = item.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"request {index} must have a non-empty string id")

    arrival_ms = _number(item, "arrival_ms", request_id)
    prompt_tokens = _positive_int(item, "prompt_tokens", request_id)
    output_tokens = _positive_int(item, "output_tokens", request_id)
    priority = item.get("priority", "normal")
    if not isinstance(priority, str) or not priority:
        raise ValueError(f"request {request_id} priority must be a non-empty string")
    deadline_ms = _optional_number(item, "deadline_ms", request_id)

    if arrival_ms < 0:
        raise ValueError(f"request {request_id} arrival_ms must be non-negative")
    if deadline_ms is not None and deadline_ms <= 0:
        raise ValueError(f"request {request_id} deadline_ms must be positive")

    return RequestSpec(
        id=request_id,
        arrival_ms=arrival_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        priority=priority,
        deadline_ms=deadline_ms,
    )


def _number(item: dict[str, Any], field: str, request_id: str) -> float:
    value = item.get(field)
    if not isinstance(value, int | float):
        raise ValueError(f"request {request_id} {field} must be numeric")
    return float(value)


def _optional_number(item: dict[str, Any], field: str, request_id: str) -> float | None:
    if field not in item:
        return None
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"request {request_id} {field} must be numeric")
    return float(value)


def _positive_int(item: dict[str, Any], field: str, request_id: str) -> int:
    value = item.get(field)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"request {request_id} {field} must be a positive integer")
    return value
