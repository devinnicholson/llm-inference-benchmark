from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root")
DEFAULT_WORKLOAD = "generated/mixed_bursty_32_seed568.json"
DEFAULT_MODEL = "llama-7b-gqa-fp16"
DEFAULT_CAPACITIES = "tight-1gb-kv"
DEFAULT_CONCURRENCY = "4"
DEFAULT_POLICIES = "deadline,memory-aware-deadline"
DEFAULT_SWEEP_OUTPUT = "results/modal-training-smoke"
DEFAULT_GPU_PROBE_OUTPUT = "results/modal-gpu-probe"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(ROOT / "src" / "llmbench", remote_path="/root/llmbench")
    .add_local_dir(ROOT / "configs", remote_path="/root/configs")
    .add_local_dir(ROOT / "workloads", remote_path="/root/workloads")
)
gpu_probe_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch", "numpy")
app = modal.App(name="llmbench-modal-training", image=image)


@app.function(timeout=300)
def run_capacity_sweep_remote(
    workload: str = DEFAULT_WORKLOAD,
    model: str = DEFAULT_MODEL,
    capacities: str = DEFAULT_CAPACITIES,
    concurrency: str = DEFAULT_CONCURRENCY,
    policies: str = DEFAULT_POLICIES,
) -> dict[str, Any]:
    from llmbench import run_sweep

    workload_path = REMOTE_ROOT / "workloads" / workload
    model_config_path = _model_config_path(model, root=REMOTE_ROOT)
    capacity_config_paths = [
        _capacity_config_path(capacity, root=REMOTE_ROOT)
        for capacity in _split_csv(capacities)
    ]
    results = run_sweep(
        workload_paths=[workload_path],
        model_config_paths=[model_config_path],
        capacity_config_paths=capacity_config_paths,
        max_concurrent_requests_values=[int(value) for value in _split_csv(concurrency)],
        scheduler_policies=_split_csv(policies),
    )
    return {
        "schema_version": 1,
        "execution": "modal",
        "results": [result.to_record() for result in results],
    }


@app.function(image=gpu_probe_image, gpu="T4", timeout=600)
def run_gpu_probe_remote() -> dict[str, Any]:
    import platform
    import time

    import torch

    nvidia_smi_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_smi_full = _run_command(["nvidia-smi"])

    cuda_available = torch.cuda.is_available()
    result: dict[str, Any] = {
        "schema_version": 1,
        "execution": "modal",
        "probe": "gpu",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": cuda_available,
        "nvidia_smi_query": nvidia_smi_query,
        "nvidia_smi_head": "\n".join(nvidia_smi_full.splitlines()[:8]),
    }
    if not cuda_available:
        return result

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    result.update(
        {
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
            "device_total_memory_mib": properties.total_memory / (1024 * 1024),
            "device_major": properties.major,
            "device_minor": properties.minor,
        }
    )

    x = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    y = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    started = time.perf_counter()
    z = x @ y
    torch.cuda.synchronize()
    result.update(
        {
            "matmul_shape": "1024x1024_fp16",
            "matmul_wall_ms": (time.perf_counter() - started) * 1000,
            "matmul_checksum": float(z[0, 0].item()),
        }
    )
    return result


@app.local_entrypoint()
def main(
    mode: str = "sweep",
    workload: str = DEFAULT_WORKLOAD,
    model: str = DEFAULT_MODEL,
    capacities: str = DEFAULT_CAPACITIES,
    concurrency: str = DEFAULT_CONCURRENCY,
    policies: str = DEFAULT_POLICIES,
    output_dir: str = "",
) -> None:
    if mode == "gpu-probe":
        payload = run_gpu_probe_remote.remote()
        output_path = Path(output_dir or DEFAULT_GPU_PROBE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "probe.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"cuda_available: {payload['cuda_available']}")
        if payload.get("device_name"):
            print(f"device: {payload['device_name']}")
        print(f"json: {json_path}")
        return

    if mode != "sweep":
        raise ValueError("mode must be 'sweep' or 'gpu-probe'")

    payload = run_capacity_sweep_remote.remote(
        workload=workload,
        model=model,
        capacities=capacities,
        concurrency=concurrency,
        policies=policies,
    )
    output_path = Path(output_dir or DEFAULT_SWEEP_OUTPUT)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "sweep-results.json"
    csv_path = output_path / "sweep-results.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_records_csv(csv_path, payload["results"])

    print(f"cases: {len(payload['results'])}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")


def _split_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("comma-separated argument must contain at least one value")
    return values


def _model_config_path(name: str, root: Path = ROOT) -> Path:
    filename = name if name.endswith(".json") else f"{name}.json"
    return root / "configs" / "models" / filename


def _capacity_config_path(name: str, root: Path = ROOT) -> Path:
    filename = name if name.endswith(".json") else f"{name}.json"
    return root / "configs" / "capacity" / filename


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0])
    extra_fields = sorted(set().union(*(record.keys() for record in records)) - set(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
        writer.writeheader()
        writer.writerows(records)


def _run_command(command: list[str]) -> str:
    import subprocess

    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as error:
        return error.output.strip()
