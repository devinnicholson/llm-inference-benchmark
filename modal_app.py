from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import time
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
DEFAULT_INFERENCE_OUTPUT = "results/modal-tiny-inference"
DEFAULT_VLLM_OUTPUT = "results/modal-vllm-inference"
DEFAULT_VLLM_STREAMING_OUTPUT = "results/modal-vllm-streaming"
DEFAULT_VLLM_CONCURRENT_OUTPUT = "results/modal-vllm-concurrent"
DEFAULT_VLLM_CACHE_METRICS_PROBE_OUTPUT = "results/modal-vllm-cache-metrics-probe"
DEFAULT_VLLM_CACHE_METRICS_SMOKE_OUTPUT = "results/modal-vllm-cache-metrics-smoke"
DEFAULT_VLLM_CAPACITY_DIAGNOSTIC_OUTPUT = "results/modal-vllm-capacity-diagnostic"
DEFAULT_VLLM_SWEEP_OUTPUT = "results/modal-vllm-sweep"
DEFAULT_VLLM_SERVER_OUTPUT = "results/modal-vllm-server-streaming"
DEFAULT_VLLM_SERVER_CONCURRENT_OUTPUT = "results/modal-vllm-server-concurrent"
DEFAULT_VLLM_SERVER_SWEEP_OUTPUT = "results/modal-vllm-server-sweep"
DEFAULT_VLLM_SERVER_CLI_HELP_OUTPUT = "results/modal-vllm-server-cli-help"
DEFAULT_VLLM_SERVER_SWEEP_COMPARE_OUTPUT = "results/modal-vllm-server-sweep-compare"
DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT = "results/modal-vllm-server-async-paired"
DEFAULT_VLLM_SERVER_ASYNC_PAIRED_SERVER_FIRST_OUTPUT = (
    "results/modal-vllm-server-async-paired-server-first"
)
DEFAULT_VLLM_SERVER_ASYNC_PHASE_ORDER_COMPARE_OUTPUT = (
    "results/modal-vllm-server-async-phase-order-compare"
)
DEFAULT_VLLM_SERVER_ASYNC_PHASE_ORDER_COMPARE_DIRS = (
    "results/modal-vllm-server-async-phase-order-compare,"
    "results/modal-vllm-server-async-phase-order-compare-trial2,"
    "results/modal-vllm-server-async-phase-order-compare-trial3"
)
DEFAULT_VLLM_SERVER_ASYNC_MULTITRIAL_OUTPUT = (
    "results/modal-vllm-server-async-multitrial-aggregate"
)
DEFAULT_VLLM_SERVER_ASYNC_LONG_PHASE_ORDER_COMPARE_OUTPUT = (
    "results/modal-vllm-server-async-phase-order-compare-long"
)
DEFAULT_VLLM_SERVER_ASYNC_WORKLOAD_COMPARE_OUTPUT = (
    "results/modal-vllm-server-async-workload-compare"
)
DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_COMPARE_OUTPUT = (
    "results/modal-vllm-server-async-cache-control-phase-order-compare"
)
DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_COMPARE_DIRS = (
    "results/modal-vllm-server-async-qwen15b-l4-n32-batched-tokens60640-server-cache-off-smoke-r1,"
    "results/modal-vllm-server-async-qwen15b-l4-n32-batched-tokens60640-server-cache-off-server-first-smoke-r1,"
    "results/modal-vllm-server-async-qwen15b-l4-n32-batched-tokens60640-server-cache-on-smoke-r1,"
    "results/modal-vllm-server-async-qwen15b-l4-n32-batched-tokens60640-server-cache-on-server-first-smoke-r1"
)
DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_MULTITRIAL_OUTPUT = (
    "results/modal-vllm-server-async-cache-control-multitrial"
)
DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_COMPARE_AGGREGATE_DIRS = (
    "results/modal-vllm-server-async-qwen15b-l4-n32-batched-tokens60640-server-cache-control-phase-order-r1,"
    "results/modal-vllm-server-async-qwen15b-l4-n32-batched-tokens60640-server-cache-control-phase-order-r2"
)
DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_SERVER_ABSOLUTE_OUTPUT = (
    "results/modal-vllm-server-async-cache-control-server-absolute"
)
DEFAULT_VLLM_SERVER_ASYNC_LOG_SUMMARY_OUTPUT = (
    "results/modal-vllm-server-async-log-summary"
)
DEFAULT_VLLM_SERVER_ASYNC_LOG_SUMMARY_DIRS = (
    f"{DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT},"
    f"{DEFAULT_VLLM_SERVER_ASYNC_PAIRED_SERVER_FIRST_OUTPUT}"
)
DEFAULT_VLLM_PREFIX_CACHE_SWEEP_OUTPUT = "results/modal-vllm-prefix-cache-sweep"
DEFAULT_VLLM_PREFIX_CACHE_COMPARE_OUTPUT = "results/modal-vllm-prefix-cache-compare"
DEFAULT_VLLM_PREFIX_CACHE_PAIRED_OUTPUT = "results/modal-vllm-prefix-cache-paired"
DEFAULT_VLLM_PREFIX_CACHE_PAIRED_CACHE_FIRST_OUTPUT = (
    "results/modal-vllm-prefix-cache-paired-cache-first"
)
DEFAULT_VLLM_PREFIX_CACHE_PHASE_ORDER_COMPARE_OUTPUT = (
    "results/modal-vllm-prefix-cache-phase-order-compare"
)
DEFAULT_VLLM_PREFIX_CACHE_PROFILE_CONTROL_OUTPUT = (
    "results/modal-vllm-prefix-cache-profile-control"
)
DEFAULT_VLLM_PREFIX_CACHE_PROFILE_CONTROL_DIRS = (
    "results/modal-vllm-prefix-cache-long-control-profile-control"
)
DEFAULT_VLLM_PREFIX_CACHE_PROFILE_MULTITRIAL_OUTPUT = (
    "results/modal-vllm-prefix-cache-long-control-multitrial"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_METRICS_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-metrics"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_MERGE_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-merged-metrics"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_WARM_WINDOW_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-warm-window"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_NEUTRAL_WARMUP_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-neutral-warmup"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_STABILITY_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-stability-summary"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_WINDOW_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-window-summary"
)
DEFAULT_VLLM_PREFIX_CACHE_PROMPT_AUDIT_OUTPUT = (
    "results/modal-vllm-prefix-cache-prompt-audit"
)
DEFAULT_VLLM_PREFIX_CACHE_PROMPT_OVERLAP_SERVER_COMPARE_OUTPUT = (
    "results/modal-vllm-prefix-cache-prompt-overlap-server-compare"
)
DEFAULT_VLLM_PREFIX_CACHE_PROMPT_OVERLAP_AUDIT_DIRS = (
    "results/modal-vllm-prefix-cache-prompt-audit-mega-long-qwen15b-n32-seed3201,"
    "results/modal-vllm-prefix-cache-prompt-audit-mega-long-qwen15b-n32-seed3301"
)
DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_SERVER_ABSOLUTE_DIR = (
    "results/modal-vllm-server-async-qwen15b-l4-n32-batched-tokens60640-server-cache-control-server-absolute-r2"
)
DEFAULT_VLLM_SWEEP_REQUEST_COUNTS = "1,2,4,8"
DEFAULT_VLLM_SWEEP_PROMPT_PROFILES = "short,long"
DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS = "16,32"
DEFAULT_VLLM_SWEEP_REPEATS = 3
DEFAULT_VLLM_SWEEP_SEED = 568
DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
EXTRA_LONG_PREFIX_CONTEXT_REPEATS = 6
ULTRA_LONG_PREFIX_CONTEXT_REPEATS = 12
MEGA_LONG_PREFIX_CONTEXT_REPEATS = 24
DEFAULT_INFERENCE_PROMPT = "Explain KV cache in LLM inference in two concise sentences."
DEFAULT_CONCURRENT_PROMPTS = (
    "Explain KV cache pressure in LLM serving in two concise sentences.",
    "Explain why batching can improve GPU utilization for LLM inference.",
    "Explain the difference between prefill and decode in LLM inference.",
    "Explain why long prompts can hurt tail latency in an inference server.",
)
VALID_VLLM_PROMPT_PROFILES = {
    "short",
    "long",
    "mixed",
    "shared_prefix",
    "shared_prefix_long",
    "shared_prefix_long_variant",
    "shared_prefix_long_no_repeat_variant",
    "shared_prefix_extra_long_no_repeat_variant",
    "shared_prefix_ultra_long_no_repeat_variant",
    "shared_prefix_mega_long_no_repeat_variant",
    "matched_unique_prefix",
    "matched_unique_prefix_variant",
    "matched_unique_prefix_no_repeat_variant",
    "matched_unique_prefix_extra_long_no_repeat_variant",
    "matched_unique_prefix_ultra_long_no_repeat_variant",
    "matched_unique_prefix_mega_long_no_repeat_variant",
    "neutral_long",
    "neutral_extra_long",
    "neutral_ultra_long",
    "neutral_mega_long",
}
HF_CACHE_PATH = "/cache"
VLLM_CACHE_PATH = "/vllm-cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(ROOT / "src" / "llmbench", remote_path="/root/llmbench")
    .add_local_dir(ROOT / "configs", remote_path="/root/configs")
    .add_local_dir(ROOT / "workloads", remote_path="/root/workloads")
)
gpu_probe_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch", "numpy")
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("accelerate", "numpy", "safetensors", "torch", "transformers")
    .env({"HF_HOME": HF_CACHE_PATH})
)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface-hub==0.36.0", "httpx==0.28.1")
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
        }
    )
)
hf_cache_volume = modal.Volume.from_name("llmbench-hf-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("llmbench-vllm-cache", create_if_missing=True)
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


@app.function(image=vllm_image, gpu="L4", timeout=900)
def run_vllm_server_cli_help_remote() -> dict[str, Any]:
    import platform
    import subprocess

    import vllm

    started = time.perf_counter()
    commands = [
        ["vllm", "serve", "--help"],
        ["vllm", "serve", "--help=CacheConfig"],
        ["vllm", "serve", "--help=all"],
    ]
    captures = []
    for command in commands:
        command_started = time.perf_counter()
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=240,
            check=False,
        )
        text = _strip_trailing_whitespace_lines(completed.stdout)
        captures.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "elapsed_ms": (time.perf_counter() - command_started) * 1000,
                "help_text": text,
                **_parse_vllm_server_cli_help(text),
            }
        )
    help_text = "\n\n".join(
        "$ " + " ".join(capture["command"]) + "\n" + capture["help_text"]
        for capture in captures
    )
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-cli-help",
        "modal_gpu": "L4",
        "commands": commands,
        "returncodes": [capture["returncode"] for capture in captures],
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "help_captures": captures,
        "help_text": help_text,
        **_parse_vllm_server_cli_help(help_text),
    }


@app.function(
    image=inference_image,
    gpu="T4",
    timeout=900,
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
def run_tiny_inference_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 32,
    warmup_new_tokens: int = 4,
) -> dict[str, Any]:
    import platform

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if warmup_new_tokens <= 0:
        raise ValueError("warmup_new_tokens must be positive")

    nvidia_smi_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    dtype = torch.float16 if cuda_available else torch.float32

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model_load_ms = (time.perf_counter() - started) * 1000
    hf_cache_volume.commit()

    tokenization_started = time.perf_counter()
    tokenized, prompt_format = _tokenize_prompt(tokenizer, prompt)
    inputs = {key: value.to(device) for key, value in tokenized.items()}
    tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    _synchronize_if_cuda(torch)
    warmup_started = time.perf_counter()
    _manual_greedy_decode(
        model,
        inputs,
        max_new_tokens=warmup_new_tokens,
        torch_module=torch,
    )
    _synchronize_if_cuda(torch)
    warmup_ms = (time.perf_counter() - warmup_started) * 1000

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    decode_result = _manual_greedy_decode(
        model,
        inputs,
        max_new_tokens=max_new_tokens,
        torch_module=torch,
    )
    generated_ids = decode_result["generated_ids"]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    generated_tokens = int(generated_ids.shape[-1])
    decode_tokens = max(generated_tokens - 1, 0)
    decode_ms = float(decode_result["decode_ms"])
    ttft_ms = float(decode_result["ttft_ms"])
    total_generation_ms = ttft_ms + decode_ms

    peak_memory_allocated_mib = None
    if cuda_available:
        peak_memory_allocated_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "tiny-inference",
        "model_id": hf_model,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": int(max_new_tokens),
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "cuda_available": bool(cuda_available),
        "nvidia_smi_query": nvidia_smi_query,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "transformers_version": str(transformers.__version__),
        "dtype": str(dtype),
        "tokenizer_load_ms": tokenizer_load_ms,
        "model_load_ms": model_load_ms,
        "tokenization_ms": tokenization_ms,
        "warmup_ms": warmup_ms,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "total_generation_ms": total_generation_ms,
        "tpot_ms": decode_ms / decode_tokens if decode_tokens else 0.0,
        "output_tokens_per_second": generated_tokens / max(total_generation_ms / 1000, 1e-9),
        "peak_memory_allocated_mib": peak_memory_allocated_mib,
    }


@app.function(
    image=inference_image,
    timeout=900,
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
def run_vllm_prefix_cache_prompt_audit_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    kv_cache_block_size: int = 16,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if kv_cache_block_size <= 0:
        raise ValueError("kv_cache_block_size must be positive")
    _validate_vllm_prompt_profiles(prompt_profile_values)

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000
    hf_cache_volume.commit()

    payload = _build_vllm_prefix_cache_prompt_audit_payload(
        tokenizer=tokenizer,
        hf_model=hf_model,
        request_count_values=request_count_values,
        prompt_profile_values=prompt_profile_values,
        output_token_values=output_token_values,
        repeats=repeats,
        scenario_seed=scenario_seed,
        kv_cache_block_size=kv_cache_block_size,
    )
    payload["execution"] = "modal"
    payload["tokenizer_load_ms"] = tokenizer_load_ms
    return payload


def _run_vllm_prefix_cache_paired_payload(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    warmup_prompt_profile: str = "",
    phase_order: str = "cold_first",
    collect_cache_metrics: bool = False,
    kv_cache_metrics_sample: float = 1.0,
    max_num_batched_tokens_override: int = 0,
    modal_gpu_label: str = "T4",
) -> dict[str, Any]:
    import asyncio
    import contextlib
    import gc
    import inspect
    import io
    import logging
    import platform
    import random
    import re
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if kv_cache_metrics_sample <= 0:
        raise ValueError("kv_cache_metrics_sample must be positive")
    phase_order = phase_order.lower().replace("-", "_")
    if phase_order not in {"cold_first", "cache_first"}:
        raise ValueError("phase_order must be cold_first or cache_first")
    _validate_vllm_prompt_profiles(prompt_profile_values)
    warmup_prompt_profile_value = warmup_prompt_profile.strip().lower().replace("-", "_")
    if warmup_prompt_profile_value:
        _validate_vllm_prompt_profiles(
            [warmup_prompt_profile_value],
            label="warmup_prompt_profile",
        )

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    prompt_format_started = time.perf_counter()

    def build_scenario_specs(
        profiles: list[str],
        scenario_id_prefix: str = "",
    ) -> list[dict[str, Any]]:
        specs = []
        for prompt_profile in profiles:
            for max_new_tokens in output_token_values:
                for request_count in request_count_values:
                    prompt_records = []
                    prompts = _select_sweep_prompts(
                        request_count,
                        prompt_profile,
                        variant_index=scenario_seed,
                    )
                    for index, prompt in enumerate(prompts):
                        formatted_prompt, prompt_format = _format_prompt_for_generation(
                            tokenizer,
                            prompt,
                        )
                        prompt_records.append(
                            {
                                "request_id": (
                                    f"{prompt_profile}-out{max_new_tokens}-"
                                    f"n{request_count}-{index:02d}"
                                ),
                                "prompt": prompt,
                                "formatted_prompt": formatted_prompt,
                                "prompt_format": prompt_format,
                                "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
                            }
                        )
                    prompt_tokens = [
                        record["prompt_tokens"] for record in prompt_records
                    ]
                    scenario_id = (
                        f"{prompt_profile}_out{max_new_tokens}_n{request_count}"
                    )
                    if scenario_id_prefix:
                        scenario_id = f"{scenario_id_prefix}_{scenario_id}"
                    specs.append(
                        {
                            "scenario_id": scenario_id,
                            "prompt_profile": prompt_profile,
                            "request_count": request_count,
                            "max_new_tokens": max_new_tokens,
                            "prompt_format": prompt_records[0]["prompt_format"],
                            "prompt_tokens_min": min(prompt_tokens),
                            "prompt_tokens_max": max(prompt_tokens),
                            "prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
                            "total_prompt_tokens": sum(prompt_tokens),
                            "prompt_records": prompt_records,
                        }
                    )
        return specs

    scenario_specs = build_scenario_specs(prompt_profile_values)
    warmup_scenario_specs = (
        build_scenario_specs(
            [warmup_prompt_profile_value],
            scenario_id_prefix="warmup",
        )
        if warmup_prompt_profile_value
        else scenario_specs
    )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    max_request_count = max(request_count_values)
    max_output_tokens = max(output_token_values)
    all_shape_specs = scenario_specs + warmup_scenario_specs
    max_prompt_tokens = max(spec["prompt_tokens_max"] for spec in all_shape_specs)
    max_model_len = max(1024, max_prompt_tokens + max_output_tokens + 32)
    default_max_num_batched_tokens = max(2048, max_request_count * max_model_len)
    max_num_batched_tokens, max_num_batched_tokens_source = (
        _resolve_max_num_batched_tokens(
            default_max_num_batched_tokens,
            max_num_batched_tokens_override,
            "max_num_batched_tokens_override",
        )
    )
    scenario_plan = []
    for repeat_index in range(repeats):
        for spec in scenario_specs:
            scenario_plan.append((repeat_index, spec))
    random.Random(scenario_seed).shuffle(scenario_plan)
    plan_summary = [
        {
            "run_order": run_order,
            "repeat_index": repeat_index,
            "scenario_id": spec["scenario_id"],
        }
        for run_order, (repeat_index, spec) in enumerate(scenario_plan)
    ]

    async def call_engine_log_stats(engine: Any, label: str) -> dict[str, Any]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        captured_logs: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_logs.append(self.format(record))

        handler = CaptureHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        capture_loggers = [
            logging.getLogger(),
            logging.getLogger("vllm"),
            logging.getLogger("vllm.v1.metrics.loggers"),
        ]
        started = time.perf_counter()
        try:
            for logger in capture_loggers:
                logger.addHandler(handler)
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                result = engine.do_log_stats()
                if inspect.isawaitable(result):
                    result = await result
            return {
                "label": label,
                "ok": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(result).__name__,
                "return_repr": repr(result)[:1000],
                "stdout": stdout_buffer.getvalue()[-4000:],
                "stderr": stderr_buffer.getvalue()[-4000:],
                "captured_logs": captured_logs[-20:],
            }
        except Exception as error:  # pragma: no cover - remote metrics probe
            return {
                "label": label,
                "ok": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(error).__name__,
                "return_repr": str(error)[:1000],
                "stdout": stdout_buffer.getvalue()[-4000:],
                "stderr": stderr_buffer.getvalue()[-4000:],
                "captured_logs": captured_logs[-20:],
            }
        finally:
            for logger in capture_loggers:
                logger.removeHandler(handler)

    def call_with_captured_logs(label: str, func: Any) -> tuple[Any, dict[str, Any]]:
        captured_logs: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_logs.append(self.format(record))

        handler = CaptureHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        capture_loggers = [
            logging.getLogger(),
            logging.getLogger("vllm"),
            logging.getLogger("vllm.v1"),
        ]
        started = time.perf_counter()
        try:
            for logger in capture_loggers:
                logger.addHandler(handler)
            result = func()
            return result, {
                "label": label,
                "ok": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(result).__name__,
                "return_repr": repr(result)[:1000],
                "stdout": "",
                "stderr": "",
                "captured_logs": captured_logs[-80:],
            }
        finally:
            for logger in capture_loggers:
                logger.removeHandler(handler)

    def parse_engine_log_metrics(log_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        seen_lines = set()
        patterns = {
            "avg_prompt_throughput_tokens_per_s": r"Avg prompt throughput: ([0-9.]+) tokens/s",
            "avg_generation_throughput_tokens_per_s": (
                r"Avg generation throughput: ([0-9.]+) tokens/s"
            ),
            "gpu_kv_cache_usage_pct": r"GPU KV cache usage: ([0-9.]+)%",
            "prefix_cache_hit_rate_pct": r"Prefix cache hit rate: ([0-9.]+)%",
        }
        for call in log_stats:
            for line in call.get("captured_logs", []):
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                row: dict[str, Any] = {
                    "label": call["label"],
                    "line": line,
                }
                matched = False
                for field, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        row[field] = float(match.group(1))
                        matched = True
                if matched:
                    rows.append(row)
        return rows

    def latest_cache_metric_value(
        metrics: list[dict[str, Any]],
        field: str,
    ) -> float | None:
        for row in reversed(metrics):
            value = row.get(field)
            if value is not None:
                return float(value)
        return None

    def snapshot_prefix_cache_counters(engine: Any, label: str) -> dict[str, Any]:
        sources = []

        def collect_metric_sources(target: Any, path: str) -> None:
            stack = [(target, path)]
            seen: set[int] = set()
            while stack:
                current, current_path = stack.pop()
                if current is None or id(current) in seen:
                    continue
                seen.add(id(current))
                metrics = getattr(current, "prefix_caching_metrics", None)
                if metrics is not None:
                    requests = getattr(metrics, "aggregated_requests", None)
                    queries = getattr(metrics, "aggregated_query_total", None)
                    hits = getattr(metrics, "aggregated_query_hit", None)
                    sources.append(
                        {
                            "path": current_path,
                            "logger_type": type(current).__name__,
                            "requests": (
                                int(requests) if requests is not None else None
                            ),
                            "queries": int(queries) if queries is not None else None,
                            "hits": int(hits) if hits is not None else None,
                            "hit_rate_pct": _prefix_cache_counter_hit_rate_pct(
                                hits,
                                queries,
                            ),
                            "empty": bool(getattr(metrics, "empty", False)),
                        }
                    )

                per_engine_loggers = getattr(
                    current,
                    "per_engine_stat_loggers",
                    None,
                )
                if isinstance(per_engine_loggers, dict):
                    for engine_index, logger_value in per_engine_loggers.items():
                        stack.append(
                            (
                                logger_value,
                                f"{current_path}.per_engine[{engine_index}]",
                            )
                        )

                child_loggers = getattr(current, "stat_loggers", None)
                if isinstance(child_loggers, list):
                    for child_index, logger_value in enumerate(child_loggers):
                        stack.append(
                            (
                                logger_value,
                                f"{current_path}.stat_loggers[{child_index}]",
                            )
                        )

        try:
            manager = getattr(engine, "logger_manager", None)
            stat_loggers = list(getattr(manager, "stat_loggers", []) or [])
            for logger_index, stat_logger in enumerate(stat_loggers):
                collect_metric_sources(
                    stat_logger,
                    f"logger_manager.stat_loggers[{logger_index}]",
                )
            total_requests = sum(
                row["requests"] or 0 for row in sources
            )
            total_queries = sum(row["queries"] or 0 for row in sources)
            total_hits = sum(row["hits"] or 0 for row in sources)
            return {
                "label": label,
                "ok": True,
                "error": "",
                "source_count": len(sources),
                "sources": sources,
                "total_requests": total_requests,
                "total_queries": total_queries,
                "total_hits": total_hits,
                "hit_rate_pct": _prefix_cache_counter_hit_rate_pct(
                    total_hits,
                    total_queries,
                ),
            }
        except Exception as error:  # pragma: no cover - remote object probe
            return {
                "label": label,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "source_count": 0,
                "sources": sources,
                "total_requests": None,
                "total_queries": None,
                "total_hits": None,
                "hit_rate_pct": None,
            }

    async def run_engine_phase(
        *,
        enable_prefix_caching: bool,
        phase_label: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        metrics_args = (
            {
                "kv_cache_metrics": True,
                "kv_cache_metrics_sample": kv_cache_metrics_sample,
                "disable_log_stats": False,
            }
            if collect_cache_metrics
            else {}
        )
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_request_count,
            gpu_memory_utilization=0.50,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=True,
            trust_remote_code=False,
            **metrics_args,
        )
        engine, engine_init_log_stats = call_with_captured_logs(
            f"{phase_label}_engine_init",
            lambda: AsyncLLM.from_engine_args(engine_args),
        )
        engine_capacity = _snapshot_vllm_engine_capacity(
            engine=engine,
            engine_args=engine_args,
            phase_label=phase_label,
            engine_init_log_stats=engine_init_log_stats,
            requested_max_model_len=max_model_len,
            requested_max_num_batched_tokens=max_num_batched_tokens,
            requested_max_num_seqs=max_request_count,
            requested_gpu_memory_utilization=0.50,
        )
        engine_load_ms = (time.perf_counter() - started) * 1000

        async def stream_one(
            run_id: str,
            record: dict[str, Any],
            sampling_params: Any,
            batch_started: float,
        ) -> dict[str, Any]:
            request_started = time.perf_counter()
            first_chunk_ms: float | None = None
            finished_ms: float | None = None
            generated_text_parts = []
            generated_tokens = 0
            chunk_count = 0

            async for output in engine.generate(
                request_id=f"{phase_label}-{run_id}-{record['request_id']}",
                prompt=record["formatted_prompt"],
                sampling_params=sampling_params,
            ):
                now = time.perf_counter()
                elapsed_ms = (now - request_started) * 1000
                chunk_count += 1
                for completion in output.outputs:
                    text = completion.text
                    token_ids = list(completion.token_ids or [])
                    if first_chunk_ms is None and (text or token_ids):
                        first_chunk_ms = elapsed_ms
                    if text:
                        generated_text_parts.append(text)
                    generated_tokens += len(token_ids)
                if output.finished:
                    finished_ms = elapsed_ms
                    break

            stream_wall_ms = finished_ms or ((time.perf_counter() - request_started) * 1000)
            decode_tokens_after_first = max(generated_tokens - 1, 0)
            decode_after_first_ms = (
                stream_wall_ms - first_chunk_ms
                if first_chunk_ms is not None
                else None
            )
            return {
                "request_id": record["request_id"],
                "prompt_tokens": record["prompt_tokens"],
                "first_chunk_ms": first_chunk_ms,
                "stream_wall_ms": stream_wall_ms,
                "decode_after_first_ms": decode_after_first_ms,
                "stream_tpot_ms": (
                    decode_after_first_ms / decode_tokens_after_first
                    if decode_after_first_ms is not None and decode_tokens_after_first
                    else None
                ),
                "generated_tokens": generated_tokens,
                "total_sequence_tokens": record["prompt_tokens"] + generated_tokens,
                "chunk_count": chunk_count,
                "batch_finished_ms": (time.perf_counter() - batch_started) * 1000,
                "generated_text": "".join(generated_text_parts),
            }

        async def run_scenario(
            spec: dict[str, Any],
            repeat_index: int,
            run_order: int,
            max_tokens_override: int | None = None,
        ) -> dict[str, Any]:
            max_tokens = max_tokens_override or spec["max_new_tokens"]
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
            run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
            counter_before = (
                snapshot_prefix_cache_counters(
                    engine,
                    f"{phase_label}_{run_id}_{spec['scenario_id']}_before",
                )
                if collect_cache_metrics
                else None
            )
            batch_started = time.perf_counter()
            request_results = await asyncio.gather(
                *(
                    stream_one(
                        run_id,
                        record,
                        sampling_params,
                        batch_started,
                    )
                    for record in spec["prompt_records"]
                )
            )
            batch_wall_ms = (time.perf_counter() - batch_started) * 1000
            counter_after = (
                snapshot_prefix_cache_counters(
                    engine,
                    f"{phase_label}_{run_id}_{spec['scenario_id']}_after",
                )
                if collect_cache_metrics
                else None
            )
            counter_delta = _prefix_cache_counter_delta(
                counter_before,
                counter_after,
            )
            summary = _summarize_stream_requests(request_results, batch_wall_ms)
            cache_log_stats = None
            cache_metrics: list[dict[str, Any]] = []
            if collect_cache_metrics:
                cache_log_stats = await call_engine_log_stats(
                    engine,
                    f"{phase_label}_{run_id}_{spec['scenario_id']}",
                )
                cache_metrics = parse_engine_log_metrics([cache_log_stats])
            return {
                "run_id": run_id,
                "repeat_index": repeat_index,
                "run_order": run_order,
                "scenario_id": spec["scenario_id"],
                "prompt_profile": spec["prompt_profile"],
                "request_count": spec["request_count"],
                "max_new_tokens": max_tokens,
                "prompt_format": spec["prompt_format"],
                "prompt_tokens_min": spec["prompt_tokens_min"],
                "prompt_tokens_max": spec["prompt_tokens_max"],
                "prompt_tokens_mean": spec["prompt_tokens_mean"],
                "total_prompt_tokens": spec["total_prompt_tokens"],
                "estimated_peak_sequence_tokens": sum(
                    request["total_sequence_tokens"] for request in request_results
                ),
                **summary,
                "cache_metrics_log_stats": cache_log_stats,
                "cache_metrics": cache_metrics,
                "prefix_cache_counter_before": counter_before,
                "prefix_cache_counter_after": counter_after,
                "prefix_cache_counter_delta": counter_delta,
                "prefix_cache_counter_requests": counter_delta["requests"],
                "prefix_cache_counter_queries": counter_delta["queries"],
                "prefix_cache_counter_hits": counter_delta["hits"],
                "prefix_cache_counter_hit_rate_pct": counter_delta["hit_rate_pct"],
                "prefix_cache_hit_rate_pct": latest_cache_metric_value(
                    cache_metrics,
                    "prefix_cache_hit_rate_pct",
                ),
                "gpu_kv_cache_usage_pct": latest_cache_metric_value(
                    cache_metrics,
                    "gpu_kv_cache_usage_pct",
                ),
                "avg_prompt_throughput_tokens_per_s": latest_cache_metric_value(
                    cache_metrics,
                    "avg_prompt_throughput_tokens_per_s",
                ),
                "avg_generation_throughput_tokens_per_s": latest_cache_metric_value(
                    cache_metrics,
                    "avg_generation_throughput_tokens_per_s",
                ),
                **_vllm_engine_capacity_run_fields(engine_capacity),
                "requests": request_results,
            }

        async def run_warmups() -> dict[str, Any]:
            started = time.perf_counter()
            warmup_summaries = []
            run_order = 0
            for warmup_index in range(warmup_runs):
                for spec in warmup_scenario_specs:
                    result = await run_scenario(
                        spec,
                        repeat_index=-(warmup_index + 1),
                        run_order=run_order,
                        max_tokens_override=1,
                    )
                    warmup_summaries.append(
                        {
                            "warmup_index": warmup_index,
                            "scenario_id": spec["scenario_id"],
                            "prompt_profile": spec["prompt_profile"],
                            "request_count": spec["request_count"],
                            "max_new_tokens": 1,
                            "prompt_tokens_mean": spec["prompt_tokens_mean"],
                            "total_prompt_tokens": spec["total_prompt_tokens"],
                            "batch_wall_ms": result["batch_wall_ms"],
                            "total_output_tokens": result["total_output_tokens"],
                        }
                    )
                    run_order += 1
            return {
                "warmup_runs": warmup_runs,
                "warmup_scenario_runs": len(warmup_summaries),
                "warmup_wall_ms": (time.perf_counter() - started) * 1000,
                "total_output_tokens": sum(row["total_output_tokens"] for row in warmup_summaries),
                "summaries": warmup_summaries,
            }

        try:
            initial_prefix_cache_counter = (
                snapshot_prefix_cache_counters(
                    engine,
                    f"{phase_label}_after_engine_load",
                )
                if collect_cache_metrics
                else None
            )
            warmup_result = await run_warmups()
            warmup_prefix_cache_counter = (
                snapshot_prefix_cache_counters(
                    engine,
                    f"{phase_label}_after_warmup",
                )
                if collect_cache_metrics
                else None
            )
            warmup_cache_log_stats = None
            warmup_cache_metrics: list[dict[str, Any]] = []
            if collect_cache_metrics:
                warmup_cache_log_stats = await call_engine_log_stats(
                    engine,
                    f"{phase_label}_after_warmup",
                )
                warmup_cache_metrics = parse_engine_log_metrics([warmup_cache_log_stats])
            scenario_runs = []
            for run_order, (repeat_index, spec) in enumerate(scenario_plan):
                scenario_runs.append(
                    await run_scenario(
                        spec,
                        repeat_index=repeat_index,
                        run_order=run_order,
                    )
                )
        finally:
            engine.shutdown()

        return {
            "phase_label": phase_label,
            "enable_prefix_caching": enable_prefix_caching,
            "engine_load_ms": engine_load_ms,
            "engine_capacity": engine_capacity,
            "engine_init_log_stats": engine_init_log_stats,
            "warmup": warmup_result,
            "initial_prefix_cache_counter": initial_prefix_cache_counter,
            "warmup_prefix_cache_counter": warmup_prefix_cache_counter,
            "warmup_prefix_cache_counter_delta": _prefix_cache_counter_delta(
                initial_prefix_cache_counter,
                warmup_prefix_cache_counter,
            ),
            "warmup_cache_metrics_log_stats": warmup_cache_log_stats,
            "warmup_cache_metrics": warmup_cache_metrics,
            "scenario_runs": scenario_runs,
        }

    def release_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2.0)

    if phase_order == "cold_first":
        cold_result = asyncio.run(
            run_engine_phase(enable_prefix_caching=False, phase_label="cold")
        )
        release_cuda_memory()
        cache_result = asyncio.run(
            run_engine_phase(enable_prefix_caching=True, phase_label="cache")
        )
    else:
        cache_result = asyncio.run(
            run_engine_phase(enable_prefix_caching=True, phase_label="cache")
        )
        release_cuda_memory()
        cold_result = asyncio.run(
            run_engine_phase(enable_prefix_caching=False, phase_label="cold")
        )

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    paired_runs = _make_vllm_prefix_cache_paired_rows(
        cold_result["scenario_runs"],
        cache_result["scenario_runs"],
    )
    paired_scenarios = _aggregate_vllm_prefix_cache_paired_scenarios(paired_runs)
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-prefix-cache-paired",
        "backend": "vllm-paired-prefix-cache",
        "modal_gpu": modal_gpu_label,
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "warmup_runs": int(warmup_runs),
        "warmup_prompt_profile": warmup_prompt_profile_value or None,
        "warmup_scenario_count": len(warmup_scenario_specs) if warmup_runs else 0,
        "phase_order": phase_order,
        "collect_cache_metrics": bool(collect_cache_metrics),
        "kv_cache_metrics_sample": float(kv_cache_metrics_sample),
        "scenario_count": len(paired_scenarios),
        "paired_run_count": len(paired_runs),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "default_max_num_batched_tokens": default_max_num_batched_tokens,
        "max_num_batched_tokens_source": max_num_batched_tokens_source,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "cold_engine_load_ms": cold_result["engine_load_ms"],
        "cache_engine_load_ms": cache_result["engine_load_ms"],
        "cold_engine_capacity": cold_result["engine_capacity"],
        "cache_engine_capacity": cache_result["engine_capacity"],
        "cold_warmup": cold_result["warmup"],
        "cache_warmup": cache_result["warmup"],
        "cold_warmup_cache_metrics": cold_result["warmup_cache_metrics"],
        "cache_warmup_cache_metrics": cache_result["warmup_cache_metrics"],
        "cold_initial_prefix_cache_counter": cold_result[
            "initial_prefix_cache_counter"
        ],
        "cache_initial_prefix_cache_counter": cache_result[
            "initial_prefix_cache_counter"
        ],
        "cold_warmup_prefix_cache_counter": cold_result[
            "warmup_prefix_cache_counter"
        ],
        "cache_warmup_prefix_cache_counter": cache_result[
            "warmup_prefix_cache_counter"
        ],
        "cold_warmup_prefix_cache_counter_delta": cold_result[
            "warmup_prefix_cache_counter_delta"
        ],
        "cache_warmup_prefix_cache_counter_delta": cache_result[
            "warmup_prefix_cache_counter_delta"
        ],
        "cold_warmup_cache_metrics_log_stats": cold_result[
            "warmup_cache_metrics_log_stats"
        ],
        "cache_warmup_cache_metrics_log_stats": cache_result[
            "warmup_cache_metrics_log_stats"
        ],
        "scenario_plan": plan_summary,
        "summary": _vllm_prefix_cache_paired_summary_rows(paired_scenarios),
        "paired_runs": paired_runs,
        "paired_scenarios": paired_scenarios,
        "cold_scenario_runs": cold_result["scenario_runs"],
        "cache_scenario_runs": cache_result["scenario_runs"],
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "mean_cache_to_cold_throughput_ratio": _mean_present(
            row["cache_to_cold_output_tokens_per_second_ratio"] for row in paired_runs
        ),
        "mean_cache_to_cold_first_event_ratio": _mean_present(
            row["cache_to_cold_p95_first_event_ms_ratio"] for row in paired_runs
        ),
        "mean_cache_to_cold_latency_ratio": _mean_present(
            row["cache_to_cold_p95_latency_ms_ratio"] for row in paired_runs
        ),
        "mean_cache_to_cold_tpot_ratio": _mean_present(
            row["cache_to_cold_p95_stream_tpot_ms_ratio"] for row in paired_runs
        ),
        "note": (
            "One Modal worker runs two in-process AsyncLLM engines with prefix "
            f"caching off and on in phase order {phase_order}. Paired rows "
            "compare matching scenario_id and repeat_index."
        ),
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_prefix_cache_paired_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    warmup_prompt_profile: str = "",
    phase_order: str = "cold_first",
    collect_cache_metrics: bool = False,
    kv_cache_metrics_sample: float = 1.0,
    max_num_batched_tokens_override: int = 0,
) -> dict[str, Any]:
    return _run_vllm_prefix_cache_paired_payload(
        hf_model=hf_model,
        request_counts=request_counts,
        prompt_profiles=prompt_profiles,
        output_tokens=output_tokens,
        repeats=repeats,
        scenario_seed=scenario_seed,
        warmup_runs=warmup_runs,
        warmup_prompt_profile=warmup_prompt_profile,
        phase_order=phase_order,
        collect_cache_metrics=collect_cache_metrics,
        kv_cache_metrics_sample=kv_cache_metrics_sample,
        max_num_batched_tokens_override=max_num_batched_tokens_override,
        modal_gpu_label="T4",
    )


@app.function(
    image=vllm_image,
    gpu="L4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_prefix_cache_paired_l4_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    warmup_prompt_profile: str = "",
    phase_order: str = "cold_first",
    collect_cache_metrics: bool = False,
    kv_cache_metrics_sample: float = 1.0,
    max_num_batched_tokens_override: int = 0,
) -> dict[str, Any]:
    return _run_vllm_prefix_cache_paired_payload(
        hf_model=hf_model,
        request_counts=request_counts,
        prompt_profiles=prompt_profiles,
        output_tokens=output_tokens,
        repeats=repeats,
        scenario_seed=scenario_seed,
        warmup_runs=warmup_runs,
        warmup_prompt_profile=warmup_prompt_profile,
        phase_order=phase_order,
        collect_cache_metrics=collect_cache_metrics,
        kv_cache_metrics_sample=kv_cache_metrics_sample,
        max_num_batched_tokens_override=max_num_batched_tokens_override,
        modal_gpu_label="L4",
    )


def _select_vllm_prefix_cache_paired_remote(modal_gpu: str) -> Any:
    normalized_gpu = modal_gpu.strip().upper().replace("_", "-")
    if normalized_gpu == "T4":
        return run_vllm_prefix_cache_paired_remote
    if normalized_gpu == "L4":
        return run_vllm_prefix_cache_paired_l4_remote
    raise ValueError("modal_gpu must be one of: T4, L4")


def _run_vllm_capacity_diagnostic_payload(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_prompt_profile: str = "",
    prefix_cache_modes: str = "false",
    max_num_batched_tokens_values: str = "",
    gpu_memory_utilization: float = 0.50,
    modal_gpu_label: str = "T4",
) -> dict[str, Any]:
    import platform
    import subprocess
    import sys
    import time

    from transformers import AutoTokenizer

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    prefix_cache_values = [
        _parse_bool_choice(value, "prefix_cache_modes")
        for value in _split_csv(prefix_cache_modes)
    ]
    override_max_num_batched_tokens_values = (
        _split_positive_int_csv(
            max_num_batched_tokens_values,
            "max_num_batched_tokens_values",
        )
        if max_num_batched_tokens_values.strip()
        else []
    )
    if gpu_memory_utilization <= 0 or gpu_memory_utilization > 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    _validate_vllm_prompt_profiles(prompt_profile_values)
    warmup_prompt_profile_value = warmup_prompt_profile.strip().lower().replace("-", "_")
    shape_profiles = list(prompt_profile_values)
    if warmup_prompt_profile_value:
        _validate_vllm_prompt_profiles(
            [warmup_prompt_profile_value],
            label="warmup_prompt_profile",
        )
        shape_profiles.append(warmup_prompt_profile_value)

    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def max_prompt_tokens_for_shape(request_count: int) -> int:
        prompt_token_counts = []
        for profile in shape_profiles:
            prompts = _select_sweep_prompts(
                request_count,
                profile,
                variant_index=scenario_seed,
            )
            for prompt in prompts:
                formatted_prompt, _ = _format_prompt_for_generation(tokenizer, prompt)
                prompt_token_counts.append(len(tokenizer.encode(formatted_prompt)))
        return max(prompt_token_counts)

    probe_script = r'''
from __future__ import annotations

import json
import sys

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


def main() -> None:
    spec = json.loads(sys.argv[1])
    engine_args = AsyncEngineArgs(
        model=spec["model"],
        dtype="half",
        max_model_len=spec["max_model_len"],
        max_num_batched_tokens=spec["max_num_batched_tokens"],
        max_num_seqs=spec["max_num_seqs"],
        gpu_memory_utilization=spec["gpu_memory_utilization"],
        enable_prefix_caching=spec["enable_prefix_caching"],
        enforce_eager=True,
        trust_remote_code=False,
        kv_cache_metrics=True,
        kv_cache_metrics_sample=1.0,
        disable_log_stats=False,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        print("LLMBENCH_CAPACITY_PROBE_READY")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
'''
    probe_path = Path("/tmp/vllm_capacity_probe.py")
    probe_path.write_text(probe_script, encoding="utf-8")

    rows = []
    for request_count in request_count_values:
        max_prompt_tokens = max_prompt_tokens_for_shape(request_count)
        for max_new_tokens in output_token_values:
            max_model_len = max(1024, max_prompt_tokens + max_new_tokens + 32)
            default_max_num_batched_tokens = max(2048, request_count * max_model_len)
            max_num_batched_token_options = (
                override_max_num_batched_tokens_values
                if override_max_num_batched_tokens_values
                else [default_max_num_batched_tokens]
            )
            for max_num_batched_tokens in max_num_batched_token_options:
                max_num_batched_tokens_source = (
                    "override"
                    if override_max_num_batched_tokens_values
                    else "default"
                )
                for enable_prefix_caching in prefix_cache_values:
                    spec = {
                        "model": hf_model,
                        "max_model_len": max_model_len,
                        "max_num_batched_tokens": max_num_batched_tokens,
                        "max_num_seqs": request_count,
                        "gpu_memory_utilization": gpu_memory_utilization,
                        "enable_prefix_caching": enable_prefix_caching,
                    }
                    started = time.perf_counter()
                    proc = subprocess.run(
                        [sys.executable, str(probe_path), json.dumps(spec)],
                        capture_output=True,
                        text=True,
                        timeout=600,
                        check=False,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    combined_output = "\n".join([proc.stdout, proc.stderr])
                    parsed = _parse_vllm_engine_capacity_log_metrics(
                        {
                            "captured_logs": [],
                            "stdout": combined_output,
                            "stderr": "",
                        }
                    )
                    rows.append(
                        {
                            "request_count": request_count,
                            "max_new_tokens": max_new_tokens,
                            "enable_prefix_caching": enable_prefix_caching,
                            "max_prompt_tokens": max_prompt_tokens,
                            "max_model_len": max_model_len,
                            "max_num_batched_tokens": max_num_batched_tokens,
                            "default_max_num_batched_tokens": (
                                default_max_num_batched_tokens
                            ),
                            "max_num_batched_tokens_source": (
                                max_num_batched_tokens_source
                            ),
                            "max_num_seqs": request_count,
                            "gpu_memory_utilization": gpu_memory_utilization,
                            "returncode": proc.returncode,
                            "ok": proc.returncode == 0,
                            "elapsed_ms": elapsed_ms,
                            "gpu_kv_cache_size_tokens": parsed[
                                "gpu_kv_cache_size_tokens"
                            ],
                            "available_kv_cache_memory_gib": parsed[
                                "available_kv_cache_memory_gib"
                            ],
                            "max_concurrency_for_request": parsed[
                                "max_concurrency_for_request"
                            ],
                            "max_concurrency_request_tokens": parsed[
                                "max_concurrency_request_tokens"
                            ],
                            "logged_max_model_len": parsed["max_model_len"],
                            "stdout_tail": proc.stdout[-12000:],
                            "stderr_tail": proc.stderr[-12000:],
                        }
                    )

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-capacity-diagnostic",
        "backend": "vllm-capacity-diagnostic",
        "modal_gpu": modal_gpu_label,
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "scenario_seed": int(scenario_seed),
        "warmup_prompt_profile": warmup_prompt_profile_value or None,
        "prefix_cache_modes": prefix_cache_values,
        "max_num_batched_tokens_values": override_max_num_batched_tokens_values,
        "gpu_memory_utilization": gpu_memory_utilization,
        "row_count": len(rows),
        "rows": rows,
        "summary": [
            {
                key: row[key]
                for key in (
                    "request_count",
                    "max_new_tokens",
                    "enable_prefix_caching",
                    "max_model_len",
                    "max_num_batched_tokens",
                    "default_max_num_batched_tokens",
                    "max_num_batched_tokens_source",
                    "max_num_seqs",
                    "gpu_memory_utilization",
                    "gpu_kv_cache_size_tokens",
                    "available_kv_cache_memory_gib",
                    "max_concurrency_for_request",
                    "max_concurrency_request_tokens",
                    "ok",
                )
            }
            for row in rows
        ],
        "vllm_version": _extract_vllm_version_from_probe(rows),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def _extract_vllm_version_from_probe(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        for text in (row.get("stdout_tail") or "", row.get("stderr_tail") or ""):
            match = re.search(r"vLLM API server version ([0-9.]+)", text)
            if match:
                return match.group(1)
    return None


def _format_vllm_capacity_diagnostic_markdown(payload: dict[str, Any]) -> str:
    rows = payload.get("summary") or []

    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return f"{value}{suffix}"
        if isinstance(value, float):
            return f"{value:.3f}{suffix}".rstrip("0").rstrip(".")
        return str(value)

    lines = [
        "# vLLM Capacity Diagnostic",
        "",
        f"Model: `{payload.get('model_id')}`",
        f"GPU: `{payload.get('modal_gpu')}`",
        f"Prompt profiles: `{', '.join(payload.get('prompt_profiles') or [])}`",
        f"Warmup prompt profile: `{payload.get('warmup_prompt_profile')}`",
        "",
        "| Requests | Prefix cache | Max model len | Max batched tokens | Source | Max seqs | GPU KV tokens | KV memory GiB | Max concurrency |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {request_count} | {prefix_cache} | {max_model_len} | {max_num_batched_tokens} | {source} | {max_num_seqs} | {gpu_tokens} | {kv_gib} | {concurrency} |".format(
                request_count=fmt(row.get("request_count")),
                prefix_cache=fmt(row.get("enable_prefix_caching")),
                max_model_len=fmt(row.get("max_model_len")),
                max_num_batched_tokens=fmt(row.get("max_num_batched_tokens")),
                source=fmt(row.get("max_num_batched_tokens_source") or "default"),
                max_num_seqs=fmt(row.get("max_num_seqs")),
                gpu_tokens=fmt(row.get("gpu_kv_cache_size_tokens")),
                kv_gib=fmt(row.get("available_kv_cache_memory_gib")),
                concurrency=fmt(row.get("max_concurrency_for_request")),
            )
        )
    if len(rows) >= 2:
        baseline = rows[0]
        candidate = rows[-1]
        baseline_tokens = baseline.get("gpu_kv_cache_size_tokens")
        candidate_tokens = candidate.get("gpu_kv_cache_size_tokens")
        baseline_batched = baseline.get("max_num_batched_tokens")
        candidate_batched = candidate.get("max_num_batched_tokens")
        if (
            isinstance(baseline_tokens, (int, float))
            and isinstance(candidate_tokens, (int, float))
            and baseline_tokens
            and isinstance(baseline_batched, (int, float))
            and isinstance(candidate_batched, (int, float))
            and baseline_batched
        ):
            lines.extend(
                [
                    "",
                    "## Reading",
                    "",
                    (
                        "The higher-request-count shape raises configured max batched "
                        f"tokens by `{candidate_batched / baseline_batched:.3f}x` "
                        "while reducing available GPU KV-cache tokens to "
                        f"`{candidate_tokens / baseline_tokens:.3f}x` of the baseline."
                    ),
                ]
            )
    lines.append("")
    return "\n".join(lines)


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_capacity_diagnostic_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_prompt_profile: str = "",
    prefix_cache_modes: str = "false",
    max_num_batched_tokens_values: str = "",
    gpu_memory_utilization: float = 0.50,
) -> dict[str, Any]:
    return _run_vllm_capacity_diagnostic_payload(
        hf_model=hf_model,
        request_counts=request_counts,
        prompt_profiles=prompt_profiles,
        output_tokens=output_tokens,
        scenario_seed=scenario_seed,
        warmup_prompt_profile=warmup_prompt_profile,
        prefix_cache_modes=prefix_cache_modes,
        max_num_batched_tokens_values=max_num_batched_tokens_values,
        gpu_memory_utilization=gpu_memory_utilization,
        modal_gpu_label="T4",
    )


@app.function(
    image=vllm_image,
    gpu="L4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_capacity_diagnostic_l4_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_prompt_profile: str = "",
    prefix_cache_modes: str = "false",
    max_num_batched_tokens_values: str = "",
    gpu_memory_utilization: float = 0.50,
) -> dict[str, Any]:
    return _run_vllm_capacity_diagnostic_payload(
        hf_model=hf_model,
        request_counts=request_counts,
        prompt_profiles=prompt_profiles,
        output_tokens=output_tokens,
        scenario_seed=scenario_seed,
        warmup_prompt_profile=warmup_prompt_profile,
        prefix_cache_modes=prefix_cache_modes,
        max_num_batched_tokens_values=max_num_batched_tokens_values,
        gpu_memory_utilization=gpu_memory_utilization,
        modal_gpu_label="L4",
    )


def _select_vllm_capacity_diagnostic_remote(modal_gpu: str) -> Any:
    normalized_gpu = modal_gpu.strip().upper().replace("_", "-")
    if normalized_gpu == "T4":
        return run_vllm_capacity_diagnostic_remote
    if normalized_gpu == "L4":
        return run_vllm_capacity_diagnostic_l4_remote
    raise ValueError("modal_gpu must be one of: T4, L4")


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1200,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_inference_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    import platform
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
    prompt_format_ms = (time.perf_counter() - started) * 1000
    prompt_tokens = len(tokenizer.encode(formatted_prompt))

    started = time.perf_counter()
    llm = LLM(
        model=hf_model,
        dtype="half",
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=False,
    )
    engine_load_ms = (time.perf_counter() - started) * 1000

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    started = time.perf_counter()
    outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
    generate_wall_ms = (time.perf_counter() - started) * 1000

    output = outputs[0]
    completion = output.outputs[0]
    generated_text = completion.text
    generated_token_ids = list(completion.token_ids)
    generated_tokens = len(generated_token_ids)
    metrics = _serialize_vllm_metrics(output)

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    metric_ttft_ms = _duration_from_metrics_ms(metrics, "arrival_time", "first_token_time")
    metric_total_ms = _duration_from_metrics_ms(metrics, "arrival_time", "finished_time")
    metric_decode_ms = _duration_from_metrics_ms(metrics, "first_token_time", "finished_time")
    decode_tokens = max(generated_tokens - 1, 0)

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-inference",
        "backend": "vllm",
        "model_id": hf_model,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": int(max_new_tokens),
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "max_model_len": 1024,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 1,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "engine_load_ms": engine_load_ms,
        "generate_wall_ms": generate_wall_ms,
        "output_tokens_per_second": generated_tokens / max(generate_wall_ms / 1000, 1e-9),
        "metric_ttft_ms": metric_ttft_ms,
        "metric_decode_ms": metric_decode_ms,
        "metric_total_ms": metric_total_ms,
        "metric_tpot_ms": metric_decode_ms / decode_tokens if metric_decode_ms and decode_tokens else None,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "vllm_metrics": metrics,
        "note": "Offline vLLM generate path; streaming TTFT will be measured in a later server-mode run.",
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1200,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_streaming_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    import asyncio
    import platform
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
    prompt_format_ms = (time.perf_counter() - started) * 1000
    prompt_tokens = len(tokenizer.encode(formatted_prompt))

    async def run_stream() -> dict[str, Any]:
        started = time.perf_counter()
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=False,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            output_kind=RequestOutputKind.DELTA,
        )
        request_id = "modal-vllm-streaming-001"
        first_chunk_ms: float | None = None
        chunks: list[dict[str, Any]] = []
        generated_text_parts: list[str] = []
        generated_tokens = 0
        stream_started = time.perf_counter()
        stream_finished: float | None = None

        try:
            async for output in engine.generate(
                request_id=request_id,
                prompt=formatted_prompt,
                sampling_params=sampling_params,
            ):
                elapsed_ms = (time.perf_counter() - stream_started) * 1000
                for completion in output.outputs:
                    text = completion.text
                    token_ids = list(completion.token_ids or [])
                    if first_chunk_ms is None and (text or token_ids):
                        first_chunk_ms = elapsed_ms
                    if text:
                        generated_text_parts.append(text)
                    generated_tokens += len(token_ids)
                    chunks.append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "text": text,
                            "token_count": len(token_ids),
                            "finished": bool(output.finished),
                        }
                    )
                if output.finished:
                    stream_finished = time.perf_counter()
                    break
            if stream_finished is None:
                stream_finished = time.perf_counter()
        finally:
            engine.shutdown()

        stream_wall_ms = (stream_finished - stream_started) * 1000
        decode_tokens_after_first = max(generated_tokens - 1, 0)
        decode_after_first_ms = (
            stream_wall_ms - first_chunk_ms
            if first_chunk_ms is not None
            else None
        )
        return {
            "engine_load_ms": engine_load_ms,
            "stream_wall_ms": stream_wall_ms,
            "first_chunk_ms": first_chunk_ms,
            "decode_after_first_ms": decode_after_first_ms,
            "stream_tpot_ms": (
                decode_after_first_ms / decode_tokens_after_first
                if decode_after_first_ms is not None and decode_tokens_after_first
                else None
            ),
            "generated_tokens": generated_tokens,
            "generated_text": "".join(generated_text_parts),
            "chunks": chunks,
        }

    stream_result = asyncio.run(run_stream())
    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    stream_wall_ms = float(stream_result["stream_wall_ms"])
    generated_tokens = int(stream_result["generated_tokens"])
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-streaming",
        "backend": "vllm",
        "model_id": hf_model,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": int(max_new_tokens),
        "max_model_len": 1024,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 1,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "engine_load_ms": stream_result["engine_load_ms"],
        "first_chunk_ms": stream_result["first_chunk_ms"],
        "stream_wall_ms": stream_wall_ms,
        "decode_after_first_ms": stream_result["decode_after_first_ms"],
        "stream_tpot_ms": stream_result["stream_tpot_ms"],
        "generated_tokens": generated_tokens,
        "generated_text": stream_result["generated_text"],
        "output_tokens_per_second": generated_tokens / max(stream_wall_ms / 1000, 1e-9),
        "chunks": stream_result["chunks"],
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "note": "AsyncLLM streaming path using RequestOutputKind.DELTA; first_chunk_ms is measured at first yielded text/token chunk.",
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1200,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_concurrent_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt_count: int = 4,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    import asyncio
    import platform
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prompt_count <= 0:
        raise ValueError("prompt_count must be positive")

    import vllm

    prompts = _select_concurrent_prompts(prompt_count)
    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    prompt_records = []
    prompt_format_started = time.perf_counter()
    for index, prompt in enumerate(prompts):
        formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
        prompt_records.append(
            {
                "request_id": f"concurrent-{index:02d}",
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "prompt_format": prompt_format,
                "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
            }
        )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    async def run_concurrent() -> dict[str, Any]:
        started = time.perf_counter()
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=1024,
            max_num_batched_tokens=2048,
            max_num_seqs=prompt_count,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=False,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            output_kind=RequestOutputKind.DELTA,
        )

        async def stream_one(record: dict[str, Any], batch_started: float) -> dict[str, Any]:
            request_started = time.perf_counter()
            first_chunk_ms: float | None = None
            finished_ms: float | None = None
            chunks = []
            generated_text_parts = []
            generated_tokens = 0
            async for output in engine.generate(
                request_id=record["request_id"],
                prompt=record["formatted_prompt"],
                sampling_params=sampling_params,
            ):
                now = time.perf_counter()
                elapsed_ms = (now - request_started) * 1000
                batch_elapsed_ms = (now - batch_started) * 1000
                for completion in output.outputs:
                    text = completion.text
                    token_ids = list(completion.token_ids or [])
                    if first_chunk_ms is None and (text or token_ids):
                        first_chunk_ms = elapsed_ms
                    if text:
                        generated_text_parts.append(text)
                    generated_tokens += len(token_ids)
                    chunks.append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "batch_elapsed_ms": batch_elapsed_ms,
                            "token_count": len(token_ids),
                            "text": text,
                            "finished": bool(output.finished),
                        }
                    )
                if output.finished:
                    finished_ms = elapsed_ms
                    break

            stream_wall_ms = finished_ms or ((time.perf_counter() - request_started) * 1000)
            decode_tokens_after_first = max(generated_tokens - 1, 0)
            decode_after_first_ms = (
                stream_wall_ms - first_chunk_ms
                if first_chunk_ms is not None
                else None
            )
            return {
                "request_id": record["request_id"],
                "prompt": record["prompt"],
                "prompt_tokens": record["prompt_tokens"],
                "first_chunk_ms": first_chunk_ms,
                "stream_wall_ms": stream_wall_ms,
                "decode_after_first_ms": decode_after_first_ms,
                "stream_tpot_ms": (
                    decode_after_first_ms / decode_tokens_after_first
                    if decode_after_first_ms is not None and decode_tokens_after_first
                    else None
                ),
                "generated_tokens": generated_tokens,
                "generated_text": "".join(generated_text_parts),
                "chunks": chunks,
            }

        batch_started = time.perf_counter()
        try:
            request_results = await asyncio.gather(
                *(stream_one(record, batch_started) for record in prompt_records)
            )
        finally:
            engine.shutdown()
        batch_wall_ms = (time.perf_counter() - batch_started) * 1000
        return {
            "engine_load_ms": engine_load_ms,
            "batch_wall_ms": batch_wall_ms,
            "requests": request_results,
        }

    batch_result = asyncio.run(run_concurrent())
    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    requests = batch_result["requests"]
    total_output_tokens = sum(request["generated_tokens"] for request in requests)
    first_chunks = [
        request["first_chunk_ms"]
        for request in requests
        if request["first_chunk_ms"] is not None
    ]
    latencies = [request["stream_wall_ms"] for request in requests]
    tpot_values = [
        request["stream_tpot_ms"]
        for request in requests
        if request["stream_tpot_ms"] is not None
    ]
    batch_wall_ms = float(batch_result["batch_wall_ms"])
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-concurrent",
        "backend": "vllm",
        "model_id": hf_model,
        "request_count": len(requests),
        "max_new_tokens": int(max_new_tokens),
        "max_model_len": 1024,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": prompt_count,
        "prompt_format_ms": prompt_format_ms,
        "tokenizer_load_ms": tokenizer_load_ms,
        "engine_load_ms": batch_result["engine_load_ms"],
        "batch_wall_ms": batch_wall_ms,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": total_output_tokens / max(batch_wall_ms / 1000, 1e-9),
        "p50_first_chunk_ms": _percentile(first_chunks, 50),
        "p95_first_chunk_ms": _percentile(first_chunks, 95),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p50_stream_tpot_ms": _percentile(tpot_values, 50),
        "p95_stream_tpot_ms": _percentile(tpot_values, 95),
        "requests": requests,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "note": "Concurrent AsyncLLM streaming workload. Request timings are measured from each coroutine start after the shared engine is loaded.",
    }


@app.function(
    image=vllm_image,
    timeout=600,
)
def run_vllm_cache_metrics_probe_remote(
    hf_model: str = DEFAULT_HF_MODEL,
) -> dict[str, Any]:
    import inspect
    import platform

    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    def signature_parameters(target: Any) -> list[str]:
        try:
            return list(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            return []

    def dataclass_fields(target: Any) -> list[str]:
        fields = getattr(target, "__dataclass_fields__", {})
        return sorted(fields) if fields else []

    def relevant_names(names: list[str]) -> list[str]:
        markers = ("cache", "metric", "observ", "stat", "log")
        return sorted(
            name for name in names
            if any(marker in name.lower() for marker in markers)
        )

    async_engine_args_parameters = signature_parameters(AsyncEngineArgs)
    async_engine_args_fields = dataclass_fields(AsyncEngineArgs)
    async_llm_relevant_methods = relevant_names(
        [name for name in dir(AsyncLLM) if not name.startswith("_")]
    )
    observability_parameters: list[str] = []
    observability_fields: list[str] = []
    observability_error = ""
    try:
        from vllm.config import ObservabilityConfig

        observability_parameters = signature_parameters(ObservabilityConfig)
        observability_fields = dataclass_fields(ObservabilityConfig)
    except Exception as error:  # pragma: no cover - remote environment probe
        observability_error = f"{type(error).__name__}: {error}"

    constructor_checks = []
    for kwargs in (
        {"kv_cache_metrics": True},
        {"kv_cache_metrics": True, "kv_cache_metrics_sample": 1.0},
        {"disable_log_stats": False},
    ):
        try:
            AsyncEngineArgs(model=hf_model, **kwargs)
            constructor_checks.append({"kwargs": kwargs, "accepted": True, "error": ""})
        except Exception as error:  # pragma: no cover - remote environment probe
            constructor_checks.append(
                {
                    "kwargs": kwargs,
                    "accepted": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-cache-metrics-probe",
        "model_id": hf_model,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "async_engine_args_parameters": async_engine_args_parameters,
        "async_engine_args_fields": async_engine_args_fields,
        "async_engine_args_relevant_parameters": relevant_names(
            async_engine_args_parameters
        ),
        "async_engine_args_relevant_fields": relevant_names(async_engine_args_fields),
        "async_llm_relevant_methods": async_llm_relevant_methods,
        "observability_config_parameters": observability_parameters,
        "observability_config_fields": observability_fields,
        "observability_config_relevant_parameters": relevant_names(
            observability_parameters
        ),
        "observability_config_relevant_fields": relevant_names(observability_fields),
        "observability_config_error": observability_error,
        "constructor_checks": constructor_checks,
        "note": (
            "Remote introspection of vLLM constructor surfaces related to cache "
            "metrics, observability, and logging. This does not run inference."
        ),
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1200,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_cache_metrics_smoke_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt_profile: str = "shared_prefix_long",
    request_count: int = 4,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    import asyncio
    import contextlib
    import inspect
    import io
    import logging
    import platform
    import re
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    import vllm

    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_profile = prompt_profile.lower().replace("-", "_")
    _validate_vllm_prompt_profiles([prompt_profile], label="prompt_profile")

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    prompts = _select_sweep_prompts(request_count, prompt_profile)
    prompt_records = []
    for index, prompt in enumerate(prompts):
        formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
        prompt_records.append(
            {
                "request_id": f"{prompt_profile}-metrics-n{request_count}-{index:02d}",
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "prompt_format": prompt_format,
                "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
            }
        )
    max_prompt_tokens = max(record["prompt_tokens"] for record in prompt_records)
    max_model_len = max(1024, max_prompt_tokens + max_new_tokens + 32)
    max_num_batched_tokens = max(2048, request_count * max_model_len)

    def relevant_object_attrs(target: Any) -> list[dict[str, Any]]:
        rows = []
        markers = ("cache", "metric", "stat", "log")
        for name in dir(target):
            if name.startswith("__") or not any(marker in name.lower() for marker in markers):
                continue
            try:
                value = getattr(target, name)
                value_type = type(value).__name__
                if callable(value):
                    value_repr = f"<callable {value_type}>"
                else:
                    value_repr = repr(value)
            except Exception as error:  # pragma: no cover - remote object probe
                value_type = type(error).__name__
                value_repr = f"<error {error}>"
            rows.append(
                {
                    "name": name,
                    "type": value_type,
                    "repr": value_repr[:500],
                }
            )
        return rows

    async def call_log_stats(engine: Any, label: str) -> dict[str, Any]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        captured_logs: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_logs.append(self.format(record))

        handler = CaptureHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        capture_loggers = [
            logging.getLogger(),
            logging.getLogger("vllm"),
            logging.getLogger("vllm.v1.metrics.loggers"),
        ]
        started = time.perf_counter()
        try:
            for logger in capture_loggers:
                logger.addHandler(handler)
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                result = engine.do_log_stats()
                if inspect.isawaitable(result):
                    result = await result
            return {
                "label": label,
                "ok": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(result).__name__,
                "return_repr": repr(result)[:1000],
                "stdout": stdout_buffer.getvalue()[-4000:],
                "stderr": stderr_buffer.getvalue()[-4000:],
                "captured_logs": captured_logs[-20:],
            }
        except Exception as error:  # pragma: no cover - remote object probe
            return {
                "label": label,
                "ok": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(error).__name__,
                "return_repr": str(error)[:1000],
                "stdout": stdout_buffer.getvalue()[-4000:],
                "stderr": stderr_buffer.getvalue()[-4000:],
                "captured_logs": captured_logs[-20:],
            }
        finally:
            for logger in capture_loggers:
                logger.removeHandler(handler)

    def parse_log_metrics(log_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        patterns = {
            "avg_prompt_throughput_tokens_per_s": r"Avg prompt throughput: ([0-9.]+) tokens/s",
            "avg_generation_throughput_tokens_per_s": (
                r"Avg generation throughput: ([0-9.]+) tokens/s"
            ),
            "gpu_kv_cache_usage_pct": r"GPU KV cache usage: ([0-9.]+)%",
            "prefix_cache_hit_rate_pct": r"Prefix cache hit rate: ([0-9.]+)%",
        }
        for call in log_stats:
            for line in call.get("captured_logs", []):
                row: dict[str, Any] = {
                    "label": call["label"],
                    "line": line,
                }
                matched = False
                for field, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        row[field] = float(match.group(1))
                        matched = True
                if matched:
                    rows.append(row)
        return rows

    async def run_smoke() -> dict[str, Any]:
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=request_count,
            gpu_memory_utilization=0.50,
            enable_prefix_caching=True,
            kv_cache_metrics=True,
            kv_cache_metrics_sample=1.0,
            disable_log_stats=False,
            enforce_eager=True,
            trust_remote_code=False,
        )
        started = time.perf_counter()
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000
        log_stats = []
        try:
            log_stats.append(await call_log_stats(engine, "after_engine_load"))
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )

            async def stream_one(record: dict[str, Any]) -> dict[str, Any]:
                request_started = time.perf_counter()
                first_chunk_ms: float | None = None
                finished_ms: float | None = None
                generated_tokens = 0
                chunk_count = 0
                async for output in engine.generate(
                    request_id=f"metrics-smoke-{record['request_id']}",
                    prompt=record["formatted_prompt"],
                    sampling_params=sampling_params,
                ):
                    elapsed_ms = (time.perf_counter() - request_started) * 1000
                    chunk_count += 1
                    for completion in output.outputs:
                        token_ids = list(completion.token_ids or [])
                        if first_chunk_ms is None and (completion.text or token_ids):
                            first_chunk_ms = elapsed_ms
                        generated_tokens += len(token_ids)
                    if output.finished:
                        finished_ms = elapsed_ms
                        break
                stream_wall_ms = finished_ms or (
                    (time.perf_counter() - request_started) * 1000
                )
                decode_tokens_after_first = max(generated_tokens - 1, 0)
                decode_after_first_ms = (
                    stream_wall_ms - first_chunk_ms
                    if first_chunk_ms is not None
                    else None
                )
                return {
                    "request_id": record["request_id"],
                    "prompt_tokens": record["prompt_tokens"],
                    "first_chunk_ms": first_chunk_ms,
                    "stream_wall_ms": stream_wall_ms,
                    "decode_after_first_ms": decode_after_first_ms,
                    "stream_tpot_ms": (
                        decode_after_first_ms / decode_tokens_after_first
                        if decode_after_first_ms is not None and decode_tokens_after_first
                        else None
                    ),
                    "generated_tokens": generated_tokens,
                    "total_sequence_tokens": record["prompt_tokens"] + generated_tokens,
                    "chunk_count": chunk_count,
                }

            batch_started = time.perf_counter()
            request_results = await asyncio.gather(
                *(stream_one(record) for record in prompt_records)
            )
            batch_wall_ms = (time.perf_counter() - batch_started) * 1000
            log_stats.append(await call_log_stats(engine, "after_shared_prefix_batch"))
            parsed_log_metrics = parse_log_metrics(log_stats)
            engine_attrs = relevant_object_attrs(engine)
            engine_core_attrs = (
                relevant_object_attrs(engine.engine_core)
                if hasattr(engine, "engine_core")
                else []
            )
        finally:
            engine.shutdown()

        return {
            "engine_load_ms": engine_load_ms,
            "request_results": request_results,
            "batch_wall_ms": batch_wall_ms,
            "summary": _summarize_stream_requests(request_results, batch_wall_ms),
            "log_stats_calls": log_stats,
            "parsed_log_metrics": parsed_log_metrics,
            "engine_relevant_attrs": engine_attrs,
            "engine_core_relevant_attrs": engine_core_attrs,
        }

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    smoke = asyncio.run(run_smoke())
    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-cache-metrics-smoke",
        "backend": "vllm-async-engine",
        "model_id": hf_model,
        "prompt_profile": prompt_profile,
        "request_count": request_count,
        "max_new_tokens": max_new_tokens,
        "prompt_tokens_min": min(record["prompt_tokens"] for record in prompt_records),
        "prompt_tokens_max": max_prompt_tokens,
        "prompt_tokens_mean": (
            sum(record["prompt_tokens"] for record in prompt_records) / len(prompt_records)
        ),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_prefix_caching": True,
        "kv_cache_metrics": True,
        "kv_cache_metrics_sample": 1.0,
        "disable_log_stats": False,
        "smoke": smoke,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "note": (
            "Tiny GPU-backed metrics smoke. The run enables vLLM KV-cache "
            "metrics, executes one shared-prefix batch, calls do_log_stats "
            "before and after the batch, and records visible cache/stat/log "
            "attributes on AsyncLLM and engine_core."
        ),
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_sweep_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    enable_prefix_caching: bool = False,
) -> dict[str, Any]:
    import asyncio
    import platform
    import random
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    _validate_vllm_prompt_profiles(prompt_profile_values)

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    scenario_specs = []
    prompt_format_started = time.perf_counter()
    for prompt_profile in prompt_profile_values:
        for max_new_tokens in output_token_values:
            for request_count in request_count_values:
                prompt_records = []
                prompts = _select_sweep_prompts(request_count, prompt_profile)
                for index, prompt in enumerate(prompts):
                    formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
                    prompt_records.append(
                        {
                            "request_id": (
                                f"{prompt_profile}-out{max_new_tokens}-n{request_count}-"
                                f"{index:02d}"
                            ),
                            "prompt": prompt,
                            "formatted_prompt": formatted_prompt,
                            "prompt_format": prompt_format,
                            "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
                        }
                    )
                prompt_tokens = [record["prompt_tokens"] for record in prompt_records]
                scenario_specs.append(
                    {
                        "scenario_id": f"{prompt_profile}_out{max_new_tokens}_n{request_count}",
                        "prompt_profile": prompt_profile,
                        "request_count": request_count,
                        "max_new_tokens": max_new_tokens,
                        "prompt_format": prompt_records[0]["prompt_format"],
                        "prompt_tokens_min": min(prompt_tokens),
                        "prompt_tokens_max": max(prompt_tokens),
                        "prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
                        "total_prompt_tokens": sum(prompt_tokens),
                        "prompt_records": prompt_records,
                    }
                )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    max_request_count = max(request_count_values)
    max_output_tokens = max(output_token_values)
    max_prompt_tokens = max(spec["prompt_tokens_max"] for spec in scenario_specs)
    max_model_len = max(1024, max_prompt_tokens + max_output_tokens + 32)
    max_num_batched_tokens = max(2048, max_request_count * max_model_len)

    async def run_streaming_sweep() -> dict[str, Any]:
        started = time.perf_counter()
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_request_count,
            gpu_memory_utilization=0.50,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=True,
            trust_remote_code=False,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000

        async def stream_one(
            scenario_id: str,
            run_id: str,
            record: dict[str, Any],
            sampling_params: Any,
            batch_started: float,
        ) -> dict[str, Any]:
            request_started = time.perf_counter()
            first_chunk_ms: float | None = None
            finished_ms: float | None = None
            generated_text_parts = []
            generated_tokens = 0
            chunk_count = 0

            async for output in engine.generate(
                request_id=f"{run_id}-{scenario_id}-{record['request_id']}",
                prompt=record["formatted_prompt"],
                sampling_params=sampling_params,
            ):
                now = time.perf_counter()
                elapsed_ms = (now - request_started) * 1000
                chunk_count += 1
                for completion in output.outputs:
                    text = completion.text
                    token_ids = list(completion.token_ids or [])
                    if first_chunk_ms is None and (text or token_ids):
                        first_chunk_ms = elapsed_ms
                    if text:
                        generated_text_parts.append(text)
                    generated_tokens += len(token_ids)
                if output.finished:
                    finished_ms = elapsed_ms
                    break

            stream_wall_ms = finished_ms or ((time.perf_counter() - request_started) * 1000)
            decode_tokens_after_first = max(generated_tokens - 1, 0)
            decode_after_first_ms = (
                stream_wall_ms - first_chunk_ms
                if first_chunk_ms is not None
                else None
            )
            generated_text = "".join(generated_text_parts)
            return {
                "request_id": record["request_id"],
                "prompt_tokens": record["prompt_tokens"],
                "first_chunk_ms": first_chunk_ms,
                "stream_wall_ms": stream_wall_ms,
                "decode_after_first_ms": decode_after_first_ms,
                "stream_tpot_ms": (
                    decode_after_first_ms / decode_tokens_after_first
                    if decode_after_first_ms is not None and decode_tokens_after_first
                    else None
                ),
                "generated_tokens": generated_tokens,
                "total_sequence_tokens": record["prompt_tokens"] + generated_tokens,
                "chunk_count": chunk_count,
                "batch_finished_ms": (time.perf_counter() - batch_started) * 1000,
                "generated_text": generated_text,
            }

        async def drain_warmup_request(
            scenario_id: str,
            record: dict[str, Any],
            sampling_params: Any,
        ) -> int:
            generated_tokens = 0
            async for output in engine.generate(
                request_id=f"warmup-{scenario_id}-{record['request_id']}",
                prompt=record["formatted_prompt"],
                sampling_params=sampling_params,
            ):
                for completion in output.outputs:
                    generated_tokens += len(list(completion.token_ids or []))
                if output.finished:
                    break
            return generated_tokens

        async def run_warmup() -> dict[str, Any]:
            sampling_params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
            started = time.perf_counter()
            generated_tokens = 0
            for spec in scenario_specs:
                warmup_counts = await asyncio.gather(
                    *(
                        drain_warmup_request(
                            spec["scenario_id"],
                            record,
                            sampling_params,
                        )
                        for record in spec["prompt_records"]
                    )
                )
                generated_tokens += sum(warmup_counts)
            return {
                "warmup_wall_ms": (time.perf_counter() - started) * 1000,
                "shape_warmup_scenarios": len(scenario_specs),
                "generated_tokens": generated_tokens,
            }

        async def run_scenario(
            spec: dict[str, Any],
            repeat_index: int,
            run_order: int,
        ) -> dict[str, Any]:
            sampling_params = SamplingParams(
                max_tokens=spec["max_new_tokens"],
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
            run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
            batch_started = time.perf_counter()
            request_results = await asyncio.gather(
                *(
                    stream_one(
                        spec["scenario_id"],
                        run_id,
                        record,
                        sampling_params,
                        batch_started,
                    )
                    for record in spec["prompt_records"]
                )
            )
            batch_wall_ms = (time.perf_counter() - batch_started) * 1000
            summary = _summarize_stream_requests(request_results, batch_wall_ms)
            return {
                "run_id": run_id,
                "repeat_index": repeat_index,
                "run_order": run_order,
                "scenario_id": spec["scenario_id"],
                "prompt_profile": spec["prompt_profile"],
                "request_count": spec["request_count"],
                "max_new_tokens": spec["max_new_tokens"],
                "prompt_format": spec["prompt_format"],
                "prompt_tokens_min": spec["prompt_tokens_min"],
                "prompt_tokens_max": spec["prompt_tokens_max"],
                "prompt_tokens_mean": spec["prompt_tokens_mean"],
                "total_prompt_tokens": spec["total_prompt_tokens"],
                "estimated_peak_sequence_tokens": sum(
                    request["total_sequence_tokens"] for request in request_results
                ),
                **summary,
                "requests": request_results,
            }

        try:
            warmup_result = await run_warmup()
            scenario_plan = []
            for repeat_index in range(repeats):
                for spec in scenario_specs:
                    scenario_plan.append((repeat_index, spec))
            random.Random(scenario_seed).shuffle(scenario_plan)
            scenario_runs = []
            for run_order, (repeat_index, spec) in enumerate(scenario_plan):
                scenario_runs.append(
                    await run_scenario(
                        spec,
                        repeat_index=repeat_index,
                        run_order=run_order,
                    )
                )
        finally:
            engine.shutdown()

        return {
            "engine_load_ms": engine_load_ms,
            "warmup": warmup_result,
            "scenario_runs": scenario_runs,
        }

    sweep_result = asyncio.run(run_streaming_sweep())
    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    scenario_runs = sweep_result["scenario_runs"]
    scenarios = _aggregate_vllm_sweep_scenarios(scenario_specs, scenario_runs)
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-sweep",
        "backend": "vllm",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "scenario_count": len(scenarios),
        "scenario_run_count": len(scenario_runs),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "enable_prefix_caching": bool(enable_prefix_caching),
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "engine_load_ms": sweep_result["engine_load_ms"],
        "warmup": sweep_result["warmup"],
        "summary": _vllm_sweep_summary_rows(scenarios),
        "run_summary": _vllm_sweep_run_rows(scenario_runs),
        "scenarios": scenarios,
        "scenario_runs": scenario_runs,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "note": (
            "One loaded AsyncLLM engine runs a seeded, repeated, shuffled grid "
            "over request count, prompt profile, and output token budget after "
            f"one-token shape warmups with prefix caching {bool(enable_prefix_caching)}."
        ),
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_streaming_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 32,
    ready_timeout_s: int = 600,
) -> dict[str, Any]:
    import json
    import platform
    import subprocess
    import threading
    import time

    import httpx
    from transformers import AutoTokenizer

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be positive")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
    prompt_tokens = len(tokenizer.encode(formatted_prompt))

    port = 8000
    base_url = f"http://127.0.0.1:{port}"
    command = [
        "vllm",
        "serve",
        hf_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "half",
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "1024",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.50",
        "--enforce-eager",
    ]

    server_logs: list[str] = []
    server_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def drain_logs() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            server_logs.append(line.rstrip())

    log_thread = threading.Thread(target=drain_logs, daemon=True)
    log_thread.start()

    ready_ms: float | None = None
    usage: dict[str, Any] | None = None
    response_status_code: int | None = None
    chunks: list[dict[str, Any]] = []
    generated_text_parts: list[str] = []

    try:
        with httpx.Client(timeout=2.0) as health_client:
            while (time.perf_counter() - server_started) < ready_timeout_s:
                if process.poll() is not None:
                    raise RuntimeError(
                        "vLLM server exited before readiness: "
                        + "\n".join(_tail_lines(server_logs, 40))
                    )
                try:
                    response = health_client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        ready_ms = (time.perf_counter() - server_started) * 1000
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1.0)
        if ready_ms is None:
            raise TimeoutError(
                "vLLM server did not become ready: "
                + "\n".join(_tail_lines(server_logs, 40))
            )

        request_body = {
            "model": hf_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        first_content_ms: float | None = None
        request_started = time.perf_counter()
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response_status_code = response.status_code
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    elapsed_ms = (time.perf_counter() - request_started) * 1000
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    choices = event.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        if content and first_content_ms is None:
                            first_content_ms = elapsed_ms
                        if content:
                            generated_text_parts.append(content)
                        chunks.append(
                            {
                                "elapsed_ms": elapsed_ms,
                                "content": content,
                                "finish_reason": choice.get("finish_reason"),
                            }
                        )
        request_wall_ms = (time.perf_counter() - request_started) * 1000
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        log_thread.join(timeout=5)

    generated_text = "".join(generated_text_parts)
    generated_tokens = (
        int(usage["completion_tokens"])
        if usage and usage.get("completion_tokens") is not None
        else len(tokenizer.encode(generated_text, add_special_tokens=False))
    )
    measured_prompt_tokens = (
        int(usage["prompt_tokens"])
        if usage and usage.get("prompt_tokens") is not None
        else prompt_tokens
    )
    decode_tokens_after_first = max(generated_tokens - 1, 0)
    decode_after_first_ms = (
        request_wall_ms - first_content_ms
        if first_content_ms is not None
        else None
    )

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-streaming",
        "backend": "vllm-openai-server",
        "model_id": hf_model,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "prompt_tokens": measured_prompt_tokens,
        "prompt_tokens_local_estimate": prompt_tokens,
        "max_new_tokens": int(max_new_tokens),
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "response_status_code": response_status_code,
        "server_ready_ms": ready_ms,
        "request_wall_ms": request_wall_ms,
        "first_content_ms": first_content_ms,
        "decode_after_first_ms": decode_after_first_ms,
        "stream_tpot_ms": (
            decode_after_first_ms / decode_tokens_after_first
            if decode_after_first_ms is not None and decode_tokens_after_first
            else None
        ),
        "output_tokens_per_second": generated_tokens / max(request_wall_ms / 1000, 1e-9),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "usage": usage,
        "server_command": command,
        "tokenizer_load_ms": tokenizer_load_ms,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "server_logs_head": server_logs[:80],
        "server_logs_tail": _tail_lines(server_logs, 80),
        "note": "OpenAI-compatible vLLM server smoke using /v1/chat/completions with SSE streaming.",
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_concurrent_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profile: str = "short",
    max_new_tokens: int = 32,
    ready_timeout_s: int = 600,
) -> dict[str, Any]:
    import asyncio
    import json
    import platform
    import subprocess
    import threading
    import time

    import httpx
    from transformers import AutoTokenizer

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile = prompt_profile.lower().replace("-", "_")
    _validate_vllm_prompt_profiles([prompt_profile], label="prompt_profile")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be positive")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    scenario_specs = []
    prompt_format_started = time.perf_counter()
    for request_count in request_count_values:
        prompt_records = []
        for index, prompt in enumerate(_select_sweep_prompts(request_count, prompt_profile)):
            formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
            prompt_records.append(
                {
                    "request_id": f"{prompt_profile}-n{request_count}-{index:02d}",
                    "prompt": prompt,
                    "formatted_prompt": formatted_prompt,
                    "prompt_format": prompt_format,
                    "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
                }
            )
        prompt_tokens = [record["prompt_tokens"] for record in prompt_records]
        scenario_specs.append(
            {
                "scenario_id": f"{prompt_profile}_out{max_new_tokens}_n{request_count}",
                "prompt_profile": prompt_profile,
                "request_count": request_count,
                "max_new_tokens": max_new_tokens,
                "prompt_format": prompt_records[0]["prompt_format"],
                "prompt_tokens_min": min(prompt_tokens),
                "prompt_tokens_max": max(prompt_tokens),
                "prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
                "total_prompt_tokens": sum(prompt_tokens),
                "prompt_records": prompt_records,
            }
        )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    max_request_count = max(request_count_values)
    max_model_len = 1024
    max_num_batched_tokens = max(2048, max_request_count * max_model_len)
    port = 8000
    base_url = f"http://127.0.0.1:{port}"
    command = [
        "vllm",
        "serve",
        hf_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "half",
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--max-num-seqs",
        str(max_request_count),
        "--gpu-memory-utilization",
        "0.50",
        "--enforce-eager",
    ]

    server_logs: list[str] = []
    server_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def drain_logs() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            server_logs.append(line.rstrip())

    log_thread = threading.Thread(target=drain_logs, daemon=True)
    log_thread.start()

    async def stream_one(
        scenario_id: str,
        record: dict[str, Any],
        batch_started: float,
    ) -> dict[str, Any]:
        request_body = {
            "model": hf_model,
            "messages": [{"role": "user", "content": record["prompt"]}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        chunks: list[dict[str, Any]] = []
        generated_text_parts: list[str] = []
        usage: dict[str, Any] | None = None
        response_status_code: int | None = None
        first_content_ms: float | None = None
        request_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response_status_code = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    elapsed_ms = (time.perf_counter() - request_started) * 1000
                    batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        if content and first_content_ms is None:
                            first_content_ms = elapsed_ms
                        if content:
                            generated_text_parts.append(content)
                        chunks.append(
                            {
                                "elapsed_ms": elapsed_ms,
                                "batch_elapsed_ms": batch_elapsed_ms,
                                "content": content,
                                "finish_reason": choice.get("finish_reason"),
                            }
                        )
        request_wall_ms = (time.perf_counter() - request_started) * 1000
        generated_text = "".join(generated_text_parts)
        generated_tokens = (
            int(usage["completion_tokens"])
            if usage and usage.get("completion_tokens") is not None
            else len(tokenizer.encode(generated_text, add_special_tokens=False))
        )
        prompt_tokens = (
            int(usage["prompt_tokens"])
            if usage and usage.get("prompt_tokens") is not None
            else record["prompt_tokens"]
        )
        decode_tokens_after_first = max(generated_tokens - 1, 0)
        decode_after_first_ms = (
            request_wall_ms - first_content_ms
            if first_content_ms is not None
            else None
        )
        return {
            "request_id": record["request_id"],
            "scenario_id": scenario_id,
            "prompt": record["prompt"],
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_local_estimate": record["prompt_tokens"],
            "generated_tokens": generated_tokens,
            "total_sequence_tokens": prompt_tokens + generated_tokens,
            "generated_text": generated_text,
            "response_status_code": response_status_code,
            "first_content_ms": first_content_ms,
            "request_wall_ms": request_wall_ms,
            "decode_after_first_ms": decode_after_first_ms,
            "stream_tpot_ms": (
                decode_after_first_ms / decode_tokens_after_first
                if decode_after_first_ms is not None and decode_tokens_after_first
                else None
            ),
            "chunk_count": len(chunks),
            "chunks": chunks,
            "usage": usage,
        }

    async def run_scenario(spec: dict[str, Any]) -> dict[str, Any]:
        batch_started = time.perf_counter()
        request_results = await asyncio.gather(
            *(
                stream_one(
                    spec["scenario_id"],
                    record,
                    batch_started,
                )
                for record in spec["prompt_records"]
            )
        )
        batch_wall_ms = (time.perf_counter() - batch_started) * 1000
        total_output_tokens = sum(request["generated_tokens"] for request in request_results)
        first_contents = [
            request["first_content_ms"]
            for request in request_results
            if request["first_content_ms"] is not None
        ]
        latencies = [request["request_wall_ms"] for request in request_results]
        tpot_values = [
            request["stream_tpot_ms"]
            for request in request_results
            if request["stream_tpot_ms"] is not None
        ]
        return {
            "scenario_id": spec["scenario_id"],
            "prompt_profile": spec["prompt_profile"],
            "request_count": spec["request_count"],
            "max_new_tokens": spec["max_new_tokens"],
            "prompt_format": spec["prompt_format"],
            "prompt_tokens_min": spec["prompt_tokens_min"],
            "prompt_tokens_max": spec["prompt_tokens_max"],
            "prompt_tokens_mean": spec["prompt_tokens_mean"],
            "total_prompt_tokens": spec["total_prompt_tokens"],
            "estimated_peak_sequence_tokens": sum(
                request["total_sequence_tokens"] for request in request_results
            ),
            "batch_wall_ms": batch_wall_ms,
            "total_output_tokens": total_output_tokens,
            "aggregate_output_tokens_per_second": total_output_tokens / max(batch_wall_ms / 1000, 1e-9),
            "p50_first_content_ms": _percentile(first_contents, 50),
            "p95_first_content_ms": _percentile(first_contents, 95),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "p50_stream_tpot_ms": _percentile(tpot_values, 50),
            "p95_stream_tpot_ms": _percentile(tpot_values, 95),
            "requests": request_results,
        }

    ready_ms: float | None = None
    scenarios: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=2.0) as health_client:
            while (time.perf_counter() - server_started) < ready_timeout_s:
                if process.poll() is not None:
                    raise RuntimeError(
                        "vLLM server exited before readiness: "
                        + "\n".join(_tail_lines(server_logs, 40))
                    )
                try:
                    response = health_client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        ready_ms = (time.perf_counter() - server_started) * 1000
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1.0)
        if ready_ms is None:
            raise TimeoutError(
                "vLLM server did not become ready: "
                + "\n".join(_tail_lines(server_logs, 40))
            )

        for spec in scenario_specs:
            scenarios.append(asyncio.run(run_scenario(spec)))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        log_thread.join(timeout=5)

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-concurrent",
        "backend": "vllm-openai-server",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profile": prompt_profile,
        "max_new_tokens": int(max_new_tokens),
        "scenario_count": len(scenarios),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "server_ready_ms": ready_ms,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "summary": _vllm_server_concurrent_summary_rows(scenarios),
        "scenarios": scenarios,
        "server_command": command,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "server_logs_head": server_logs[:80],
        "server_logs_tail": _tail_lines(server_logs, 80),
        "note": "OpenAI-compatible vLLM server concurrent streaming workload using /v1/chat/completions.",
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=2400,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_sweep_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    ready_timeout_s: int = 600,
) -> dict[str, Any]:
    import asyncio
    import json
    import platform
    import random
    import subprocess
    import threading
    import time

    import httpx
    from transformers import AutoTokenizer

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be positive")
    _validate_vllm_prompt_profiles(prompt_profile_values)

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    prompt_format_started = time.perf_counter()
    scenario_specs = _build_vllm_prompt_scenario_specs(
        tokenizer=tokenizer,
        request_count_values=request_count_values,
        prompt_profile_values=prompt_profile_values,
        output_token_values=output_token_values,
        variant_index=scenario_seed,
    )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    max_request_count = max(request_count_values)
    max_output_tokens = max(output_token_values)
    max_prompt_tokens = max(spec["prompt_tokens_max"] for spec in scenario_specs)
    max_model_len = max(1024, max_prompt_tokens + max_output_tokens + 32)
    max_num_batched_tokens = max(2048, max_request_count * max_model_len)
    port = 8000
    base_url = f"http://127.0.0.1:{port}"
    command = [
        "vllm",
        "serve",
        hf_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "half",
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--max-num-seqs",
        str(max_request_count),
        "--gpu-memory-utilization",
        "0.50",
        "--enforce-eager",
    ]

    server_logs: list[str] = []
    server_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def drain_logs() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            server_logs.append(line.rstrip())

    log_thread = threading.Thread(target=drain_logs, daemon=True)
    log_thread.start()

    async def stream_one(
        client: httpx.AsyncClient,
        scenario_id: str,
        record: dict[str, Any],
        max_tokens: int,
        batch_started: float,
    ) -> dict[str, Any]:
        request_body = {
            "model": hf_model,
            "messages": [{"role": "user", "content": record["prompt"]}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        chunks: list[dict[str, Any]] = []
        generated_text_parts: list[str] = []
        usage: dict[str, Any] | None = None
        response_status_code: int | None = None
        first_content_ms: float | None = None
        request_started = time.perf_counter()
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=request_body,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response_status_code = response.status_code
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                now = time.perf_counter()
                elapsed_ms = (now - request_started) * 1000
                batch_elapsed_ms = (now - batch_started) * 1000
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    if content and first_content_ms is None:
                        first_content_ms = elapsed_ms
                    if content:
                        generated_text_parts.append(content)
                    chunks.append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "batch_elapsed_ms": batch_elapsed_ms,
                            "content": content,
                            "finish_reason": choice.get("finish_reason"),
                        }
                    )
        request_wall_ms = (time.perf_counter() - request_started) * 1000
        generated_text = "".join(generated_text_parts)
        generated_tokens = (
            int(usage["completion_tokens"])
            if usage and usage.get("completion_tokens") is not None
            else len(tokenizer.encode(generated_text, add_special_tokens=False))
        )
        prompt_tokens = (
            int(usage["prompt_tokens"])
            if usage and usage.get("prompt_tokens") is not None
            else record["prompt_tokens"]
        )
        decode_tokens_after_first = max(generated_tokens - 1, 0)
        decode_after_first_ms = (
            request_wall_ms - first_content_ms
            if first_content_ms is not None
            else None
        )
        return {
            "request_id": record["request_id"],
            "scenario_id": scenario_id,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_local_estimate": record["prompt_tokens"],
            "generated_tokens": generated_tokens,
            "total_sequence_tokens": prompt_tokens + generated_tokens,
            "generated_text": generated_text,
            "response_status_code": response_status_code,
            "first_content_ms": first_content_ms,
            "request_wall_ms": request_wall_ms,
            "decode_after_first_ms": decode_after_first_ms,
            "stream_tpot_ms": (
                decode_after_first_ms / decode_tokens_after_first
                if decode_after_first_ms is not None and decode_tokens_after_first
                else None
            ),
            "batch_finished_ms": (time.perf_counter() - batch_started) * 1000,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "usage": usage,
        }

    async def run_scenario(
        spec: dict[str, Any],
        repeat_index: int,
        run_order: int,
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        max_tokens = max_tokens_override or spec["max_new_tokens"]
        run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
        batch_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=None) as client:
            request_results = await asyncio.gather(
                *(
                    stream_one(
                        client,
                        spec["scenario_id"],
                        record,
                        max_tokens,
                        batch_started,
                    )
                    for record in spec["prompt_records"]
                )
            )
        batch_wall_ms = (time.perf_counter() - batch_started) * 1000
        summary = _summarize_server_stream_requests(request_results, batch_wall_ms)
        return {
            "run_id": run_id,
            "repeat_index": repeat_index,
            "run_order": run_order,
            "scenario_id": spec["scenario_id"],
            "prompt_profile": spec["prompt_profile"],
            "request_count": spec["request_count"],
            "max_new_tokens": max_tokens,
            "prompt_format": spec["prompt_format"],
            "prompt_tokens_min": spec["prompt_tokens_min"],
            "prompt_tokens_max": spec["prompt_tokens_max"],
            "prompt_tokens_mean": spec["prompt_tokens_mean"],
            "total_prompt_tokens": spec["total_prompt_tokens"],
            "estimated_peak_sequence_tokens": sum(
                request["total_sequence_tokens"] for request in request_results
            ),
            **summary,
            "requests": request_results,
        }

    async def run_warmups() -> dict[str, Any]:
        started = time.perf_counter()
        warmup_summaries = []
        run_order = 0
        for warmup_index in range(warmup_runs):
            for spec in scenario_specs:
                result = await run_scenario(
                    spec,
                    repeat_index=-(warmup_index + 1),
                    run_order=run_order,
                    max_tokens_override=1,
                )
                warmup_summaries.append(
                    {
                        "warmup_index": warmup_index,
                        "scenario_id": spec["scenario_id"],
                        "request_count": spec["request_count"],
                        "batch_wall_ms": result["batch_wall_ms"],
                        "total_output_tokens": result["total_output_tokens"],
                    }
                )
                run_order += 1
        return {
            "warmup_runs": warmup_runs,
            "warmup_scenario_runs": len(warmup_summaries),
            "warmup_wall_ms": (time.perf_counter() - started) * 1000,
            "total_output_tokens": sum(row["total_output_tokens"] for row in warmup_summaries),
            "summaries": warmup_summaries,
        }

    ready_ms: float | None = None
    warmup_result: dict[str, Any] = {}
    scenario_runs: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=2.0) as health_client:
            while (time.perf_counter() - server_started) < ready_timeout_s:
                if process.poll() is not None:
                    raise RuntimeError(
                        "vLLM server exited before readiness: "
                        + "\n".join(_tail_lines(server_logs, 40))
                    )
                try:
                    response = health_client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        ready_ms = (time.perf_counter() - server_started) * 1000
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1.0)
        if ready_ms is None:
            raise TimeoutError(
                "vLLM server did not become ready: "
                + "\n".join(_tail_lines(server_logs, 40))
            )

        warmup_result = asyncio.run(run_warmups())
        scenario_plan = []
        for repeat_index in range(repeats):
            for spec in scenario_specs:
                scenario_plan.append((repeat_index, spec))
        random.Random(scenario_seed).shuffle(scenario_plan)
        for run_order, (repeat_index, spec) in enumerate(scenario_plan):
            scenario_runs.append(
                asyncio.run(
                    run_scenario(
                        spec,
                        repeat_index=repeat_index,
                        run_order=run_order,
                    )
                )
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        log_thread.join(timeout=5)

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    scenarios = _aggregate_vllm_server_scenarios(scenario_specs, scenario_runs)
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-sweep",
        "backend": "vllm-openai-server",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "warmup_runs": int(warmup_runs),
        "scenario_count": len(scenarios),
        "scenario_run_count": len(scenario_runs),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "server_ready_ms": ready_ms,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "warmup": warmup_result,
        "summary": _vllm_server_sweep_summary_rows(scenarios),
        "run_summary": _vllm_server_sweep_run_rows(scenario_runs),
        "scenarios": scenarios,
        "scenario_runs": scenario_runs,
        "server_command": command,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "server_logs_head": server_logs[:80],
        "server_logs_tail": _tail_lines(server_logs, 80),
        "note": (
            "One OpenAI-compatible vLLM server runs warmup/discard requests, "
            "then a seeded, repeated, shuffled concurrency sweep over "
            "request count, prompt profile, and output token budget."
        ),
    }


def _run_vllm_server_async_paired_payload(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    phase_order: str = "async_first",
    ready_timeout_s: int = 600,
    max_num_batched_tokens_override: int = 0,
    server_prefix_caching: str = "default",
    gpu_memory_utilization: float = 0.50,
    modal_gpu_label: str = "T4",
) -> dict[str, Any]:
    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be positive")
    if gpu_memory_utilization <= 0 or gpu_memory_utilization > 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    phase_order = phase_order.lower().replace("-", "_")
    if phase_order not in {"async_first", "server_first"}:
        raise ValueError("phase_order must be async_first or server_first")
    server_prefix_caching_mode = _normalize_server_prefix_caching_choice(
        server_prefix_caching
    )
    _validate_vllm_prompt_profiles(prompt_profile_values)

    import asyncio
    import gc
    import json
    import platform
    import random
    import subprocess
    import threading
    import time

    import httpx
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    prompt_format_started = time.perf_counter()
    scenario_specs = _build_vllm_prompt_scenario_specs(
        tokenizer=tokenizer,
        request_count_values=request_count_values,
        prompt_profile_values=prompt_profile_values,
        output_token_values=output_token_values,
        variant_index=scenario_seed,
    )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    max_request_count = max(request_count_values)
    max_output_tokens = max(output_token_values)
    max_prompt_tokens = max(spec["prompt_tokens_max"] for spec in scenario_specs)
    max_model_len = max(1024, max_prompt_tokens + max_output_tokens + 32)
    default_max_num_batched_tokens = max(2048, max_request_count * max_model_len)
    max_num_batched_tokens, max_num_batched_tokens_source = (
        _resolve_max_num_batched_tokens(
            default_max_num_batched_tokens,
            max_num_batched_tokens_override,
            "max_num_batched_tokens_override",
        )
    )
    scenario_plan = []
    for repeat_index in range(repeats):
        for spec in scenario_specs:
            scenario_plan.append((repeat_index, spec))
    random.Random(scenario_seed).shuffle(scenario_plan)
    plan_summary = [
        {
            "run_order": run_order,
            "repeat_index": repeat_index,
            "scenario_id": spec["scenario_id"],
        }
        for run_order, (repeat_index, spec) in enumerate(scenario_plan)
    ]

    async def run_async_phase() -> dict[str, Any]:
        started = time.perf_counter()
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_request_count,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            enforce_eager=True,
            trust_remote_code=False,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000

        async def stream_one(
            run_id: str,
            record: dict[str, Any],
            sampling_params: Any,
            batch_started: float,
        ) -> dict[str, Any]:
            request_started = time.perf_counter()
            first_chunk_ms: float | None = None
            finished_ms: float | None = None
            generated_text_parts = []
            generated_tokens = 0
            chunk_count = 0

            async for output in engine.generate(
                request_id=f"paired-async-{run_id}-{record['request_id']}",
                prompt=record["formatted_prompt"],
                sampling_params=sampling_params,
            ):
                now = time.perf_counter()
                elapsed_ms = (now - request_started) * 1000
                chunk_count += 1
                for completion in output.outputs:
                    text = completion.text
                    token_ids = list(completion.token_ids or [])
                    if first_chunk_ms is None and (text or token_ids):
                        first_chunk_ms = elapsed_ms
                    if text:
                        generated_text_parts.append(text)
                    generated_tokens += len(token_ids)
                if output.finished:
                    finished_ms = elapsed_ms
                    break

            stream_wall_ms = finished_ms or ((time.perf_counter() - request_started) * 1000)
            decode_tokens_after_first = max(generated_tokens - 1, 0)
            decode_after_first_ms = (
                stream_wall_ms - first_chunk_ms
                if first_chunk_ms is not None
                else None
            )
            return {
                "request_id": record["request_id"],
                "prompt_tokens": record["prompt_tokens"],
                "first_chunk_ms": first_chunk_ms,
                "stream_wall_ms": stream_wall_ms,
                "decode_after_first_ms": decode_after_first_ms,
                "stream_tpot_ms": (
                    decode_after_first_ms / decode_tokens_after_first
                    if decode_after_first_ms is not None and decode_tokens_after_first
                    else None
                ),
                "generated_tokens": generated_tokens,
                "total_sequence_tokens": record["prompt_tokens"] + generated_tokens,
                "chunk_count": chunk_count,
                "batch_finished_ms": (time.perf_counter() - batch_started) * 1000,
                "generated_text": "".join(generated_text_parts),
            }

        async def run_scenario(
            spec: dict[str, Any],
            repeat_index: int,
            run_order: int,
            max_tokens_override: int | None = None,
        ) -> dict[str, Any]:
            max_tokens = max_tokens_override or spec["max_new_tokens"]
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
            run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
            batch_started = time.perf_counter()
            request_results = await asyncio.gather(
                *(
                    stream_one(
                        run_id,
                        record,
                        sampling_params,
                        batch_started,
                    )
                    for record in spec["prompt_records"]
                )
            )
            batch_wall_ms = (time.perf_counter() - batch_started) * 1000
            summary = _summarize_stream_requests(request_results, batch_wall_ms)
            return {
                "run_id": run_id,
                "repeat_index": repeat_index,
                "run_order": run_order,
                "scenario_id": spec["scenario_id"],
                "prompt_profile": spec["prompt_profile"],
                "request_count": spec["request_count"],
                "max_new_tokens": max_tokens,
                "prompt_format": spec["prompt_format"],
                "prompt_tokens_min": spec["prompt_tokens_min"],
                "prompt_tokens_max": spec["prompt_tokens_max"],
                "prompt_tokens_mean": spec["prompt_tokens_mean"],
                "total_prompt_tokens": spec["total_prompt_tokens"],
                "estimated_peak_sequence_tokens": sum(
                    request["total_sequence_tokens"] for request in request_results
                ),
                **summary,
                "requests": request_results,
            }

        async def run_warmups() -> dict[str, Any]:
            started = time.perf_counter()
            warmup_summaries = []
            run_order = 0
            for warmup_index in range(warmup_runs):
                for spec in scenario_specs:
                    result = await run_scenario(
                        spec,
                        repeat_index=-(warmup_index + 1),
                        run_order=run_order,
                        max_tokens_override=1,
                    )
                    warmup_summaries.append(
                        {
                            "warmup_index": warmup_index,
                            "scenario_id": spec["scenario_id"],
                            "request_count": spec["request_count"],
                            "batch_wall_ms": result["batch_wall_ms"],
                            "total_output_tokens": result["total_output_tokens"],
                        }
                    )
                    run_order += 1
            return {
                "warmup_runs": warmup_runs,
                "warmup_scenario_runs": len(warmup_summaries),
                "warmup_wall_ms": (time.perf_counter() - started) * 1000,
                "total_output_tokens": sum(row["total_output_tokens"] for row in warmup_summaries),
                "summaries": warmup_summaries,
            }

        try:
            warmup_result = await run_warmups()
            scenario_runs = []
            for run_order, (repeat_index, spec) in enumerate(scenario_plan):
                scenario_runs.append(
                    await run_scenario(
                        spec,
                        repeat_index=repeat_index,
                        run_order=run_order,
                    )
                )
        finally:
            engine.shutdown()

        return {
            "engine_load_ms": engine_load_ms,
            "warmup": warmup_result,
            "scenario_runs": scenario_runs,
        }

    def run_server_phase() -> dict[str, Any]:
        port = 8000
        base_url = f"http://127.0.0.1:{port}"
        command = [
            "vllm",
            "serve",
            hf_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            "half",
            "--max-model-len",
            str(max_model_len),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--max-num-seqs",
            str(max_request_count),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--enforce-eager",
        ]
        command.extend(
            _vllm_server_prefix_caching_args(server_prefix_caching_mode)
        )
        server_logs: list[str] = []
        server_started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def drain_logs() -> None:
            if process.stdout is None:
                return
            for line in process.stdout:
                server_logs.append(line.rstrip())

        log_thread = threading.Thread(target=drain_logs, daemon=True)
        log_thread.start()

        async def stream_one(
            client: httpx.AsyncClient,
            record: dict[str, Any],
            max_tokens: int,
            batch_started: float,
        ) -> dict[str, Any]:
            request_body = {
                "model": hf_model,
                "messages": [{"role": "user", "content": record["prompt"]}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            generated_text_parts: list[str] = []
            usage: dict[str, Any] | None = None
            response_status_code: int | None = None
            first_content_ms: float | None = None
            chunk_count = 0
            request_started = time.perf_counter()
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response_status_code = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    elapsed_ms = (time.perf_counter() - request_started) * 1000
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        if content and first_content_ms is None:
                            first_content_ms = elapsed_ms
                        if content:
                            generated_text_parts.append(content)
                        chunk_count += 1
            request_wall_ms = (time.perf_counter() - request_started) * 1000
            generated_text = "".join(generated_text_parts)
            generated_tokens = (
                int(usage["completion_tokens"])
                if usage and usage.get("completion_tokens") is not None
                else len(tokenizer.encode(generated_text, add_special_tokens=False))
            )
            prompt_tokens = (
                int(usage["prompt_tokens"])
                if usage and usage.get("prompt_tokens") is not None
                else record["prompt_tokens"]
            )
            decode_tokens_after_first = max(generated_tokens - 1, 0)
            decode_after_first_ms = (
                request_wall_ms - first_content_ms
                if first_content_ms is not None
                else None
            )
            return {
                "request_id": record["request_id"],
                "prompt_tokens": prompt_tokens,
                "prompt_tokens_local_estimate": record["prompt_tokens"],
                "generated_tokens": generated_tokens,
                "total_sequence_tokens": prompt_tokens + generated_tokens,
                "generated_text": generated_text,
                "response_status_code": response_status_code,
                "first_content_ms": first_content_ms,
                "request_wall_ms": request_wall_ms,
                "decode_after_first_ms": decode_after_first_ms,
                "stream_tpot_ms": (
                    decode_after_first_ms / decode_tokens_after_first
                    if decode_after_first_ms is not None and decode_tokens_after_first
                    else None
                ),
                "batch_finished_ms": (time.perf_counter() - batch_started) * 1000,
                "chunk_count": chunk_count,
                "usage": usage,
            }

        async def snapshot_server_metrics(
            client: httpx.AsyncClient,
            label: str,
        ) -> dict[str, Any]:
            started = time.perf_counter()
            try:
                response = await client.get(f"{base_url}/metrics")
                parsed = _parse_vllm_server_prometheus_metrics(response.text)
                return {
                    "label": label,
                    "ok": response.status_code == 200,
                    "status_code": response.status_code,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    **parsed,
                }
            except Exception as error:  # pragma: no cover - remote server probe
                return {
                    "label": label,
                    "ok": False,
                    "status_code": None,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                    "sample_count": 0,
                    "prefix_related_sample_count": 0,
                    "prefix_related_samples": [],
                    "total_requests": None,
                    "total_queries": None,
                    "total_hits": None,
                    "hit_rate_pct": None,
                    "selected_request_metric": None,
                    "selected_query_metric": None,
                    "selected_hit_metric": None,
                }

        async def run_scenario(
            spec: dict[str, Any],
            repeat_index: int,
            run_order: int,
            max_tokens_override: int | None = None,
        ) -> dict[str, Any]:
            max_tokens = max_tokens_override or spec["max_new_tokens"]
            run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
            async with httpx.AsyncClient(timeout=None) as client:
                metrics_before = await snapshot_server_metrics(
                    client,
                    f"{run_id}-before",
                )
                batch_started = time.perf_counter()
                request_results = await asyncio.gather(
                    *(
                        stream_one(
                            client,
                            record,
                            max_tokens,
                            batch_started,
                        )
                        for record in spec["prompt_records"]
                    )
                )
                batch_wall_ms = (time.perf_counter() - batch_started) * 1000
                metrics_after = await snapshot_server_metrics(
                    client,
                    f"{run_id}-after",
                )
            counter_delta = _prefix_cache_counter_delta(
                metrics_before,
                metrics_after,
            )
            summary = _summarize_server_stream_requests(request_results, batch_wall_ms)
            return {
                "run_id": run_id,
                "repeat_index": repeat_index,
                "run_order": run_order,
                "scenario_id": spec["scenario_id"],
                "prompt_profile": spec["prompt_profile"],
                "request_count": spec["request_count"],
                "max_new_tokens": max_tokens,
                "prompt_format": spec["prompt_format"],
                "prompt_tokens_min": spec["prompt_tokens_min"],
                "prompt_tokens_max": spec["prompt_tokens_max"],
                "prompt_tokens_mean": spec["prompt_tokens_mean"],
                "total_prompt_tokens": spec["total_prompt_tokens"],
                "estimated_peak_sequence_tokens": sum(
                    request["total_sequence_tokens"] for request in request_results
                ),
                **summary,
                "server_metrics_before": metrics_before,
                "server_metrics_after": metrics_after,
                "server_prefix_cache_counter_delta": counter_delta,
                "server_prefix_cache_counter_requests": counter_delta["requests"],
                "server_prefix_cache_counter_queries": counter_delta["queries"],
                "server_prefix_cache_counter_hits": counter_delta["hits"],
                "server_prefix_cache_counter_hit_rate_pct": counter_delta[
                    "hit_rate_pct"
                ],
                "server_metrics_before_ok": metrics_before["ok"],
                "server_metrics_after_ok": metrics_after["ok"],
                "server_metrics_before_prefix_related_sample_count": metrics_before[
                    "prefix_related_sample_count"
                ],
                "server_metrics_after_prefix_related_sample_count": metrics_after[
                    "prefix_related_sample_count"
                ],
                "server_metrics_selected_query_metric": metrics_after[
                    "selected_query_metric"
                ],
                "server_metrics_selected_hit_metric": metrics_after[
                    "selected_hit_metric"
                ],
                "requests": request_results,
            }

        async def run_warmups() -> dict[str, Any]:
            started = time.perf_counter()
            warmup_summaries = []
            run_order = 0
            for warmup_index in range(warmup_runs):
                for spec in scenario_specs:
                    result = await run_scenario(
                        spec,
                        repeat_index=-(warmup_index + 1),
                        run_order=run_order,
                        max_tokens_override=1,
                    )
                    warmup_summaries.append(
                        {
                            "warmup_index": warmup_index,
                            "scenario_id": spec["scenario_id"],
                            "request_count": spec["request_count"],
                            "batch_wall_ms": result["batch_wall_ms"],
                            "total_output_tokens": result["total_output_tokens"],
                        }
                    )
                    run_order += 1
            return {
                "warmup_runs": warmup_runs,
                "warmup_scenario_runs": len(warmup_summaries),
                "warmup_wall_ms": (time.perf_counter() - started) * 1000,
                "total_output_tokens": sum(row["total_output_tokens"] for row in warmup_summaries),
                "summaries": warmup_summaries,
            }

        ready_ms: float | None = None
        warmup_result: dict[str, Any] = {}
        scenario_runs: list[dict[str, Any]] = []
        try:
            with httpx.Client(timeout=2.0) as health_client:
                while (time.perf_counter() - server_started) < ready_timeout_s:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "vLLM server exited before readiness: "
                            + "\n".join(_tail_lines(server_logs, 40))
                        )
                    try:
                        response = health_client.get(f"{base_url}/health")
                        if response.status_code == 200:
                            ready_ms = (time.perf_counter() - server_started) * 1000
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(1.0)
            if ready_ms is None:
                raise TimeoutError(
                    "vLLM server did not become ready: "
                    + "\n".join(_tail_lines(server_logs, 40))
                )

            warmup_result = asyncio.run(run_warmups())
            for run_order, (repeat_index, spec) in enumerate(scenario_plan):
                scenario_runs.append(
                    asyncio.run(
                        run_scenario(
                            spec,
                            repeat_index=repeat_index,
                            run_order=run_order,
                        )
                    )
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
            log_thread.join(timeout=5)

        return {
            "server_ready_ms": ready_ms,
            "warmup": warmup_result,
            "scenario_runs": scenario_runs,
            "server_command": command,
            "server_logs_head": server_logs[:80],
            "server_logs_tail": _tail_lines(server_logs, 80),
        }

    def release_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2.0)

    if phase_order == "async_first":
        async_result = asyncio.run(run_async_phase())
        release_cuda_memory()
        server_result = run_server_phase()
    else:
        server_result = run_server_phase()
        release_cuda_memory()
        async_result = asyncio.run(run_async_phase())

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    paired_runs = _make_vllm_server_async_paired_rows(
        async_result["scenario_runs"],
        server_result["scenario_runs"],
    )
    paired_scenarios = _aggregate_vllm_server_async_paired_scenarios(paired_runs)
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-async-paired",
        "backend": "vllm-paired-async-and-openai-server",
        "modal_gpu": modal_gpu_label,
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "warmup_runs": int(warmup_runs),
        "phase_order": phase_order,
        "scenario_count": len(paired_scenarios),
        "paired_run_count": len(paired_runs),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "default_max_num_batched_tokens": default_max_num_batched_tokens,
        "max_num_batched_tokens_source": max_num_batched_tokens_source,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": gpu_memory_utilization,
        "async_enable_prefix_caching_configured": False,
        "server_prefix_caching_configured": server_prefix_caching_mode,
        "server_prefix_caching_flag": _vllm_server_prefix_caching_args(
            server_prefix_caching_mode
        ),
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "async_engine_load_ms": async_result["engine_load_ms"],
        "server_ready_ms": server_result["server_ready_ms"],
        "async_warmup": async_result["warmup"],
        "server_warmup": server_result["warmup"],
        "scenario_plan": plan_summary,
        "summary": _vllm_server_async_paired_summary_rows(paired_scenarios),
        "paired_runs": paired_runs,
        "paired_scenarios": paired_scenarios,
        "async_scenario_runs": async_result["scenario_runs"],
        "server_scenario_runs": server_result["scenario_runs"],
        "server_command": server_result["server_command"],
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "server_logs_head": server_result["server_logs_head"],
        "server_logs_tail": server_result["server_logs_tail"],
        "mean_server_to_async_throughput_ratio": _mean_present(
            row["server_to_async_output_tokens_per_second_ratio"] for row in paired_runs
        ),
        "mean_server_to_async_first_event_ratio": _mean_present(
            row["server_to_async_p95_first_event_ms_ratio"] for row in paired_runs
        ),
        "mean_server_to_async_latency_ratio": _mean_present(
            row["server_to_async_p95_latency_ms_ratio"] for row in paired_runs
        ),
        "mean_server_to_async_tpot_ratio": _mean_present(
            row["server_to_async_p95_stream_tpot_ms_ratio"] for row in paired_runs
        ),
        "note": (
            "One Modal worker runs in-process AsyncLLM and the OpenAI-compatible "
            f"vLLM server in phase order {phase_order} with the same scenario plan. "
            "Paired rows compare matching scenario_id and repeat_index. The "
            f"server prefix-cache mode is {server_prefix_caching_mode}."
        ),
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_async_paired_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    phase_order: str = "async_first",
    ready_timeout_s: int = 600,
    max_num_batched_tokens_override: int = 0,
    server_prefix_caching: str = "default",
    gpu_memory_utilization: float = 0.50,
) -> dict[str, Any]:
    return _run_vllm_server_async_paired_payload(
        hf_model=hf_model,
        request_counts=request_counts,
        prompt_profiles=prompt_profiles,
        output_tokens=output_tokens,
        repeats=repeats,
        scenario_seed=scenario_seed,
        warmup_runs=warmup_runs,
        phase_order=phase_order,
        ready_timeout_s=ready_timeout_s,
        max_num_batched_tokens_override=max_num_batched_tokens_override,
        server_prefix_caching=server_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization,
        modal_gpu_label="T4",
    )


@app.function(
    image=vllm_image,
    gpu="L4",
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_async_paired_l4_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    phase_order: str = "async_first",
    ready_timeout_s: int = 600,
    max_num_batched_tokens_override: int = 0,
    server_prefix_caching: str = "default",
    gpu_memory_utilization: float = 0.50,
) -> dict[str, Any]:
    return _run_vllm_server_async_paired_payload(
        hf_model=hf_model,
        request_counts=request_counts,
        prompt_profiles=prompt_profiles,
        output_tokens=output_tokens,
        repeats=repeats,
        scenario_seed=scenario_seed,
        warmup_runs=warmup_runs,
        phase_order=phase_order,
        ready_timeout_s=ready_timeout_s,
        max_num_batched_tokens_override=max_num_batched_tokens_override,
        server_prefix_caching=server_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization,
        modal_gpu_label="L4",
    )


def _select_vllm_server_async_paired_remote(modal_gpu: str) -> Any:
    normalized_gpu = modal_gpu.strip().upper().replace("_", "-")
    if normalized_gpu == "T4":
        return run_vllm_server_async_paired_remote
    if normalized_gpu == "L4":
        return run_vllm_server_async_paired_l4_remote
    raise ValueError("modal_gpu must be one of: T4, L4")


@app.local_entrypoint()
def main(
    mode: str = "sweep",
    workload: str = DEFAULT_WORKLOAD,
    model: str = DEFAULT_MODEL,
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    capacities: str = DEFAULT_CAPACITIES,
    concurrency: str = DEFAULT_CONCURRENCY,
    policies: str = DEFAULT_POLICIES,
    max_new_tokens: int = 32,
    prompt_count: int = 4,
    prompt_profile: str = "short",
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    warmup_runs: int = 1,
    warmup_prompt_profile: str = "",
    phase_order: str = "async_first",
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    prefix_caching: str = "off",
    cache_metrics: str = "off",
    kv_cache_metrics_sample: float = 1.0,
    kv_cache_block_size: int = 16,
    modal_gpu: str = "T4",
    capacity_prefix_cache_modes: str = "false",
    capacity_max_num_batched_tokens: str = "",
    server_async_max_num_batched_tokens: int = 0,
    server_async_prefix_caching: str = "default",
    prefix_cache_max_num_batched_tokens: int = 0,
    gpu_memory_utilization: float = 0.50,
    cold_sweep_dir: str = DEFAULT_VLLM_SWEEP_OUTPUT,
    prefix_sweep_dir: str = DEFAULT_VLLM_PREFIX_CACHE_SWEEP_OUTPUT,
    async_sweep_dir: str = DEFAULT_VLLM_SWEEP_OUTPUT,
    server_sweep_dir: str = DEFAULT_VLLM_SERVER_SWEEP_OUTPUT,
    async_first_paired_dir: str = DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT,
    server_first_paired_dir: str = DEFAULT_VLLM_SERVER_ASYNC_PAIRED_SERVER_FIRST_OUTPUT,
    phase_order_compare_dirs: str = DEFAULT_VLLM_SERVER_ASYNC_PHASE_ORDER_COMPARE_DIRS,
    server_async_log_summary_dirs: str = DEFAULT_VLLM_SERVER_ASYNC_LOG_SUMMARY_DIRS,
    server_async_cache_control_dirs: str = DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_COMPARE_DIRS,
    server_async_cache_control_compare_dirs: str = DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_COMPARE_AGGREGATE_DIRS,
    server_async_cache_control_server_absolute_compare_dirs: str = (
        DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_COMPARE_AGGREGATE_DIRS
    ),
    short_multitrial_dir: str = DEFAULT_VLLM_SERVER_ASYNC_MULTITRIAL_OUTPUT,
    long_phase_order_compare_dir: str = DEFAULT_VLLM_SERVER_ASYNC_LONG_PHASE_ORDER_COMPARE_OUTPUT,
    prefix_cache_cold_first_paired_dir: str = DEFAULT_VLLM_PREFIX_CACHE_PAIRED_OUTPUT,
    prefix_cache_cache_first_paired_dir: str = DEFAULT_VLLM_PREFIX_CACHE_PAIRED_CACHE_FIRST_OUTPUT,
    prefix_cache_phase_order_compare_dir: str = DEFAULT_VLLM_PREFIX_CACHE_PHASE_ORDER_COMPARE_OUTPUT,
    prefix_cache_profile_control_dirs: str = DEFAULT_VLLM_PREFIX_CACHE_PROFILE_CONTROL_DIRS,
    prefix_cache_isolated_metrics_dir: str = DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_METRICS_OUTPUT,
    prefix_cache_isolated_merge_dirs: str = "",
    prefix_cache_prompt_audit_dirs: str = DEFAULT_VLLM_PREFIX_CACHE_PROMPT_OVERLAP_AUDIT_DIRS,
    prefix_cache_shared_profile: str = "shared_prefix_long",
    prefix_cache_control_profile: str = "matched_unique_prefix",
    server_async_cache_control_server_absolute_dir: str = DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_SERVER_ABSOLUTE_DIR,
    output_dir: str = "",
) -> None:
    def normalized_prefix_cache_control_profiles() -> tuple[str, str]:
        shared_profile = prefix_cache_shared_profile.strip().lower().replace("-", "_")
        control_profile = prefix_cache_control_profile.strip().lower().replace("-", "_")
        _validate_vllm_prompt_profiles(
            [shared_profile],
            label="prefix_cache_shared_profile",
        )
        _validate_vllm_prompt_profiles(
            [control_profile],
            label="prefix_cache_control_profile",
        )
        return shared_profile, control_profile

    if mode == "gpu-probe":
        payload = run_gpu_probe_remote.remote()
        output_path = Path(output_dir or DEFAULT_GPU_PROBE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "probe.json"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(f"cuda_available: {payload['cuda_available']}")
        if payload.get("device_name"):
            print(f"device: {payload['device_name']}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-server-cli-help":
        payload = run_vllm_server_cli_help_remote.remote()
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_CLI_HELP_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-server-cli-help.json"
        text_path = output_path / "vllm-server-cli-help.txt"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        text_path.write_text(payload["help_text"], encoding="utf-8")

        print(f"vllm_version: {payload['vllm_version']}")
        print("prefix_related_flags: " + ",".join(payload["prefix_related_flags"]))
        print(
            "has_no_enable_prefix_caching_flag: "
            f"{payload['has_no_enable_prefix_caching_flag']}"
        )
        print(f"json: {json_path}")
        print(f"text: {text_path}")
        return

    if mode == "tiny-inference":
        payload = run_tiny_inference_remote.remote(
            hf_model=hf_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_INFERENCE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "inference.json"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(f"model: {payload['model_id']}")
        print(f"device: {payload['device_name']}")
        print(f"ttft_ms: {payload['ttft_ms']:.3f}")
        print(f"tpot_ms: {payload['tpot_ms']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-prefix-cache-prompt-audit":
        payload = run_vllm_prefix_cache_prompt_audit_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            kv_cache_block_size=kv_cache_block_size,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_PROMPT_AUDIT_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-prompt-audit.json"
        scenario_csv_path = output_path / "prefix-cache-prompt-audit-scenarios.csv"
        prompt_csv_path = output_path / "prefix-cache-prompt-audit-prompts.csv"
        profile_csv_path = output_path / "prefix-cache-prompt-audit-profile-control.csv"
        markdown_path = output_path / "prefix-cache-prompt-audit.md"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_records_csv(scenario_csv_path, payload["scenario_rows"])
        _write_records_csv(prompt_csv_path, payload["prompt_rows"])
        _write_records_csv(profile_csv_path, payload["profile_control_rows"])
        markdown_path.write_text(payload["markdown"], encoding="utf-8")

        print(f"scenarios: {payload['scenario_count']}")
        print(f"prompts: {payload['prompt_count']}")
        print(f"block_size: {payload['kv_cache_block_size']}")
        shared_blocks = payload["summary"].get("shared_common_prefix_full_blocks_mean")
        control_blocks = payload["summary"].get("control_common_prefix_full_blocks_mean")
        reusable_delta = payload["summary"].get(
            "shared_minus_control_reusable_block_tokens_mean"
        )
        shared_blocks_text = (
            f"{shared_blocks:.3f}" if shared_blocks is not None else "n/a"
        )
        control_blocks_text = (
            f"{control_blocks:.3f}" if control_blocks is not None else "n/a"
        )
        reusable_delta_text = (
            f"{reusable_delta:.3f}" if reusable_delta is not None else "n/a"
        )
        print(f"mean_shared_common_prefix_full_blocks: {shared_blocks_text}")
        print(f"mean_control_common_prefix_full_blocks: {control_blocks_text}")
        print(
            "mean_shared_minus_control_reusable_block_tokens: "
            f"{reusable_delta_text}"
        )
        print(f"json: {json_path}")
        print(f"scenario_csv: {scenario_csv_path}")
        print(f"prompt_csv: {prompt_csv_path}")
        print(f"profile_csv: {profile_csv_path}")
        print(f"markdown: {markdown_path}")
        return

    if mode == "vllm-cache-metrics-probe":
        payload = run_vllm_cache_metrics_probe_remote.remote(hf_model=hf_model)
        output_path = Path(output_dir or DEFAULT_VLLM_CACHE_METRICS_PROBE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "cache-metrics-probe.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"vllm_version: {payload['vllm_version']}")
        print(
            "async_engine_args_relevant_parameters: "
            f"{','.join(payload['async_engine_args_relevant_parameters'])}"
        )
        print(
            "observability_config_relevant_parameters: "
            f"{','.join(payload['observability_config_relevant_parameters'])}"
        )
        for check in payload["constructor_checks"]:
            print(f"constructor_check {check['kwargs']}: accepted={check['accepted']}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-capacity-diagnostic":
        capacity_remote = _select_vllm_capacity_diagnostic_remote(modal_gpu)
        payload = capacity_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            scenario_seed=scenario_seed,
            warmup_prompt_profile=warmup_prompt_profile,
            prefix_cache_modes=capacity_prefix_cache_modes,
            max_num_batched_tokens_values=capacity_max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_CAPACITY_DIAGNOSTIC_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "capacity-diagnostic.json"
        csv_path = output_path / "capacity-diagnostic.csv"
        markdown_path = output_path / "capacity-diagnostic.md"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_records_csv(csv_path, payload["summary"])
        markdown_path.write_text(
            _format_vllm_capacity_diagnostic_markdown(payload),
            encoding="utf-8",
        )

        print(f"model: {payload['model_id']}")
        print(f"modal_gpu: {payload['modal_gpu']}")
        print(f"rows: {payload['row_count']}")
        for row in payload["summary"]:
            print(
                "capacity "
                f"n={row['request_count']} "
                f"prefix_cache={row['enable_prefix_caching']} "
                f"max_num_batched_tokens={row['max_num_batched_tokens']} "
                f"source={row['max_num_batched_tokens_source']} "
                f"gpu_kv_cache_size_tokens={row['gpu_kv_cache_size_tokens']} "
                f"max_concurrency={row['max_concurrency_for_request']}"
            )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"markdown: {markdown_path}")
        return

    if mode == "vllm-cache-metrics-smoke":
        payload = run_vllm_cache_metrics_smoke_remote.remote(
            hf_model=hf_model,
            prompt_profile=prompt_profile,
            request_count=prompt_count,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_CACHE_METRICS_SMOKE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "cache-metrics-smoke.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"vllm_version: {payload['vllm_version']}")
        print(f"prompt_profile: {payload['prompt_profile']}")
        print(f"request_count: {payload['request_count']}")
        print(f"engine_load_ms: {payload['smoke']['engine_load_ms']:.3f}")
        for call in payload["smoke"]["log_stats_calls"]:
            print(
                f"log_stats {call['label']}: ok={call['ok']} "
                f"return_type={call['return_type']}"
            )
        print(
            "engine_relevant_attrs: "
            f"{len(payload['smoke']['engine_relevant_attrs'])}"
        )
        print(
            "engine_core_relevant_attrs: "
            f"{len(payload['smoke']['engine_core_relevant_attrs'])}"
        )
        print(f"json: {json_path}")
        return

    if mode == "vllm-inference":
        payload = run_vllm_inference_remote.remote(
            hf_model=hf_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-inference.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"generate_wall_ms: {payload['generate_wall_ms']:.3f}")
        print(f"output_tokens_per_second: {payload['output_tokens_per_second']:.3f}")
        if payload.get("metric_ttft_ms") is not None:
            print(f"metric_ttft_ms: {payload['metric_ttft_ms']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-streaming":
        payload = run_vllm_streaming_remote.remote(
            hf_model=hf_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_STREAMING_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-streaming.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        if payload.get("first_chunk_ms") is not None:
            print(f"first_chunk_ms: {payload['first_chunk_ms']:.3f}")
        if payload.get("stream_tpot_ms") is not None:
            print(f"stream_tpot_ms: {payload['stream_tpot_ms']:.3f}")
        print(f"output_tokens_per_second: {payload['output_tokens_per_second']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-concurrent":
        payload = run_vllm_concurrent_remote.remote(
            hf_model=hf_model,
            prompt_count=prompt_count,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_CONCURRENT_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-concurrent.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"requests: {payload['request_count']}")
        print(f"batch_wall_ms: {payload['batch_wall_ms']:.3f}")
        print(f"p95_first_chunk_ms: {payload['p95_first_chunk_ms']:.3f}")
        print(f"aggregate_output_tokens_per_second: {payload['aggregate_output_tokens_per_second']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-server-streaming":
        payload = run_vllm_server_streaming_remote.remote(
            hf_model=hf_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-server-streaming.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"server_ready_ms: {payload['server_ready_ms']:.3f}")
        if payload.get("first_content_ms") is not None:
            print(f"first_content_ms: {payload['first_content_ms']:.3f}")
        if payload.get("stream_tpot_ms") is not None:
            print(f"stream_tpot_ms: {payload['stream_tpot_ms']:.3f}")
        print(f"output_tokens_per_second: {payload['output_tokens_per_second']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-server-concurrent":
        payload = run_vllm_server_concurrent_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profile=prompt_profile,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_CONCURRENT_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-server-concurrent.json"
        csv_path = output_path / "vllm-server-concurrent.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"server_ready_ms: {payload['server_ready_ms']:.3f}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"max_num_seqs: {payload['max_num_seqs']}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-sweep":
        payload = run_vllm_server_sweep_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            warmup_runs=warmup_runs,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_SWEEP_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-server-sweep.json"
        csv_path = output_path / "vllm-server-sweep.csv"
        run_csv_path = output_path / "vllm-server-sweep-runs.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])
        _write_records_csv(run_csv_path, payload["run_summary"])

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"server_ready_ms: {payload['server_ready_ms']:.3f}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"scenario_runs: {payload['scenario_run_count']}")
        print(f"repeats: {payload['repeats']}")
        print(f"warmup_runs: {payload['warmup_runs']}")
        print(f"scenario_seed: {payload['scenario_seed']}")
        print(f"max_num_seqs: {payload['max_num_seqs']}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"runs_csv: {run_csv_path}")
        return

    if mode == "vllm-server-sweep-compare":
        payload = _compare_vllm_server_async_csvs(
            async_csv=Path(async_sweep_dir) / "vllm-sweep.csv",
            server_csv=Path(server_sweep_dir) / "vllm-server-sweep.csv",
            async_label="in_process_async_llm",
            server_label="openai_compatible_server",
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_SWEEP_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "server-vs-async.json"
        csv_path = output_path / "server-vs-async.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"scenarios: {payload['scenario_count']}")
        print(
            "mean_server_to_async_throughput_ratio: "
            f"{payload['mean_server_to_async_throughput_ratio']:.3f}"
        )
        print(
            "mean_server_to_async_latency_ratio: "
            f"{payload['mean_server_to_async_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-async-paired":
        server_async_remote = _select_vllm_server_async_paired_remote(modal_gpu)
        payload = server_async_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            warmup_runs=warmup_runs,
            phase_order=phase_order,
            max_num_batched_tokens_override=server_async_max_num_batched_tokens,
            server_prefix_caching=server_async_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        phase_order_slug = phase_order.lower().replace("_", "-")
        default_output = (
            DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT
            if phase_order_slug == "async-first"
            else f"{DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT}-{phase_order_slug}"
        )
        output_path = Path(output_dir or default_output)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "paired-server-async.json"
        summary_csv_path = output_path / "paired-server-async-summary.csv"
        run_csv_path = output_path / "paired-server-async-runs.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(summary_csv_path, payload["summary"])
        _write_records_csv(run_csv_path, payload["paired_runs"])

        print(f"model: {payload['model_id']}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"paired_runs: {payload['paired_run_count']}")
        print(f"repeats: {payload['repeats']}")
        print(f"warmup_runs: {payload['warmup_runs']}")
        print(f"phase_order: {payload['phase_order']}")
        print(f"modal_gpu: {payload['modal_gpu']}")
        print(
            "server_prefix_caching_configured: "
            f"{payload['server_prefix_caching_configured']}"
        )
        print(f"max_num_batched_tokens: {payload['max_num_batched_tokens']}")
        print(
            "max_num_batched_tokens_source: "
            f"{payload['max_num_batched_tokens_source']}"
        )
        print(f"gpu_memory_utilization: {payload['gpu_memory_utilization']}")
        print(
            "mean_server_to_async_throughput_ratio: "
            f"{payload['mean_server_to_async_throughput_ratio']:.3f}"
        )
        print(
            "mean_server_to_async_latency_ratio: "
            f"{payload['mean_server_to_async_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"summary_csv: {summary_csv_path}")
        print(f"runs_csv: {run_csv_path}")
        return

    if mode == "vllm-server-async-log-summary":
        payload = _summarize_vllm_server_async_log_metrics(
            [Path(path) for path in _split_csv(server_async_log_summary_dirs)]
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_ASYNC_LOG_SUMMARY_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "server-async-log-summary.json"
        csv_path = output_path / "server-async-log-summary.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"rows: {payload['row_count']}")
        print(
            "all_server_enable_prefix_caching_observed: "
            f"{payload['all_server_enable_prefix_caching_observed']}"
        )
        print(
            "server_enable_prefix_caching_observed_true_count: "
            f"{payload['server_enable_prefix_caching_observed_true_count']}"
        )
        print(
            "server_enable_prefix_caching_observed_false_count: "
            f"{payload['server_enable_prefix_caching_observed_false_count']}"
        )
        print(
            "mean_server_latest_prefix_cache_hit_rate_pct: "
            f"{_format_optional_float(payload['mean_server_latest_prefix_cache_hit_rate_pct'])}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-async-phase-order-compare":
        payload = _compare_vllm_server_async_phase_orders(
            async_first_dir=Path(async_first_paired_dir),
            server_first_dir=Path(server_first_paired_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_ASYNC_PHASE_ORDER_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "phase-order-compare.json"
        csv_path = output_path / "phase-order-compare.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"scenarios: {payload['scenario_count']}")
        print(
            "async_first_mean_server_to_async_throughput_ratio: "
            f"{payload['async_first']['mean_server_to_async_throughput_ratio']:.3f}"
        )
        print(
            "server_first_mean_server_to_async_throughput_ratio: "
            f"{payload['server_first']['mean_server_to_async_throughput_ratio']:.3f}"
        )
        print(
            "async_first_mean_server_to_async_latency_ratio: "
            f"{payload['async_first']['mean_server_to_async_latency_ratio']:.3f}"
        )
        print(
            "server_first_mean_server_to_async_latency_ratio: "
            f"{payload['server_first']['mean_server_to_async_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-async-multitrial-aggregate":
        payload = _aggregate_vllm_server_async_phase_order_trials(
            compare_dirs=[
                Path(path)
                for path in _split_csv(phase_order_compare_dirs)
            ],
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_ASYNC_MULTITRIAL_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "phase-order-multitrial.json"
        csv_path = output_path / "phase-order-multitrial.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])

        print(f"trials: {payload['trial_count']}")
        for row in payload["summary"]:
            print(
                f"{row['metric']}: async_first_mean={row['async_first_mean']:.3f} "
                f"server_first_mean={row['server_first_mean']:.3f} "
                f"delta_mean={row['server_first_minus_async_first_mean']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-async-workload-compare":
        payload = _compare_vllm_server_async_workload_profiles(
            short_multitrial_dir=Path(short_multitrial_dir),
            long_phase_order_compare_dir=Path(long_phase_order_compare_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_ASYNC_WORKLOAD_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "workload-compare.json"
        csv_path = output_path / "workload-compare.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"metrics: {payload['metric_count']}")
        print(f"short_trials: {payload['short_trial_count']}")
        print(f"long_paired_runs: {payload['long_paired_run_count']}")
        for row in payload["rows"]:
            print(
                f"{row['metric']}: short_delta_mean="
                f"{row['short_order_effect_delta_mean']:.3f} long_delta_mean="
                f"{row['long_order_effect_delta_mean']:.3f} "
                f"long_minus_short_delta_mean="
                f"{row['long_minus_short_order_effect_delta_mean']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-async-cache-control-compare":
        payload = _compare_vllm_server_async_cache_control_matrix(
            [Path(path) for path in _split_csv(server_async_cache_control_dirs)]
        )
        output_path = Path(
            output_dir or DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_COMPARE_OUTPUT
        )
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "cache-control-phase-order-compare.json"
        matrix_csv_path = output_path / "cache-control-matrix.csv"
        contrast_csv_path = output_path / "cache-control-contrasts.csv"
        summary_csv_path = output_path / "cache-control-mode-summary.csv"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_records_csv(matrix_csv_path, payload["matrix_rows"])
        _write_records_csv(contrast_csv_path, payload["phase_order_cache_contrasts"])
        _write_records_csv(summary_csv_path, payload["cache_mode_summary"])

        print(f"matrix_rows: {payload['row_count']}")
        for row in payload["phase_order_cache_contrasts"]:
            print(
                f"{row['phase_order']}: on_minus_off_throughput_ratio="
                f"{row['cache_on_minus_off_throughput_ratio']:.3f} "
                f"on_minus_off_latency_ratio="
                f"{row['cache_on_minus_off_latency_ratio']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"matrix_csv: {matrix_csv_path}")
        print(f"contrast_csv: {contrast_csv_path}")
        print(f"summary_csv: {summary_csv_path}")
        return

    if mode == "vllm-server-async-cache-control-multitrial":
        payload = _aggregate_vllm_server_async_cache_control_trials(
            [
                Path(path)
                for path in _split_csv(server_async_cache_control_compare_dirs)
            ]
        )
        output_path = Path(
            output_dir or DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_MULTITRIAL_OUTPUT
        )
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "cache-control-multitrial.json"
        trials_csv_path = output_path / "cache-control-multitrial-trials.csv"
        summary_csv_path = output_path / "cache-control-multitrial-summary.csv"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_records_csv(trials_csv_path, payload["trial_rows"])
        _write_records_csv(summary_csv_path, payload["summary"])

        print(f"trials: {payload['trial_count']}")
        for row in payload["summary"]:
            if row["metric"] != "cache_on_minus_off_throughput_ratio":
                continue
            print(
                f"{row['phase_order']}: throughput_delta_mean="
                f"{row['mean']:.3f} bootstrap_p05={row['bootstrap_mean_p05']:.3f} "
                f"bootstrap_p95={row['bootstrap_mean_p95']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"trials_csv: {trials_csv_path}")
        print(f"summary_csv: {summary_csv_path}")
        return

    if mode == "vllm-server-async-cache-control-server-absolute":
        payload = _compare_vllm_server_async_cache_control_server_absolute(
            [
                Path(path)
                for path in _split_csv(
                    server_async_cache_control_server_absolute_compare_dirs
                )
            ]
        )
        output_path = Path(
            output_dir
            or DEFAULT_VLLM_SERVER_ASYNC_CACHE_CONTROL_SERVER_ABSOLUTE_OUTPUT
        )
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "server-cache-control-absolute.json"
        trials_csv_path = output_path / "server-cache-control-absolute-trials.csv"
        summary_csv_path = output_path / "server-cache-control-absolute-summary.csv"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_records_csv(trials_csv_path, payload["trial_rows"])
        _write_records_csv(summary_csv_path, payload["summary"])

        print(f"trial_contrasts: {payload['contrast_count']}")
        for row in payload["summary"]:
            if row["metric"] != "cache_on_div_off_server_output_tokens_per_second":
                continue
            print(
                f"{row['phase_order']} {row['prompt_profile']}: "
                f"throughput_ratio_mean={row['mean']:.3f} "
                f"bootstrap_p05={row['bootstrap_mean_p05']:.3f} "
                f"bootstrap_p95={row['bootstrap_mean_p95']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"trials_csv: {trials_csv_path}")
        print(f"summary_csv: {summary_csv_path}")
        return

    if mode == "vllm-prefix-cache-prompt-overlap-server-compare":
        payload = _compare_vllm_prefix_cache_prompt_overlap_with_server_cache_control(
            prompt_audit_dirs=[
                Path(path)
                for path in _split_csv(prefix_cache_prompt_audit_dirs)
            ],
            server_absolute_dir=Path(server_async_cache_control_server_absolute_dir),
        )
        output_path = Path(
            output_dir
            or DEFAULT_VLLM_PREFIX_CACHE_PROMPT_OVERLAP_SERVER_COMPARE_OUTPUT
        )
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prompt-overlap-server-compare.json"
        audit_csv_path = output_path / "prompt-overlap-audit-summary.csv"
        server_csv_path = output_path / "prompt-overlap-server-summary.csv"
        markdown_path = output_path / "prompt-overlap-server-compare.md"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_records_csv(audit_csv_path, payload["audit_profile_summary_rows"])
        _write_records_csv(server_csv_path, payload["server_join_rows"])
        markdown_path.write_text(payload["markdown"], encoding="utf-8")

        print(f"audit_trials: {payload['audit_trial_count']}")
        print(
            "control_common_prefix_full_blocks_mean: "
            f"{payload['summary']['control_common_prefix_full_blocks_mean']:.3f}"
        )
        print(
            "matched_unique_min_server_throughput_ratio_mean: "
            f"{payload['summary']['matched_unique_min_server_throughput_ratio_mean']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"audit_csv: {audit_csv_path}")
        print(f"server_csv: {server_csv_path}")
        print(f"markdown: {markdown_path}")
        return

    if mode == "vllm-sweep":
        enable_prefix_caching = _parse_bool_choice(prefix_caching, "prefix_caching")
        payload = run_vllm_sweep_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            enable_prefix_caching=enable_prefix_caching,
        )
        default_output = (
            DEFAULT_VLLM_PREFIX_CACHE_SWEEP_OUTPUT
            if enable_prefix_caching
            else DEFAULT_VLLM_SWEEP_OUTPUT
        )
        output_path = Path(output_dir or default_output)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-sweep.json"
        csv_path = output_path / "vllm-sweep.csv"
        run_csv_path = output_path / "vllm-sweep-runs.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])
        _write_records_csv(run_csv_path, payload["run_summary"])

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"scenario_runs: {payload['scenario_run_count']}")
        print(f"repeats: {payload['repeats']}")
        print(f"scenario_seed: {payload['scenario_seed']}")
        print(f"enable_prefix_caching: {payload['enable_prefix_caching']}")
        print(f"max_num_seqs: {payload['max_num_seqs']}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"runs_csv: {run_csv_path}")
        return

    if mode == "vllm-prefix-cache-isolated-window-summary":
        shared_profile, control_profile = normalized_prefix_cache_control_profiles()
        payload = _summarize_vllm_prefix_cache_isolated_window(
            isolated_metrics_dir=Path(prefix_cache_isolated_metrics_dir),
            shared_profile=shared_profile,
            control_profile=control_profile,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_WINDOW_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-window-summary.json"
        csv_path = output_path / "prefix-cache-window-summary.csv"
        profile_csv_path = output_path / "prefix-cache-window-profile-control.csv"
        markdown_path = output_path / "prefix-cache-window-summary.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["scenario_rows"])
        _write_records_csv(profile_csv_path, payload["profile_control_rows"])
        markdown_path.write_text(payload["markdown"], encoding="utf-8")

        print(f"scenario_rows: {payload['scenario_count']}")
        print(f"profile_control_rows: {payload['profile_control_row_count']}")
        print(
            "mean_shared_minus_control_estimated_measured_cache_hit_rate_pct: "
            f"{payload['summary']['mean_shared_minus_control_estimated_measured_cache_hit_rate_pct']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"profile_csv: {profile_csv_path}")
        print(f"markdown: {markdown_path}")
        return

    if mode == "vllm-prefix-cache-isolated-stability-summary":
        shared_profile, control_profile = normalized_prefix_cache_control_profiles()
        payload = _summarize_vllm_prefix_cache_isolated_stability(
            isolated_metrics_dir=Path(prefix_cache_isolated_metrics_dir),
            shared_profile=shared_profile,
            control_profile=control_profile,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_STABILITY_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-isolated-stability-summary.json"
        csv_path = output_path / "prefix-cache-isolated-stability-summary.csv"
        profile_csv_path = output_path / "prefix-cache-isolated-stability-profile-control.csv"
        markdown_path = output_path / "prefix-cache-isolated-stability-summary.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["scenario_rows"])
        _write_records_csv(profile_csv_path, payload["profile_control_rows"])
        markdown_path.write_text(payload["markdown"], encoding="utf-8")

        print(f"scenario_rows: {payload['scenario_count']}")
        print(f"profile_control_rows: {payload['profile_control_row_count']}")
        print(
            "mean_shared_minus_control_cache_hit_rate_pct: "
            f"{payload['summary']['mean_shared_minus_control_cache_hit_rate_pct']:.3f}"
        )
        print(
            "max_cache_hit_rate_population_stdev: "
            f"{payload['summary']['max_cache_hit_rate_population_stdev']:.3f}"
        )
        counter_summary = payload["summary"].get(
            "mean_shared_minus_control_cache_counter_hit_rate_pct"
        )
        if counter_summary is not None:
            print(
                "mean_shared_minus_control_cache_counter_hit_rate_pct: "
                f"{counter_summary:.3f}"
            )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"profile_csv: {profile_csv_path}")
        print(f"markdown: {markdown_path}")
        return

    if mode == "vllm-prefix-cache-isolated-merge":
        shared_profile, control_profile = normalized_prefix_cache_control_profiles()
        payload = _merge_vllm_prefix_cache_isolated_metrics(
            source_dirs=[
                Path(path)
                for path in _split_csv(prefix_cache_isolated_merge_dirs)
            ],
            shared_profile=shared_profile,
            control_profile=control_profile,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_MERGE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-isolated-metrics.json"
        summary_csv_path = output_path / "prefix-cache-isolated-metrics-summary.csv"
        runs_csv_path = output_path / "prefix-cache-isolated-metrics-runs.csv"
        profile_csv_path = output_path / "prefix-cache-isolated-profile-control.csv"
        chunks_csv_path = output_path / "prefix-cache-isolated-merge-sources.csv"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_records_csv(summary_csv_path, payload["summary"])
        _write_records_csv(runs_csv_path, payload["paired_runs"])
        _write_records_csv(profile_csv_path, payload["profile_control"]["rows"])
        _write_records_csv(chunks_csv_path, payload["source_chunks"])

        print(f"source_chunks: {payload['source_chunk_count']}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"paired_runs: {payload['paired_run_count']}")
        print(f"remote_calls: {payload['remote_call_count']}")
        print(f"checkpoint_complete: {payload['checkpoint_complete']}")
        print(f"json: {json_path}")
        print(f"summary_csv: {summary_csv_path}")
        print(f"runs_csv: {runs_csv_path}")
        print(f"profile_csv: {profile_csv_path}")
        print(f"sources_csv: {chunks_csv_path}")
        return

    if mode in {
        "vllm-prefix-cache-isolated-metrics",
        "vllm-prefix-cache-isolated-warm-window",
        "vllm-prefix-cache-isolated-neutral-warmup",
    }:
        paired_remote = _select_vllm_prefix_cache_paired_remote(modal_gpu)
        modal_gpu_value = modal_gpu.strip().upper().replace("_", "-")
        shared_profile, control_profile = normalized_prefix_cache_control_profiles()
        request_count_values = _split_positive_int_csv(request_counts, "request_counts")
        prompt_profile_values = [
            profile.lower().replace("-", "_")
            for profile in _split_csv(prompt_profiles)
        ]
        output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        _validate_vllm_prompt_profiles(prompt_profile_values)
        isolated_phase_order = phase_order.lower().replace("-", "_")
        if isolated_phase_order not in {"cold_first", "cache_first"}:
            isolated_phase_order = "cold_first"
        effective_warmup_runs = (
            1
            if mode
            in {
                "vllm-prefix-cache-isolated-warm-window",
                "vllm-prefix-cache-isolated-neutral-warmup",
            }
            else 0
        )
        effective_warmup_prompt_profile = (
            warmup_prompt_profile.strip().lower().replace("-", "_")
            if warmup_prompt_profile.strip()
            else "neutral_long"
            if mode == "vllm-prefix-cache-isolated-neutral-warmup"
            else ""
        )

        default_output = (
            DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_WARM_WINDOW_OUTPUT
            if mode == "vllm-prefix-cache-isolated-warm-window"
            else DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_NEUTRAL_WARMUP_OUTPUT
            if mode == "vllm-prefix-cache-isolated-neutral-warmup"
            else DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_METRICS_OUTPUT
        )
        output_path = Path(output_dir or default_output)
        output_path.mkdir(parents=True, exist_ok=True)
        planned_remote_call_count = (
            int(repeats)
            * len(prompt_profile_values)
            * len(output_token_values)
            * len(request_count_values)
        )

        paired_runs = []
        remote_call_summaries = []

        def build_isolated_payload(checkpoint_complete: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            paired_scenarios = _aggregate_vllm_prefix_cache_paired_scenarios(
                paired_runs
            )
            summary_rows = _vllm_prefix_cache_paired_summary_rows(paired_scenarios)
            profile_control_rows = _vllm_prefix_cache_isolated_profile_control_rows(
                paired_scenarios,
                shared_profile=shared_profile,
                control_profile=control_profile,
            )
            payload = {
                "schema_version": 1,
                "execution": "modal",
                "mode": mode,
                "backend": "vllm-paired-prefix-cache",
                "modal_gpu": modal_gpu_value,
                "model_id": hf_model,
                "request_counts": request_count_values,
                "prompt_profiles": prompt_profile_values,
                "output_tokens": output_token_values,
                "repeats": int(repeats),
                "scenario_seed": int(scenario_seed),
                "phase_order": isolated_phase_order,
                "warmup_runs": effective_warmup_runs,
                "warmup_prompt_profile": effective_warmup_prompt_profile or None,
                "collect_cache_metrics": True,
                "kv_cache_metrics_sample": float(kv_cache_metrics_sample),
                "max_num_batched_tokens_override": int(
                    prefix_cache_max_num_batched_tokens
                ),
                "checkpoint_complete": checkpoint_complete,
                "planned_remote_call_count": planned_remote_call_count,
                "isolation_method": (
                    "Each scenario/repeat is executed by a separate remote paired "
                    "benchmark call with fresh cold/cache AsyncLLM engines. The "
                    f"run uses {effective_warmup_runs} warmup scenario run(s) "
                    "inside each fresh engine before the measured scenario. "
                    f"Warmup prompt profile: {effective_warmup_prompt_profile or 'measured scenario'}."
                ),
                "remote_call_count": len(remote_call_summaries),
                "scenario_count": len(paired_scenarios),
                "paired_run_count": len(paired_runs),
                "summary": summary_rows,
                "paired_runs": paired_runs,
                "paired_scenarios": paired_scenarios,
                "profile_control": {
                    "shared_profile": shared_profile,
                    "control_profile": control_profile,
                    "row_count": len(profile_control_rows),
                    "summary": _vllm_prefix_cache_isolated_profile_control_summary(
                        profile_control_rows
                    ),
                    "rows": profile_control_rows,
                },
                "remote_call_summaries": remote_call_summaries,
                "mean_cache_to_cold_throughput_ratio": _mean_present(
                    row["cache_to_cold_output_tokens_per_second_ratio"]
                    for row in paired_runs
                ),
                "mean_cache_to_cold_first_event_ratio": _mean_present(
                    row["cache_to_cold_p95_first_event_ms_ratio"]
                    for row in paired_runs
                ),
                "mean_cache_to_cold_latency_ratio": _mean_present(
                    row["cache_to_cold_p95_latency_ms_ratio"] for row in paired_runs
                ),
                "mean_cache_to_cold_tpot_ratio": _mean_present(
                    row["cache_to_cold_p95_stream_tpot_ms_ratio"]
                    for row in paired_runs
                ),
                "note": (
                    "This is a metric-isolation harness. The warm-window mode adds "
                    "a throwaway warmup window before the measured scenario; the "
                    "default isolated mode runs with no warmup."
                ),
            }
            return payload, profile_control_rows

        def write_isolated_outputs(
            payload: dict[str, Any],
            profile_control_rows: list[dict[str, Any]],
            partial: bool,
        ) -> tuple[Path, Path, Path, Path]:
            suffix = ".partial" if partial else ""
            json_path = output_path / f"prefix-cache-isolated-metrics{suffix}.json"
            summary_csv_path = (
                output_path / f"prefix-cache-isolated-metrics-summary{suffix}.csv"
            )
            runs_csv_path = (
                output_path / f"prefix-cache-isolated-metrics-runs{suffix}.csv"
            )
            profile_csv_path = (
                output_path / f"prefix-cache-isolated-profile-control{suffix}.csv"
            )
            json_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _write_records_csv(summary_csv_path, payload["summary"])
            _write_records_csv(runs_csv_path, payload["paired_runs"])
            _write_records_csv(profile_csv_path, profile_control_rows)
            return json_path, summary_csv_path, runs_csv_path, profile_csv_path

        call_index = 0
        for repeat_index in range(repeats):
            for prompt_profile_value in prompt_profile_values:
                for max_tokens in output_token_values:
                    for request_count_value in request_count_values:
                        single_payload = paired_remote.remote(
                            hf_model=hf_model,
                            request_counts=str(request_count_value),
                            prompt_profiles=prompt_profile_value,
                            output_tokens=str(max_tokens),
                            repeats=1,
                            scenario_seed=scenario_seed + repeat_index,
                            warmup_runs=effective_warmup_runs,
                            warmup_prompt_profile=effective_warmup_prompt_profile,
                            phase_order=isolated_phase_order,
                            collect_cache_metrics=True,
                            kv_cache_metrics_sample=kv_cache_metrics_sample,
                            max_num_batched_tokens_override=(
                                prefix_cache_max_num_batched_tokens
                            ),
                        )
                        if single_payload["paired_run_count"] != 1:
                            raise ValueError(
                                "Expected one paired run from isolated metrics call"
                            )
                        row = dict(single_payload["paired_runs"][0])
                        row["modal_gpu"] = single_payload.get("modal_gpu")
                        row["repeat_index"] = repeat_index
                        row["pair_id"] = (
                            f"{row['scenario_id']}_isolated_rep{repeat_index:02d}"
                        )
                        row["isolation_call_index"] = call_index
                        row["isolation_phase_order"] = single_payload["phase_order"]
                        row["isolation_warmup_runs"] = effective_warmup_runs
                        row["isolation_warmup_prompt_profile"] = (
                            effective_warmup_prompt_profile or None
                        )
                        paired_runs.append(row)

                        cold_run = single_payload["cold_scenario_runs"][0]
                        cache_run = single_payload["cache_scenario_runs"][0]
                        remote_call_summaries.append(
                            {
                                "isolation_call_index": call_index,
                                "modal_gpu": single_payload.get("modal_gpu"),
                                "repeat_index": repeat_index,
                                "scenario_id": row["scenario_id"],
                                "prompt_profile": row["prompt_profile"],
                                "request_count": row["request_count"],
                                "max_new_tokens": row["max_new_tokens"],
                                "max_num_batched_tokens": single_payload[
                                    "max_num_batched_tokens"
                                ],
                                "default_max_num_batched_tokens": single_payload[
                                    "default_max_num_batched_tokens"
                                ],
                                "max_num_batched_tokens_source": single_payload[
                                    "max_num_batched_tokens_source"
                                ],
                                "phase_order": single_payload["phase_order"],
                                "warmup_prompt_profile": single_payload[
                                    "warmup_prompt_profile"
                                ],
                                "warmup_scenario_count": single_payload[
                                    "warmup_scenario_count"
                                ],
                                "cold_engine_load_ms": single_payload[
                                    "cold_engine_load_ms"
                                ],
                                "cache_engine_load_ms": single_payload[
                                    "cache_engine_load_ms"
                                ],
                                "cold_engine_capacity": single_payload.get(
                                    "cold_engine_capacity"
                                ),
                                "cache_engine_capacity": single_payload.get(
                                    "cache_engine_capacity"
                                ),
                                "nvidia_smi_before": single_payload.get(
                                    "nvidia_smi_before"
                                ),
                                "nvidia_smi_after": single_payload.get(
                                    "nvidia_smi_after"
                                ),
                                "cold_prefix_cache_hit_rate_pct": row[
                                    "cold_prefix_cache_hit_rate_pct"
                                ],
                                "cache_prefix_cache_hit_rate_pct": row[
                                    "cache_prefix_cache_hit_rate_pct"
                                ],
                                "cache_to_cold_prefix_cache_hit_rate_pct_delta": row[
                                    "cache_to_cold_prefix_cache_hit_rate_pct_delta"
                                ],
                                "cold_prefix_cache_counter_queries": row[
                                    "cold_prefix_cache_counter_queries"
                                ],
                                "cache_prefix_cache_counter_queries": row[
                                    "cache_prefix_cache_counter_queries"
                                ],
                                "cold_prefix_cache_counter_hits": row[
                                    "cold_prefix_cache_counter_hits"
                                ],
                                "cache_prefix_cache_counter_hits": row[
                                    "cache_prefix_cache_counter_hits"
                                ],
                                "cold_prefix_cache_counter_hit_rate_pct": row[
                                    "cold_prefix_cache_counter_hit_rate_pct"
                                ],
                                "cache_prefix_cache_counter_hit_rate_pct": row[
                                    "cache_prefix_cache_counter_hit_rate_pct"
                                ],
                                "cache_to_cold_prefix_cache_counter_hit_rate_pct_delta": row[
                                    "cache_to_cold_prefix_cache_counter_hit_rate_pct_delta"
                                ],
                                "cache_to_cold_output_tokens_per_second_ratio": row[
                                    "cache_to_cold_output_tokens_per_second_ratio"
                                ],
                                "cache_to_cold_p95_latency_ms_ratio": row[
                                    "cache_to_cold_p95_latency_ms_ratio"
                                ],
                                "warmup_runs": effective_warmup_runs,
                                "cold_warmup": single_payload["cold_warmup"],
                                "cache_warmup": single_payload["cache_warmup"],
                                "cold_warmup_cache_metrics": single_payload[
                                    "cold_warmup_cache_metrics"
                                ],
                                "cache_warmup_cache_metrics": single_payload[
                                    "cache_warmup_cache_metrics"
                                ],
                                "cold_initial_prefix_cache_counter": single_payload[
                                    "cold_initial_prefix_cache_counter"
                                ],
                                "cache_initial_prefix_cache_counter": single_payload[
                                    "cache_initial_prefix_cache_counter"
                                ],
                                "cold_warmup_prefix_cache_counter": single_payload[
                                    "cold_warmup_prefix_cache_counter"
                                ],
                                "cache_warmup_prefix_cache_counter": single_payload[
                                    "cache_warmup_prefix_cache_counter"
                                ],
                                "cold_warmup_prefix_cache_counter_delta": single_payload[
                                    "cold_warmup_prefix_cache_counter_delta"
                                ],
                                "cache_warmup_prefix_cache_counter_delta": single_payload[
                                    "cache_warmup_prefix_cache_counter_delta"
                                ],
                                "cold_warmup_cache_metrics_log_stats": single_payload[
                                    "cold_warmup_cache_metrics_log_stats"
                                ],
                                "cache_warmup_cache_metrics_log_stats": single_payload[
                                    "cache_warmup_cache_metrics_log_stats"
                                ],
                                "cold_cache_metrics": cold_run.get("cache_metrics"),
                                "cache_cache_metrics": cache_run.get("cache_metrics"),
                                "cold_cache_metrics_log_stats": cold_run.get(
                                    "cache_metrics_log_stats"
                                ),
                                "cache_cache_metrics_log_stats": cache_run.get(
                                    "cache_metrics_log_stats"
                                ),
                                "cold_prefix_cache_counter_before": cold_run.get(
                                    "prefix_cache_counter_before"
                                ),
                                "cold_prefix_cache_counter_after": cold_run.get(
                                    "prefix_cache_counter_after"
                                ),
                                "cold_prefix_cache_counter_delta": cold_run.get(
                                    "prefix_cache_counter_delta"
                                ),
                                "cache_prefix_cache_counter_before": cache_run.get(
                                    "prefix_cache_counter_before"
                                ),
                                "cache_prefix_cache_counter_after": cache_run.get(
                                    "prefix_cache_counter_after"
                                ),
                                "cache_prefix_cache_counter_delta": cache_run.get(
                                    "prefix_cache_counter_delta"
                                ),
                            }
                        )
                        call_index += 1
                        partial_payload, partial_profile_control_rows = (
                            build_isolated_payload(checkpoint_complete=False)
                        )
                        write_isolated_outputs(
                            partial_payload,
                            partial_profile_control_rows,
                            partial=True,
                        )
                        print(
                            "checkpoint_remote_calls: "
                            f"{partial_payload['remote_call_count']}/"
                            f"{partial_payload['planned_remote_call_count']}"
                        )

        payload, profile_control_rows = build_isolated_payload(checkpoint_complete=True)
        json_path, summary_csv_path, runs_csv_path, profile_csv_path = write_isolated_outputs(
            payload,
            profile_control_rows,
            partial=False,
        )

        print(f"scenarios: {payload['scenario_count']}")
        print(f"paired_runs: {payload['paired_run_count']}")
        print(f"remote_calls: {payload['remote_call_count']}")
        print(f"phase_order: {payload['phase_order']}")
        print(f"modal_gpu: {payload['modal_gpu']}")
        print(f"warmup_runs: {payload['warmup_runs']}")
        print(f"warmup_prompt_profile: {payload['warmup_prompt_profile']}")
        print(
            "max_num_batched_tokens_override: "
            f"{payload['max_num_batched_tokens_override']}"
        )
        print(
            "mean_cache_to_cold_throughput_ratio: "
            f"{payload['mean_cache_to_cold_throughput_ratio']:.3f}"
        )
        print(
            "mean_cache_to_cold_latency_ratio: "
            f"{payload['mean_cache_to_cold_latency_ratio']:.3f}"
        )
        profile_summary = payload["profile_control"]["summary"]
        if profile_summary:
            print(
                "mean_shared_minus_control_cache_hit_rate_delta: "
                f"{profile_summary['mean_shared_minus_control_cache_prefix_cache_hit_rate_pct_median']:.3f}"
            )
            counter_key = (
                "mean_shared_minus_control_"
                "cache_prefix_cache_counter_hit_rate_pct_median"
            )
            if profile_summary.get(counter_key) is not None:
                print(
                    "mean_shared_minus_control_cache_counter_hit_rate_delta: "
                    f"{profile_summary[counter_key]:.3f}"
                )
        print(f"json: {json_path}")
        print(f"summary_csv: {summary_csv_path}")
        print(f"runs_csv: {runs_csv_path}")
        print(f"profile_csv: {profile_csv_path}")
        return

    if mode == "vllm-prefix-cache-paired":
        collect_cache_metrics = _parse_bool_choice(cache_metrics, "cache_metrics")
        paired_remote = _select_vllm_prefix_cache_paired_remote(modal_gpu)
        payload = paired_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            warmup_runs=warmup_runs,
            warmup_prompt_profile=warmup_prompt_profile,
            phase_order=phase_order,
            collect_cache_metrics=collect_cache_metrics,
            kv_cache_metrics_sample=kv_cache_metrics_sample,
            max_num_batched_tokens_override=prefix_cache_max_num_batched_tokens,
        )
        phase_order_slug = payload["phase_order"].lower().replace("_", "-")
        default_output = (
            DEFAULT_VLLM_PREFIX_CACHE_PAIRED_OUTPUT
            if phase_order_slug == "cold-first"
            else f"{DEFAULT_VLLM_PREFIX_CACHE_PAIRED_OUTPUT}-{phase_order_slug}"
        )
        output_path = Path(output_dir or default_output)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "paired-prefix-cache.json"
        summary_csv_path = output_path / "paired-prefix-cache-summary.csv"
        run_csv_path = output_path / "paired-prefix-cache-runs.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(summary_csv_path, payload["summary"])
        _write_records_csv(run_csv_path, payload["paired_runs"])

        print(f"model: {payload['model_id']}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"paired_runs: {payload['paired_run_count']}")
        print(f"repeats: {payload['repeats']}")
        print(f"warmup_runs: {payload['warmup_runs']}")
        print(f"warmup_prompt_profile: {payload['warmup_prompt_profile']}")
        print(f"phase_order: {payload['phase_order']}")
        print(f"modal_gpu: {payload['modal_gpu']}")
        print(f"collect_cache_metrics: {payload['collect_cache_metrics']}")
        print(f"max_num_batched_tokens: {payload['max_num_batched_tokens']}")
        print(
            "max_num_batched_tokens_source: "
            f"{payload['max_num_batched_tokens_source']}"
        )
        print(
            "mean_cache_to_cold_throughput_ratio: "
            f"{payload['mean_cache_to_cold_throughput_ratio']:.3f}"
        )
        print(
            "mean_cache_to_cold_latency_ratio: "
            f"{payload['mean_cache_to_cold_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"summary_csv: {summary_csv_path}")
        print(f"runs_csv: {run_csv_path}")
        return

    if mode == "vllm-prefix-cache-compare":
        payload = _compare_vllm_sweep_csvs(
            cold_csv=Path(cold_sweep_dir) / "vllm-sweep.csv",
            prefix_csv=Path(prefix_sweep_dir) / "vllm-sweep.csv",
            cold_label="prefix_caching_off",
            prefix_label="prefix_caching_on",
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-compare.json"
        csv_path = output_path / "prefix-cache-compare.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"scenarios: {payload['scenario_count']}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-prefix-cache-phase-order-compare":
        payload = _compare_vllm_prefix_cache_phase_orders(
            cold_first_dir=Path(prefix_cache_cold_first_paired_dir),
            cache_first_dir=Path(prefix_cache_cache_first_paired_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_PHASE_ORDER_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-phase-order-compare.json"
        csv_path = output_path / "prefix-cache-phase-order-compare.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"scenarios: {payload['scenario_count']}")
        print(
            "cold_first_mean_cache_to_cold_throughput_ratio: "
            f"{payload['cold_first']['mean_cache_to_cold_throughput_ratio']:.3f}"
        )
        print(
            "cache_first_mean_cache_to_cold_throughput_ratio: "
            f"{payload['cache_first']['mean_cache_to_cold_throughput_ratio']:.3f}"
        )
        print(
            "cold_first_mean_cache_to_cold_latency_ratio: "
            f"{payload['cold_first']['mean_cache_to_cold_latency_ratio']:.3f}"
        )
        print(
            "cache_first_mean_cache_to_cold_latency_ratio: "
            f"{payload['cache_first']['mean_cache_to_cold_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-prefix-cache-profile-control":
        payload = _compare_vllm_prefix_cache_profile_controls(
            phase_order_compare_dir=Path(prefix_cache_phase_order_compare_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_PROFILE_CONTROL_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-profile-control.json"
        csv_path = output_path / "prefix-cache-profile-control.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"profile_control_rows: {payload['row_count']}")
        print(
            "mean_shared_minus_control_cold_first_throughput_ratio: "
            f"{payload['summary']['mean_shared_minus_control_cold_first_output_tokens_per_second_ratio']:.3f}"
        )
        print(
            "mean_shared_minus_control_cache_first_throughput_ratio: "
            f"{payload['summary']['mean_shared_minus_control_cache_first_output_tokens_per_second_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-prefix-cache-profile-multitrial":
        payload = _aggregate_vllm_prefix_cache_profile_control_trials(
            profile_control_dirs=[
                Path(path)
                for path in _split_csv(prefix_cache_profile_control_dirs)
            ],
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_PROFILE_MULTITRIAL_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-profile-multitrial.json"
        csv_path = output_path / "prefix-cache-profile-multitrial.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])

        print(f"trials: {payload['trial_count']}")
        for row in payload["summary"]:
            print(
                f"{row['phase_order']} {row['metric']}: "
                f"mean_delta={row['shared_minus_control_mean']:.3f} "
                f"min={row['shared_minus_control_min']:.3f} "
                f"max={row['shared_minus_control_max']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode != "sweep":
        raise ValueError(
            "mode must be 'sweep', 'gpu-probe', 'tiny-inference', "
            "'vllm-cache-metrics-probe', 'vllm-inference', "
            "'vllm-cache-metrics-smoke', 'vllm-streaming', 'vllm-concurrent', "
            "'vllm-server-streaming', 'vllm-server-concurrent', "
            "'vllm-server-sweep', 'vllm-server-sweep-compare', "
            "'vllm-server-cli-help', "
            "'vllm-server-async-paired', "
            "'vllm-server-async-log-summary', "
            "'vllm-server-async-phase-order-compare', "
            "'vllm-server-async-multitrial-aggregate', "
            "'vllm-server-async-workload-compare', "
            "'vllm-server-async-cache-control-compare', "
            "'vllm-server-async-cache-control-multitrial', "
            "'vllm-server-async-cache-control-server-absolute', "
            "'vllm-sweep', or "
            "'vllm-prefix-cache-isolated-window-summary', "
            "'vllm-prefix-cache-isolated-stability-summary', "
            "'vllm-prefix-cache-prompt-overlap-server-compare', "
            "'vllm-prefix-cache-prompt-audit', "
            "'vllm-prefix-cache-isolated-merge', "
            "'vllm-prefix-cache-isolated-neutral-warmup', "
            "'vllm-prefix-cache-isolated-warm-window', "
            "'vllm-prefix-cache-isolated-metrics', "
            "'vllm-prefix-cache-paired', 'vllm-prefix-cache-compare', or "
            "'vllm-prefix-cache-phase-order-compare', or "
            "'vllm-prefix-cache-profile-control', or "
            "'vllm-prefix-cache-profile-multitrial'"
        )

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


def _split_positive_int_csv(value: str, label: str) -> list[int]:
    raw_values = _split_csv(value)
    parsed_values = []
    for raw_value in raw_values:
        try:
            parsed_value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"{label} must contain positive integers") from error
        if parsed_value <= 0:
            raise ValueError(f"{label} must contain positive integers")
        parsed_values.append(parsed_value)
    return parsed_values


def _resolve_max_num_batched_tokens(
    default_value: int,
    override_value: int,
    label: str,
) -> tuple[int, str]:
    if default_value <= 0:
        raise ValueError("default max_num_batched_tokens must be positive")
    if override_value < 0:
        raise ValueError(f"{label} must be zero or a positive integer")
    if override_value == 0:
        return default_value, "default"
    return override_value, "override"


def _normalize_server_prefix_caching_choice(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"default", "auto"}:
        return "default"
    if normalized in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
        return "on"
    if normalized in {"0", "false", "no", "n", "off", "disabled", "disable"}:
        return "off"
    raise ValueError("server_prefix_caching must be default, on, or off")


def _vllm_server_prefix_caching_args(mode: str) -> list[str]:
    normalized_mode = _normalize_server_prefix_caching_choice(mode)
    if normalized_mode == "default":
        return []
    if normalized_mode == "on":
        return ["--enable-prefix-caching"]
    return ["--no-enable-prefix-caching"]


def _build_vllm_prompt_scenario_specs(
    *,
    tokenizer: Any,
    request_count_values: list[int],
    prompt_profile_values: list[str],
    output_token_values: list[int],
    variant_index: int,
) -> list[dict[str, Any]]:
    scenario_specs = []
    for prompt_profile in prompt_profile_values:
        for max_new_tokens in output_token_values:
            for request_count in request_count_values:
                prompt_records = []
                prompts = _select_sweep_prompts(
                    request_count,
                    prompt_profile,
                    variant_index=variant_index,
                )
                for index, prompt in enumerate(prompts):
                    formatted_prompt, prompt_format = _format_prompt_for_generation(
                        tokenizer,
                        prompt,
                    )
                    prompt_records.append(
                        {
                            "request_id": (
                                f"{prompt_profile}-out{max_new_tokens}-"
                                f"n{request_count}-{index:02d}"
                            ),
                            "prompt": prompt,
                            "formatted_prompt": formatted_prompt,
                            "prompt_format": prompt_format,
                            "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
                        }
                    )
                prompt_tokens = [record["prompt_tokens"] for record in prompt_records]
                scenario_specs.append(
                    {
                        "scenario_id": (
                            f"{prompt_profile}_out{max_new_tokens}_n{request_count}"
                        ),
                        "prompt_profile": prompt_profile,
                        "request_count": request_count,
                        "max_new_tokens": max_new_tokens,
                        "prompt_format": prompt_records[0]["prompt_format"],
                        "prompt_tokens_min": min(prompt_tokens),
                        "prompt_tokens_max": max(prompt_tokens),
                        "prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
                        "total_prompt_tokens": sum(prompt_tokens),
                        "prompt_records": prompt_records,
                    }
                )
    return scenario_specs


def _validate_vllm_prompt_profiles(profiles: list[str], label: str = "prompt_profiles") -> None:
    invalid_profiles = [
        profile for profile in profiles
        if profile not in VALID_VLLM_PROMPT_PROFILES
    ]
    if invalid_profiles:
        valid_profiles = ", ".join(sorted(VALID_VLLM_PROMPT_PROFILES))
        invalid_list = ", ".join(invalid_profiles)
        raise ValueError(
            f"{label} contains unsupported profiles: {invalid_list}. "
            f"Valid profiles: {valid_profiles}"
        )


def _parse_bool_choice(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    raise ValueError(f"{label} must be on or off")


def _select_concurrent_prompts(prompt_count: int) -> list[str]:
    prompts = []
    for index in range(prompt_count):
        base_prompt = DEFAULT_CONCURRENT_PROMPTS[index % len(DEFAULT_CONCURRENT_PROMPTS)]
        if index < len(DEFAULT_CONCURRENT_PROMPTS):
            prompts.append(base_prompt)
        else:
            prompts.append(f"{base_prompt} Variation {index + 1}.")
    return prompts


def _select_sweep_prompts(
    prompt_count: int,
    prompt_profile: str,
    variant_index: int = 0,
) -> list[str]:
    prompt_profile = prompt_profile.lower().replace("-", "_")
    short_prompts = _select_concurrent_prompts(prompt_count)
    if prompt_profile == "short":
        return short_prompts
    long_prompts = [
        _extend_prompt_for_context(prompt, index)
        for index, prompt in enumerate(short_prompts)
    ]
    if prompt_profile == "long":
        return long_prompts
    if prompt_profile == "mixed":
        return [
            prompt if index % 2 == 0 else long_prompts[index]
            for index, prompt in enumerate(short_prompts)
        ]
    if prompt_profile == "shared_prefix":
        return _select_shared_prefix_prompts(prompt_count)
    if prompt_profile == "shared_prefix_long":
        return _select_shared_prefix_prompts(prompt_count, long_context=True)
    if prompt_profile == "shared_prefix_long_variant":
        return _select_shared_prefix_variant_prompts(prompt_count, variant_index)
    if prompt_profile == "shared_prefix_long_no_repeat_variant":
        return _select_shared_prefix_variant_prompts(
            prompt_count,
            variant_index,
            allow_prompt_repeats=False,
        )
    if prompt_profile == "shared_prefix_extra_long_no_repeat_variant":
        return _select_shared_prefix_variant_prompts(
            prompt_count,
            variant_index,
            allow_prompt_repeats=False,
            extra_context_repeats=EXTRA_LONG_PREFIX_CONTEXT_REPEATS,
        )
    if prompt_profile == "shared_prefix_ultra_long_no_repeat_variant":
        return _select_shared_prefix_variant_prompts(
            prompt_count,
            variant_index,
            allow_prompt_repeats=False,
            extra_context_repeats=ULTRA_LONG_PREFIX_CONTEXT_REPEATS,
        )
    if prompt_profile == "shared_prefix_mega_long_no_repeat_variant":
        return _select_shared_prefix_variant_prompts(
            prompt_count,
            variant_index,
            allow_prompt_repeats=False,
            extra_context_repeats=MEGA_LONG_PREFIX_CONTEXT_REPEATS,
        )
    if prompt_profile == "matched_unique_prefix":
        return _select_matched_unique_prefix_prompts(prompt_count)
    if prompt_profile == "matched_unique_prefix_variant":
        return _select_matched_unique_prefix_variant_prompts(
            prompt_count,
            variant_index,
        )
    if prompt_profile == "matched_unique_prefix_no_repeat_variant":
        return _select_matched_unique_prefix_variant_prompts(
            prompt_count,
            variant_index,
            allow_prompt_repeats=False,
        )
    if prompt_profile == "matched_unique_prefix_extra_long_no_repeat_variant":
        return _select_matched_unique_prefix_variant_prompts(
            prompt_count,
            variant_index,
            allow_prompt_repeats=False,
            extra_context_repeats=EXTRA_LONG_PREFIX_CONTEXT_REPEATS,
        )
    if prompt_profile == "matched_unique_prefix_ultra_long_no_repeat_variant":
        return _select_matched_unique_prefix_variant_prompts(
            prompt_count,
            variant_index,
            allow_prompt_repeats=False,
            extra_context_repeats=ULTRA_LONG_PREFIX_CONTEXT_REPEATS,
        )
    if prompt_profile == "matched_unique_prefix_mega_long_no_repeat_variant":
        return _select_matched_unique_prefix_variant_prompts(
            prompt_count,
            variant_index,
            allow_prompt_repeats=False,
            extra_context_repeats=MEGA_LONG_PREFIX_CONTEXT_REPEATS,
        )
    if prompt_profile == "neutral_long":
        return _select_neutral_long_prompts(prompt_count)
    if prompt_profile == "neutral_extra_long":
        return _select_neutral_long_prompts(
            prompt_count,
            extra_context_repeats=EXTRA_LONG_PREFIX_CONTEXT_REPEATS,
            allow_prompt_repeats=False,
        )
    if prompt_profile == "neutral_ultra_long":
        return _select_neutral_long_prompts(
            prompt_count,
            extra_context_repeats=ULTRA_LONG_PREFIX_CONTEXT_REPEATS,
            allow_prompt_repeats=False,
        )
    if prompt_profile == "neutral_mega_long":
        return _select_neutral_long_prompts(
            prompt_count,
            extra_context_repeats=MEGA_LONG_PREFIX_CONTEXT_REPEATS,
            allow_prompt_repeats=False,
        )
    valid_profiles = ", ".join(sorted(VALID_VLLM_PROMPT_PROFILES))
    raise ValueError(f"prompt_profile must be one of: {valid_profiles}")


def _select_shared_prefix_prompts(
    prompt_count: int,
    long_context: bool = False,
) -> list[str]:
    common_prefix = (
        "You are evaluating a GPU inference platform during a capacity incident. "
        "Every request in this batch shares the same incident brief, model "
        "deployment notes, and operational constraints. The service runs an "
        "OpenAI-compatible endpoint backed by vLLM on one memory-constrained GPU. "
        "Traffic arrives in bursts, responses stream to users, prefill competes "
        "with decode for scheduler time, and the KV cache can dominate available "
        "memory when many long contexts overlap. Operators care about p95 time to "
        "first token, p95 end-to-end latency, output tokens per second, and the "
        "amount of reusable prefix work. The common incident facts are: model "
        "weights are already resident, requests use deterministic decoding, "
        "prefix cache blocks can be reused only when leading prompt tokens match "
        "exactly, and request-specific details appear only after this shared "
        "brief. Use the shared context as ground truth and answer the final task."
    )
    if long_context:
        common_prefix = f"{common_prefix}\n\n{_long_prefix_cache_context()}"
    suffixes = [
        "Task A: explain how prefix caching changes prefill cost for this batch.",
        "Task B: identify which metric should move first if KV reuse is effective.",
        "Task C: describe the failure mode if cache blocks fragment under load.",
        "Task D: compare throughput impact at one, two, four, and eight requests.",
        "Task E: explain why decode may dominate after prefill is avoided.",
        "Task F: name one measurement that would separate cache reuse from noise.",
        "Task G: summarize how phase order could confound this experiment.",
        "Task H: recommend the next benchmark to validate the observed effect.",
    ]
    prompts = []
    for index in range(prompt_count):
        suffix = suffixes[index % len(suffixes)]
        prompts.append(f"{common_prefix}\n\n{suffix}")
    return prompts


def _select_matched_unique_prefix_prompts(prompt_count: int) -> list[str]:
    suffixes = [
        "Task A: explain how prefix caching changes prefill cost for this batch.",
        "Task B: identify which metric should move first if KV reuse is effective.",
        "Task C: describe the failure mode if cache blocks fragment under load.",
        "Task D: compare throughput impact at one, two, four, and eight requests.",
        "Task E: explain why decode may dominate after prefill is avoided.",
        "Task F: name one measurement that would separate cache reuse from noise.",
        "Task G: summarize how phase order could confound this experiment.",
        "Task H: recommend the next benchmark to validate the observed effect.",
    ]
    prompts = []
    for index in range(prompt_count):
        suffix = suffixes[index % len(suffixes)]
        unique_context = _matched_unique_prefix_context(index)
        prompts.append(f"{unique_context}\n\n{suffix}")
    return prompts


def _select_shared_prefix_variant_prompts(
    prompt_count: int,
    variant_index: int,
    allow_prompt_repeats: bool = True,
    extra_context_repeats: int = 0,
) -> list[str]:
    family = _prefix_cache_variant_family(variant_index)
    common_prefix = (
        f"{family['title']} shared incident packet. Every request in this batch "
        f"begins with the same {family['system']} runbook, rollout timeline, "
        "model-serving envelope, and measurement checklist. The platform is "
        "testing whether vLLM can reuse exact leading prompt blocks when the "
        "request-specific task appears only after a long stable context. The "
        f"shared operational facts are: {family['facts']} The service uses "
        "deterministic decoding, a fixed output budget, a single warm model, "
        "and one memory-constrained GPU. The evaluator should treat this shared "
        "packet as the identical reusable prefix for all requests in the batch.\n\n"
        f"{family['detail']}\n\n"
        "Cache decision rule. Exact prefix reuse should reduce repeated prefill "
        "work only after complete cache blocks have been populated. The metric "
        "of record is the measured-window prefix-cache query and hit counter; "
        "latency and throughput are secondary because this small model can be "
        "dominated by scheduler and kernel noise. A good result separates "
        "large exact-prefix reuse from generic warm-engine effects."
    )
    extra_context = _extra_long_shared_variant_context(
        family,
        extra_context_repeats,
    )
    if extra_context:
        common_prefix = f"{common_prefix}\n\n{extra_context}"
    suffixes = _variant_task_suffixes(
        family,
        prompt_count,
        allow_repeats=allow_prompt_repeats,
    )
    prompts = []
    for index in range(prompt_count):
        suffix = (
            suffixes[index % len(suffixes)]
            if allow_prompt_repeats
            else suffixes[index]
        )
        prompts.append(f"{common_prefix}\n\n{suffix}")
    return prompts


def _select_matched_unique_prefix_variant_prompts(
    prompt_count: int,
    variant_index: int,
    allow_prompt_repeats: bool = True,
    extra_context_repeats: int = 0,
) -> list[str]:
    family = _prefix_cache_variant_family(variant_index)
    suffixes = _variant_task_suffixes(
        family,
        prompt_count,
        allow_repeats=allow_prompt_repeats,
    )
    labels = _variant_control_labels(prompt_count, allow_repeats=allow_prompt_repeats)
    prompts = []
    for index in range(prompt_count):
        label = (
            labels[index % len(labels)]
            if allow_prompt_repeats
            else labels[index]
        )
        suffix = (
            suffixes[index % len(suffixes)]
            if allow_prompt_repeats
            else suffixes[index]
        )
        extra_context = _extra_long_control_variant_context(
            family,
            label,
            extra_context_repeats,
        )
        extra_context_block = f"{extra_context}\n\n" if extra_context else ""
        unique_context = (
            f"{label} {family['title']} control packet. This request starts "
            "with a distinct tenant marker, dashboard route, owner alias, "
            "timeline summary, and mitigation note before it reaches the common "
            f"{family['system']} vocabulary. It intentionally avoids the long "
            "identical leading block used by the shared-prefix workload while "
            "keeping the same serving shape, output budget, benchmark topic, "
            "and deterministic decode settings.\n\n"
            f"Control background. {family['facts']} The request still discusses "
            "prefill, decode, cache blocks, GPU memory pressure, and p95 "
            "latency, but each prompt has a different first paragraph and a "
            "different incident identifier. That makes the beginning of every "
            "request diverge before a reusable prefix block can form.\n\n"
            f"{family['detail']}\n\n"
            f"{extra_context_block}"
            "Control decision rule. If this unique-prefix control improves by "
            "the same amount as the shared-prefix profile, the benchmark is "
            "probably measuring warm allocation state or generic shape effects. "
            "If the control stays low while the shared profile reports high "
            "direct counter reuse, the result supports an exact-prefix KV-cache "
            "claim."
        )
        prompts.append(f"{unique_context}\n\n{suffix}")
    return prompts


def _prefix_cache_variant_family(variant_index: int) -> dict[str, str]:
    families = [
        {
            "title": "Payments Relay",
            "system": "payment authorization gateway",
            "facts": (
                "issuer retries are spiking, idempotency keys must be preserved, "
                "the fraud-score service is healthy, and queue delay is highest "
                "during prefill-heavy bursts."
            ),
            "detail": (
                "The packet tracks card-network routing, regional failover, "
                "tenant-specific retry budgets, rate-limit headers, settlement "
                "windows, and a rollout guard that prevents unsafe provider "
                "switches. Operators compare prompt prefill pressure, decode "
                "throughput, queue admission order, and prefix-cache reuse before "
                "changing the serving policy."
            ),
            "object": "payment batch",
        },
        {
            "title": "Search Indexer",
            "system": "search indexing pipeline",
            "facts": (
                "freshness lag is rising, replica compaction is delayed, shard "
                "ownership is stable, and query serving must remain online while "
                "background indexing catches up."
            ),
            "detail": (
                "The packet covers crawler checkpoints, segment merge pressure, "
                "document normalization, hot-shard routing, cache invalidation, "
                "and a staged rollback plan. Operators compare prompt prefill "
                "pressure, decode throughput, queue admission order, and "
                "prefix-cache reuse before changing the serving policy."
            ),
            "object": "indexing batch",
        },
        {
            "title": "Telemetry Warehouse",
            "system": "telemetry warehouse ingestion service",
            "facts": (
                "schema drift warnings are elevated, late partitions are being "
                "replayed, checksum mismatches are isolated to one region, and "
                "dashboard freshness must stay within the incident objective."
            ),
            "detail": (
                "The packet tracks stream offsets, warehouse compaction, table "
                "ownership, retry windows, quality gates, and dashboard publish "
                "order. Operators compare prompt prefill pressure, decode "
                "throughput, queue admission order, and prefix-cache reuse before "
                "changing the serving policy."
            ),
            "object": "telemetry batch",
        },
    ]
    return families[variant_index % len(families)]


def _variant_control_labels(
    prompt_count: int,
    allow_repeats: bool = True,
) -> list[str]:
    labels = [
        "Atlas-01",
        "Beacon-02",
        "Cinder-03",
        "Drift-04",
        "Ember-05",
        "Flux-06",
        "Graph-07",
        "Helix-08",
    ]
    if allow_repeats or prompt_count <= len(labels):
        return labels

    expanded = list(labels)
    stems = [
        "Ion",
        "Juno",
        "Kilo",
        "Lumen",
        "Mica",
        "Nova",
        "Orion",
        "Pulse",
        "Quill",
        "Rivet",
        "Sol",
        "Trace",
        "Umbra",
        "Vector",
        "Warden",
        "Xeno",
    ]
    for index in range(len(expanded), prompt_count):
        stem = stems[(index - len(labels)) % len(stems)]
        expanded.append(f"{stem}-{index + 1:02d}")
    return expanded


def _variant_task_suffixes(
    family: dict[str, str],
    prompt_count: int | None = None,
    allow_repeats: bool = True,
) -> list[str]:
    suffixes = [
        f"Task A: explain how prefix caching changes prefill cost for this {family['object']}.",
        "Task B: identify which direct counter should move if exact KV reuse is effective.",
        "Task C: describe one failure mode if cache blocks fragment under this workload.",
        "Task D: compare the expected effect at two, four, and eight concurrent requests.",
        "Task E: explain why decode can dominate after repeated prefill work is avoided.",
        "Task F: name one timing metric that should not be overclaimed from this run.",
        "Task G: summarize how a matched unique-prefix control protects the conclusion.",
        "Task H: recommend the next benchmark variation to validate this cache signal.",
    ]
    if allow_repeats or prompt_count is None or prompt_count <= len(suffixes):
        return suffixes

    extra_templates = [
        "Task I: list two scheduler effects that can obscure a prefix-cache timing win.",
        "Task J: explain why a duplicate prompt is a different control than a unique prefix.",
        "Task K: describe how block size changes the observed reusable-token estimate.",
        "Task L: identify one Modal run parameter needed for a reproducible artifact.",
        "Task M: compare direct counter evidence with cumulative log-stat evidence.",
        "Task N: state why a small model can understate prefill savings in timing.",
        "Task O: recommend a request-count sweep that avoids duplicate prompt reuse.",
        "Task P: summarize the expected direct-counter gap for a no-repeat control.",
        "Task Q: explain how prompt hashing should be audited before claiming speedup.",
        "Task R: name one benchmark disclosure needed for ML infrastructure interviews.",
        "Task S: describe a failure case involving tokenizer template scaffolding.",
        "Task T: explain how exact-prefix reuse differs from semantic prompt similarity.",
        "Task U: propose one larger-model follow-up for this cache signal.",
        "Task V: describe how p95 latency should be interpreted in paired runs.",
        "Task W: explain why request ordering can change cache hit opportunities.",
        "Task X: state one reason to keep raw per-run CSV artifacts.",
    ]
    while len(suffixes) < prompt_count:
        template = extra_templates[(len(suffixes) - 8) % len(extra_templates)]
        suffixes.append(f"{template} Scenario item {len(suffixes) + 1}.")
    return suffixes


def _extra_long_shared_variant_context(
    family: dict[str, str],
    repeats: int,
) -> str:
    sections = []
    for index in range(max(repeats, 0)):
        sections.append(
            f"Extended shared prefill section {index + 1}. The {family['object']} "
            f"uses the same {family['system']} terminology, incident objective, "
            "traffic envelope, rollout guard, failure budget, and benchmark "
            "measurement plan for every request in this batch. The paragraph is "
            "intentionally stable across requests so complete leading cache "
            "blocks can be populated once and queried by later requests. The "
            "operator review repeats the same fields: admission timestamp, "
            "tenant priority, retry policy, backpressure reason, model routing "
            "state, tokenizer policy, deterministic decode setting, expected "
            "output budget, and trace identifiers for prefill, decode, stream "
            "TPOT, and p95 first-event latency. The final task remains short "
            "and request-specific only after this shared section ends."
        )
    return "\n\n".join(sections)


def _extra_long_control_variant_context(
    family: dict[str, str],
    label: str,
    repeats: int,
) -> str:
    sections = []
    for index in range(max(repeats, 0)):
        sections.append(
            f"Extended control prefill section {index + 1} for {label}. This "
            f"{family['object']} uses a distinct route marker, dashboard owner, "
            "incident objective, traffic envelope, rollout guard, failure "
            "budget, and benchmark measurement plan before it reaches language "
            "that resembles the shared profile. The paragraph is intentionally "
            "long but not a shared leading prefix across requests. It repeats "
            "comparable fields for admission timestamp, tenant priority, retry "
            "policy, backpressure reason, model routing state, tokenizer policy, "
            "deterministic decode setting, expected output budget, and trace "
            "identifiers for prefill, decode, stream TPOT, and p95 first-event "
            "latency. The final task remains short and request-specific only "
            "after this distinct control section ends."
        )
    return "\n\n".join(sections)


def _extra_long_neutral_context(label: str, repeats: int) -> str:
    sections = []
    for index in range(max(repeats, 0)):
        sections.append(
            f"Extended neutral warmup section {index + 1} for {label}. This "
            "calibration text describes an unrelated analytics review with a "
            "separate owner, route marker, freshness objective, retry plan, "
            "validation checklist, and publication decision. It is long enough "
            "to exercise prefill allocation and scheduling, but it avoids the "
            "measured incident packet, the shared prefix language, and the "
            "matched-control decision rule. The paragraph repeats comparable "
            "serving fields for admission timestamp, tenant priority, model "
            "routing state, tokenizer policy, deterministic decode setting, "
            "output budget, prefill trace, decode trace, stream TPOT, and p95 "
            "first-event latency so the warmup shape is close without seeding "
            "the measured prefix-cache claim."
        )
    return "\n\n".join(sections)


def _select_neutral_long_prompts(
    prompt_count: int,
    extra_context_repeats: int = 0,
    allow_prompt_repeats: bool = True,
) -> list[str]:
    suffixes = [
        "Calibration task A: classify which queue should receive this packet.",
        "Calibration task B: summarize the dominant constraint in one sentence.",
        "Calibration task C: identify one field that would be useful to log.",
        "Calibration task D: compare the packet with a steady-state request.",
        "Calibration task E: explain how admission order affects the report.",
        "Calibration task F: name one sanity check for the measurement window.",
        "Calibration task G: describe the expected scheduler bookkeeping.",
        "Calibration task H: recommend the next diagnostic counter to inspect.",
    ]
    prompts = []
    for index in range(prompt_count):
        suffix = suffixes[index % len(suffixes)]
        unique_context = _neutral_long_context(index)
        if not allow_prompt_repeats:
            suffix = f"{suffix} Calibration item {index + 1}."
            unique_context = (
                f"{unique_context}\n\n"
                f"Neutral no-repeat marker {index + 1}. This calibration request "
                "keeps the same warmup role but carries a unique item number so "
                "the extra-long warmup batch does not contain exact duplicates."
            )
        extra_context = _extra_long_neutral_context(
            unique_context.split(" ", 1)[0],
            extra_context_repeats,
        )
        if extra_context:
            unique_context = f"{unique_context}\n\n{extra_context}"
        prompts.append(f"{unique_context}\n\n{suffix}")
    return prompts


def _long_prefix_cache_context() -> str:
    return (
        "Extended incident packet. The workload trace contains one warm model, "
        "one tokenizer, a fixed CUDA graph policy, and an arrival burst where "
        "all user requests begin with the same runbook, tenant limits, model "
        "deployment notes, and safety constraints. The platform team suspects "
        "prefill dominates early latency because every prompt repeats the same "
        "policy and trace summary before reaching a short request-specific task. "
        "The scheduler admits all requests at nearly the same time, so reusable "
        "prefix blocks should reduce repeated attention work during prefill while "
        "leaving decode token generation mostly unchanged.\n\n"
        "Cache mechanics. Prefix-cache reuse is only valid when leading tokens "
        "match exactly through complete cache blocks. A small edit near the front "
        "of the prompt should destroy most reuse, while a long identical incident "
        "brief should allow later requests to skip repeated prefix computation "
        "after blocks are populated. This benchmark therefore watches first-token "
        "latency, end-to-end p95 latency, output tokens per second, and time per "
        "output token. A true cache effect should be strongest when multiple "
        "concurrent requests share the same long leading context.\n\n"
        "Operational constraints. The GPU is a memory-constrained T4, the model "
        "is small enough to run cheaply but still exercises vLLM scheduling, "
        "decoding is deterministic, output lengths are fixed by scenario, and "
        "the paired experiment runs prefix caching off and on inside one Modal "
        "worker. The evidence should separate prefix reuse from warm-engine "
        "effects, phase order, prompt length, and random service noise.\n\n"
        "Decision rule. If the shared-prefix profile improves with caching while "
        "the matched unique-prefix profile does not, the result supports a real "
        "prefix-cache claim. If both profiles improve similarly, the benchmark is "
        "probably measuring warm state or shape effects instead of cache reuse."
    )


def _matched_unique_prefix_context(index: int) -> str:
    labels = [
        "Alpha-17",
        "Bravo-26",
        "Crimson-35",
        "Delta-44",
        "Ember-53",
        "Falcon-62",
        "Graphite-71",
        "Harbor-80",
    ]
    label = labels[index % len(labels)]
    return (
        f"{label} incident packet. This request intentionally starts with a "
        "different leading identifier, tenant trace, service owner, and rollout "
        "history so the large reusable prefix from the shared-context workload "
        "is not present. The workload still contains one warm model, one "
        "tokenizer, a fixed CUDA graph policy, and an arrival burst where user "
        "requests have comparable runbooks, tenant limits, model deployment "
        "notes, and safety constraints. The platform team suspects prefill "
        "dominates early latency because each prompt carries a long operational "
        "brief before reaching a short request-specific task. The scheduler "
        "admits all requests at nearly the same time, but the leading tokens "
        "diverge before a reusable cache block can form.\n\n"
        "Control mechanics. Prefix-cache reuse is only valid when leading tokens "
        "match exactly through complete cache blocks. This control keeps the "
        "prompt length, topic, vocabulary, and requested output shape close to "
        "the shared-prefix case while changing the beginning of every prompt. "
        "The benchmark therefore watches first-token latency, end-to-end p95 "
        "latency, output tokens per second, and time per output token. A true "
        "cache effect should be weaker here than in the shared-prefix profile.\n\n"
        "Operational constraints. The GPU is a memory-constrained T4, the model "
        "is small enough to run cheaply but still exercises vLLM scheduling, "
        "decoding is deterministic, output lengths are fixed by scenario, and "
        "the paired experiment runs prefix caching off and on inside one Modal "
        "worker. The evidence should separate prefix reuse from warm-engine "
        "effects, phase order, prompt length, and random service noise.\n\n"
        "Synthetic trace detail. The tenant has eight queued requests, each "
        "request references a different dashboard, rollout window, service "
        "owner, alert identifier, and mitigation note, and each request carries "
        "a separate prefill-heavy runbook. The content is intentionally similar "
        "in topic and size to the shared-prefix workload, but it is arranged so "
        "the beginning of every prompt diverges immediately. The benchmark "
        "should therefore preserve comparable context length, vocabulary, output "
        "budget, scheduler pressure, and decode work without providing a long "
        "identical leading block for vLLM to reuse. Any remaining speedup in "
        "this control is more likely to be phase order, warm allocation state, "
        "or generic shape reuse than exact-prefix KV-cache reuse. The control "
        "also keeps the same measurement vocabulary around prefill, decode, "
        "tail latency, throughput, cache blocks, and paired phase ordering.\n\n"
        "Decision rule. If this unique-prefix control improves by the same amount "
        "as the shared-prefix profile, the result is probably not caused by "
        "large exact-prefix reuse. If this control is flat while shared-prefix "
        "improves, the benchmark has a stronger KV-cache signal."
    )


def _neutral_long_context(index: int) -> str:
    labels = [
        "Quartz-11",
        "Lumen-22",
        "Atlas-33",
        "Nimbus-44",
        "Vector-55",
        "Solace-66",
        "Orchid-77",
        "Zenith-88",
    ]
    label = labels[index % len(labels)]
    return (
        f"{label} calibration packet. This warmup request is deliberately "
        "unrelated to the measured incident profiles. It begins with a unique "
        "marker, a separate routing note, a different document title, and an "
        "independent checklist so it should not seed the measured prefix-cache "
        "state beyond short tokenizer and chat-template scaffolding. The packet "
        "is only meant to exercise a comparable long-prefill request shape.\n\n"
        "Calibration background. The synthetic document describes a nightly "
        "data-quality audit for a batch analytics service. The audit tracks "
        "schema drift, missing partitions, retry budgets, timestamp skew, "
        "checksum mismatches, and dashboard refresh order. The worker pool has "
        "already loaded its dependencies, the input tables are fixed for the "
        "trial, and every request asks for a deterministic short answer. The "
        "content is verbose on purpose so the serving engine allocates and "
        "schedules a long prompt before producing only a small number of output "
        "tokens.\n\n"
        "Shape notes. Each calibration request keeps a similar envelope: one "
        "long context, one short task, one deterministic decode setting, and a "
        "batch admitted at nearly the same time. The text avoids the measured "
        "shared incident brief, tenant rollout language, prefix-cache decision "
        "rule, and benchmark-specific control paragraphs. Its job is to touch "
        "the same general prefill and decode machinery while avoiding the exact "
        "leading token sequences that the measured cache experiment evaluates.\n\n"
        "Audit details. The batch has eight possible records, each record uses "
        "a separate source table, owner alias, retry window, freshness target, "
        "and downstream report. The reviewer must decide whether the record is "
        "ready for publication, should be quarantined for another validation "
        "pass, or should be escalated because a threshold changed. The details "
        "are intentionally repetitive in structure but different in leading "
        "tokens, which keeps the warmup useful for request-shape coverage "
        "without constructing a large common prefix for the measured workload.\n\n"
        "Calibration rule. Treat this packet as a warmup-only document. The "
        "answer should be concise, deterministic, and based only on the audit "
        "fields in this packet. Do not assume it shares state, document text, "
        "or reusable leading context with any later measured request."
    )


def _extend_prompt_for_context(prompt: str, index: int) -> str:
    return (
        f"{prompt} Consider request group {index + 1} in a bursty production "
        "serving workload with mixed prompt lengths, streaming responses, and a "
        "fixed GPU memory budget. Compare prefill cost, decode cost, scheduler "
        "queueing, KV-cache growth, cache block fragmentation, and tail latency. "
        "Use one concrete systems example and identify the bottleneck that would "
        "matter most on a memory-constrained GPU."
    )


def _summarize_stream_requests(
    requests: list[dict[str, Any]],
    batch_wall_ms: float,
) -> dict[str, Any]:
    total_output_tokens = sum(request["generated_tokens"] for request in requests)
    first_chunks = [
        request["first_chunk_ms"]
        for request in requests
        if request["first_chunk_ms"] is not None
    ]
    latencies = [request["stream_wall_ms"] for request in requests]
    tpot_values = [
        request["stream_tpot_ms"]
        for request in requests
        if request["stream_tpot_ms"] is not None
    ]
    return {
        "batch_wall_ms": batch_wall_ms,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": total_output_tokens / max(batch_wall_ms / 1000, 1e-9),
        "p50_first_chunk_ms": _percentile(first_chunks, 50),
        "p95_first_chunk_ms": _percentile(first_chunks, 95),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p50_stream_tpot_ms": _percentile(tpot_values, 50),
        "p95_stream_tpot_ms": _percentile(tpot_values, 95),
    }


def _summarize_server_stream_requests(
    requests: list[dict[str, Any]],
    batch_wall_ms: float,
) -> dict[str, Any]:
    total_output_tokens = sum(request["generated_tokens"] for request in requests)
    first_contents = [
        request["first_content_ms"]
        for request in requests
        if request["first_content_ms"] is not None
    ]
    latencies = [request["request_wall_ms"] for request in requests]
    tpot_values = [
        request["stream_tpot_ms"]
        for request in requests
        if request["stream_tpot_ms"] is not None
    ]
    return {
        "batch_wall_ms": batch_wall_ms,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": total_output_tokens / max(batch_wall_ms / 1000, 1e-9),
        "p50_first_content_ms": _percentile(first_contents, 50),
        "p95_first_content_ms": _percentile(first_contents, 95),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p50_stream_tpot_ms": _percentile(tpot_values, 50),
        "p95_stream_tpot_ms": _percentile(tpot_values, 95),
    }


def _aggregate_vllm_sweep_scenarios(
    scenario_specs: list[dict[str, Any]],
    scenario_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for run in scenario_runs:
        runs_by_scenario.setdefault(run["scenario_id"], []).append(run)

    scenarios = []
    for spec in scenario_specs:
        runs = sorted(
            runs_by_scenario.get(spec["scenario_id"], []),
            key=lambda run: (run["repeat_index"], run["run_order"]),
        )
        scenario: dict[str, Any] = {
            "scenario_id": spec["scenario_id"],
            "prompt_profile": spec["prompt_profile"],
            "request_count": spec["request_count"],
            "max_new_tokens": spec["max_new_tokens"],
            "prompt_format": spec["prompt_format"],
            "prompt_tokens_min": spec["prompt_tokens_min"],
            "prompt_tokens_max": spec["prompt_tokens_max"],
            "prompt_tokens_mean": spec["prompt_tokens_mean"],
            "total_prompt_tokens": spec["total_prompt_tokens"],
            "repeats": len(runs),
            "repeat_indices": [run["repeat_index"] for run in runs],
            "run_orders": [run["run_order"] for run in runs],
        }
        for field in (
            "estimated_peak_sequence_tokens",
            "batch_wall_ms",
            "total_output_tokens",
            "aggregate_output_tokens_per_second",
            "p50_first_chunk_ms",
            "p95_first_chunk_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p50_stream_tpot_ms",
            "p95_stream_tpot_ms",
        ):
            scenario.update(_metric_distribution(runs, field))
        scenarios.append(scenario)
    return scenarios


def _aggregate_vllm_server_scenarios(
    scenario_specs: list[dict[str, Any]],
    scenario_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for run in scenario_runs:
        runs_by_scenario.setdefault(run["scenario_id"], []).append(run)

    scenarios = []
    for spec in scenario_specs:
        runs = sorted(
            runs_by_scenario.get(spec["scenario_id"], []),
            key=lambda run: (run["repeat_index"], run["run_order"]),
        )
        scenario: dict[str, Any] = {
            "scenario_id": spec["scenario_id"],
            "prompt_profile": spec["prompt_profile"],
            "request_count": spec["request_count"],
            "max_new_tokens": spec["max_new_tokens"],
            "prompt_format": spec["prompt_format"],
            "prompt_tokens_min": spec["prompt_tokens_min"],
            "prompt_tokens_max": spec["prompt_tokens_max"],
            "prompt_tokens_mean": spec["prompt_tokens_mean"],
            "total_prompt_tokens": spec["total_prompt_tokens"],
            "repeats": len(runs),
            "repeat_indices": [run["repeat_index"] for run in runs],
            "run_orders": [run["run_order"] for run in runs],
        }
        for field in (
            "estimated_peak_sequence_tokens",
            "batch_wall_ms",
            "total_output_tokens",
            "aggregate_output_tokens_per_second",
            "p50_first_content_ms",
            "p95_first_content_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p50_stream_tpot_ms",
            "p95_stream_tpot_ms",
        ):
            scenario.update(_metric_distribution(runs, field))
        scenarios.append(scenario)
    return scenarios


def _vllm_sweep_summary_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "prompt_tokens_mean",
        "total_prompt_tokens",
        "repeats",
        "estimated_peak_sequence_tokens_median",
        "estimated_peak_sequence_tokens_p95",
        "batch_wall_ms_median",
        "batch_wall_ms_p95",
        "batch_wall_ms_min",
        "batch_wall_ms_max",
        "batch_wall_ms_cv",
        "total_output_tokens_median",
        "aggregate_output_tokens_per_second_median",
        "aggregate_output_tokens_per_second_p95",
        "aggregate_output_tokens_per_second_min",
        "aggregate_output_tokens_per_second_max",
        "aggregate_output_tokens_per_second_cv",
        "p95_first_chunk_ms_median",
        "p95_first_chunk_ms_p95",
        "p95_first_chunk_ms_min",
        "p95_first_chunk_ms_max",
        "p95_first_chunk_ms_cv",
        "p95_latency_ms_median",
        "p95_latency_ms_p95",
        "p95_latency_ms_min",
        "p95_latency_ms_max",
        "p95_latency_ms_cv",
        "p95_stream_tpot_ms_median",
        "p95_stream_tpot_ms_p95",
        "p95_stream_tpot_ms_min",
        "p95_stream_tpot_ms_max",
        "p95_stream_tpot_ms_cv",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in scenarios
    ]


def _vllm_sweep_run_rows(scenario_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_fields = (
        "run_id",
        "repeat_index",
        "run_order",
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "prompt_tokens_mean",
        "total_prompt_tokens",
        "estimated_peak_sequence_tokens",
        "batch_wall_ms",
        "total_output_tokens",
        "aggregate_output_tokens_per_second",
        "p50_first_chunk_ms",
        "p95_first_chunk_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p50_stream_tpot_ms",
        "p95_stream_tpot_ms",
    )
    return [
        {field: run.get(field) for field in row_fields}
        for run in sorted(scenario_runs, key=lambda run: run["run_order"])
    ]


def _vllm_server_sweep_summary_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "prompt_tokens_mean",
        "total_prompt_tokens",
        "repeats",
        "estimated_peak_sequence_tokens_median",
        "estimated_peak_sequence_tokens_p95",
        "batch_wall_ms_median",
        "batch_wall_ms_p95",
        "batch_wall_ms_min",
        "batch_wall_ms_max",
        "batch_wall_ms_cv",
        "total_output_tokens_median",
        "aggregate_output_tokens_per_second_median",
        "aggregate_output_tokens_per_second_p95",
        "aggregate_output_tokens_per_second_min",
        "aggregate_output_tokens_per_second_max",
        "aggregate_output_tokens_per_second_cv",
        "p95_first_content_ms_median",
        "p95_first_content_ms_p95",
        "p95_first_content_ms_min",
        "p95_first_content_ms_max",
        "p95_first_content_ms_cv",
        "p95_latency_ms_median",
        "p95_latency_ms_p95",
        "p95_latency_ms_min",
        "p95_latency_ms_max",
        "p95_latency_ms_cv",
        "p95_stream_tpot_ms_median",
        "p95_stream_tpot_ms_p95",
        "p95_stream_tpot_ms_min",
        "p95_stream_tpot_ms_max",
        "p95_stream_tpot_ms_cv",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in scenarios
    ]


def _vllm_server_sweep_run_rows(scenario_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_fields = (
        "run_id",
        "repeat_index",
        "run_order",
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "prompt_tokens_mean",
        "total_prompt_tokens",
        "estimated_peak_sequence_tokens",
        "batch_wall_ms",
        "total_output_tokens",
        "aggregate_output_tokens_per_second",
        "p50_first_content_ms",
        "p95_first_content_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p50_stream_tpot_ms",
        "p95_stream_tpot_ms",
    )
    return [
        {field: run.get(field) for field in row_fields}
        for run in sorted(scenario_runs, key=lambda run: run["run_order"])
    ]


def _vllm_server_concurrent_summary_rows(
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "prompt_tokens_mean",
        "total_prompt_tokens",
        "estimated_peak_sequence_tokens",
        "batch_wall_ms",
        "total_output_tokens",
        "aggregate_output_tokens_per_second",
        "p50_first_content_ms",
        "p95_first_content_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p50_stream_tpot_ms",
        "p95_stream_tpot_ms",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in scenarios
    ]


def _metric_distribution(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, float | int | None]:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    if not values:
        return {
            f"{field}_count": 0,
            f"{field}_min": None,
            f"{field}_max": None,
            f"{field}_mean": None,
            f"{field}_median": None,
            f"{field}_p95": None,
            f"{field}_cv": None,
        }

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    stddev = variance**0.5
    return {
        f"{field}_count": len(values),
        f"{field}_min": min(values),
        f"{field}_max": max(values),
        f"{field}_mean": mean_value,
        f"{field}_median": _percentile(values, 50),
        f"{field}_p95": _percentile(values, 95),
        f"{field}_cv": stddev / abs(mean_value) if mean_value else None,
    }


def _parse_vllm_server_cli_help(help_text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in help_text.splitlines()]
    prefix_related_lines = [
        line
        for line in lines
        if "prefix" in line.lower()
    ]
    prefix_related_flags = sorted(
        {
            flag
            for flag in re.findall(r"--[A-Za-z0-9][A-Za-z0-9-]*", help_text)
            if "prefix" in flag.lower()
        }
    )
    return {
        "prefix_related_flags": prefix_related_flags,
        "prefix_related_lines": prefix_related_lines,
        "has_enable_prefix_caching_flag": (
            "--enable-prefix-caching" in prefix_related_flags
        ),
        "has_no_enable_prefix_caching_flag": (
            "--no-enable-prefix-caching" in prefix_related_flags
        ),
        "has_disable_prefix_caching_flag": (
            "--disable-prefix-caching" in prefix_related_flags
        ),
    }


def _strip_trailing_whitespace_lines(text: str) -> str:
    stripped = "\n".join(line.rstrip() for line in text.splitlines())
    return stripped + ("\n" if text.endswith("\n") else "")


def _parse_prometheus_text_samples(text: str) -> list[dict[str, Any]]:
    samples = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(
            r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)",
            line,
        )
        if not match:
            continue
        name, labels, value = match.groups()
        try:
            numeric_value = float(value)
        except ValueError:
            continue
        samples.append(
            {
                "name": name,
                "labels": labels or "",
                "value": numeric_value,
            }
        )
    return samples


def _select_prometheus_metric_sum(
    samples: list[dict[str, Any]],
    required_terms: tuple[str, ...],
    excluded_terms: tuple[str, ...] = (),
) -> dict[str, Any]:
    grouped: dict[str, float] = {}
    for sample in samples:
        name = str(sample["name"])
        lowered = name.lower()
        if not all(term in lowered for term in required_terms):
            continue
        if any(term in lowered for term in excluded_terms):
            continue
        grouped[name] = grouped.get(name, 0.0) + float(sample["value"])
    if not grouped:
        return {"name": None, "value": None}

    def rank(item: tuple[str, float]) -> tuple[int, int, str]:
        name = item[0].lower()
        total_rank = 0 if name.endswith("_total") or "total" in name else 1
        return (total_rank, len(name), item[0])

    selected_name, selected_value = sorted(grouped.items(), key=rank)[0]
    return {"name": selected_name, "value": selected_value}


def _parse_vllm_server_prometheus_metrics(text: str) -> dict[str, Any]:
    samples = _parse_prometheus_text_samples(text)
    prefix_samples = [
        sample
        for sample in samples
        if "prefix" in str(sample["name"]).lower()
    ]
    requests = _select_prometheus_metric_sum(
        prefix_samples,
        ("prefix", "cache", "request"),
        excluded_terms=("created", "rate", "percent", "percentage"),
    )
    queries = _select_prometheus_metric_sum(
        prefix_samples,
        ("prefix", "cache", "quer"),
        excluded_terms=("created", "rate", "percent", "percentage"),
    )
    hits = _select_prometheus_metric_sum(
        prefix_samples,
        ("prefix", "cache", "hit"),
        excluded_terms=("created", "rate", "percent", "percentage"),
    )
    return {
        "sample_count": len(samples),
        "prefix_related_sample_count": len(prefix_samples),
        "prefix_related_samples": prefix_samples[:80],
        "total_requests": requests["value"],
        "total_queries": queries["value"],
        "total_hits": hits["value"],
        "hit_rate_pct": _prefix_cache_counter_hit_rate_pct(
            hits["value"],
            queries["value"],
        ),
        "selected_request_metric": requests["name"],
        "selected_query_metric": queries["name"],
        "selected_hit_metric": hits["name"],
    }


def _parse_vllm_server_log_metrics(log_lines: list[str] | None) -> dict[str, Any]:
    lines = [str(line) for line in (log_lines or [])]
    text = "\n".join(lines)

    def number(pattern: str) -> float | None:
        match = re.search(pattern, text)
        if not match:
            return None
        return float(match.group(1).replace(",", ""))

    def bool_value(pattern: str) -> bool | None:
        match = re.search(pattern, text)
        if not match:
            return None
        return match.group(1).lower() == "true"

    capacity = _parse_vllm_engine_capacity_log_metrics(
        {"captured_logs": lines, "stdout": "", "stderr": ""}
    )
    runtime_matches = list(
        re.finditer(
            r"Avg prompt throughput:\s*([0-9.]+)\s*tokens/s, "
            r"Avg generation throughput:\s*([0-9.]+)\s*tokens/s, .*?"
            r"GPU KV cache usage:\s*([0-9.]+)%, "
            r"Prefix cache hit rate:\s*([0-9.]+)%",
            text,
        )
    )
    latest_runtime = runtime_matches[-1] if runtime_matches else None
    return {
        **capacity,
        "enable_prefix_caching": bool_value(r"enable_prefix_caching=(True|False)"),
        "max_num_batched_tokens": (
            int(value)
            if (value := number(r"max_num_batched_tokens['\"]?:?\s*=?\s*([0-9,]+)"))
            is not None
            else None
        ),
        "runtime_metric_count": len(runtime_matches),
        "latest_avg_prompt_throughput_tokens_per_s": (
            float(latest_runtime.group(1)) if latest_runtime else None
        ),
        "latest_avg_generation_throughput_tokens_per_s": (
            float(latest_runtime.group(2)) if latest_runtime else None
        ),
        "latest_gpu_kv_cache_usage_pct": (
            float(latest_runtime.group(3)) if latest_runtime else None
        ),
        "latest_prefix_cache_hit_rate_pct": (
            float(latest_runtime.group(4)) if latest_runtime else None
        ),
    }


def _summarize_vllm_server_async_log_metrics(
    paired_dirs: list[Path],
) -> dict[str, Any]:
    rows = []
    for paired_dir in paired_dirs:
        json_path = paired_dir / "paired-server-async.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing paired-server-async.json: {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        server_logs = (payload.get("server_logs_head") or []) + (
            payload.get("server_logs_tail") or []
        )
        metrics = _parse_vllm_server_log_metrics(server_logs)
        server_command = payload.get("server_command") or []
        rows.append(
            {
                "source_dir": str(paired_dir),
                "phase_order": payload.get("phase_order"),
                "model_id": payload.get("model_id"),
                "modal_gpu": payload.get("modal_gpu"),
                "request_counts": ",".join(
                    str(value) for value in payload.get("request_counts", [])
                ),
                "prompt_profiles": ",".join(payload.get("prompt_profiles", [])),
                "output_tokens": ",".join(
                    str(value) for value in payload.get("output_tokens", [])
                ),
                "repeats": payload.get("repeats"),
                "max_model_len": payload.get("max_model_len"),
                "max_num_batched_tokens": payload.get("max_num_batched_tokens"),
                "default_max_num_batched_tokens": payload.get(
                    "default_max_num_batched_tokens"
                ),
                "max_num_batched_tokens_source": payload.get(
                    "max_num_batched_tokens_source"
                ),
                "async_enable_prefix_caching_configured": payload.get(
                    "async_enable_prefix_caching_configured",
                    False,
                ),
                "server_prefix_caching_configured": payload.get(
                    "server_prefix_caching_configured",
                    "unknown",
                ),
                "server_prefix_caching_flag": ",".join(
                    payload.get("server_prefix_caching_flag") or []
                ),
                "server_enable_prefix_caching_observed": metrics.get(
                    "enable_prefix_caching"
                ),
                "server_command_has_enable_prefix_caching_flag": (
                    "--enable-prefix-caching" in server_command
                ),
                "server_command_has_disable_prefix_caching_flag": any(
                    flag in server_command
                    for flag in (
                        "--no-enable-prefix-caching",
                        "--disable-prefix-caching",
                    )
                ),
                "server_log_max_num_batched_tokens": metrics.get(
                    "max_num_batched_tokens"
                ),
                "server_gpu_kv_cache_size_tokens": metrics.get(
                    "gpu_kv_cache_size_tokens"
                ),
                "server_available_kv_cache_memory_gib": metrics.get(
                    "available_kv_cache_memory_gib"
                ),
                "server_max_concurrency_request_tokens": metrics.get(
                    "max_concurrency_request_tokens"
                ),
                "server_max_concurrency_for_request": metrics.get(
                    "max_concurrency_for_request"
                ),
                "server_runtime_metric_count": metrics.get("runtime_metric_count"),
                "server_latest_prefix_cache_hit_rate_pct": metrics.get(
                    "latest_prefix_cache_hit_rate_pct"
                ),
                "server_latest_gpu_kv_cache_usage_pct": metrics.get(
                    "latest_gpu_kv_cache_usage_pct"
                ),
                "server_latest_avg_prompt_throughput_tokens_per_s": metrics.get(
                    "latest_avg_prompt_throughput_tokens_per_s"
                ),
                "server_latest_avg_generation_throughput_tokens_per_s": metrics.get(
                    "latest_avg_generation_throughput_tokens_per_s"
                ),
                "mean_server_to_async_throughput_ratio": payload.get(
                    "mean_server_to_async_throughput_ratio"
                ),
                "mean_server_to_async_latency_ratio": payload.get(
                    "mean_server_to_async_latency_ratio"
                ),
                "mean_server_to_async_tpot_ratio": payload.get(
                    "mean_server_to_async_tpot_ratio"
                ),
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-log-summary",
        "source_dirs": [str(path) for path in paired_dirs],
        "row_count": len(rows),
        "rows": rows,
        "server_enable_prefix_caching_observed_true_count": sum(
            row["server_enable_prefix_caching_observed"] is True for row in rows
        ),
        "server_enable_prefix_caching_observed_false_count": sum(
            row["server_enable_prefix_caching_observed"] is False for row in rows
        ),
        "server_enable_prefix_caching_observed_unknown_count": sum(
            row["server_enable_prefix_caching_observed"] is None for row in rows
        ),
        "server_prefix_caching_configured_modes": sorted(
            {
                str(row["server_prefix_caching_configured"])
                for row in rows
                if row.get("server_prefix_caching_configured") is not None
            }
        ),
        "mean_server_latest_prefix_cache_hit_rate_pct": _mean_present_or_none(
            row["server_latest_prefix_cache_hit_rate_pct"] for row in rows
        ),
        "all_server_enable_prefix_caching_observed": (
            all(row["server_enable_prefix_caching_observed"] is True for row in rows)
            if rows
            else None
        ),
        "all_server_disable_prefix_caching_observed": (
            all(row["server_enable_prefix_caching_observed"] is False for row in rows)
            if rows
            else None
        ),
        "any_server_command_has_explicit_prefix_cache_flag": any(
            row["server_command_has_enable_prefix_caching_flag"]
            or row["server_command_has_disable_prefix_caching_flag"]
            for row in rows
        ),
    }


def _compare_vllm_server_async_cache_control_matrix(
    paired_dirs: list[Path],
) -> dict[str, Any]:
    log_summary = _summarize_vllm_server_async_log_metrics(paired_dirs)
    rows = []
    seen_keys: set[tuple[str, str]] = set()
    for row in log_summary["rows"]:
        cache_mode = str(row.get("server_prefix_caching_configured") or "unknown")
        phase_order = str(row.get("phase_order") or "unknown")
        key = (cache_mode, phase_order)
        if key in seen_keys:
            raise ValueError(
                "Duplicate cache-control matrix cell: "
                f"cache_mode={cache_mode} phase_order={phase_order}"
            )
        seen_keys.add(key)
        rows.append(
            {
                "server_prefix_caching_configured": cache_mode,
                "phase_order": phase_order,
                "source_dir": row["source_dir"],
                "model_id": row["model_id"],
                "modal_gpu": row["modal_gpu"],
                "request_counts": row["request_counts"],
                "prompt_profiles": row["prompt_profiles"],
                "output_tokens": row["output_tokens"],
                "repeats": row["repeats"],
                "max_model_len": row["max_model_len"],
                "max_num_batched_tokens": row["max_num_batched_tokens"],
                "max_num_batched_tokens_source": row[
                    "max_num_batched_tokens_source"
                ],
                "async_enable_prefix_caching_configured": row[
                    "async_enable_prefix_caching_configured"
                ],
                "server_prefix_caching_flag": row["server_prefix_caching_flag"],
                "server_enable_prefix_caching_observed": row[
                    "server_enable_prefix_caching_observed"
                ],
                "server_runtime_metric_count": row["server_runtime_metric_count"],
                "server_latest_prefix_cache_hit_rate_pct": row[
                    "server_latest_prefix_cache_hit_rate_pct"
                ],
                "server_gpu_kv_cache_size_tokens": row[
                    "server_gpu_kv_cache_size_tokens"
                ],
                "server_max_concurrency_for_request": row[
                    "server_max_concurrency_for_request"
                ],
                "mean_server_to_async_throughput_ratio": row[
                    "mean_server_to_async_throughput_ratio"
                ],
                "mean_server_to_async_latency_ratio": row[
                    "mean_server_to_async_latency_ratio"
                ],
                "mean_server_to_async_tpot_ratio": row[
                    "mean_server_to_async_tpot_ratio"
                ],
            }
        )

    rows.sort(
        key=lambda row: (
            str(row["server_prefix_caching_configured"]),
            str(row["phase_order"]),
        )
    )
    by_key = {
        (row["server_prefix_caching_configured"], row["phase_order"]): row
        for row in rows
    }
    cache_modes = sorted({row["server_prefix_caching_configured"] for row in rows})
    phase_orders = sorted({row["phase_order"] for row in rows})

    cache_mode_summary = []
    for cache_mode in cache_modes:
        mode_rows = [
            row for row in rows
            if row["server_prefix_caching_configured"] == cache_mode
        ]
        async_first = by_key.get((cache_mode, "async_first"))
        server_first = by_key.get((cache_mode, "server_first"))
        cache_mode_summary.append(
            {
                "server_prefix_caching_configured": cache_mode,
                "phase_order_count": len({row["phase_order"] for row in mode_rows}),
                "mean_server_latest_prefix_cache_hit_rate_pct": _mean_present_or_none(
                    row["server_latest_prefix_cache_hit_rate_pct"]
                    for row in mode_rows
                ),
                "mean_server_to_async_throughput_ratio": _mean_present(
                    row["mean_server_to_async_throughput_ratio"]
                    for row in mode_rows
                ),
                "mean_server_to_async_latency_ratio": _mean_present(
                    row["mean_server_to_async_latency_ratio"]
                    for row in mode_rows
                ),
                "mean_server_to_async_tpot_ratio": _mean_present(
                    row["mean_server_to_async_tpot_ratio"] for row in mode_rows
                ),
                "async_first_throughput_ratio": (
                    async_first["mean_server_to_async_throughput_ratio"]
                    if async_first
                    else None
                ),
                "server_first_throughput_ratio": (
                    server_first["mean_server_to_async_throughput_ratio"]
                    if server_first
                    else None
                ),
                "server_first_minus_async_first_throughput_ratio": (
                    _delta(
                        server_first["mean_server_to_async_throughput_ratio"],
                        async_first["mean_server_to_async_throughput_ratio"],
                    )
                    if async_first and server_first
                    else None
                ),
                "async_first_latency_ratio": (
                    async_first["mean_server_to_async_latency_ratio"]
                    if async_first
                    else None
                ),
                "server_first_latency_ratio": (
                    server_first["mean_server_to_async_latency_ratio"]
                    if server_first
                    else None
                ),
                "server_first_minus_async_first_latency_ratio": (
                    _delta(
                        server_first["mean_server_to_async_latency_ratio"],
                        async_first["mean_server_to_async_latency_ratio"],
                    )
                    if async_first and server_first
                    else None
                ),
            }
        )

    phase_order_cache_contrasts = []
    for phase_order in phase_orders:
        off_row = by_key.get(("off", phase_order))
        on_row = by_key.get(("on", phase_order))
        if not off_row or not on_row:
            continue
        phase_order_cache_contrasts.append(
            {
                "phase_order": phase_order,
                "off_source_dir": off_row["source_dir"],
                "on_source_dir": on_row["source_dir"],
                "off_server_enable_prefix_caching_observed": off_row[
                    "server_enable_prefix_caching_observed"
                ],
                "on_server_enable_prefix_caching_observed": on_row[
                    "server_enable_prefix_caching_observed"
                ],
                "off_latest_prefix_cache_hit_rate_pct": off_row[
                    "server_latest_prefix_cache_hit_rate_pct"
                ],
                "on_latest_prefix_cache_hit_rate_pct": on_row[
                    "server_latest_prefix_cache_hit_rate_pct"
                ],
                "cache_on_minus_off_prefix_cache_hit_rate_pct": _delta(
                    on_row["server_latest_prefix_cache_hit_rate_pct"],
                    off_row["server_latest_prefix_cache_hit_rate_pct"],
                ),
                "off_throughput_ratio": off_row[
                    "mean_server_to_async_throughput_ratio"
                ],
                "on_throughput_ratio": on_row[
                    "mean_server_to_async_throughput_ratio"
                ],
                "cache_on_minus_off_throughput_ratio": _delta(
                    on_row["mean_server_to_async_throughput_ratio"],
                    off_row["mean_server_to_async_throughput_ratio"],
                ),
                "cache_on_div_off_throughput_ratio": _ratio(
                    on_row["mean_server_to_async_throughput_ratio"],
                    off_row["mean_server_to_async_throughput_ratio"],
                ),
                "off_latency_ratio": off_row["mean_server_to_async_latency_ratio"],
                "on_latency_ratio": on_row["mean_server_to_async_latency_ratio"],
                "cache_on_minus_off_latency_ratio": _delta(
                    on_row["mean_server_to_async_latency_ratio"],
                    off_row["mean_server_to_async_latency_ratio"],
                ),
                "cache_on_div_off_latency_ratio": _ratio(
                    on_row["mean_server_to_async_latency_ratio"],
                    off_row["mean_server_to_async_latency_ratio"],
                ),
                "off_tpot_ratio": off_row["mean_server_to_async_tpot_ratio"],
                "on_tpot_ratio": on_row["mean_server_to_async_tpot_ratio"],
                "cache_on_minus_off_tpot_ratio": _delta(
                    on_row["mean_server_to_async_tpot_ratio"],
                    off_row["mean_server_to_async_tpot_ratio"],
                ),
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-cache-control-compare",
        "source_dirs": [str(path) for path in paired_dirs],
        "row_count": len(rows),
        "cache_modes": cache_modes,
        "phase_orders": phase_orders,
        "matrix_rows": rows,
        "cache_mode_summary": cache_mode_summary,
        "phase_order_cache_contrasts": phase_order_cache_contrasts,
        "log_summary": log_summary,
    }


def _aggregate_vllm_server_async_cache_control_trials(
    compare_dirs: list[Path],
) -> dict[str, Any]:
    if not compare_dirs:
        raise ValueError("compare_dirs must not be empty")

    metric_fields = (
        "cache_on_minus_off_prefix_cache_hit_rate_pct",
        "cache_on_minus_off_throughput_ratio",
        "cache_on_div_off_throughput_ratio",
        "cache_on_minus_off_latency_ratio",
        "cache_on_div_off_latency_ratio",
        "cache_on_minus_off_tpot_ratio",
        "off_throughput_ratio",
        "on_throughput_ratio",
        "off_latency_ratio",
        "on_latency_ratio",
        "off_latest_prefix_cache_hit_rate_pct",
        "on_latest_prefix_cache_hit_rate_pct",
    )
    trial_rows = []
    compare_payloads = []
    for trial_index, compare_dir in enumerate(compare_dirs, start=1):
        compare_json = compare_dir / "cache-control-phase-order-compare.json"
        payload = json.loads(compare_json.read_text(encoding="utf-8"))
        compare_payloads.append(payload)
        for contrast in payload.get("phase_order_cache_contrasts", []):
            trial_row: dict[str, Any] = {
                "trial_index": trial_index,
                "trial_label": compare_dir.name,
                "compare_json": str(compare_json),
                "phase_order": contrast["phase_order"],
                "off_source_dir": contrast["off_source_dir"],
                "on_source_dir": contrast["on_source_dir"],
                "off_server_enable_prefix_caching_observed": contrast[
                    "off_server_enable_prefix_caching_observed"
                ],
                "on_server_enable_prefix_caching_observed": contrast[
                    "on_server_enable_prefix_caching_observed"
                ],
            }
            for field in metric_fields:
                trial_row[field] = contrast.get(field)
            trial_rows.append(trial_row)

    phase_orders = sorted({row["phase_order"] for row in trial_rows})
    summary = []
    for phase_order in phase_orders:
        phase_rows = [
            row for row in trial_rows
            if row["phase_order"] == phase_order
        ]
        for metric in metric_fields:
            stats = _metric_distribution(phase_rows, metric)
            values = [
                float(row[metric])
                for row in phase_rows
                if row.get(metric) is not None
            ]
            interval = _bootstrap_mean_interval(values)
            summary.append(
                {
                    "phase_order": phase_order,
                    "metric": metric,
                    "trial_count": len(values),
                    "mean": stats[f"{metric}_mean"],
                    "min": stats[f"{metric}_min"],
                    "max": stats[f"{metric}_max"],
                    "median": stats[f"{metric}_median"],
                    "p95": stats[f"{metric}_p95"],
                    "cv": stats[f"{metric}_cv"],
                    "bootstrap_mean_p05": interval["mean_p05"],
                    "bootstrap_mean_p50": interval["mean_p50"],
                    "bootstrap_mean_p95": interval["mean_p95"],
                }
            )

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-cache-control-multitrial",
        "trial_count": len(compare_dirs),
        "compare_dirs": [str(path) for path in compare_dirs],
        "phase_orders": phase_orders,
        "trial_rows": trial_rows,
        "summary": summary,
        "all_off_server_disable_observed": (
            all(
                row["off_server_enable_prefix_caching_observed"] is False
                for row in trial_rows
            )
            if trial_rows
            else None
        ),
        "all_on_server_enable_observed": (
            all(
                row["on_server_enable_prefix_caching_observed"] is True
                for row in trial_rows
            )
            if trial_rows
            else None
        ),
        "compare_payloads": compare_payloads,
    }


def _compare_vllm_server_async_cache_control_server_absolute(
    compare_dirs: list[Path],
) -> dict[str, Any]:
    if not compare_dirs:
        raise ValueError("compare_dirs must not be empty")

    metric_names = (
        "server_output_tokens_per_second",
        "server_p95_latency_ms",
        "server_p95_first_content_ms",
        "server_p95_stream_tpot_ms",
        "server_batch_wall_ms",
    )
    counter_names = (
        "server_prefix_cache_counter_requests",
        "server_prefix_cache_counter_queries",
        "server_prefix_cache_counter_hits",
        "server_prefix_cache_counter_hit_rate_pct",
    )
    summary_metrics = tuple(
        field
        for metric in metric_names
        for field in (
            f"cache_on_minus_off_{metric}",
            f"cache_on_div_off_{metric}",
        )
    ) + (
        "on_server_prefix_cache_counter_queries",
        "on_server_prefix_cache_counter_hits",
        "on_server_prefix_cache_counter_hit_rate_pct",
        "cache_on_minus_off_server_prefix_cache_counter_hit_rate_pct",
    )

    run_rows = []
    compare_payloads = []
    for trial_index, compare_dir in enumerate(compare_dirs, start=1):
        compare_json = compare_dir / "cache-control-phase-order-compare.json"
        compare_payload = json.loads(compare_json.read_text(encoding="utf-8"))
        compare_payloads.append(compare_payload)
        source_dirs = sorted(
            {
                Path(row["source_dir"])
                for row in compare_payload.get("matrix_rows", [])
            },
            key=str,
        )
        for source_dir in source_dirs:
            paired_json = source_dir / "paired-server-async.json"
            paired_payload = json.loads(paired_json.read_text(encoding="utf-8"))
            server_logs = (paired_payload.get("server_logs_head") or []) + (
                paired_payload.get("server_logs_tail") or []
            )
            log_metrics = _parse_vllm_server_log_metrics(server_logs)
            cache_mode = str(
                paired_payload.get("server_prefix_caching_configured") or "unknown"
            )
            phase_order = str(paired_payload.get("phase_order") or "unknown")
            for run in paired_payload.get("paired_runs", []):
                row = {
                    "trial_index": trial_index,
                    "trial_label": compare_dir.name,
                    "compare_json": str(compare_json),
                    "source_dir": str(source_dir),
                    "server_prefix_caching_configured": cache_mode,
                    "phase_order": phase_order,
                    "prompt_profile": run.get("prompt_profile"),
                    "scenario_id": run.get("scenario_id"),
                    "request_count": run.get("request_count"),
                    "max_new_tokens": run.get("max_new_tokens"),
                    "repeat_index": run.get("repeat_index"),
                    "server_run_order": run.get("server_run_order"),
                    "prompt_tokens_mean": run.get("prompt_tokens_mean"),
                    "model_id": paired_payload.get("model_id"),
                    "modal_gpu": paired_payload.get("modal_gpu"),
                    "gpu_memory_utilization": paired_payload.get(
                        "gpu_memory_utilization"
                    ),
                    "max_model_len": paired_payload.get("max_model_len"),
                    "max_num_batched_tokens": paired_payload.get(
                        "max_num_batched_tokens"
                    ),
                    "server_enable_prefix_caching_observed": log_metrics.get(
                        "enable_prefix_caching"
                    ),
                    "server_latest_prefix_cache_hit_rate_pct": log_metrics.get(
                        "latest_prefix_cache_hit_rate_pct"
                    ),
                    "server_gpu_kv_cache_size_tokens": log_metrics.get(
                        "gpu_kv_cache_size_tokens"
                    ),
                    "server_max_concurrency_for_request": log_metrics.get(
                        "max_concurrency_for_request"
                    ),
                    "server_available_kv_cache_memory_gib": log_metrics.get(
                        "available_kv_cache_memory_gib"
                    ),
                }
                for metric in metric_names:
                    row[metric] = run.get(metric)
                for counter_name in counter_names:
                    row[counter_name] = run.get(counter_name)
                run_rows.append(row)

    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in run_rows:
        key = (
            row["trial_index"],
            row["server_prefix_caching_configured"],
            row["phase_order"],
            row["prompt_profile"],
            row["request_count"],
            row["max_new_tokens"],
            row["repeat_index"],
        )
        if key in by_key:
            raise ValueError(f"Duplicate server cache-control run row: {key}")
        by_key[key] = row

    contrast_keys = sorted(
        {
            (
                row["trial_index"],
                row["phase_order"],
                row["prompt_profile"],
                row["request_count"],
                row["max_new_tokens"],
                row["repeat_index"],
            )
            for row in run_rows
        }
    )
    trial_rows = []
    for key in contrast_keys:
        (
            trial_index,
            phase_order,
            prompt_profile,
            request_count,
            max_new_tokens,
            repeat_index,
        ) = key
        off_row = by_key.get(
            (
                trial_index,
                "off",
                phase_order,
                prompt_profile,
                request_count,
                max_new_tokens,
                repeat_index,
            )
        )
        on_row = by_key.get(
            (
                trial_index,
                "on",
                phase_order,
                prompt_profile,
                request_count,
                max_new_tokens,
                repeat_index,
            )
        )
        if not off_row or not on_row:
            continue

        contrast: dict[str, Any] = {
            "trial_index": trial_index,
            "trial_label": off_row["trial_label"],
            "phase_order": phase_order,
            "prompt_profile": prompt_profile,
            "scenario_id": off_row["scenario_id"],
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "repeat_index": repeat_index,
            "prompt_tokens_mean": off_row["prompt_tokens_mean"],
            "estimated_prompt_tokens": (
                off_row["prompt_tokens_mean"] * request_count
                if off_row["prompt_tokens_mean"] is not None
                else None
            ),
            "gpu_memory_utilization": off_row["gpu_memory_utilization"],
            "max_model_len": off_row["max_model_len"],
            "max_num_batched_tokens": off_row["max_num_batched_tokens"],
            "off_source_dir": off_row["source_dir"],
            "on_source_dir": on_row["source_dir"],
            "off_server_enable_prefix_caching_observed": off_row[
                "server_enable_prefix_caching_observed"
            ],
            "on_server_enable_prefix_caching_observed": on_row[
                "server_enable_prefix_caching_observed"
            ],
            "off_server_latest_prefix_cache_hit_rate_pct": off_row[
                "server_latest_prefix_cache_hit_rate_pct"
            ],
            "on_server_latest_prefix_cache_hit_rate_pct": on_row[
                "server_latest_prefix_cache_hit_rate_pct"
            ],
            "cache_on_minus_off_server_latest_prefix_cache_hit_rate_pct": _delta(
                on_row["server_latest_prefix_cache_hit_rate_pct"],
                off_row["server_latest_prefix_cache_hit_rate_pct"],
            ),
            "off_server_gpu_kv_cache_size_tokens": off_row[
                "server_gpu_kv_cache_size_tokens"
            ],
            "on_server_gpu_kv_cache_size_tokens": on_row[
                "server_gpu_kv_cache_size_tokens"
            ],
            "off_server_max_concurrency_for_request": off_row[
                "server_max_concurrency_for_request"
            ],
            "on_server_max_concurrency_for_request": on_row[
                "server_max_concurrency_for_request"
            ],
            "off_server_available_kv_cache_memory_gib": off_row[
                "server_available_kv_cache_memory_gib"
            ],
            "on_server_available_kv_cache_memory_gib": on_row[
                "server_available_kv_cache_memory_gib"
            ],
        }
        for metric in metric_names:
            off_value = off_row.get(metric)
            on_value = on_row.get(metric)
            contrast[f"off_{metric}"] = off_value
            contrast[f"on_{metric}"] = on_value
            contrast[f"cache_on_minus_off_{metric}"] = _delta(
                on_value,
                off_value,
            )
            contrast[f"cache_on_div_off_{metric}"] = _ratio(
                on_value,
                off_value,
            )
        for counter_name in counter_names:
            off_value = off_row.get(counter_name)
            on_value = on_row.get(counter_name)
            contrast[f"off_{counter_name}"] = off_value
            contrast[f"on_{counter_name}"] = on_value
            contrast[f"cache_on_minus_off_{counter_name}"] = _delta(
                on_value,
                off_value,
            )
        trial_rows.append(contrast)

    phase_orders = sorted({row["phase_order"] for row in trial_rows})
    prompt_profiles = sorted({row["prompt_profile"] for row in trial_rows})
    summary = []
    for phase_order in phase_orders:
        for prompt_profile in prompt_profiles:
            grouped_rows = [
                row
                for row in trial_rows
                if row["phase_order"] == phase_order
                and row["prompt_profile"] == prompt_profile
            ]
            if not grouped_rows:
                continue
            for metric in summary_metrics:
                stats = _metric_distribution(grouped_rows, metric)
                values = [
                    float(row[metric])
                    for row in grouped_rows
                    if row.get(metric) is not None
                ]
                interval = _bootstrap_mean_interval(values)
                summary.append(
                    {
                        "phase_order": phase_order,
                        "prompt_profile": prompt_profile,
                        "metric": metric,
                        "trial_count": len(values),
                        "mean": stats[f"{metric}_mean"],
                        "min": stats[f"{metric}_min"],
                        "max": stats[f"{metric}_max"],
                        "median": stats[f"{metric}_median"],
                        "p95": stats[f"{metric}_p95"],
                        "cv": stats[f"{metric}_cv"],
                        "bootstrap_mean_p05": interval["mean_p05"],
                        "bootstrap_mean_p50": interval["mean_p50"],
                        "bootstrap_mean_p95": interval["mean_p95"],
                    }
                )

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-cache-control-server-absolute",
        "trial_count": len(compare_dirs),
        "compare_dirs": [str(path) for path in compare_dirs],
        "run_row_count": len(run_rows),
        "contrast_count": len(trial_rows),
        "phase_orders": phase_orders,
        "prompt_profiles": prompt_profiles,
        "run_rows": run_rows,
        "trial_rows": trial_rows,
        "summary": summary,
        "all_off_server_disable_observed": (
            all(
                row["off_server_enable_prefix_caching_observed"] is False
                for row in trial_rows
            )
            if trial_rows
            else None
        ),
        "all_on_server_enable_observed": (
            all(
                row["on_server_enable_prefix_caching_observed"] is True
                for row in trial_rows
            )
            if trial_rows
            else None
        ),
        "compare_payloads": compare_payloads,
    }


def _compare_vllm_server_async_csvs(
    async_csv: Path,
    server_csv: Path,
    async_label: str,
    server_label: str,
) -> dict[str, Any]:
    async_rows = _read_csv_by_key(async_csv, "scenario_id")
    server_rows = _read_csv_by_key(server_csv, "scenario_id")
    scenario_ids = sorted(set(async_rows) & set(server_rows))
    if not scenario_ids:
        raise ValueError("No matching scenario_id values found for comparison")

    rows = []
    for scenario_id in scenario_ids:
        async_row = async_rows[scenario_id]
        server_row = server_rows[scenario_id]

        async_throughput = _float_field(
            async_row,
            "aggregate_output_tokens_per_second_median",
        )
        server_throughput = _float_field(
            server_row,
            "aggregate_output_tokens_per_second_median",
        )
        async_first_event = _float_field(async_row, "p95_first_chunk_ms_median")
        server_first_event = _float_field(server_row, "p95_first_content_ms_median")
        async_latency = _float_field(async_row, "p95_latency_ms_median")
        server_latency = _float_field(server_row, "p95_latency_ms_median")
        async_tpot = _float_field(async_row, "p95_stream_tpot_ms_median")
        server_tpot = _float_field(server_row, "p95_stream_tpot_ms_median")
        async_batch_wall = _float_field(async_row, "batch_wall_ms_median")
        server_batch_wall = _float_field(server_row, "batch_wall_ms_median")

        rows.append(
            {
                "scenario_id": scenario_id,
                "prompt_profile": async_row["prompt_profile"],
                "request_count": int(async_row["request_count"]),
                "max_new_tokens": int(async_row["max_new_tokens"]),
                "async_repeats": int(async_row["repeats"]),
                "server_repeats": int(server_row["repeats"]),
                "async_prompt_tokens_mean": _float_field(async_row, "prompt_tokens_mean"),
                "server_prompt_tokens_mean": _float_field(server_row, "prompt_tokens_mean"),
                "async_output_tokens_per_second_median": async_throughput,
                "server_output_tokens_per_second_median": server_throughput,
                "server_to_async_output_tokens_per_second_delta": _delta(
                    server_throughput,
                    async_throughput,
                ),
                "server_to_async_output_tokens_per_second_ratio": _ratio(
                    server_throughput,
                    async_throughput,
                ),
                "async_p95_first_event_ms_median": async_first_event,
                "server_p95_first_content_ms_median": server_first_event,
                "server_to_async_p95_first_event_ms_delta": _delta(
                    server_first_event,
                    async_first_event,
                ),
                "server_to_async_p95_first_event_ms_ratio": _ratio(
                    server_first_event,
                    async_first_event,
                ),
                "async_p95_latency_ms_median": async_latency,
                "server_p95_latency_ms_median": server_latency,
                "server_to_async_p95_latency_ms_delta": _delta(
                    server_latency,
                    async_latency,
                ),
                "server_to_async_p95_latency_ms_ratio": _ratio(
                    server_latency,
                    async_latency,
                ),
                "async_p95_stream_tpot_ms_median": async_tpot,
                "server_p95_stream_tpot_ms_median": server_tpot,
                "server_to_async_p95_stream_tpot_ms_delta": _delta(
                    server_tpot,
                    async_tpot,
                ),
                "server_to_async_p95_stream_tpot_ms_ratio": _ratio(
                    server_tpot,
                    async_tpot,
                ),
                "async_batch_wall_ms_median": async_batch_wall,
                "server_batch_wall_ms_median": server_batch_wall,
                "server_to_async_batch_wall_ms_delta": _delta(
                    server_batch_wall,
                    async_batch_wall,
                ),
                "server_to_async_batch_wall_ms_ratio": _ratio(
                    server_batch_wall,
                    async_batch_wall,
                ),
                "async_throughput_cv": _float_field(
                    async_row,
                    "aggregate_output_tokens_per_second_cv",
                ),
                "server_throughput_cv": _float_field(
                    server_row,
                    "aggregate_output_tokens_per_second_cv",
                ),
                "async_latency_cv": _float_field(async_row, "p95_latency_ms_cv"),
                "server_latency_cv": _float_field(server_row, "p95_latency_ms_cv"),
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-server-sweep-compare",
        "async_label": async_label,
        "server_label": server_label,
        "async_csv": str(async_csv),
        "server_csv": str(server_csv),
        "scenario_count": len(rows),
        "mean_server_to_async_throughput_ratio": _mean_present(
            row["server_to_async_output_tokens_per_second_ratio"] for row in rows
        ),
        "mean_server_to_async_first_event_ratio": _mean_present(
            row["server_to_async_p95_first_event_ms_ratio"] for row in rows
        ),
        "mean_server_to_async_latency_ratio": _mean_present(
            row["server_to_async_p95_latency_ms_ratio"] for row in rows
        ),
        "mean_server_to_async_tpot_ratio": _mean_present(
            row["server_to_async_p95_stream_tpot_ms_ratio"] for row in rows
        ),
        "rows": rows,
    }


def _compare_vllm_server_async_phase_orders(
    async_first_dir: Path,
    server_first_dir: Path,
) -> dict[str, Any]:
    async_first_json = async_first_dir / "paired-server-async.json"
    server_first_json = server_first_dir / "paired-server-async.json"
    async_first_csv = async_first_dir / "paired-server-async-summary.csv"
    server_first_csv = server_first_dir / "paired-server-async-summary.csv"

    async_first_payload = json.loads(async_first_json.read_text(encoding="utf-8"))
    server_first_payload = json.loads(server_first_json.read_text(encoding="utf-8"))
    async_first_rows = _read_csv_by_key(async_first_csv, "scenario_id")
    server_first_rows = _read_csv_by_key(server_first_csv, "scenario_id")
    scenario_ids = sorted(set(async_first_rows) & set(server_first_rows))
    if not scenario_ids:
        raise ValueError("No matching scenario_id values found for phase-order comparison")

    ratio_fields = (
        "server_to_async_output_tokens_per_second_ratio_median",
        "server_to_async_p95_first_event_ms_ratio_median",
        "server_to_async_p95_latency_ms_ratio_median",
        "server_to_async_p95_stream_tpot_ms_ratio_median",
        "server_to_async_batch_wall_ms_ratio_median",
    )
    rows = []
    for scenario_id in scenario_ids:
        async_first = async_first_rows[scenario_id]
        server_first = server_first_rows[scenario_id]
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "prompt_profile": async_first["prompt_profile"],
            "request_count": int(async_first["request_count"]),
            "max_new_tokens": int(async_first["max_new_tokens"]),
            "pairs": int(async_first["pairs"]),
        }
        for field in ratio_fields:
            async_first_value = _float_field(async_first, field)
            server_first_value = _float_field(server_first, field)
            short_field = field.removeprefix("server_to_async_").removesuffix("_median")
            row[f"async_first_{short_field}"] = async_first_value
            row[f"server_first_{short_field}"] = server_first_value
            row[f"server_first_minus_async_first_{short_field}"] = _delta(
                server_first_value,
                async_first_value,
            )
        rows.append(row)

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-phase-order-compare",
        "async_first_dir": str(async_first_dir),
        "server_first_dir": str(server_first_dir),
        "async_first_json": str(async_first_json),
        "server_first_json": str(server_first_json),
        "async_first_csv": str(async_first_csv),
        "server_first_csv": str(server_first_csv),
        "scenario_count": len(rows),
        "async_first": {
            "phase_order": async_first_payload.get("phase_order", "async_first"),
            "paired_run_count": async_first_payload["paired_run_count"],
            "server_ready_ms": async_first_payload["server_ready_ms"],
            "async_engine_load_ms": async_first_payload["async_engine_load_ms"],
            "mean_server_to_async_throughput_ratio": async_first_payload[
                "mean_server_to_async_throughput_ratio"
            ],
            "mean_server_to_async_first_event_ratio": async_first_payload[
                "mean_server_to_async_first_event_ratio"
            ],
            "mean_server_to_async_latency_ratio": async_first_payload[
                "mean_server_to_async_latency_ratio"
            ],
            "mean_server_to_async_tpot_ratio": async_first_payload[
                "mean_server_to_async_tpot_ratio"
            ],
        },
        "server_first": {
            "phase_order": server_first_payload.get("phase_order", "server_first"),
            "paired_run_count": server_first_payload["paired_run_count"],
            "server_ready_ms": server_first_payload["server_ready_ms"],
            "async_engine_load_ms": server_first_payload["async_engine_load_ms"],
            "mean_server_to_async_throughput_ratio": server_first_payload[
                "mean_server_to_async_throughput_ratio"
            ],
            "mean_server_to_async_first_event_ratio": server_first_payload[
                "mean_server_to_async_first_event_ratio"
            ],
            "mean_server_to_async_latency_ratio": server_first_payload[
                "mean_server_to_async_latency_ratio"
            ],
            "mean_server_to_async_tpot_ratio": server_first_payload[
                "mean_server_to_async_tpot_ratio"
            ],
        },
        "rows": rows,
    }


def _aggregate_vllm_server_async_phase_order_trials(
    compare_dirs: list[Path],
) -> dict[str, Any]:
    if not compare_dirs:
        raise ValueError("compare_dirs must not be empty")

    metric_sources = {
        "throughput_ratio": "mean_server_to_async_throughput_ratio",
        "first_event_ratio": "mean_server_to_async_first_event_ratio",
        "latency_ratio": "mean_server_to_async_latency_ratio",
        "tpot_ratio": "mean_server_to_async_tpot_ratio",
    }
    trials = []
    for index, compare_dir in enumerate(compare_dirs, start=1):
        path = compare_dir / "phase-order-compare.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        trial: dict[str, Any] = {
            "trial_index": index,
            "trial_label": compare_dir.name,
            "compare_json": str(path),
            "scenario_count": payload["scenario_count"],
        }
        for metric, source_field in metric_sources.items():
            async_first_value = payload["async_first"][source_field]
            server_first_value = payload["server_first"][source_field]
            trial[f"async_first_{metric}"] = async_first_value
            trial[f"server_first_{metric}"] = server_first_value
            trial[f"server_first_minus_async_first_{metric}"] = (
                server_first_value - async_first_value
            )
        trials.append(trial)

    summary = []
    for metric in metric_sources:
        async_stats = _metric_distribution(trials, f"async_first_{metric}")
        server_stats = _metric_distribution(trials, f"server_first_{metric}")
        delta_stats = _metric_distribution(
            trials,
            f"server_first_minus_async_first_{metric}",
        )
        delta_values = [
            trial[f"server_first_minus_async_first_{metric}"]
            for trial in trials
            if trial.get(f"server_first_minus_async_first_{metric}") is not None
        ]
        bootstrap_interval = _bootstrap_mean_interval(delta_values)
        summary.append(
            {
                "metric": metric,
                "trial_count": len(trials),
                "async_first_mean": async_stats[f"async_first_{metric}_mean"],
                "async_first_min": async_stats[f"async_first_{metric}_min"],
                "async_first_max": async_stats[f"async_first_{metric}_max"],
                "async_first_cv": async_stats[f"async_first_{metric}_cv"],
                "server_first_mean": server_stats[f"server_first_{metric}_mean"],
                "server_first_min": server_stats[f"server_first_{metric}_min"],
                "server_first_max": server_stats[f"server_first_{metric}_max"],
                "server_first_cv": server_stats[f"server_first_{metric}_cv"],
                "server_first_minus_async_first_mean": delta_stats[
                    f"server_first_minus_async_first_{metric}_mean"
                ],
                "server_first_minus_async_first_min": delta_stats[
                    f"server_first_minus_async_first_{metric}_min"
                ],
                "server_first_minus_async_first_max": delta_stats[
                    f"server_first_minus_async_first_{metric}_max"
                ],
                "server_first_minus_async_first_cv": delta_stats[
                    f"server_first_minus_async_first_{metric}_cv"
                ],
                "server_first_minus_async_first_bootstrap_mean_p05": bootstrap_interval[
                    "mean_p05"
                ],
                "server_first_minus_async_first_bootstrap_mean_p50": bootstrap_interval[
                    "mean_p50"
                ],
                "server_first_minus_async_first_bootstrap_mean_p95": bootstrap_interval[
                    "mean_p95"
                ],
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-multitrial-aggregate",
        "trial_count": len(trials),
        "compare_dirs": [str(path) for path in compare_dirs],
        "trials": trials,
        "summary": summary,
    }


def _compare_vllm_server_async_workload_profiles(
    short_multitrial_dir: Path,
    long_phase_order_compare_dir: Path,
) -> dict[str, Any]:
    short_json = short_multitrial_dir / "phase-order-multitrial.json"
    short_csv = short_multitrial_dir / "phase-order-multitrial.csv"
    long_json = long_phase_order_compare_dir / "phase-order-compare.json"
    long_csv = long_phase_order_compare_dir / "phase-order-compare.csv"

    short_payload = json.loads(short_json.read_text(encoding="utf-8"))
    long_payload = json.loads(long_json.read_text(encoding="utf-8"))
    short_csv_rows = _read_csv_by_key(short_csv, "metric")
    long_csv_rows = _read_csv_by_key(long_csv, "scenario_id")
    long_profiles = sorted(
        {
            row["prompt_profile"]
            for row in long_csv_rows.values()
            if row.get("prompt_profile")
        }
    )
    long_profile = ",".join(long_profiles) if long_profiles else "long"

    metric_sources = {
        "throughput_ratio": "mean_server_to_async_throughput_ratio",
        "first_event_ratio": "mean_server_to_async_first_event_ratio",
        "latency_ratio": "mean_server_to_async_latency_ratio",
        "tpot_ratio": "mean_server_to_async_tpot_ratio",
    }
    short_summary_by_metric = {
        row["metric"]: row
        for row in short_payload.get("summary", [])
    }
    missing_metrics = sorted(set(metric_sources) - set(short_summary_by_metric))
    if missing_metrics:
        raise ValueError(
            "short multitrial aggregate is missing metrics: "
            + ", ".join(missing_metrics)
        )
    missing_csv_metrics = sorted(set(metric_sources) - set(short_csv_rows))
    if missing_csv_metrics:
        raise ValueError(
            "short multitrial csv is missing metrics: "
            + ", ".join(missing_csv_metrics)
        )

    rows = []
    for metric, source_field in metric_sources.items():
        short_row = short_summary_by_metric[metric]
        short_async_mean = short_row["async_first_mean"]
        short_server_mean = short_row["server_first_mean"]
        short_order_effect_delta_mean = short_row[
            "server_first_minus_async_first_mean"
        ]
        long_async_mean = long_payload["async_first"][source_field]
        long_server_mean = long_payload["server_first"][source_field]
        long_order_effect_delta_mean = _delta(long_server_mean, long_async_mean)

        rows.append(
            {
                "metric": metric,
                "short_profile": "short",
                "long_profile": long_profile,
                "short_trial_count": short_payload["trial_count"],
                "long_scenario_count": long_payload["scenario_count"],
                "long_paired_run_count": min(
                    long_payload["async_first"]["paired_run_count"],
                    long_payload["server_first"]["paired_run_count"],
                ),
                "short_async_first_mean": short_async_mean,
                "long_async_first_mean": long_async_mean,
                "long_minus_short_async_first_mean": _delta(
                    long_async_mean,
                    short_async_mean,
                ),
                "short_server_first_mean": short_server_mean,
                "long_server_first_mean": long_server_mean,
                "long_minus_short_server_first_mean": _delta(
                    long_server_mean,
                    short_server_mean,
                ),
                "short_order_effect_delta_mean": short_order_effect_delta_mean,
                "long_order_effect_delta_mean": long_order_effect_delta_mean,
                "long_minus_short_order_effect_delta_mean": _delta(
                    long_order_effect_delta_mean,
                    short_order_effect_delta_mean,
                ),
                "short_order_effect_bootstrap_mean_p05": short_row[
                    "server_first_minus_async_first_bootstrap_mean_p05"
                ],
                "short_order_effect_bootstrap_mean_p50": short_row[
                    "server_first_minus_async_first_bootstrap_mean_p50"
                ],
                "short_order_effect_bootstrap_mean_p95": short_row[
                    "server_first_minus_async_first_bootstrap_mean_p95"
                ],
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-workload-compare",
        "short_profile": "short",
        "long_profile": long_profile,
        "short_multitrial_dir": str(short_multitrial_dir),
        "long_phase_order_compare_dir": str(long_phase_order_compare_dir),
        "short_json": str(short_json),
        "short_csv": str(short_csv),
        "long_json": str(long_json),
        "long_csv": str(long_csv),
        "metric_count": len(rows),
        "short_trial_count": short_payload["trial_count"],
        "long_scenario_count": long_payload["scenario_count"],
        "long_paired_run_count": min(
            long_payload["async_first"]["paired_run_count"],
            long_payload["server_first"]["paired_run_count"],
        ),
        "rows": rows,
    }


def _bootstrap_mean_interval(
    values: list[float],
    samples: int = 4096,
    seed: int = DEFAULT_VLLM_SWEEP_SEED,
) -> dict[str, float | None]:
    if not values:
        return {"mean_p05": None, "mean_p50": None, "mean_p95": None}
    if len(values) == 1:
        return {
            "mean_p05": values[0],
            "mean_p50": values[0],
            "mean_p95": values[0],
        }

    import random

    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    return {
        "mean_p05": _percentile(means, 5),
        "mean_p50": _percentile(means, 50),
        "mean_p95": _percentile(means, 95),
    }


def _make_vllm_server_async_paired_rows(
    async_runs: list[dict[str, Any]],
    server_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    async_by_key = {
        (run["scenario_id"], run["repeat_index"]): run
        for run in async_runs
    }
    server_by_key = {
        (run["scenario_id"], run["repeat_index"]): run
        for run in server_runs
    }
    keys = sorted(
        set(async_by_key) & set(server_by_key),
        key=lambda key: (key[0], key[1]),
    )
    if not keys:
        raise ValueError("No matching paired runs found")

    paired_rows = []
    for scenario_id, repeat_index in keys:
        async_run = async_by_key[(scenario_id, repeat_index)]
        server_run = server_by_key[(scenario_id, repeat_index)]
        async_throughput = async_run.get("aggregate_output_tokens_per_second")
        server_throughput = server_run.get("aggregate_output_tokens_per_second")
        async_first_event = async_run.get("p95_first_chunk_ms")
        server_first_event = server_run.get("p95_first_content_ms")
        async_latency = async_run.get("p95_latency_ms")
        server_latency = server_run.get("p95_latency_ms")
        async_tpot = async_run.get("p95_stream_tpot_ms")
        server_tpot = server_run.get("p95_stream_tpot_ms")
        async_batch_wall = async_run.get("batch_wall_ms")
        server_batch_wall = server_run.get("batch_wall_ms")
        server_counter_hit_rate = server_run.get(
            "server_prefix_cache_counter_hit_rate_pct"
        )
        paired_rows.append(
            {
                "pair_id": f"{scenario_id}_rep{repeat_index:02d}",
                "scenario_id": scenario_id,
                "prompt_profile": async_run["prompt_profile"],
                "request_count": async_run["request_count"],
                "max_new_tokens": async_run["max_new_tokens"],
                "repeat_index": repeat_index,
                "async_run_order": async_run["run_order"],
                "server_run_order": server_run["run_order"],
                "prompt_tokens_mean": async_run["prompt_tokens_mean"],
                "estimated_peak_sequence_tokens_async": async_run.get(
                    "estimated_peak_sequence_tokens"
                ),
                "estimated_peak_sequence_tokens_server": server_run.get(
                    "estimated_peak_sequence_tokens"
                ),
                "async_output_tokens_per_second": async_throughput,
                "server_output_tokens_per_second": server_throughput,
                "server_to_async_output_tokens_per_second_delta": _delta(
                    server_throughput,
                    async_throughput,
                ),
                "server_to_async_output_tokens_per_second_ratio": _ratio(
                    server_throughput,
                    async_throughput,
                ),
                "async_p95_first_event_ms": async_first_event,
                "server_p95_first_content_ms": server_first_event,
                "server_to_async_p95_first_event_ms_delta": _delta(
                    server_first_event,
                    async_first_event,
                ),
                "server_to_async_p95_first_event_ms_ratio": _ratio(
                    server_first_event,
                    async_first_event,
                ),
                "async_p95_latency_ms": async_latency,
                "server_p95_latency_ms": server_latency,
                "server_to_async_p95_latency_ms_delta": _delta(
                    server_latency,
                    async_latency,
                ),
                "server_to_async_p95_latency_ms_ratio": _ratio(
                    server_latency,
                    async_latency,
                ),
                "async_p95_stream_tpot_ms": async_tpot,
                "server_p95_stream_tpot_ms": server_tpot,
                "server_to_async_p95_stream_tpot_ms_delta": _delta(
                    server_tpot,
                    async_tpot,
                ),
                "server_to_async_p95_stream_tpot_ms_ratio": _ratio(
                    server_tpot,
                    async_tpot,
                ),
                "async_batch_wall_ms": async_batch_wall,
                "server_batch_wall_ms": server_batch_wall,
                "server_to_async_batch_wall_ms_delta": _delta(
                    server_batch_wall,
                    async_batch_wall,
                ),
                "server_to_async_batch_wall_ms_ratio": _ratio(
                    server_batch_wall,
                    async_batch_wall,
                ),
                "server_prefix_cache_counter_requests": server_run.get(
                    "server_prefix_cache_counter_requests"
                ),
                "server_prefix_cache_counter_queries": server_run.get(
                    "server_prefix_cache_counter_queries"
                ),
                "server_prefix_cache_counter_hits": server_run.get(
                    "server_prefix_cache_counter_hits"
                ),
                "server_prefix_cache_counter_hit_rate_pct": server_counter_hit_rate,
                "server_metrics_before_ok": server_run.get(
                    "server_metrics_before_ok"
                ),
                "server_metrics_after_ok": server_run.get(
                    "server_metrics_after_ok"
                ),
                "server_metrics_before_prefix_related_sample_count": server_run.get(
                    "server_metrics_before_prefix_related_sample_count"
                ),
                "server_metrics_after_prefix_related_sample_count": server_run.get(
                    "server_metrics_after_prefix_related_sample_count"
                ),
                "server_metrics_selected_query_metric": server_run.get(
                    "server_metrics_selected_query_metric"
                ),
                "server_metrics_selected_hit_metric": server_run.get(
                    "server_metrics_selected_hit_metric"
                ),
            }
        )
    return paired_rows


def _aggregate_vllm_server_async_paired_scenarios(
    paired_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in paired_runs:
        pairs_by_scenario.setdefault(row["scenario_id"], []).append(row)

    scenarios = []
    for scenario_id, pairs in sorted(pairs_by_scenario.items()):
        first = pairs[0]
        scenario: dict[str, Any] = {
            "scenario_id": scenario_id,
            "prompt_profile": first["prompt_profile"],
            "request_count": first["request_count"],
            "max_new_tokens": first["max_new_tokens"],
            "pairs": len(pairs),
            "prompt_tokens_mean": first["prompt_tokens_mean"],
        }
        for field in (
            "server_to_async_output_tokens_per_second_ratio",
            "server_to_async_output_tokens_per_second_delta",
            "server_to_async_p95_first_event_ms_ratio",
            "server_to_async_p95_first_event_ms_delta",
            "server_to_async_p95_latency_ms_ratio",
            "server_to_async_p95_latency_ms_delta",
            "server_to_async_p95_stream_tpot_ms_ratio",
            "server_to_async_p95_stream_tpot_ms_delta",
            "server_to_async_batch_wall_ms_ratio",
            "server_to_async_batch_wall_ms_delta",
            "async_output_tokens_per_second",
            "server_output_tokens_per_second",
            "async_p95_first_event_ms",
            "server_p95_first_content_ms",
            "async_p95_latency_ms",
            "server_p95_latency_ms",
            "async_p95_stream_tpot_ms",
            "server_p95_stream_tpot_ms",
            "server_prefix_cache_counter_requests",
            "server_prefix_cache_counter_queries",
            "server_prefix_cache_counter_hits",
            "server_prefix_cache_counter_hit_rate_pct",
            "server_metrics_before_prefix_related_sample_count",
            "server_metrics_after_prefix_related_sample_count",
        ):
            scenario.update(_metric_distribution(pairs, field))
        scenarios.append(scenario)
    return scenarios


def _vllm_server_async_paired_summary_rows(
    paired_scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "pairs",
        "prompt_tokens_mean",
        "server_to_async_output_tokens_per_second_ratio_median",
        "server_to_async_output_tokens_per_second_ratio_p95",
        "server_to_async_p95_first_event_ms_ratio_median",
        "server_to_async_p95_first_event_ms_ratio_p95",
        "server_to_async_p95_latency_ms_ratio_median",
        "server_to_async_p95_latency_ms_ratio_p95",
        "server_to_async_p95_stream_tpot_ms_ratio_median",
        "server_to_async_p95_stream_tpot_ms_ratio_p95",
        "server_to_async_batch_wall_ms_ratio_median",
        "server_to_async_batch_wall_ms_ratio_p95",
        "async_output_tokens_per_second_median",
        "server_output_tokens_per_second_median",
        "async_p95_first_event_ms_median",
        "server_p95_first_content_ms_median",
        "async_p95_latency_ms_median",
        "server_p95_latency_ms_median",
        "async_p95_stream_tpot_ms_median",
        "server_p95_stream_tpot_ms_median",
        "server_prefix_cache_counter_requests_median",
        "server_prefix_cache_counter_queries_median",
        "server_prefix_cache_counter_hits_median",
        "server_prefix_cache_counter_hit_rate_pct_median",
        "server_metrics_before_prefix_related_sample_count_median",
        "server_metrics_after_prefix_related_sample_count_median",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in paired_scenarios
    ]


def _make_vllm_prefix_cache_paired_rows(
    cold_runs: list[dict[str, Any]],
    cache_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cold_by_key = {
        (run["scenario_id"], run["repeat_index"]): run
        for run in cold_runs
    }
    cache_by_key = {
        (run["scenario_id"], run["repeat_index"]): run
        for run in cache_runs
    }
    keys = sorted(
        set(cold_by_key) & set(cache_by_key),
        key=lambda key: (key[0], key[1]),
    )
    if not keys:
        raise ValueError("No matching paired prefix-cache runs found")

    paired_rows = []
    for scenario_id, repeat_index in keys:
        cold_run = cold_by_key[(scenario_id, repeat_index)]
        cache_run = cache_by_key[(scenario_id, repeat_index)]
        cold_throughput = cold_run.get("aggregate_output_tokens_per_second")
        cache_throughput = cache_run.get("aggregate_output_tokens_per_second")
        cold_first_event = cold_run.get("p95_first_chunk_ms")
        cache_first_event = cache_run.get("p95_first_chunk_ms")
        cold_latency = cold_run.get("p95_latency_ms")
        cache_latency = cache_run.get("p95_latency_ms")
        cold_tpot = cold_run.get("p95_stream_tpot_ms")
        cache_tpot = cache_run.get("p95_stream_tpot_ms")
        cold_batch_wall = cold_run.get("batch_wall_ms")
        cache_batch_wall = cache_run.get("batch_wall_ms")
        cold_prefix_hit_rate = cold_run.get("prefix_cache_hit_rate_pct")
        cache_prefix_hit_rate = cache_run.get("prefix_cache_hit_rate_pct")
        cold_counter_hit_rate = cold_run.get("prefix_cache_counter_hit_rate_pct")
        cache_counter_hit_rate = cache_run.get("prefix_cache_counter_hit_rate_pct")
        cold_gpu_kv_usage = cold_run.get("gpu_kv_cache_usage_pct")
        cache_gpu_kv_usage = cache_run.get("gpu_kv_cache_usage_pct")
        capacity_fields = (
            "engine_max_model_len",
            "engine_max_num_batched_tokens",
            "engine_max_num_seqs",
            "engine_gpu_memory_utilization",
            "engine_gpu_kv_cache_size_tokens",
            "engine_available_kv_cache_memory_gib",
            "engine_max_concurrency_for_request",
            "engine_max_concurrency_request_tokens",
            "engine_derived_gpu_kv_cache_size_tokens",
        )
        paired_rows.append(
            {
                "pair_id": f"{scenario_id}_rep{repeat_index:02d}",
                "scenario_id": scenario_id,
                "prompt_profile": cold_run["prompt_profile"],
                "request_count": cold_run["request_count"],
                "max_new_tokens": cold_run["max_new_tokens"],
                "repeat_index": repeat_index,
                "cold_run_order": cold_run["run_order"],
                "cache_run_order": cache_run["run_order"],
                "prompt_tokens_mean": cold_run["prompt_tokens_mean"],
                "estimated_peak_sequence_tokens_cold": cold_run.get(
                    "estimated_peak_sequence_tokens"
                ),
                "estimated_peak_sequence_tokens_cache": cache_run.get(
                    "estimated_peak_sequence_tokens"
                ),
                "cold_output_tokens_per_second": cold_throughput,
                "cache_output_tokens_per_second": cache_throughput,
                "cache_to_cold_output_tokens_per_second_delta": _delta(
                    cache_throughput,
                    cold_throughput,
                ),
                "cache_to_cold_output_tokens_per_second_ratio": _ratio(
                    cache_throughput,
                    cold_throughput,
                ),
                "cold_p95_first_event_ms": cold_first_event,
                "cache_p95_first_event_ms": cache_first_event,
                "cache_to_cold_p95_first_event_ms_delta": _delta(
                    cache_first_event,
                    cold_first_event,
                ),
                "cache_to_cold_p95_first_event_ms_ratio": _ratio(
                    cache_first_event,
                    cold_first_event,
                ),
                "cold_p95_latency_ms": cold_latency,
                "cache_p95_latency_ms": cache_latency,
                "cache_to_cold_p95_latency_ms_delta": _delta(
                    cache_latency,
                    cold_latency,
                ),
                "cache_to_cold_p95_latency_ms_ratio": _ratio(
                    cache_latency,
                    cold_latency,
                ),
                "cold_p95_stream_tpot_ms": cold_tpot,
                "cache_p95_stream_tpot_ms": cache_tpot,
                "cache_to_cold_p95_stream_tpot_ms_delta": _delta(
                    cache_tpot,
                    cold_tpot,
                ),
                "cache_to_cold_p95_stream_tpot_ms_ratio": _ratio(
                    cache_tpot,
                    cold_tpot,
                ),
                "cold_batch_wall_ms": cold_batch_wall,
                "cache_batch_wall_ms": cache_batch_wall,
                "cache_to_cold_batch_wall_ms_delta": _delta(
                    cache_batch_wall,
                    cold_batch_wall,
                ),
                "cache_to_cold_batch_wall_ms_ratio": _ratio(
                    cache_batch_wall,
                    cold_batch_wall,
                ),
                "cold_prefix_cache_hit_rate_pct": cold_prefix_hit_rate,
                "cache_prefix_cache_hit_rate_pct": cache_prefix_hit_rate,
                "cache_to_cold_prefix_cache_hit_rate_pct_delta": _delta(
                    cache_prefix_hit_rate,
                    cold_prefix_hit_rate,
                ),
                "cold_prefix_cache_counter_requests": cold_run.get(
                    "prefix_cache_counter_requests"
                ),
                "cache_prefix_cache_counter_requests": cache_run.get(
                    "prefix_cache_counter_requests"
                ),
                "cold_prefix_cache_counter_queries": cold_run.get(
                    "prefix_cache_counter_queries"
                ),
                "cache_prefix_cache_counter_queries": cache_run.get(
                    "prefix_cache_counter_queries"
                ),
                "cold_prefix_cache_counter_hits": cold_run.get(
                    "prefix_cache_counter_hits"
                ),
                "cache_prefix_cache_counter_hits": cache_run.get(
                    "prefix_cache_counter_hits"
                ),
                "cold_prefix_cache_counter_hit_rate_pct": cold_counter_hit_rate,
                "cache_prefix_cache_counter_hit_rate_pct": cache_counter_hit_rate,
                "cache_to_cold_prefix_cache_counter_hit_rate_pct_delta": _delta(
                    cache_counter_hit_rate,
                    cold_counter_hit_rate,
                ),
                "cold_gpu_kv_cache_usage_pct": cold_gpu_kv_usage,
                "cache_gpu_kv_cache_usage_pct": cache_gpu_kv_usage,
                "cache_to_cold_gpu_kv_cache_usage_pct_delta": _delta(
                    cache_gpu_kv_usage,
                    cold_gpu_kv_usage,
                ),
                **{
                    f"cold_{field}": cold_run.get(field)
                    for field in capacity_fields
                },
                **{
                    f"cache_{field}": cache_run.get(field)
                    for field in capacity_fields
                },
            }
        )
    return paired_rows


def _aggregate_vllm_prefix_cache_paired_scenarios(
    paired_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in paired_runs:
        pairs_by_scenario.setdefault(row["scenario_id"], []).append(row)

    scenarios = []
    for scenario_id, pairs in sorted(pairs_by_scenario.items()):
        first = pairs[0]
        scenario: dict[str, Any] = {
            "scenario_id": scenario_id,
            "prompt_profile": first["prompt_profile"],
            "request_count": first["request_count"],
            "max_new_tokens": first["max_new_tokens"],
            "pairs": len(pairs),
            "prompt_tokens_mean": first["prompt_tokens_mean"],
        }
        for field in (
            "cache_to_cold_output_tokens_per_second_ratio",
            "cache_to_cold_output_tokens_per_second_delta",
            "cache_to_cold_p95_first_event_ms_ratio",
            "cache_to_cold_p95_first_event_ms_delta",
            "cache_to_cold_p95_latency_ms_ratio",
            "cache_to_cold_p95_latency_ms_delta",
            "cache_to_cold_p95_stream_tpot_ms_ratio",
            "cache_to_cold_p95_stream_tpot_ms_delta",
            "cache_to_cold_batch_wall_ms_ratio",
            "cache_to_cold_batch_wall_ms_delta",
            "cold_output_tokens_per_second",
            "cache_output_tokens_per_second",
            "cold_p95_first_event_ms",
            "cache_p95_first_event_ms",
            "cold_p95_latency_ms",
            "cache_p95_latency_ms",
            "cold_p95_stream_tpot_ms",
            "cache_p95_stream_tpot_ms",
            "cold_prefix_cache_hit_rate_pct",
            "cache_prefix_cache_hit_rate_pct",
            "cache_to_cold_prefix_cache_hit_rate_pct_delta",
            "cold_prefix_cache_counter_requests",
            "cache_prefix_cache_counter_requests",
            "cold_prefix_cache_counter_queries",
            "cache_prefix_cache_counter_queries",
            "cold_prefix_cache_counter_hits",
            "cache_prefix_cache_counter_hits",
            "cold_prefix_cache_counter_hit_rate_pct",
            "cache_prefix_cache_counter_hit_rate_pct",
            "cache_to_cold_prefix_cache_counter_hit_rate_pct_delta",
            "cold_gpu_kv_cache_usage_pct",
            "cache_gpu_kv_cache_usage_pct",
            "cache_to_cold_gpu_kv_cache_usage_pct_delta",
            "cold_engine_max_model_len",
            "cache_engine_max_model_len",
            "cold_engine_max_num_batched_tokens",
            "cache_engine_max_num_batched_tokens",
            "cold_engine_max_num_seqs",
            "cache_engine_max_num_seqs",
            "cold_engine_gpu_memory_utilization",
            "cache_engine_gpu_memory_utilization",
            "cold_engine_gpu_kv_cache_size_tokens",
            "cache_engine_gpu_kv_cache_size_tokens",
            "cold_engine_available_kv_cache_memory_gib",
            "cache_engine_available_kv_cache_memory_gib",
            "cold_engine_max_concurrency_for_request",
            "cache_engine_max_concurrency_for_request",
            "cold_engine_max_concurrency_request_tokens",
            "cache_engine_max_concurrency_request_tokens",
            "cold_engine_derived_gpu_kv_cache_size_tokens",
            "cache_engine_derived_gpu_kv_cache_size_tokens",
        ):
            scenario.update(_metric_distribution(pairs, field))
        scenarios.append(scenario)
    return scenarios


def _vllm_prefix_cache_paired_summary_rows(
    paired_scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "pairs",
        "prompt_tokens_mean",
        "cache_to_cold_output_tokens_per_second_ratio_median",
        "cache_to_cold_output_tokens_per_second_ratio_p95",
        "cache_to_cold_p95_first_event_ms_ratio_median",
        "cache_to_cold_p95_first_event_ms_ratio_p95",
        "cache_to_cold_p95_latency_ms_ratio_median",
        "cache_to_cold_p95_latency_ms_ratio_p95",
        "cache_to_cold_p95_stream_tpot_ms_ratio_median",
        "cache_to_cold_p95_stream_tpot_ms_ratio_p95",
        "cache_to_cold_batch_wall_ms_ratio_median",
        "cache_to_cold_batch_wall_ms_ratio_p95",
        "cold_output_tokens_per_second_median",
        "cache_output_tokens_per_second_median",
        "cold_p95_first_event_ms_median",
        "cache_p95_first_event_ms_median",
        "cold_p95_latency_ms_median",
        "cache_p95_latency_ms_median",
        "cold_p95_stream_tpot_ms_median",
        "cache_p95_stream_tpot_ms_median",
        "cold_prefix_cache_hit_rate_pct_median",
        "cache_prefix_cache_hit_rate_pct_median",
        "cache_to_cold_prefix_cache_hit_rate_pct_delta_median",
        "cold_prefix_cache_counter_queries_median",
        "cache_prefix_cache_counter_queries_median",
        "cold_prefix_cache_counter_hits_median",
        "cache_prefix_cache_counter_hits_median",
        "cold_prefix_cache_counter_hit_rate_pct_median",
        "cache_prefix_cache_counter_hit_rate_pct_median",
        "cache_to_cold_prefix_cache_counter_hit_rate_pct_delta_median",
        "cold_gpu_kv_cache_usage_pct_median",
        "cache_gpu_kv_cache_usage_pct_median",
        "cache_to_cold_gpu_kv_cache_usage_pct_delta_median",
        "cold_engine_max_model_len_median",
        "cache_engine_max_model_len_median",
        "cold_engine_max_num_batched_tokens_median",
        "cache_engine_max_num_batched_tokens_median",
        "cold_engine_max_num_seqs_median",
        "cache_engine_max_num_seqs_median",
        "cold_engine_gpu_memory_utilization_median",
        "cache_engine_gpu_memory_utilization_median",
        "cold_engine_gpu_kv_cache_size_tokens_median",
        "cache_engine_gpu_kv_cache_size_tokens_median",
        "cold_engine_available_kv_cache_memory_gib_median",
        "cache_engine_available_kv_cache_memory_gib_median",
        "cold_engine_max_concurrency_for_request_median",
        "cache_engine_max_concurrency_for_request_median",
        "cold_engine_max_concurrency_request_tokens_median",
        "cache_engine_max_concurrency_request_tokens_median",
        "cold_engine_derived_gpu_kv_cache_size_tokens_median",
        "cache_engine_derived_gpu_kv_cache_size_tokens_median",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in paired_scenarios
    ]


def _vllm_prefix_cache_isolated_profile_control_rows(
    paired_scenarios: list[dict[str, Any]],
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> list[dict[str, Any]]:
    by_profile_and_shape = {
        (
            scenario["prompt_profile"],
            int(scenario["request_count"]),
            int(scenario["max_new_tokens"]),
        ): scenario
        for scenario in paired_scenarios
    }
    shared_keys = {
        (request_count, max_new_tokens)
        for profile, request_count, max_new_tokens in by_profile_and_shape
        if profile == shared_profile
    }
    control_keys = {
        (request_count, max_new_tokens)
        for profile, request_count, max_new_tokens in by_profile_and_shape
        if profile == control_profile
    }
    comparison_keys = sorted(shared_keys & control_keys)
    if not comparison_keys:
        return []

    metric_fields = (
        "cache_to_cold_output_tokens_per_second_ratio_median",
        "cache_to_cold_p95_first_event_ms_ratio_median",
        "cache_to_cold_p95_latency_ms_ratio_median",
        "cache_to_cold_p95_stream_tpot_ms_ratio_median",
        "cache_to_cold_batch_wall_ms_ratio_median",
        "cold_prefix_cache_hit_rate_pct_median",
        "cache_prefix_cache_hit_rate_pct_median",
        "cache_to_cold_prefix_cache_hit_rate_pct_delta_median",
        "cold_prefix_cache_counter_queries_median",
        "cache_prefix_cache_counter_queries_median",
        "cold_prefix_cache_counter_hits_median",
        "cache_prefix_cache_counter_hits_median",
        "cold_prefix_cache_counter_hit_rate_pct_median",
        "cache_prefix_cache_counter_hit_rate_pct_median",
        "cache_to_cold_prefix_cache_counter_hit_rate_pct_delta_median",
        "cold_gpu_kv_cache_usage_pct_median",
        "cache_gpu_kv_cache_usage_pct_median",
        "cache_to_cold_gpu_kv_cache_usage_pct_delta_median",
    )
    rows = []
    for request_count, max_new_tokens in comparison_keys:
        shared = by_profile_and_shape[(shared_profile, request_count, max_new_tokens)]
        control = by_profile_and_shape[(control_profile, request_count, max_new_tokens)]
        shared_prompt_tokens = shared.get("prompt_tokens_mean")
        control_prompt_tokens = control.get("prompt_tokens_mean")
        row: dict[str, Any] = {
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "shared_profile": shared_profile,
            "control_profile": control_profile,
            "shared_scenario_id": shared["scenario_id"],
            "control_scenario_id": control["scenario_id"],
            "shared_prompt_tokens_mean": shared_prompt_tokens,
            "control_prompt_tokens_mean": control_prompt_tokens,
            "shared_to_control_prompt_tokens_mean_ratio": _ratio(
                shared_prompt_tokens,
                control_prompt_tokens,
            ),
        }
        for field in metric_fields:
            shared_value = shared.get(field)
            control_value = control.get(field)
            row[f"shared_{field}"] = shared_value
            row[f"control_{field}"] = control_value
            row[f"shared_minus_control_{field}"] = _delta(
                shared_value,
                control_value,
            )
        rows.append(row)
    return rows


def _vllm_prefix_cache_isolated_profile_control_summary(
    profile_control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not profile_control_rows:
        return {}
    summary_fields = [
        field
        for field in profile_control_rows[0]
        if field.startswith("shared_minus_control_")
    ]
    summary = {}
    for field in summary_fields:
        values = [
            row[field]
            for row in profile_control_rows
            if row.get(field) is not None
        ]
        summary[f"mean_{field}"] = (
            sum(float(value) for value in values) / len(values)
            if values
            else None
        )
    summary["mean_shared_to_control_prompt_tokens_mean_ratio"] = _mean_present(
        row["shared_to_control_prompt_tokens_mean_ratio"]
        for row in profile_control_rows
    )
    return summary


def _latest_cache_metric_field(
    metrics: list[dict[str, Any]] | None,
    field: str,
) -> float | None:
    for row in reversed(metrics or []):
        value = row.get(field)
        if value is not None:
            return float(value)
    return None


def _parse_vllm_engine_capacity_log_metrics(
    log_stats: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any]:
    calls = log_stats if isinstance(log_stats, list) else [log_stats]
    text_parts: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        for line in call.get("captured_logs", []) or []:
            text_parts.append(str(line))
        for key in ("stdout", "stderr", "return_repr"):
            value = call.get(key)
            if value:
                text_parts.append(str(value))
    text = "\n".join(text_parts)

    def number(pattern: str) -> float | None:
        match = re.search(pattern, text)
        if not match:
            return None
        return float(match.group(1).replace(",", ""))

    gpu_tokens = number(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")
    request_tokens = number(
        r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:"
    )
    max_concurrency = number(
        r"Maximum concurrency for\s*[0-9,]+\s*tokens per request:\s*([0-9.]+)x"
    )
    max_model_len = number(r"Using max model len\s*([0-9,]+)")
    available_kv_cache_memory_gib = number(
        r"Available KV cache memory:\s*([0-9.]+)\s*GiB"
    )
    return {
        "gpu_kv_cache_size_tokens": (
            int(gpu_tokens) if gpu_tokens is not None else None
        ),
        "max_concurrency_request_tokens": (
            int(request_tokens) if request_tokens is not None else None
        ),
        "max_concurrency_for_request": max_concurrency,
        "max_model_len": int(max_model_len) if max_model_len is not None else None,
        "available_kv_cache_memory_gib": available_kv_cache_memory_gib,
    }


def _safe_get_attr_path(target: Any, attr_path: tuple[str, ...]) -> Any:
    current = target
    for attr in attr_path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(attr)
        else:
            current = getattr(current, attr, None)
    return current


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value if _json_scalar(item) is not None]
    return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _discover_vllm_engine_config_values(
    *,
    engine: Any,
    engine_args: Any,
) -> dict[str, Any]:
    paths: dict[str, tuple[Any, tuple[str, ...]]] = {
        "engine_args.max_model_len": (engine_args, ("max_model_len",)),
        "engine_args.max_num_batched_tokens": (
            engine_args,
            ("max_num_batched_tokens",),
        ),
        "engine_args.max_num_seqs": (engine_args, ("max_num_seqs",)),
        "engine_args.gpu_memory_utilization": (
            engine_args,
            ("gpu_memory_utilization",),
        ),
        "engine.engine_core.engine_core_config.model_config.max_model_len": (
            engine,
            ("engine_core", "engine_core_config", "model_config", "max_model_len"),
        ),
        "engine.engine_core.engine_core_config.scheduler_config.max_num_batched_tokens": (
            engine,
            (
                "engine_core",
                "engine_core_config",
                "scheduler_config",
                "max_num_batched_tokens",
            ),
        ),
        "engine.engine_core.engine_core_config.scheduler_config.max_num_seqs": (
            engine,
            ("engine_core", "engine_core_config", "scheduler_config", "max_num_seqs"),
        ),
        "engine.engine_core.engine_core_config.cache_config.block_size": (
            engine,
            ("engine_core", "engine_core_config", "cache_config", "block_size"),
        ),
        "engine.engine_core.engine_core_config.cache_config.num_gpu_blocks": (
            engine,
            ("engine_core", "engine_core_config", "cache_config", "num_gpu_blocks"),
        ),
        "engine.engine_core.engine_core_config.cache_config.gpu_memory_utilization": (
            engine,
            (
                "engine_core",
                "engine_core_config",
                "cache_config",
                "gpu_memory_utilization",
            ),
        ),
        "engine.engine_core.engine_core_config.cache_config.kv_cache_memory_bytes": (
            engine,
            (
                "engine_core",
                "engine_core_config",
                "cache_config",
                "kv_cache_memory_bytes",
            ),
        ),
        "engine.llm_engine.scheduler_config.max_num_batched_tokens": (
            engine,
            ("llm_engine", "scheduler_config", "max_num_batched_tokens"),
        ),
        "engine.llm_engine.scheduler_config.max_num_seqs": (
            engine,
            ("llm_engine", "scheduler_config", "max_num_seqs"),
        ),
        "engine.llm_engine.cache_config.block_size": (
            engine,
            ("llm_engine", "cache_config", "block_size"),
        ),
        "engine.llm_engine.cache_config.num_gpu_blocks": (
            engine,
            ("llm_engine", "cache_config", "num_gpu_blocks"),
        ),
        "engine.llm_engine.model_config.max_model_len": (
            engine,
            ("llm_engine", "model_config", "max_model_len"),
        ),
    }
    discovered = {}
    for label, (root, attr_path) in paths.items():
        value = _json_scalar(_safe_get_attr_path(root, attr_path))
        if value is not None:
            discovered[label] = value
    return discovered


def _snapshot_vllm_engine_capacity(
    *,
    engine: Any,
    engine_args: Any,
    phase_label: str,
    engine_init_log_stats: dict[str, Any] | None,
    requested_max_model_len: int,
    requested_max_num_batched_tokens: int,
    requested_max_num_seqs: int,
    requested_gpu_memory_utilization: float,
) -> dict[str, Any]:
    log_metrics = _parse_vllm_engine_capacity_log_metrics(engine_init_log_stats)
    discovered = _discover_vllm_engine_config_values(
        engine=engine,
        engine_args=engine_args,
    )
    block_size = _first_number(
        discovered.get("engine.engine_core.engine_core_config.cache_config.block_size"),
        discovered.get("engine.llm_engine.cache_config.block_size"),
    )
    num_gpu_blocks = _first_number(
        discovered.get("engine.engine_core.engine_core_config.cache_config.num_gpu_blocks"),
        discovered.get("engine.llm_engine.cache_config.num_gpu_blocks"),
    )
    derived_gpu_tokens = (
        int(block_size * num_gpu_blocks)
        if block_size is not None and num_gpu_blocks is not None
        else None
    )
    gpu_kv_cache_size_tokens = (
        log_metrics.get("gpu_kv_cache_size_tokens") or derived_gpu_tokens
    )
    max_model_len = (
        log_metrics.get("max_model_len")
        or _first_number(
            discovered.get("engine.engine_core.engine_core_config.model_config.max_model_len"),
            discovered.get("engine.llm_engine.model_config.max_model_len"),
            discovered.get("engine_args.max_model_len"),
        )
        or requested_max_model_len
    )
    max_concurrency = log_metrics.get("max_concurrency_for_request")
    if max_concurrency is None and gpu_kv_cache_size_tokens and max_model_len:
        max_concurrency = float(gpu_kv_cache_size_tokens) / float(max_model_len)

    max_num_batched_tokens = (
        _first_number(
            discovered.get(
                "engine.engine_core.engine_core_config.scheduler_config.max_num_batched_tokens"
            ),
            discovered.get("engine.llm_engine.scheduler_config.max_num_batched_tokens"),
            discovered.get("engine_args.max_num_batched_tokens"),
        )
        or requested_max_num_batched_tokens
    )
    max_num_seqs = (
        _first_number(
            discovered.get("engine.engine_core.engine_core_config.scheduler_config.max_num_seqs"),
            discovered.get("engine.llm_engine.scheduler_config.max_num_seqs"),
            discovered.get("engine_args.max_num_seqs"),
        )
        or requested_max_num_seqs
    )
    gpu_memory_utilization = (
        _first_number(
            discovered.get(
                "engine.engine_core.engine_core_config.cache_config.gpu_memory_utilization"
            ),
            discovered.get("engine_args.gpu_memory_utilization"),
        )
        or requested_gpu_memory_utilization
    )
    return {
        "phase_label": phase_label,
        "requested_max_model_len": int(requested_max_model_len),
        "requested_max_num_batched_tokens": int(requested_max_num_batched_tokens),
        "requested_max_num_seqs": int(requested_max_num_seqs),
        "requested_gpu_memory_utilization": float(requested_gpu_memory_utilization),
        "max_model_len": int(max_model_len) if max_model_len is not None else None,
        "max_num_batched_tokens": int(max_num_batched_tokens)
        if max_num_batched_tokens is not None
        else None,
        "max_num_seqs": int(max_num_seqs) if max_num_seqs is not None else None,
        "gpu_memory_utilization": float(gpu_memory_utilization)
        if gpu_memory_utilization is not None
        else None,
        "gpu_kv_cache_size_tokens": gpu_kv_cache_size_tokens,
        "available_kv_cache_memory_gib": log_metrics.get(
            "available_kv_cache_memory_gib"
        ),
        "max_concurrency_for_request": max_concurrency,
        "max_concurrency_request_tokens": (
            log_metrics.get("max_concurrency_request_tokens") or int(max_model_len)
        )
        if max_model_len is not None
        else None,
        "derived_gpu_kv_cache_size_tokens": derived_gpu_tokens,
        "discovered_config_values": discovered,
        "engine_init_log_stats": engine_init_log_stats,
    }


def _vllm_engine_capacity_run_fields(
    capacity: dict[str, Any] | None,
) -> dict[str, Any]:
    capacity = capacity or {}
    return {
        "engine_max_model_len": capacity.get("max_model_len"),
        "engine_max_num_batched_tokens": capacity.get("max_num_batched_tokens"),
        "engine_max_num_seqs": capacity.get("max_num_seqs"),
        "engine_gpu_memory_utilization": capacity.get("gpu_memory_utilization"),
        "engine_gpu_kv_cache_size_tokens": capacity.get("gpu_kv_cache_size_tokens"),
        "engine_available_kv_cache_memory_gib": capacity.get(
            "available_kv_cache_memory_gib"
        ),
        "engine_max_concurrency_for_request": capacity.get(
            "max_concurrency_for_request"
        ),
        "engine_max_concurrency_request_tokens": capacity.get(
            "max_concurrency_request_tokens"
        ),
        "engine_derived_gpu_kv_cache_size_tokens": capacity.get(
            "derived_gpu_kv_cache_size_tokens"
        ),
    }


def _estimated_window_rate_pct(
    *,
    warmup_rate_pct: float | None,
    after_rate_pct: float | None,
    warmup_tokens: float | None,
    measured_tokens: float | None,
) -> float | None:
    if (
        warmup_rate_pct is None
        or after_rate_pct is None
        or warmup_tokens is None
        or measured_tokens is None
        or measured_tokens <= 0
    ):
        return None
    total_tokens = warmup_tokens + measured_tokens
    estimated = (
        (after_rate_pct / 100.0 * total_tokens)
        - (warmup_rate_pct / 100.0 * warmup_tokens)
    ) / measured_tokens * 100.0
    return max(0.0, min(100.0, estimated))


def _prefix_cache_isolated_metrics_source_paths(
    isolated_metrics_dir: Path,
) -> tuple[Path, Path]:
    final_json = isolated_metrics_dir / "prefix-cache-isolated-metrics.json"
    final_runs_csv = isolated_metrics_dir / "prefix-cache-isolated-metrics-runs.csv"
    partial_json = isolated_metrics_dir / "prefix-cache-isolated-metrics.partial.json"
    partial_runs_csv = (
        isolated_metrics_dir / "prefix-cache-isolated-metrics-runs.partial.csv"
    )
    if final_json.exists() or not partial_json.exists():
        return final_json, final_runs_csv
    return partial_json, partial_runs_csv


def _coerce_csv_scalar(value: str) -> Any:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_isolated_metric_run_rows(
    source_payload: dict[str, Any],
    source_runs_csv: Path,
) -> list[dict[str, Any]]:
    if source_runs_csv.exists():
        with source_runs_csv.open("r", encoding="utf-8", newline="") as handle:
            return [
                {
                    field: _coerce_csv_scalar(value)
                    for field, value in row.items()
                }
                for row in csv.DictReader(handle)
            ]
    return [dict(row) for row in source_payload.get("paired_runs", [])]


def _ordered_unique(values: Any) -> list[Any]:
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _same_value_or_list(values: Any) -> Any:
    unique = _ordered_unique(values)
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else unique


def _merge_vllm_prefix_cache_isolated_metrics(
    source_dirs: list[Path],
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> dict[str, Any]:
    if not source_dirs:
        raise ValueError("At least one isolated metrics source directory is required")

    paired_runs: list[dict[str, Any]] = []
    remote_call_summaries: list[dict[str, Any]] = []
    source_chunks: list[dict[str, Any]] = []
    source_payloads: list[dict[str, Any]] = []
    next_repeat_index = 0
    next_call_index = 0
    planned_remote_call_count = 0

    for chunk_index, source_dir in enumerate(source_dirs):
        source_json, source_runs_csv = _prefix_cache_isolated_metrics_source_paths(
            source_dir
        )
        source_payload = json.loads(source_json.read_text(encoding="utf-8"))
        source_rows = _read_isolated_metric_run_rows(source_payload, source_runs_csv)
        if not source_rows:
            raise ValueError(f"No isolated metric rows found in {source_dir}")

        source_payloads.append(source_payload)
        source_repeat_values = sorted(
            {
                int(row.get("repeat_index") or 0)
                for row in source_rows
            }
        )
        repeat_index_map = {
            source_repeat: next_repeat_index + offset
            for offset, source_repeat in enumerate(source_repeat_values)
        }

        source_call_index_map: dict[Any, int] = {}
        source_remote_calls = source_payload.get("remote_call_summaries") or []
        for call_ordinal, call in enumerate(source_remote_calls):
            source_call_index = int(call.get("isolation_call_index", call_ordinal))
            global_call_index = next_call_index
            next_call_index += 1
            source_call_index_map[source_call_index] = global_call_index

            merged_call = dict(call)
            merged_call["source_chunk_index"] = chunk_index
            merged_call["source_dir"] = str(source_dir)
            merged_call["source_isolation_call_index"] = source_call_index
            merged_call["isolation_call_index"] = global_call_index
            remote_call_summaries.append(merged_call)

        for row_ordinal, row in enumerate(source_rows):
            source_repeat_index = int(row.get("repeat_index") or 0)
            global_repeat_index = repeat_index_map[source_repeat_index]
            source_call_value = row.get("isolation_call_index")
            source_call_key: Any
            if source_call_value is None:
                source_call_key = f"row:{row_ordinal}"
                source_call_index = None
            else:
                source_call_index = int(source_call_value)
                source_call_key = source_call_index

            if source_call_key not in source_call_index_map:
                source_call_index_map[source_call_key] = next_call_index
                remote_call_summaries.append(
                    {
                        "isolation_call_index": next_call_index,
                        "source_chunk_index": chunk_index,
                        "source_dir": str(source_dir),
                        "source_isolation_call_index": source_call_index,
                        "source_row_ordinal": row_ordinal,
                        "synthetic": True,
                    }
                )
                next_call_index += 1
            global_call_index = source_call_index_map[source_call_key]

            merged_row = dict(row)
            merged_row["source_chunk_index"] = chunk_index
            merged_row["source_dir"] = str(source_dir)
            merged_row["source_json"] = str(source_json)
            merged_row["source_runs_csv"] = str(source_runs_csv)
            merged_row["source_repeat_index"] = source_repeat_index
            merged_row["source_isolation_call_index"] = source_call_index
            merged_row["repeat_index"] = global_repeat_index
            merged_row["isolation_call_index"] = global_call_index
            merged_row["pair_id"] = (
                f"{merged_row['scenario_id']}_merged_rep{global_repeat_index:02d}"
            )
            paired_runs.append(merged_row)

        chunk_complete = source_payload.get("checkpoint_complete")
        if chunk_complete is None:
            chunk_complete = source_json.name == "prefix-cache-isolated-metrics.json"
        source_chunks.append(
            {
                "source_chunk_index": chunk_index,
                "source_dir": str(source_dir),
                "source_json": str(source_json),
                "source_runs_csv": str(source_runs_csv),
                "source_mode": source_payload.get("mode"),
                "source_checkpoint_complete": chunk_complete,
                "source_repeats": source_payload.get("repeats"),
                "source_repeat_count": len(source_repeat_values),
                "global_repeat_start": min(repeat_index_map.values()),
                "global_repeat_end": max(repeat_index_map.values()),
                "source_paired_run_count": len(source_rows),
                "source_remote_call_count": len(source_remote_calls),
            }
        )
        planned_remote_call_count += int(
            source_payload.get("planned_remote_call_count") or len(source_rows)
        )
        next_repeat_index += len(source_repeat_values)

    paired_scenarios = _aggregate_vllm_prefix_cache_paired_scenarios(paired_runs)
    summary_rows = _vllm_prefix_cache_paired_summary_rows(paired_scenarios)
    profile_control_rows = _vllm_prefix_cache_isolated_profile_control_rows(
        paired_scenarios,
        shared_profile=shared_profile,
        control_profile=control_profile,
    )

    request_count_values = sorted(
        {int(row["request_count"]) for row in paired_runs}
    )
    prompt_profile_values = _ordered_unique(row["prompt_profile"] for row in paired_runs)
    output_token_values = sorted(
        {int(row["max_new_tokens"]) for row in paired_runs}
    )
    checkpoint_complete = all(
        bool(row["source_checkpoint_complete"]) for row in source_chunks
    )
    return {
        "schema_version": 1,
        "execution": "local",
        "mode": "vllm-prefix-cache-isolated-merge",
        "backend": "vllm-paired-prefix-cache",
        "model_id": _same_value_or_list(
            payload.get("model_id") for payload in source_payloads
        ),
        "source_chunk_count": len(source_chunks),
        "source_chunks": source_chunks,
        "source_modes": _ordered_unique(
            payload.get("mode") for payload in source_payloads
        ),
        "source_dirs": [str(path) for path in source_dirs],
        "source_checkpoint_complete": checkpoint_complete,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": next_repeat_index,
        "scenario_seed": _same_value_or_list(
            payload.get("scenario_seed") for payload in source_payloads
        ),
        "phase_order": _same_value_or_list(
            payload.get("phase_order") for payload in source_payloads
        ),
        "warmup_runs": _same_value_or_list(
            payload.get("warmup_runs") for payload in source_payloads
        ),
        "warmup_prompt_profile": _same_value_or_list(
            payload.get("warmup_prompt_profile") for payload in source_payloads
        ),
        "collect_cache_metrics": _same_value_or_list(
            payload.get("collect_cache_metrics") for payload in source_payloads
        ),
        "kv_cache_metrics_sample": _same_value_or_list(
            payload.get("kv_cache_metrics_sample") for payload in source_payloads
        ),
        "checkpoint_complete": checkpoint_complete,
        "planned_remote_call_count": planned_remote_call_count,
        "remote_call_count": len(remote_call_summaries),
        "scenario_count": len(paired_scenarios),
        "paired_run_count": len(paired_runs),
        "summary": summary_rows,
        "paired_runs": paired_runs,
        "paired_scenarios": paired_scenarios,
        "profile_control": {
            "shared_profile": shared_profile,
            "control_profile": control_profile,
            "row_count": len(profile_control_rows),
            "summary": _vllm_prefix_cache_isolated_profile_control_summary(
                profile_control_rows
            ),
            "rows": profile_control_rows,
        },
        "remote_call_summaries": remote_call_summaries,
        "mean_cache_to_cold_throughput_ratio": _mean_present(
            row["cache_to_cold_output_tokens_per_second_ratio"]
            for row in paired_runs
        ),
        "mean_cache_to_cold_first_event_ratio": _mean_present(
            row["cache_to_cold_p95_first_event_ms_ratio"]
            for row in paired_runs
        ),
        "mean_cache_to_cold_latency_ratio": _mean_present(
            row["cache_to_cold_p95_latency_ms_ratio"] for row in paired_runs
        ),
        "mean_cache_to_cold_tpot_ratio": _mean_present(
            row["cache_to_cold_p95_stream_tpot_ms_ratio"]
            for row in paired_runs
        ),
        "note": (
            "Merged isolated metrics artifact. Source chunks may be final or "
            "partial checkpoint directories; repeat_index and isolation_call_index "
            "are reindexed globally so stability summaries can pair observations "
            "across chunks."
        ),
    }


def _summarize_vllm_prefix_cache_isolated_window(
    isolated_metrics_dir: Path,
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> dict[str, Any]:
    source_json, _source_runs_csv = _prefix_cache_isolated_metrics_source_paths(
        isolated_metrics_dir
    )
    source_payload = json.loads(source_json.read_text(encoding="utf-8"))
    paired_runs = source_payload["paired_runs"]
    remote_calls = {
        int(row["isolation_call_index"]): row
        for row in source_payload["remote_call_summaries"]
    }
    window_rows = []
    for paired in paired_runs:
        call = remote_calls[int(paired["isolation_call_index"])]
        warmup_summaries = call.get("cache_warmup", {}).get("summaries", [])
        warmup_prompt_tokens = sum(
            float(row.get("total_prompt_tokens") or 0.0)
            for row in warmup_summaries
        )
        measured_prompt_tokens = (
            float(paired["prompt_tokens_mean"]) * int(paired["request_count"])
        )
        cache_warmup_hit = _latest_cache_metric_field(
            call.get("cache_warmup_cache_metrics"),
            "prefix_cache_hit_rate_pct",
        )
        cache_after_hit = paired.get("cache_prefix_cache_hit_rate_pct")
        cold_warmup_hit = _latest_cache_metric_field(
            call.get("cold_warmup_cache_metrics"),
            "prefix_cache_hit_rate_pct",
        )
        cold_after_hit = paired.get("cold_prefix_cache_hit_rate_pct")
        cache_estimated = _estimated_window_rate_pct(
            warmup_rate_pct=cache_warmup_hit,
            after_rate_pct=cache_after_hit,
            warmup_tokens=warmup_prompt_tokens,
            measured_tokens=measured_prompt_tokens,
        )
        cold_estimated = _estimated_window_rate_pct(
            warmup_rate_pct=cold_warmup_hit,
            after_rate_pct=cold_after_hit,
            warmup_tokens=warmup_prompt_tokens,
            measured_tokens=measured_prompt_tokens,
        )
        window_rows.append(
            {
                "scenario_id": paired["scenario_id"],
                "prompt_profile": paired["prompt_profile"],
                "request_count": int(paired["request_count"]),
                "max_new_tokens": int(paired["max_new_tokens"]),
                "repeat_index": int(paired["repeat_index"]),
                "isolation_call_index": int(paired["isolation_call_index"]),
                "warmup_prompt_profile": call.get("warmup_prompt_profile"),
                "warmup_prompt_tokens": warmup_prompt_tokens,
                "measured_prompt_tokens": measured_prompt_tokens,
                "cache_warmup_prefix_cache_hit_rate_pct": cache_warmup_hit,
                "cache_after_prefix_cache_hit_rate_pct": cache_after_hit,
                "cache_estimated_measured_prefix_cache_hit_rate_pct": cache_estimated,
                "cold_warmup_prefix_cache_hit_rate_pct": cold_warmup_hit,
                "cold_after_prefix_cache_hit_rate_pct": cold_after_hit,
                "cold_estimated_measured_prefix_cache_hit_rate_pct": cold_estimated,
                "cache_to_cold_estimated_measured_prefix_cache_hit_rate_pct_delta": _delta(
                    cache_estimated,
                    cold_estimated,
                ),
                "cache_to_cold_output_tokens_per_second_ratio": paired[
                    "cache_to_cold_output_tokens_per_second_ratio"
                ],
                "cache_to_cold_p95_latency_ms_ratio": paired[
                    "cache_to_cold_p95_latency_ms_ratio"
                ],
            }
        )

    def add_stats(row: dict[str, Any], prefix: str, values: list[float]) -> None:
        row[f"{prefix}_mean"] = sum(values) / len(values) if values else None
        row[f"{prefix}_min"] = min(values) if values else None
        row[f"{prefix}_max"] = max(values) if values else None
        row[f"{prefix}_population_stdev"] = (
            statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
        )

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in window_rows:
        key = (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        )
        grouped.setdefault(key, []).append(row)

    scenario_rows = []
    for key, rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][2], item[0][1]),
    ):
        prompt_profile, request_count, max_new_tokens = key
        first = rows[0]
        scenario: dict[str, Any] = {
            "scenario_id": first["scenario_id"],
            "prompt_profile": prompt_profile,
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "runs": len(rows),
            "warmup_prompt_profile": first["warmup_prompt_profile"],
            "warmup_prompt_tokens_mean": _mean_present(
                row["warmup_prompt_tokens"] for row in rows
            ),
            "measured_prompt_tokens_mean": _mean_present(
                row["measured_prompt_tokens"] for row in rows
            ),
        }
        for field in (
            "cache_warmup_prefix_cache_hit_rate_pct",
            "cache_after_prefix_cache_hit_rate_pct",
            "cache_estimated_measured_prefix_cache_hit_rate_pct",
            "cold_estimated_measured_prefix_cache_hit_rate_pct",
            "cache_to_cold_estimated_measured_prefix_cache_hit_rate_pct_delta",
            "cache_to_cold_output_tokens_per_second_ratio",
            "cache_to_cold_p95_latency_ms_ratio",
        ):
            values = [
                float(row[field])
                for row in rows
                if row.get(field) is not None
            ]
            add_stats(scenario, field, values)
        scenario_rows.append(scenario)

    by_profile_and_shape = {
        (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        ): row
        for row in scenario_rows
    }
    comparison_keys = sorted(
        {
            (request_count, max_new_tokens)
            for profile, request_count, max_new_tokens in by_profile_and_shape
            if profile == shared_profile
        }
        & {
            (request_count, max_new_tokens)
            for profile, request_count, max_new_tokens in by_profile_and_shape
            if profile == control_profile
        }
    )
    profile_control_rows = []
    for request_count, max_new_tokens in comparison_keys:
        shared = by_profile_and_shape[(shared_profile, request_count, max_new_tokens)]
        control = by_profile_and_shape[(control_profile, request_count, max_new_tokens)]
        shared_estimated = shared[
            "cache_estimated_measured_prefix_cache_hit_rate_pct_mean"
        ]
        control_estimated = control[
            "cache_estimated_measured_prefix_cache_hit_rate_pct_mean"
        ]
        shared_after = shared["cache_after_prefix_cache_hit_rate_pct_mean"]
        control_after = control["cache_after_prefix_cache_hit_rate_pct_mean"]
        shared_throughput = shared[
            "cache_to_cold_output_tokens_per_second_ratio_mean"
        ]
        control_throughput = control[
            "cache_to_cold_output_tokens_per_second_ratio_mean"
        ]
        shared_latency = shared["cache_to_cold_p95_latency_ms_ratio_mean"]
        control_latency = control["cache_to_cold_p95_latency_ms_ratio_mean"]
        profile_control_rows.append(
            {
                "request_count": request_count,
                "max_new_tokens": max_new_tokens,
                "shared_profile": shared_profile,
                "control_profile": control_profile,
                "shared_estimated_measured_cache_hit_rate_pct": shared_estimated,
                "control_estimated_measured_cache_hit_rate_pct": control_estimated,
                "shared_minus_control_estimated_measured_cache_hit_rate_pct": _delta(
                    shared_estimated,
                    control_estimated,
                ),
                "shared_after_cache_hit_rate_pct": shared_after,
                "control_after_cache_hit_rate_pct": control_after,
                "shared_minus_control_after_cache_hit_rate_pct": _delta(
                    shared_after,
                    control_after,
                ),
                "shared_cache_to_cold_throughput_ratio_mean": shared_throughput,
                "control_cache_to_cold_throughput_ratio_mean": control_throughput,
                "shared_minus_control_cache_to_cold_throughput_ratio_mean": _delta(
                    shared_throughput,
                    control_throughput,
                ),
                "shared_cache_to_cold_p95_latency_ratio_mean": shared_latency,
                "control_cache_to_cold_p95_latency_ratio_mean": control_latency,
                "shared_minus_control_cache_to_cold_p95_latency_ratio_mean": _delta(
                    shared_latency,
                    control_latency,
                ),
            }
        )

    estimated_stdevs = [
        row["cache_estimated_measured_prefix_cache_hit_rate_pct_population_stdev"]
        for row in scenario_rows
        if row["cache_estimated_measured_prefix_cache_hit_rate_pct_population_stdev"]
        is not None
    ]
    summary = {
        "mean_shared_minus_control_estimated_measured_cache_hit_rate_pct": _mean_present(
            row["shared_minus_control_estimated_measured_cache_hit_rate_pct"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_after_cache_hit_rate_pct": _mean_present(
            row["shared_minus_control_after_cache_hit_rate_pct"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_throughput_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_throughput_ratio_mean"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_p95_latency_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_p95_latency_ratio_mean"]
            for row in profile_control_rows
        ),
        "max_estimated_measured_cache_hit_rate_population_stdev": (
            max(estimated_stdevs) if estimated_stdevs else None
        ),
    }
    markdown = _format_vllm_prefix_cache_isolated_window_markdown(
        source_dir=isolated_metrics_dir,
        source_mode=source_payload.get("mode"),
        source_warmup_runs=source_payload.get("warmup_runs"),
        source_warmup_prompt_profile=source_payload.get("warmup_prompt_profile"),
        scenario_rows=scenario_rows,
        profile_control_rows=profile_control_rows,
        summary=summary,
    )
    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-isolated-window-summary",
        "source_dir": str(isolated_metrics_dir),
        "source_json": str(source_json),
        "source_mode": source_payload.get("mode"),
        "source_checkpoint_complete": source_payload.get("checkpoint_complete"),
        "source_warmup_runs": source_payload.get("warmup_runs"),
        "source_warmup_prompt_profile": source_payload.get("warmup_prompt_profile"),
        "source_scenario_count": source_payload["scenario_count"],
        "source_paired_run_count": source_payload["paired_run_count"],
        "source_remote_call_count": source_payload["remote_call_count"],
        "estimate_method": (
            "Token-weighted before/after rate estimate: "
            "after_rate * (warmup_prompt_tokens + measured_prompt_tokens) "
            "minus warmup_rate * warmup_prompt_tokens, divided by measured_prompt_tokens."
        ),
        "scenario_count": len(scenario_rows),
        "profile_control_row_count": len(profile_control_rows),
        "shared_profile": shared_profile,
        "control_profile": control_profile,
        "summary": summary,
        "window_rows": window_rows,
        "scenario_rows": scenario_rows,
        "profile_control_rows": profile_control_rows,
        "markdown": markdown,
    }


def _format_vllm_prefix_cache_isolated_window_markdown(
    source_dir: Path,
    source_mode: Any,
    source_warmup_runs: Any,
    source_warmup_prompt_profile: Any,
    scenario_rows: list[dict[str, Any]],
    profile_control_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}{suffix}"

    lines = [
        "# Prefix-Cache Measured-Window Estimate",
        "",
        f"Source: `{source_dir}`",
        f"Source mode: `{source_mode}`",
        f"Warmup runs: `{source_warmup_runs}`",
        f"Warmup prompt profile: `{source_warmup_prompt_profile}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (
            "| Mean shared-minus-control estimated measured cache hit rate | "
            f"{fmt(summary['mean_shared_minus_control_estimated_measured_cache_hit_rate_pct'], ' pp')} |"
        ),
        (
            "| Mean shared-minus-control logged after-window cache hit rate | "
            f"{fmt(summary['mean_shared_minus_control_after_cache_hit_rate_pct'], ' pp')} |"
        ),
        "",
        "## Scenario Estimates",
        "",
        "| Profile | Requests | Runs | Warmup hit | Logged after | Estimated measured |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenario_rows:
        lines.append(
            "| "
            f"`{row['prompt_profile']}` | "
            f"{row['request_count']} | "
            f"{row['runs']} | "
            f"{fmt(row['cache_warmup_prefix_cache_hit_rate_pct_mean'], '%')} | "
            f"{fmt(row['cache_after_prefix_cache_hit_rate_pct_mean'], '%')} | "
            f"{fmt(row['cache_estimated_measured_prefix_cache_hit_rate_pct_mean'], '%')} |"
        )
    lines.extend(
        [
            "",
            "## Shared Vs Control",
            "",
            "| Requests | Estimated Shared | Estimated Control | Delta | Throughput Delta | p95 Latency Delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in profile_control_rows:
        lines.append(
            "| "
            f"{row['request_count']} | "
            f"{fmt(row['shared_estimated_measured_cache_hit_rate_pct'], '%')} | "
            f"{fmt(row['control_estimated_measured_cache_hit_rate_pct'], '%')} | "
            f"{fmt(row['shared_minus_control_estimated_measured_cache_hit_rate_pct'], ' pp')} | "
            f"{fmt(row['shared_minus_control_cache_to_cold_throughput_ratio_mean'])} | "
            f"{fmt(row['shared_minus_control_cache_to_cold_p95_latency_ratio_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Estimate method: token-weighted before/after rate delta. This is an approximation until direct vLLM hit/miss counters are captured.",
            "",
        ]
    )
    return "\n".join(lines)


def _value_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "paired_observation_count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "population_stdev": None,
            "bootstrap_mean_p05": None,
            "bootstrap_mean_p50": None,
            "bootstrap_mean_p95": None,
        }
    interval = _bootstrap_mean_interval(values)
    return {
        "paired_observation_count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "population_stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "bootstrap_mean_p05": interval["mean_p05"],
        "bootstrap_mean_p50": interval["mean_p50"],
        "bootstrap_mean_p95": interval["mean_p95"],
    }


def _add_distribution_fields(
    row: dict[str, Any],
    prefix: str,
    values: list[float],
) -> None:
    distribution = _value_distribution(values)
    for field, value in distribution.items():
        row[f"{prefix}_{field}"] = value


def _paired_profile_delta_values(
    run_rows: list[dict[str, Any]],
    shared_profile: str,
    control_profile: str,
    request_count: int,
    max_new_tokens: int,
    field: str,
) -> list[float]:
    shared_by_repeat: dict[int, dict[str, Any]] = {}
    control_by_repeat: dict[int, dict[str, Any]] = {}
    for row in run_rows:
        if int(row["request_count"]) != request_count:
            continue
        if int(row["max_new_tokens"]) != max_new_tokens:
            continue
        if row["prompt_profile"] not in {shared_profile, control_profile}:
            continue
        repeat_index = int(row.get("repeat_index") or 0)
        if row["prompt_profile"] == shared_profile:
            shared_by_repeat[repeat_index] = row
        else:
            control_by_repeat[repeat_index] = row

    values = []
    for repeat_index in sorted(set(shared_by_repeat) & set(control_by_repeat)):
        delta = _delta(
            _float_field(shared_by_repeat[repeat_index], field),
            _float_field(control_by_repeat[repeat_index], field),
        )
        if delta is not None:
            values.append(delta)
    return values


def _summarize_vllm_prefix_cache_isolated_stability(
    isolated_metrics_dir: Path,
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> dict[str, Any]:
    source_json, source_runs_csv = _prefix_cache_isolated_metrics_source_paths(
        isolated_metrics_dir
    )
    source_payload = json.loads(source_json.read_text(encoding="utf-8"))
    with source_runs_csv.open("r", encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    if not run_rows:
        raise ValueError("No isolated metric run rows found")

    def present_values(rows: list[dict[str, Any]], field: str) -> list[float]:
        values = []
        for row in rows:
            value = _float_field(row, field)
            if value is not None:
                values.append(value)
        return values

    def add_stats(row: dict[str, Any], prefix: str, values: list[float]) -> None:
        row[f"{prefix}_mean"] = sum(values) / len(values) if values else None
        row[f"{prefix}_min"] = min(values) if values else None
        row[f"{prefix}_max"] = max(values) if values else None
        row[f"{prefix}_population_stdev"] = (
            statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
        )

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        )
        grouped.setdefault(key, []).append(row)

    scenario_rows = []
    for key, rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][2], item[0][1]),
    ):
        prompt_profile, request_count, max_new_tokens = key
        first = rows[0]
        scenario: dict[str, Any] = {
            "scenario_id": first["scenario_id"],
            "prompt_profile": prompt_profile,
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "runs": len(rows),
            "prompt_tokens_mean": _float_field(first, "prompt_tokens_mean"),
        }
        for field in (
            "cold_prefix_cache_hit_rate_pct",
            "cache_prefix_cache_hit_rate_pct",
            "cache_to_cold_prefix_cache_hit_rate_pct_delta",
            "cold_prefix_cache_counter_queries",
            "cache_prefix_cache_counter_queries",
            "cold_prefix_cache_counter_hits",
            "cache_prefix_cache_counter_hits",
            "cold_prefix_cache_counter_hit_rate_pct",
            "cache_prefix_cache_counter_hit_rate_pct",
            "cache_to_cold_prefix_cache_counter_hit_rate_pct_delta",
            "cache_to_cold_output_tokens_per_second_ratio",
            "cache_to_cold_p95_first_event_ms_ratio",
            "cache_to_cold_p95_latency_ms_ratio",
            "cache_to_cold_p95_stream_tpot_ms_ratio",
        ):
            add_stats(scenario, field, present_values(rows, field))
        scenario_rows.append(scenario)

    by_profile_and_shape = {
        (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        ): row
        for row in scenario_rows
    }
    comparison_keys = sorted(
        {
            (request_count, max_new_tokens)
            for profile, request_count, max_new_tokens in by_profile_and_shape
            if profile == shared_profile
        }
        & {
            (request_count, max_new_tokens)
            for profile, request_count, max_new_tokens in by_profile_and_shape
            if profile == control_profile
        }
    )
    profile_control_rows = []
    paired_delta_values_by_metric: dict[str, list[float]] = {}
    for request_count, max_new_tokens in comparison_keys:
        shared = by_profile_and_shape[(shared_profile, request_count, max_new_tokens)]
        control = by_profile_and_shape[(control_profile, request_count, max_new_tokens)]
        shared_hit = shared["cache_prefix_cache_hit_rate_pct_mean"]
        control_hit = control["cache_prefix_cache_hit_rate_pct_mean"]
        shared_counter_hit = shared["cache_prefix_cache_counter_hit_rate_pct_mean"]
        control_counter_hit = control["cache_prefix_cache_counter_hit_rate_pct_mean"]
        shared_throughput = shared["cache_to_cold_output_tokens_per_second_ratio_mean"]
        control_throughput = control["cache_to_cold_output_tokens_per_second_ratio_mean"]
        shared_first_event = shared["cache_to_cold_p95_first_event_ms_ratio_mean"]
        control_first_event = control["cache_to_cold_p95_first_event_ms_ratio_mean"]
        shared_latency = shared["cache_to_cold_p95_latency_ms_ratio_mean"]
        control_latency = control["cache_to_cold_p95_latency_ms_ratio_mean"]
        shared_tpot = shared["cache_to_cold_p95_stream_tpot_ms_ratio_mean"]
        control_tpot = control["cache_to_cold_p95_stream_tpot_ms_ratio_mean"]
        profile_control_row = {
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "shared_profile": shared_profile,
            "control_profile": control_profile,
            "shared_cache_hit_rate_pct_mean": shared_hit,
            "control_cache_hit_rate_pct_mean": control_hit,
            "shared_minus_control_cache_hit_rate_pct": _delta(
                shared_hit,
                control_hit,
            ),
            "shared_cache_counter_queries_mean": shared[
                "cache_prefix_cache_counter_queries_mean"
            ],
            "control_cache_counter_queries_mean": control[
                "cache_prefix_cache_counter_queries_mean"
            ],
            "shared_cache_counter_hits_mean": shared[
                "cache_prefix_cache_counter_hits_mean"
            ],
            "control_cache_counter_hits_mean": control[
                "cache_prefix_cache_counter_hits_mean"
            ],
            "shared_cache_counter_hit_rate_pct_mean": shared_counter_hit,
            "control_cache_counter_hit_rate_pct_mean": control_counter_hit,
            "shared_minus_control_cache_counter_hit_rate_pct": _delta(
                shared_counter_hit,
                control_counter_hit,
            ),
            "shared_cache_counter_hit_rate_pct_population_stdev": shared[
                "cache_prefix_cache_counter_hit_rate_pct_population_stdev"
            ],
            "control_cache_counter_hit_rate_pct_population_stdev": control[
                "cache_prefix_cache_counter_hit_rate_pct_population_stdev"
            ],
            "shared_cache_hit_rate_pct_population_stdev": shared[
                "cache_prefix_cache_hit_rate_pct_population_stdev"
            ],
            "control_cache_hit_rate_pct_population_stdev": control[
                "cache_prefix_cache_hit_rate_pct_population_stdev"
            ],
            "shared_cache_to_cold_throughput_ratio_mean": shared_throughput,
            "control_cache_to_cold_throughput_ratio_mean": control_throughput,
            "shared_minus_control_cache_to_cold_throughput_ratio_mean": _delta(
                shared_throughput,
                control_throughput,
            ),
            "shared_cache_to_cold_p95_first_event_ratio_mean": shared_first_event,
            "control_cache_to_cold_p95_first_event_ratio_mean": control_first_event,
            "shared_minus_control_cache_to_cold_p95_first_event_ratio_mean": _delta(
                shared_first_event,
                control_first_event,
            ),
            "shared_cache_to_cold_p95_latency_ratio_mean": shared_latency,
            "control_cache_to_cold_p95_latency_ratio_mean": control_latency,
            "shared_minus_control_cache_to_cold_p95_latency_ratio_mean": _delta(
                shared_latency,
                control_latency,
            ),
            "shared_cache_to_cold_p95_stream_tpot_ratio_mean": shared_tpot,
            "control_cache_to_cold_p95_stream_tpot_ratio_mean": control_tpot,
            "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_mean": _delta(
                shared_tpot,
                control_tpot,
            ),
        }
        for output_prefix, source_field in (
            (
                "shared_minus_control_cache_hit_rate_pct",
                "cache_prefix_cache_hit_rate_pct",
            ),
            (
                "shared_minus_control_cache_counter_hit_rate_pct",
                "cache_prefix_cache_counter_hit_rate_pct",
            ),
            (
                "shared_minus_control_cache_to_cold_throughput_ratio",
                "cache_to_cold_output_tokens_per_second_ratio",
            ),
            (
                "shared_minus_control_cache_to_cold_p95_first_event_ratio",
                "cache_to_cold_p95_first_event_ms_ratio",
            ),
            (
                "shared_minus_control_cache_to_cold_p95_latency_ratio",
                "cache_to_cold_p95_latency_ms_ratio",
            ),
            (
                "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio",
                "cache_to_cold_p95_stream_tpot_ms_ratio",
            ),
        ):
            values = _paired_profile_delta_values(
                run_rows=run_rows,
                shared_profile=shared_profile,
                control_profile=control_profile,
                request_count=request_count,
                max_new_tokens=max_new_tokens,
                field=source_field,
            )
            paired_delta_values_by_metric.setdefault(output_prefix, []).extend(values)
            _add_distribution_fields(profile_control_row, output_prefix, values)
        profile_control_rows.append(profile_control_row)

    cache_stdevs = [
        row["cache_prefix_cache_hit_rate_pct_population_stdev"]
        for row in scenario_rows
        if row["cache_prefix_cache_hit_rate_pct_population_stdev"] is not None
    ]
    counter_stdevs = [
        row["cache_prefix_cache_counter_hit_rate_pct_population_stdev"]
        for row in scenario_rows
        if row["cache_prefix_cache_counter_hit_rate_pct_population_stdev"] is not None
    ]
    summary = {
        "mean_shared_minus_control_cache_hit_rate_pct": _mean_present(
            row["shared_minus_control_cache_hit_rate_pct"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_throughput_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_throughput_ratio_mean"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_p95_first_event_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_p95_first_event_ratio_mean"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_p95_latency_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_p95_latency_ratio_mean"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_p95_stream_tpot_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_mean"]
            for row in profile_control_rows
        ),
        "max_cache_hit_rate_population_stdev": max(cache_stdevs) if cache_stdevs else None,
    }
    counter_deltas = [
        row["shared_minus_control_cache_counter_hit_rate_pct"]
        for row in profile_control_rows
        if row["shared_minus_control_cache_counter_hit_rate_pct"] is not None
    ]
    if counter_deltas:
        summary["mean_shared_minus_control_cache_counter_hit_rate_pct"] = (
            sum(counter_deltas) / len(counter_deltas)
        )
        summary["max_cache_counter_hit_rate_population_stdev"] = (
            max(counter_stdevs) if counter_stdevs else None
        )
    for summary_prefix, metric_prefix in (
        (
            "mean_shared_minus_control_cache_hit_rate_pct",
            "shared_minus_control_cache_hit_rate_pct",
        ),
        (
            "mean_shared_minus_control_cache_counter_hit_rate_pct",
            "shared_minus_control_cache_counter_hit_rate_pct",
        ),
        (
            "mean_shared_minus_control_cache_to_cold_throughput_ratio",
            "shared_minus_control_cache_to_cold_throughput_ratio",
        ),
        (
            "mean_shared_minus_control_cache_to_cold_p95_first_event_ratio",
            "shared_minus_control_cache_to_cold_p95_first_event_ratio",
        ),
        (
            "mean_shared_minus_control_cache_to_cold_p95_latency_ratio",
            "shared_minus_control_cache_to_cold_p95_latency_ratio",
        ),
        (
            "mean_shared_minus_control_cache_to_cold_p95_stream_tpot_ratio",
            "shared_minus_control_cache_to_cold_p95_stream_tpot_ratio",
        ),
    ):
        values = paired_delta_values_by_metric.get(metric_prefix, [])
        if values:
            _add_distribution_fields(summary, summary_prefix, values)
    source_mode = source_payload.get("mode")
    source_warmup_runs = source_payload.get("warmup_runs")
    source_warmup_prompt_profile = source_payload.get("warmup_prompt_profile")
    source_repeats = source_payload.get("repeats")
    source_phase_order = source_payload.get("phase_order")
    markdown = _format_vllm_prefix_cache_isolated_stability_markdown(
        source_dir=isolated_metrics_dir,
        source_mode=source_mode,
        source_warmup_runs=source_warmup_runs,
        source_warmup_prompt_profile=source_warmup_prompt_profile,
        source_repeats=source_repeats,
        source_phase_order=source_phase_order,
        scenario_rows=scenario_rows,
        profile_control_rows=profile_control_rows,
        summary=summary,
    )
    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-isolated-stability-summary",
        "source_dir": str(isolated_metrics_dir),
        "source_json": str(source_json),
        "source_runs_csv": str(source_runs_csv),
        "source_scenario_count": source_payload["scenario_count"],
        "source_paired_run_count": source_payload["paired_run_count"],
        "source_remote_call_count": source_payload["remote_call_count"],
        "source_checkpoint_complete": source_payload.get("checkpoint_complete"),
        "source_mode": source_mode,
        "source_warmup_runs": source_warmup_runs,
        "source_warmup_prompt_profile": source_warmup_prompt_profile,
        "source_repeats": source_repeats,
        "source_phase_order": source_phase_order,
        "scenario_count": len(scenario_rows),
        "profile_control_row_count": len(profile_control_rows),
        "shared_profile": shared_profile,
        "control_profile": control_profile,
        "summary": summary,
        "scenario_rows": scenario_rows,
        "profile_control_rows": profile_control_rows,
        "markdown": markdown,
    }


def _format_vllm_prefix_cache_isolated_stability_markdown(
    source_dir: Path,
    source_mode: Any,
    source_warmup_runs: Any,
    source_warmup_prompt_profile: Any,
    source_repeats: Any,
    source_phase_order: Any,
    scenario_rows: list[dict[str, Any]],
    profile_control_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}{suffix}"

    def fmt_ci(row: dict[str, Any], prefix: str, suffix: str = "") -> str:
        low = row.get(f"{prefix}_bootstrap_mean_p05")
        high = row.get(f"{prefix}_bootstrap_mean_p95")
        if low is None or high is None:
            return "n/a"
        return f"{fmt(low, suffix)} to {fmt(high, suffix)}"

    has_counter = any(
        row.get("cache_prefix_cache_counter_hit_rate_pct_mean") is not None
        for row in scenario_rows
    )
    lines = [
        "# Prefix-Cache Isolated Stability Summary",
        "",
        f"Source: `{source_dir}`",
        f"Source mode: `{source_mode}`",
        f"Warmup runs: `{source_warmup_runs}`",
        f"Warmup prompt profile: `{source_warmup_prompt_profile}`",
        f"Repeats: `{source_repeats}`",
        f"Phase order: `{source_phase_order}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (
            "| Mean shared-minus-control cache hit rate | "
            f"{fmt(summary['mean_shared_minus_control_cache_hit_rate_pct'], ' pp')} |"
        ),
        (
            "| Max cache-hit population stdev | "
            f"{fmt(summary['max_cache_hit_rate_population_stdev'])} |"
        ),
    ]
    if has_counter:
        lines.extend(
            [
                (
                    "| Mean shared-minus-control direct counter hit rate | "
                    f"{fmt(summary.get('mean_shared_minus_control_cache_counter_hit_rate_pct'), ' pp')} |"
                ),
                (
                    "| Direct counter hit-rate 90% bootstrap interval | "
                    f"{fmt_ci(summary, 'mean_shared_minus_control_cache_counter_hit_rate_pct', ' pp')} |"
                ),
                (
                    "| Throughput-ratio delta 90% bootstrap interval | "
                    f"{fmt_ci(summary, 'mean_shared_minus_control_cache_to_cold_throughput_ratio')} |"
                ),
                (
                    "| p95 first-event/TTFT ratio delta 90% bootstrap interval | "
                    f"{fmt_ci(summary, 'mean_shared_minus_control_cache_to_cold_p95_first_event_ratio')} |"
                ),
                (
                    "| p95 latency-ratio delta 90% bootstrap interval | "
                    f"{fmt_ci(summary, 'mean_shared_minus_control_cache_to_cold_p95_latency_ratio')} |"
                ),
                (
                    "| p95 stream TPOT ratio delta 90% bootstrap interval | "
                    f"{fmt_ci(summary, 'mean_shared_minus_control_cache_to_cold_p95_stream_tpot_ratio')} |"
                ),
                (
                    "| Max direct counter hit-rate population stdev | "
                    f"{fmt(summary.get('max_cache_counter_hit_rate_population_stdev'))} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Scenario Stability",
            "",
        ]
    )
    if has_counter:
        lines.extend(
            [
                (
                    "| Profile | Requests | Runs | Logged Hit Mean | "
                    "Direct Counter Hit Mean | Direct Queries | Direct Hits |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Profile | Requests | Runs | Cache Hit Mean | Min | Max | Stdev |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in scenario_rows:
        if has_counter:
            lines.append(
                "| "
                f"`{row['prompt_profile']}` | "
                f"{row['request_count']} | "
                f"{row['runs']} | "
                f"{fmt(row['cache_prefix_cache_hit_rate_pct_mean'], '%')} | "
                f"{fmt(row['cache_prefix_cache_counter_hit_rate_pct_mean'], '%')} | "
                f"{fmt(row['cache_prefix_cache_counter_queries_mean'])} | "
                f"{fmt(row['cache_prefix_cache_counter_hits_mean'])} |"
            )
        else:
            lines.append(
                "| "
                f"`{row['prompt_profile']}` | "
                f"{row['request_count']} | "
                f"{row['runs']} | "
                f"{fmt(row['cache_prefix_cache_hit_rate_pct_mean'], '%')} | "
                f"{fmt(row['cache_prefix_cache_hit_rate_pct_min'], '%')} | "
                f"{fmt(row['cache_prefix_cache_hit_rate_pct_max'], '%')} | "
                f"{fmt(row['cache_prefix_cache_hit_rate_pct_population_stdev'])} |"
            )
    lines.extend(["", "## Shared Vs Control", ""])
    if has_counter:
        lines.extend(
            [
                (
                    "| Requests | Paired Obs | Logged Shared Hit Mean | "
                    "Logged Control Hit Mean | Direct Counter Delta | "
                    "Direct Counter 90% CI | Throughput Delta | "
                    "Throughput 90% CI | p95 First-Event Delta | "
                    "p95 First-Event 90% CI | p95 Latency Delta | "
                    "p95 Latency 90% CI | p95 Stream TPOT Delta | "
                    "p95 Stream TPOT 90% CI |"
                ),
                (
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                    "---: | ---: | ---: | ---: | ---: | ---: | ---: |"
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "| Requests | Shared Hit Mean | Control Hit Mean | "
                    "Delta | Throughput Delta | p95 Latency Delta |"
                ),
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in profile_control_rows:
        logged_delta = row["shared_minus_control_cache_hit_rate_pct"]
        counter_delta = row.get("shared_minus_control_cache_counter_hit_rate_pct")
        delta = counter_delta if has_counter and counter_delta is not None else logged_delta
        if has_counter:
            lines.append(
                "| "
                f"{row['request_count']} | "
                f"{row.get('shared_minus_control_cache_counter_hit_rate_pct_paired_observation_count')} | "
                f"{fmt(row['shared_cache_hit_rate_pct_mean'], '%')} | "
                f"{fmt(row['control_cache_hit_rate_pct_mean'], '%')} | "
                f"{fmt(delta, ' pp')} | "
                f"{fmt_ci(row, 'shared_minus_control_cache_counter_hit_rate_pct', ' pp')} | "
                f"{fmt(row['shared_minus_control_cache_to_cold_throughput_ratio_mean'])} | "
                f"{fmt_ci(row, 'shared_minus_control_cache_to_cold_throughput_ratio')} | "
                f"{fmt(row['shared_minus_control_cache_to_cold_p95_first_event_ratio_mean'])} | "
                f"{fmt_ci(row, 'shared_minus_control_cache_to_cold_p95_first_event_ratio')} | "
                f"{fmt(row['shared_minus_control_cache_to_cold_p95_latency_ratio_mean'])} | "
                f"{fmt_ci(row, 'shared_minus_control_cache_to_cold_p95_latency_ratio')} | "
                f"{fmt(row['shared_minus_control_cache_to_cold_p95_stream_tpot_ratio_mean'])} | "
                f"{fmt_ci(row, 'shared_minus_control_cache_to_cold_p95_stream_tpot_ratio')} |"
            )
        else:
            lines.append(
                "| "
                f"{row['request_count']} | "
                f"{fmt(row['shared_cache_hit_rate_pct_mean'], '%')} | "
                f"{fmt(row['control_cache_hit_rate_pct_mean'], '%')} | "
                f"{fmt(delta, ' pp')} | "
                f"{fmt(row['shared_minus_control_cache_to_cold_throughput_ratio_mean'])} | "
                f"{fmt(row['shared_minus_control_cache_to_cold_p95_latency_ratio_mean'])} |"
            )
    if has_counter:
        lines.extend(
            [
                "",
                "Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _compare_vllm_prefix_cache_phase_orders(
    cold_first_dir: Path,
    cache_first_dir: Path,
) -> dict[str, Any]:
    cold_first_json = cold_first_dir / "paired-prefix-cache.json"
    cache_first_json = cache_first_dir / "paired-prefix-cache.json"
    cold_first_csv = cold_first_dir / "paired-prefix-cache-summary.csv"
    cache_first_csv = cache_first_dir / "paired-prefix-cache-summary.csv"

    cold_first_payload = json.loads(cold_first_json.read_text(encoding="utf-8"))
    cache_first_payload = json.loads(cache_first_json.read_text(encoding="utf-8"))
    cold_first_rows = _read_csv_by_key(cold_first_csv, "scenario_id")
    cache_first_rows = _read_csv_by_key(cache_first_csv, "scenario_id")
    scenario_ids = sorted(set(cold_first_rows) & set(cache_first_rows))
    if not scenario_ids:
        raise ValueError("No matching scenario_id values found for prefix-cache phase comparison")

    ratio_fields = (
        "cache_to_cold_output_tokens_per_second_ratio_median",
        "cache_to_cold_p95_first_event_ms_ratio_median",
        "cache_to_cold_p95_latency_ms_ratio_median",
        "cache_to_cold_p95_stream_tpot_ms_ratio_median",
        "cache_to_cold_batch_wall_ms_ratio_median",
        "cold_prefix_cache_hit_rate_pct_median",
        "cache_prefix_cache_hit_rate_pct_median",
        "cache_to_cold_prefix_cache_hit_rate_pct_delta_median",
        "cold_gpu_kv_cache_usage_pct_median",
        "cache_gpu_kv_cache_usage_pct_median",
        "cache_to_cold_gpu_kv_cache_usage_pct_delta_median",
    )
    rows = []
    for scenario_id in scenario_ids:
        cold_first = cold_first_rows[scenario_id]
        cache_first = cache_first_rows[scenario_id]
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "prompt_profile": cold_first["prompt_profile"],
            "request_count": int(cold_first["request_count"]),
            "max_new_tokens": int(cold_first["max_new_tokens"]),
            "pairs": int(cold_first["pairs"]),
            "prompt_tokens_mean": _float_field(cold_first, "prompt_tokens_mean"),
        }
        for field in ratio_fields:
            cold_first_value = _float_field(cold_first, field)
            cache_first_value = _float_field(cache_first, field)
            short_field = field.removesuffix("_median")
            if short_field.startswith("cache_to_cold_"):
                short_field = short_field.removeprefix("cache_to_cold_")
            row[f"cold_first_{short_field}"] = cold_first_value
            row[f"cache_first_{short_field}"] = cache_first_value
            row[f"cache_first_minus_cold_first_{short_field}"] = _delta(
                cache_first_value,
                cold_first_value,
            )
        rows.append(row)

    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-phase-order-compare",
        "cold_first_dir": str(cold_first_dir),
        "cache_first_dir": str(cache_first_dir),
        "cold_first_json": str(cold_first_json),
        "cache_first_json": str(cache_first_json),
        "cold_first_csv": str(cold_first_csv),
        "cache_first_csv": str(cache_first_csv),
        "scenario_count": len(rows),
        "cold_first": {
            "phase_order": cold_first_payload.get("phase_order", "cold_first"),
            "paired_run_count": cold_first_payload["paired_run_count"],
            "cold_engine_load_ms": cold_first_payload["cold_engine_load_ms"],
            "cache_engine_load_ms": cold_first_payload["cache_engine_load_ms"],
            "mean_cache_to_cold_throughput_ratio": cold_first_payload[
                "mean_cache_to_cold_throughput_ratio"
            ],
            "mean_cache_to_cold_first_event_ratio": cold_first_payload[
                "mean_cache_to_cold_first_event_ratio"
            ],
            "mean_cache_to_cold_latency_ratio": cold_first_payload[
                "mean_cache_to_cold_latency_ratio"
            ],
            "mean_cache_to_cold_tpot_ratio": cold_first_payload[
                "mean_cache_to_cold_tpot_ratio"
            ],
        },
        "cache_first": {
            "phase_order": cache_first_payload.get("phase_order", "cache_first"),
            "paired_run_count": cache_first_payload["paired_run_count"],
            "cold_engine_load_ms": cache_first_payload["cold_engine_load_ms"],
            "cache_engine_load_ms": cache_first_payload["cache_engine_load_ms"],
            "mean_cache_to_cold_throughput_ratio": cache_first_payload[
                "mean_cache_to_cold_throughput_ratio"
            ],
            "mean_cache_to_cold_first_event_ratio": cache_first_payload[
                "mean_cache_to_cold_first_event_ratio"
            ],
            "mean_cache_to_cold_latency_ratio": cache_first_payload[
                "mean_cache_to_cold_latency_ratio"
            ],
            "mean_cache_to_cold_tpot_ratio": cache_first_payload[
                "mean_cache_to_cold_tpot_ratio"
            ],
        },
        "rows": rows,
    }


def _compare_vllm_prefix_cache_profile_controls(
    phase_order_compare_dir: Path,
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> dict[str, Any]:
    phase_json = phase_order_compare_dir / "prefix-cache-phase-order-compare.json"
    phase_csv = phase_order_compare_dir / "prefix-cache-phase-order-compare.csv"
    phase_payload = json.loads(phase_json.read_text(encoding="utf-8"))
    with phase_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_profile_and_shape = {
        (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        ): row
        for row in rows
    }
    shared_keys = {
        (request_count, max_new_tokens)
        for profile, request_count, max_new_tokens in by_profile_and_shape
        if profile == shared_profile
    }
    control_keys = {
        (request_count, max_new_tokens)
        for profile, request_count, max_new_tokens in by_profile_and_shape
        if profile == control_profile
    }
    comparison_keys = sorted(shared_keys & control_keys)
    if not comparison_keys:
        raise ValueError(
            "No matching shared/control profile rows found in prefix-cache phase comparison"
        )

    metric_fields = (
        "output_tokens_per_second_ratio",
        "p95_first_event_ms_ratio",
        "p95_latency_ms_ratio",
        "p95_stream_tpot_ms_ratio",
        "batch_wall_ms_ratio",
        "cold_prefix_cache_hit_rate_pct",
        "cache_prefix_cache_hit_rate_pct",
        "prefix_cache_hit_rate_pct_delta",
        "cold_gpu_kv_cache_usage_pct",
        "cache_gpu_kv_cache_usage_pct",
        "gpu_kv_cache_usage_pct_delta",
    )
    phase_labels = ("cold_first", "cache_first")
    comparison_rows = []
    for request_count, max_new_tokens in comparison_keys:
        shared = by_profile_and_shape[(shared_profile, request_count, max_new_tokens)]
        control = by_profile_and_shape[(control_profile, request_count, max_new_tokens)]
        shared_prompt_tokens = _float_field(shared, "prompt_tokens_mean")
        control_prompt_tokens = _float_field(control, "prompt_tokens_mean")
        row: dict[str, Any] = {
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "shared_profile": shared_profile,
            "control_profile": control_profile,
            "shared_scenario_id": shared["scenario_id"],
            "control_scenario_id": control["scenario_id"],
            "shared_prompt_tokens_mean": shared_prompt_tokens,
            "control_prompt_tokens_mean": control_prompt_tokens,
            "shared_to_control_prompt_tokens_mean_ratio": _ratio(
                shared_prompt_tokens,
                control_prompt_tokens,
            ),
        }
        for phase_label in phase_labels:
            for metric_field in metric_fields:
                source_field = f"{phase_label}_{metric_field}"
                shared_value = _float_field(shared, source_field)
                control_value = _float_field(control, source_field)
                row[f"shared_{source_field}"] = shared_value
                row[f"control_{source_field}"] = control_value
                row[f"shared_minus_control_{source_field}"] = _delta(
                    shared_value,
                    control_value,
                )
        comparison_rows.append(row)

    summary_fields = [
        field
        for field in comparison_rows[0]
        if field.startswith("shared_minus_control_")
    ]
    summary = {
        f"mean_{field}": _mean_present(row[field] for row in comparison_rows)
        for field in summary_fields
    }
    summary["mean_shared_to_control_prompt_tokens_mean_ratio"] = _mean_present(
        row["shared_to_control_prompt_tokens_mean_ratio"]
        for row in comparison_rows
    )

    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-profile-control",
        "phase_order_compare_dir": str(phase_order_compare_dir),
        "phase_order_compare_json": str(phase_json),
        "phase_order_compare_csv": str(phase_csv),
        "shared_profile": shared_profile,
        "control_profile": control_profile,
        "row_count": len(comparison_rows),
        "phase_order_scenario_count": phase_payload["scenario_count"],
        "summary": summary,
        "rows": comparison_rows,
    }


def _aggregate_vllm_prefix_cache_profile_control_trials(
    profile_control_dirs: list[Path],
) -> dict[str, Any]:
    if not profile_control_dirs:
        raise ValueError("profile_control_dirs must not be empty")

    metric_sources = (
        (
            "cold_first",
            "throughput_ratio",
            "mean_shared_minus_control_cold_first_output_tokens_per_second_ratio",
            "higher_is_better",
        ),
        (
            "cold_first",
            "first_event_ratio",
            "mean_shared_minus_control_cold_first_p95_first_event_ms_ratio",
            "lower_is_better",
        ),
        (
            "cold_first",
            "latency_ratio",
            "mean_shared_minus_control_cold_first_p95_latency_ms_ratio",
            "lower_is_better",
        ),
        (
            "cold_first",
            "tpot_ratio",
            "mean_shared_minus_control_cold_first_p95_stream_tpot_ms_ratio",
            "lower_is_better",
        ),
        (
            "cache_first",
            "throughput_ratio",
            "mean_shared_minus_control_cache_first_output_tokens_per_second_ratio",
            "higher_is_better",
        ),
        (
            "cache_first",
            "first_event_ratio",
            "mean_shared_minus_control_cache_first_p95_first_event_ms_ratio",
            "lower_is_better",
        ),
        (
            "cache_first",
            "latency_ratio",
            "mean_shared_minus_control_cache_first_p95_latency_ms_ratio",
            "lower_is_better",
        ),
        (
            "cache_first",
            "tpot_ratio",
            "mean_shared_minus_control_cache_first_p95_stream_tpot_ms_ratio",
            "lower_is_better",
        ),
    )
    trials = []
    for index, profile_control_dir in enumerate(profile_control_dirs, start=1):
        path = profile_control_dir / "prefix-cache-profile-control.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        trial: dict[str, Any] = {
            "trial_index": index,
            "trial_label": profile_control_dir.name,
            "profile_control_json": str(path),
            "row_count": payload["row_count"],
            "shared_profile": payload["shared_profile"],
            "control_profile": payload["control_profile"],
            "mean_shared_to_control_prompt_tokens_mean_ratio": summary[
                "mean_shared_to_control_prompt_tokens_mean_ratio"
            ],
        }
        for phase_order, metric, source_field, _direction in metric_sources:
            trial[f"{phase_order}_{metric}"] = summary[source_field]
        trials.append(trial)

    summary_rows = []
    for phase_order, metric, _source_field, direction in metric_sources:
        field = f"{phase_order}_{metric}"
        stats = _metric_distribution(trials, field)
        values = [
            trial[field]
            for trial in trials
            if trial.get(field) is not None
        ]
        bootstrap_interval = _bootstrap_mean_interval(values)
        summary_rows.append(
            {
                "phase_order": phase_order,
                "metric": metric,
                "direction": direction,
                "trial_count": len(trials),
                "shared_minus_control_mean": stats[f"{field}_mean"],
                "shared_minus_control_min": stats[f"{field}_min"],
                "shared_minus_control_max": stats[f"{field}_max"],
                "shared_minus_control_cv": stats[f"{field}_cv"],
                "shared_minus_control_bootstrap_mean_p05": bootstrap_interval[
                    "mean_p05"
                ],
                "shared_minus_control_bootstrap_mean_p50": bootstrap_interval[
                    "mean_p50"
                ],
                "shared_minus_control_bootstrap_mean_p95": bootstrap_interval[
                    "mean_p95"
                ],
            }
        )

    prompt_ratio_stats = _metric_distribution(
        trials,
        "mean_shared_to_control_prompt_tokens_mean_ratio",
    )
    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-profile-multitrial",
        "trial_count": len(trials),
        "profile_control_dirs": [str(path) for path in profile_control_dirs],
        "shared_profile": trials[0]["shared_profile"],
        "control_profile": trials[0]["control_profile"],
        "prompt_token_ratio_mean": prompt_ratio_stats[
            "mean_shared_to_control_prompt_tokens_mean_ratio_mean"
        ],
        "prompt_token_ratio_min": prompt_ratio_stats[
            "mean_shared_to_control_prompt_tokens_mean_ratio_min"
        ],
        "prompt_token_ratio_max": prompt_ratio_stats[
            "mean_shared_to_control_prompt_tokens_mean_ratio_max"
        ],
        "trials": trials,
        "summary": summary_rows,
    }


def _compare_vllm_sweep_csvs(
    cold_csv: Path,
    prefix_csv: Path,
    cold_label: str,
    prefix_label: str,
) -> dict[str, Any]:
    cold_rows = _read_csv_by_key(cold_csv, "scenario_id")
    prefix_rows = _read_csv_by_key(prefix_csv, "scenario_id")
    scenario_ids = sorted(set(cold_rows) & set(prefix_rows))
    if not scenario_ids:
        raise ValueError("No matching scenario_id values found for comparison")

    rows = []
    for scenario_id in scenario_ids:
        cold = cold_rows[scenario_id]
        prefix = prefix_rows[scenario_id]
        throughput_cold = _float_field(cold, "aggregate_output_tokens_per_second_median")
        throughput_prefix = _float_field(prefix, "aggregate_output_tokens_per_second_median")
        first_chunk_cold = _float_field(cold, "p95_first_chunk_ms_median")
        first_chunk_prefix = _float_field(prefix, "p95_first_chunk_ms_median")
        latency_cold = _float_field(cold, "p95_latency_ms_median")
        latency_prefix = _float_field(prefix, "p95_latency_ms_median")
        tpot_cold = _float_field(cold, "p95_stream_tpot_ms_median")
        tpot_prefix = _float_field(prefix, "p95_stream_tpot_ms_median")
        rows.append(
            {
                "scenario_id": scenario_id,
                "prompt_profile": cold["prompt_profile"],
                "request_count": int(cold["request_count"]),
                "max_new_tokens": int(cold["max_new_tokens"]),
                "prompt_tokens_mean": _float_field(cold, "prompt_tokens_mean"),
                "estimated_peak_sequence_tokens_median": _float_field(
                    cold,
                    "estimated_peak_sequence_tokens_median",
                ),
                "cold_output_tokens_per_second_median": throughput_cold,
                "prefix_output_tokens_per_second_median": throughput_prefix,
                "output_tokens_per_second_delta": _delta(throughput_prefix, throughput_cold),
                "output_tokens_per_second_ratio": _ratio(throughput_prefix, throughput_cold),
                "cold_p95_first_chunk_ms_median": first_chunk_cold,
                "prefix_p95_first_chunk_ms_median": first_chunk_prefix,
                "p95_first_chunk_ms_delta": _delta(first_chunk_prefix, first_chunk_cold),
                "p95_first_chunk_ms_ratio": _ratio(first_chunk_prefix, first_chunk_cold),
                "cold_p95_latency_ms_median": latency_cold,
                "prefix_p95_latency_ms_median": latency_prefix,
                "p95_latency_ms_delta": _delta(latency_prefix, latency_cold),
                "p95_latency_ms_ratio": _ratio(latency_prefix, latency_cold),
                "cold_p95_stream_tpot_ms_median": tpot_cold,
                "prefix_p95_stream_tpot_ms_median": tpot_prefix,
                "p95_stream_tpot_ms_delta": _delta(tpot_prefix, tpot_cold),
                "p95_stream_tpot_ms_ratio": _ratio(tpot_prefix, tpot_cold),
                "cold_throughput_cv": _float_field(
                    cold,
                    "aggregate_output_tokens_per_second_cv",
                ),
                "prefix_throughput_cv": _float_field(
                    prefix,
                    "aggregate_output_tokens_per_second_cv",
                ),
                "cold_latency_cv": _float_field(cold, "p95_latency_ms_cv"),
                "prefix_latency_cv": _float_field(prefix, "p95_latency_ms_cv"),
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-compare",
        "cold_label": cold_label,
        "prefix_label": prefix_label,
        "cold_csv": str(cold_csv),
        "prefix_csv": str(prefix_csv),
        "scenario_count": len(rows),
        "rows": rows,
    }


def _read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row[key]: row for row in reader}


def _float_field(row: dict[str, str], field: str) -> float | None:
    value = row.get(field)
    if value in {None, ""}:
        return None
    return float(value)


def _prefix_cache_counter_hit_rate_pct(
    hits: int | float | None,
    queries: int | float | None,
) -> float | None:
    if hits is None or queries is None:
        return None
    hits_value = float(hits)
    queries_value = float(queries)
    if queries_value == 0:
        return 0.0 if hits_value == 0 else None
    return hits_value / queries_value * 100.0


def _prefix_cache_counter_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, int | float | None]:
    if not before or not after:
        return {
            "requests": None,
            "queries": None,
            "hits": None,
            "hit_rate_pct": None,
        }
    requests = _counter_delta_value(
        before.get("total_requests"),
        after.get("total_requests"),
    )
    queries = _counter_delta_value(
        before.get("total_queries"),
        after.get("total_queries"),
    )
    hits = _counter_delta_value(before.get("total_hits"), after.get("total_hits"))
    return {
        "requests": requests,
        "queries": queries,
        "hits": hits,
        "hit_rate_pct": _prefix_cache_counter_hit_rate_pct(hits, queries),
    }


def _counter_delta_value(
    before_value: int | float | None,
    after_value: int | float | None,
) -> int | None:
    if before_value is None or after_value is None:
        return None
    delta = int(after_value) - int(before_value)
    return delta if delta >= 0 else None


def _delta(new_value: float | None, baseline_value: float | None) -> float | None:
    if new_value is None or baseline_value is None:
        return None
    return new_value - baseline_value


def _ratio(new_value: float | None, baseline_value: float | None) -> float | None:
    if new_value is None or baseline_value in {None, 0.0}:
        return None
    return new_value / baseline_value


def _mean_present(values: Any) -> float:
    present = [float(value) for value in values if value is not None]
    if not present:
        raise ValueError("Cannot compute mean for empty values")
    return sum(present) / len(present)


def _mean_present_or_none(values: Any) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _format_optional_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "None"
    return f"{float(value):.{digits}f}"


def _tail_lines(lines: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    return lines[-count:]


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
        writer = csv.DictWriter(
            handle,
            fieldnames=[*fieldnames, *extra_fields],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def _build_vllm_prefix_cache_prompt_audit_payload(
    *,
    tokenizer: Any,
    hf_model: str,
    request_count_values: list[int],
    prompt_profile_values: list[str],
    output_token_values: list[int],
    repeats: int,
    scenario_seed: int,
    kv_cache_block_size: int,
) -> dict[str, Any]:
    scenario_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []

    for repeat_index in range(repeats):
        variant_index = scenario_seed + repeat_index
        for prompt_profile in prompt_profile_values:
            for max_new_tokens in output_token_values:
                for request_count in request_count_values:
                    prompts = _select_sweep_prompts(
                        request_count,
                        prompt_profile,
                        variant_index=variant_index,
                    )
                    internal_prompt_rows = []
                    for prompt_index, prompt in enumerate(prompts):
                        formatted_prompt, prompt_format = _format_prompt_for_generation(
                            tokenizer,
                            prompt,
                        )
                        token_ids = list(tokenizer.encode(formatted_prompt))
                        internal_prompt_rows.append(
                            {
                                "prompt_index": prompt_index,
                                "prompt": prompt,
                                "formatted_prompt": formatted_prompt,
                                "prompt_format": prompt_format,
                                "token_ids": token_ids,
                                "prompt_tokens": len(token_ids),
                            }
                        )

                    token_id_rows = [
                        row["token_ids"] for row in internal_prompt_rows
                    ]
                    common_prefix_tokens = _common_prefix_token_count(token_id_rows)
                    common_prefix_full_blocks = _token_block_count(
                        common_prefix_tokens,
                        kv_cache_block_size,
                    )
                    common_prefix_block_tokens = (
                        common_prefix_full_blocks * kv_cache_block_size
                    )
                    prompt_token_counts = [
                        row["prompt_tokens"] for row in internal_prompt_rows
                    ]
                    suffix_after_common_tokens = [
                        max(count - common_prefix_tokens, 0)
                        for count in prompt_token_counts
                    ]
                    total_prompt_tokens = sum(prompt_token_counts)
                    estimated_reusable_block_tokens = (
                        common_prefix_block_tokens * max(request_count - 1, 0)
                    )
                    duplicate_reuse = _exact_duplicate_prompt_reuse_stats(
                        internal_prompt_rows,
                        kv_cache_block_size,
                    )
                    duplicate_group_sizes: dict[int, int] = {}
                    duplicate_ordinals: dict[int, int] = {}
                    duplicate_groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
                    for row in internal_prompt_rows:
                        duplicate_groups.setdefault(tuple(row["token_ids"]), []).append(
                            row
                        )
                    for rows in duplicate_groups.values():
                        group_size = len(rows)
                        for ordinal, row in enumerate(rows):
                            duplicate_group_sizes[int(row["prompt_index"])] = group_size
                            duplicate_ordinals[int(row["prompt_index"])] = ordinal
                    scenario_id = (
                        f"{prompt_profile}_out{max_new_tokens}_n{request_count}"
                        f"_rep{repeat_index:02d}"
                    )
                    common_prefix_text_preview = ""
                    if token_id_rows and hasattr(tokenizer, "decode"):
                        common_token_ids = token_id_rows[0][:common_prefix_tokens]
                        common_prefix_text_preview = str(
                            tokenizer.decode(common_token_ids)
                        )[:240]

                    scenario_row = {
                        "scenario_id": scenario_id,
                        "prompt_profile": prompt_profile,
                        "repeat_index": repeat_index,
                        "variant_index": variant_index,
                        "request_count": request_count,
                        "max_new_tokens": max_new_tokens,
                        "prompt_format": internal_prompt_rows[0]["prompt_format"],
                        "kv_cache_block_size": kv_cache_block_size,
                        "prompt_tokens_min": min(prompt_token_counts),
                        "prompt_tokens_max": max(prompt_token_counts),
                        "prompt_tokens_mean": _mean_present(prompt_token_counts),
                        "total_prompt_tokens": total_prompt_tokens,
                        "common_prefix_tokens": common_prefix_tokens,
                        "common_prefix_full_blocks": common_prefix_full_blocks,
                        "common_prefix_block_tokens": common_prefix_block_tokens,
                        "common_prefix_remainder_tokens": (
                            common_prefix_tokens % kv_cache_block_size
                        ),
                        "suffix_after_common_tokens_min": min(
                            suffix_after_common_tokens
                        ),
                        "suffix_after_common_tokens_max": max(
                            suffix_after_common_tokens
                        ),
                        "suffix_after_common_tokens_mean": _mean_present(
                            suffix_after_common_tokens
                        ),
                        "unique_prompt_count": duplicate_reuse[
                            "unique_prompt_count"
                        ],
                        "exact_duplicate_prompt_group_count": duplicate_reuse[
                            "exact_duplicate_prompt_group_count"
                        ],
                        "exact_duplicate_prompt_repeated_count": duplicate_reuse[
                            "exact_duplicate_prompt_repeated_count"
                        ],
                        "exact_duplicate_prompt_max_group_size": duplicate_reuse[
                            "exact_duplicate_prompt_max_group_size"
                        ],
                        "estimated_exact_duplicate_reusable_block_tokens": (
                            duplicate_reuse[
                                "estimated_exact_duplicate_reusable_block_tokens"
                            ]
                        ),
                        "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt": (
                            duplicate_reuse[
                                "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt"
                            ]
                        ),
                        "estimated_reusable_block_tokens": estimated_reusable_block_tokens,
                        "estimated_reusable_block_token_fraction_of_total_prompt": (
                            estimated_reusable_block_tokens / total_prompt_tokens
                            if total_prompt_tokens
                            else None
                        ),
                        "common_prefix_token_fraction_of_prompt_mean": (
                            common_prefix_tokens
                            / max(_mean_present(prompt_token_counts) or 1, 1e-9)
                        ),
                        "common_prefix_text_preview": common_prefix_text_preview,
                    }
                    scenario_rows.append(scenario_row)

                    for row in internal_prompt_rows:
                        formatted_prompt = row["formatted_prompt"]
                        token_ids = row["token_ids"]
                        prompt_rows.append(
                            {
                                "scenario_id": scenario_id,
                                "prompt_profile": prompt_profile,
                                "repeat_index": repeat_index,
                                "variant_index": variant_index,
                                "request_count": request_count,
                                "max_new_tokens": max_new_tokens,
                                "prompt_index": row["prompt_index"],
                                "prompt_format": row["prompt_format"],
                                "prompt_tokens": row["prompt_tokens"],
                                "suffix_after_common_tokens": max(
                                    row["prompt_tokens"] - common_prefix_tokens,
                                    0,
                                ),
                                "formatted_prompt_duplicate_group_size": (
                                    duplicate_group_sizes[int(row["prompt_index"])]
                                ),
                                "formatted_prompt_duplicate_ordinal": (
                                    duplicate_ordinals[int(row["prompt_index"])]
                                ),
                                "formatted_prompt_sha256_16": hashlib.sha256(
                                    formatted_prompt.encode("utf-8")
                                ).hexdigest()[:16],
                                "token_ids_head": json.dumps(token_ids[:16]),
                                "token_ids_tail": json.dumps(token_ids[-16:]),
                                "prompt_preview": row["prompt"][:240],
                                "formatted_prompt_preview": formatted_prompt[:240],
                            }
                        )

    shared_profile, control_profile = _infer_vllm_prefix_cache_prompt_audit_pair(
        prompt_profile_values
    )
    profile_control_rows = _vllm_prefix_cache_prompt_audit_profile_control_rows(
        scenario_rows,
        shared_profile=shared_profile,
        control_profile=control_profile,
    )
    summary = _vllm_prefix_cache_prompt_audit_summary(
        scenario_rows,
        profile_control_rows,
        shared_profile=shared_profile,
        control_profile=control_profile,
    )
    payload = {
        "schema_version": 1,
        "execution": "local",
        "mode": "vllm-prefix-cache-prompt-audit",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "shared_profile": shared_profile,
        "control_profile": control_profile,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "kv_cache_block_size": int(kv_cache_block_size),
        "scenario_count": len(scenario_rows),
        "prompt_count": len(prompt_rows),
        "profile_control_row_count": len(profile_control_rows),
        "summary": summary,
        "scenario_rows": scenario_rows,
        "prompt_rows": prompt_rows,
        "profile_control_rows": profile_control_rows,
        "note": (
            "This tokenizer audit estimates reusable full KV-cache blocks from "
            "exact leading token overlap. It does not execute vLLM; it explains "
            "the prompt shape behind the measured prefix-cache counter results."
        ),
    }
    payload["markdown"] = _format_vllm_prefix_cache_prompt_audit_markdown(payload)
    return payload


def _common_prefix_token_count(token_rows: list[list[int]]) -> int:
    if not token_rows:
        return 0
    shortest = min(len(row) for row in token_rows)
    count = 0
    for index in range(shortest):
        token = token_rows[0][index]
        if any(row[index] != token for row in token_rows[1:]):
            break
        count += 1
    return count


def _token_block_count(token_count: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return max(token_count, 0) // block_size


def _exact_duplicate_prompt_reuse_stats(
    prompt_rows: list[dict[str, Any]],
    block_size: int,
) -> dict[str, Any]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in prompt_rows:
        groups.setdefault(tuple(row["token_ids"]), []).append(row)

    reusable_block_tokens = 0
    duplicate_group_count = 0
    repeated_count = 0
    max_group_size = 0
    for token_ids, rows in groups.items():
        group_size = len(rows)
        max_group_size = max(max_group_size, group_size)
        if group_size <= 1:
            continue
        duplicate_group_count += 1
        repeated_count += group_size - 1
        reusable_block_tokens += (
            _token_block_count(len(token_ids), block_size)
            * block_size
            * (group_size - 1)
        )

    total_prompt_tokens = sum(int(row["prompt_tokens"]) for row in prompt_rows)
    return {
        "unique_prompt_count": len(groups),
        "exact_duplicate_prompt_group_count": duplicate_group_count,
        "exact_duplicate_prompt_repeated_count": repeated_count,
        "exact_duplicate_prompt_max_group_size": max_group_size,
        "estimated_exact_duplicate_reusable_block_tokens": reusable_block_tokens,
        "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt": (
            reusable_block_tokens / total_prompt_tokens
            if total_prompt_tokens
            else None
        ),
    }


def _infer_vllm_prefix_cache_prompt_audit_pair(
    prompt_profiles: list[str],
) -> tuple[str, str]:
    profile_set = set(prompt_profiles)
    candidate_pairs = [
        (
            "shared_prefix_mega_long_no_repeat_variant",
            "matched_unique_prefix_mega_long_no_repeat_variant",
        ),
        (
            "shared_prefix_ultra_long_no_repeat_variant",
            "matched_unique_prefix_ultra_long_no_repeat_variant",
        ),
        (
            "shared_prefix_extra_long_no_repeat_variant",
            "matched_unique_prefix_extra_long_no_repeat_variant",
        ),
        (
            "shared_prefix_long_no_repeat_variant",
            "matched_unique_prefix_no_repeat_variant",
        ),
        ("shared_prefix_long_variant", "matched_unique_prefix_variant"),
        ("shared_prefix_long", "matched_unique_prefix"),
        ("shared_prefix", "matched_unique_prefix"),
    ]
    for shared_profile, control_profile in candidate_pairs:
        if shared_profile in profile_set and control_profile in profile_set:
            return shared_profile, control_profile
    return "shared_prefix_long_variant", "matched_unique_prefix_variant"


def _vllm_prefix_cache_prompt_audit_profile_control_rows(
    scenario_rows: list[dict[str, Any]],
    shared_profile: str = "shared_prefix_long_variant",
    control_profile: str = "matched_unique_prefix_variant",
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    for row in scenario_rows:
        key = (
            int(row["repeat_index"]),
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        )
        by_key.setdefault(key, {})[str(row["prompt_profile"])] = row

    profile_control_rows = []
    for key in sorted(by_key):
        repeat_index, request_count, max_new_tokens = key
        rows = by_key[key]
        shared = rows.get(shared_profile)
        control = rows.get(control_profile)
        if shared is None or control is None:
            continue
        profile_control_rows.append(
            {
                "repeat_index": repeat_index,
                "request_count": request_count,
                "max_new_tokens": max_new_tokens,
                "shared_profile": shared_profile,
                "control_profile": control_profile,
                "shared_variant_index": shared["variant_index"],
                "control_variant_index": control["variant_index"],
                "shared_common_prefix_tokens": shared["common_prefix_tokens"],
                "control_common_prefix_tokens": control["common_prefix_tokens"],
                "shared_minus_control_common_prefix_tokens": _delta(
                    shared["common_prefix_tokens"],
                    control["common_prefix_tokens"],
                ),
                "shared_common_prefix_full_blocks": shared[
                    "common_prefix_full_blocks"
                ],
                "control_common_prefix_full_blocks": control[
                    "common_prefix_full_blocks"
                ],
                "shared_minus_control_common_prefix_full_blocks": _delta(
                    shared["common_prefix_full_blocks"],
                    control["common_prefix_full_blocks"],
                ),
                "shared_estimated_reusable_block_tokens": shared[
                    "estimated_reusable_block_tokens"
                ],
                "control_estimated_reusable_block_tokens": control[
                    "estimated_reusable_block_tokens"
                ],
                "shared_minus_control_estimated_reusable_block_tokens": _delta(
                    shared["estimated_reusable_block_tokens"],
                    control["estimated_reusable_block_tokens"],
                ),
                "shared_reusable_block_token_fraction": shared[
                    "estimated_reusable_block_token_fraction_of_total_prompt"
                ],
                "control_reusable_block_token_fraction": control[
                    "estimated_reusable_block_token_fraction_of_total_prompt"
                ],
                "shared_minus_control_reusable_block_token_fraction": _delta(
                    shared[
                        "estimated_reusable_block_token_fraction_of_total_prompt"
                    ],
                    control[
                        "estimated_reusable_block_token_fraction_of_total_prompt"
                    ],
                ),
                "shared_unique_prompt_count": shared["unique_prompt_count"],
                "control_unique_prompt_count": control["unique_prompt_count"],
                "shared_exact_duplicate_prompt_repeated_count": shared[
                    "exact_duplicate_prompt_repeated_count"
                ],
                "control_exact_duplicate_prompt_repeated_count": control[
                    "exact_duplicate_prompt_repeated_count"
                ],
                "shared_estimated_exact_duplicate_reusable_block_tokens": shared[
                    "estimated_exact_duplicate_reusable_block_tokens"
                ],
                "control_estimated_exact_duplicate_reusable_block_tokens": control[
                    "estimated_exact_duplicate_reusable_block_tokens"
                ],
                "shared_minus_control_estimated_exact_duplicate_reusable_block_tokens": _delta(
                    shared["estimated_exact_duplicate_reusable_block_tokens"],
                    control["estimated_exact_duplicate_reusable_block_tokens"],
                ),
                "shared_exact_duplicate_reusable_block_token_fraction": shared[
                    "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt"
                ],
                "control_exact_duplicate_reusable_block_token_fraction": control[
                    "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt"
                ],
                "shared_minus_control_exact_duplicate_reusable_block_token_fraction": _delta(
                    shared[
                        "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt"
                    ],
                    control[
                        "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt"
                    ],
                ),
            }
        )
    return profile_control_rows


def _vllm_prefix_cache_prompt_audit_summary(
    scenario_rows: list[dict[str, Any]],
    profile_control_rows: list[dict[str, Any]],
    shared_profile: str = "shared_prefix_long_variant",
    control_profile: str = "matched_unique_prefix_variant",
) -> dict[str, Any]:
    shared_rows = [
        row
        for row in scenario_rows
        if row["prompt_profile"] == shared_profile
    ]
    control_rows = [
        row
        for row in scenario_rows
        if row["prompt_profile"] == control_profile
    ]
    return {
        "scenario_count": len(scenario_rows),
        "profile_control_row_count": len(profile_control_rows),
        "shared_common_prefix_tokens_mean": _mean_present(
            row["common_prefix_tokens"] for row in shared_rows
        ),
        "control_common_prefix_tokens_mean": _mean_present(
            row["common_prefix_tokens"] for row in control_rows
        ),
        "shared_common_prefix_full_blocks_mean": _mean_present(
            row["common_prefix_full_blocks"] for row in shared_rows
        ),
        "control_common_prefix_full_blocks_mean": _mean_present(
            row["common_prefix_full_blocks"] for row in control_rows
        ),
        "shared_estimated_reusable_block_tokens_mean": _mean_present(
            row["estimated_reusable_block_tokens"] for row in shared_rows
        ),
        "control_estimated_reusable_block_tokens_mean": _mean_present(
            row["estimated_reusable_block_tokens"] for row in control_rows
        ),
        "shared_estimated_exact_duplicate_reusable_block_tokens_mean": _mean_present(
            row["estimated_exact_duplicate_reusable_block_tokens"]
            for row in shared_rows
        ),
        "control_estimated_exact_duplicate_reusable_block_tokens_mean": _mean_present(
            row["estimated_exact_duplicate_reusable_block_tokens"]
            for row in control_rows
        ),
        "shared_exact_duplicate_reusable_block_fraction_mean": _mean_present(
            row[
                "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt"
            ]
            for row in shared_rows
        ),
        "control_exact_duplicate_reusable_block_fraction_mean": _mean_present(
            row[
                "estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt"
            ]
            for row in control_rows
        ),
        "shared_minus_control_common_prefix_blocks_mean": _mean_present(
            row["shared_minus_control_common_prefix_full_blocks"]
            for row in profile_control_rows
        ),
        "shared_minus_control_reusable_block_tokens_mean": _mean_present(
            row["shared_minus_control_estimated_reusable_block_tokens"]
            for row in profile_control_rows
        ),
        "shared_minus_control_reusable_block_fraction_mean": _mean_present(
            row["shared_minus_control_reusable_block_token_fraction"]
            for row in profile_control_rows
        ),
        "shared_minus_control_exact_duplicate_reusable_block_tokens_mean": _mean_present(
            row[
                "shared_minus_control_estimated_exact_duplicate_reusable_block_tokens"
            ]
            for row in profile_control_rows
        ),
        "shared_minus_control_exact_duplicate_reusable_block_fraction_mean": _mean_present(
            row["shared_minus_control_exact_duplicate_reusable_block_token_fraction"]
            for row in profile_control_rows
        ),
    }


def _compare_vllm_prefix_cache_prompt_overlap_with_server_cache_control(
    *,
    prompt_audit_dirs: list[Path],
    server_absolute_dir: Path,
) -> dict[str, Any]:
    if not prompt_audit_dirs:
        raise ValueError("prompt_audit_dirs must not be empty")

    prompt_payloads = []
    audit_scenario_rows = []
    audit_profile_control_rows = []
    for audit_dir in prompt_audit_dirs:
        audit_json = audit_dir / "prefix-cache-prompt-audit.json"
        payload = json.loads(audit_json.read_text(encoding="utf-8"))
        prompt_payloads.append(
            {
                "audit_dir": str(audit_dir),
                "audit_json": str(audit_json),
                "scenario_seed": payload.get("scenario_seed"),
                "model_id": payload.get("model_id"),
                "request_counts": payload.get("request_counts"),
                "prompt_profiles": payload.get("prompt_profiles"),
                "output_tokens": payload.get("output_tokens"),
                "kv_cache_block_size": payload.get("kv_cache_block_size"),
                "summary": payload.get("summary"),
            }
        )
        for row in payload.get("scenario_rows", []):
            audit_scenario_rows.append(
                {
                    "audit_dir": str(audit_dir),
                    "scenario_seed": payload.get("scenario_seed"),
                    "prompt_profile": row["prompt_profile"],
                    "variant_index": row["variant_index"],
                    "request_count": row["request_count"],
                    "max_new_tokens": row["max_new_tokens"],
                    "prompt_tokens_mean": row["prompt_tokens_mean"],
                    "common_prefix_tokens": row["common_prefix_tokens"],
                    "common_prefix_full_blocks": row[
                        "common_prefix_full_blocks"
                    ],
                    "estimated_reusable_block_tokens": row[
                        "estimated_reusable_block_tokens"
                    ],
                    "estimated_reusable_block_token_fraction_of_total_prompt": row[
                        "estimated_reusable_block_token_fraction_of_total_prompt"
                    ],
                    "unique_prompt_count": row["unique_prompt_count"],
                    "exact_duplicate_prompt_repeated_count": row[
                        "exact_duplicate_prompt_repeated_count"
                    ],
                    "estimated_exact_duplicate_reusable_block_tokens": row[
                        "estimated_exact_duplicate_reusable_block_tokens"
                    ],
                }
            )
        for row in payload.get("profile_control_rows", []):
            audit_profile_control_rows.append(
                {
                    "audit_dir": str(audit_dir),
                    "scenario_seed": payload.get("scenario_seed"),
                    **row,
                }
            )

    server_json = server_absolute_dir / "server-cache-control-absolute.json"
    server_payload = json.loads(server_json.read_text(encoding="utf-8"))
    server_summary_by_key = {
        (row["phase_order"], row["prompt_profile"], row["metric"]): row
        for row in server_payload.get("summary", [])
    }

    audit_profile_summary_rows = []
    for prompt_profile in sorted(
        {row["prompt_profile"] for row in audit_scenario_rows}
    ):
        profile_rows = [
            row
            for row in audit_scenario_rows
            if row["prompt_profile"] == prompt_profile
        ]
        summary_row = {
            "prompt_profile": prompt_profile,
            "audit_trial_count": len(profile_rows),
        }
        for metric in (
            "prompt_tokens_mean",
            "common_prefix_tokens",
            "common_prefix_full_blocks",
            "estimated_reusable_block_tokens",
            "estimated_reusable_block_token_fraction_of_total_prompt",
            "unique_prompt_count",
            "exact_duplicate_prompt_repeated_count",
            "estimated_exact_duplicate_reusable_block_tokens",
        ):
            stats = _metric_distribution(profile_rows, metric)
            summary_row[f"{metric}_mean"] = stats[f"{metric}_mean"]
            summary_row[f"{metric}_min"] = stats[f"{metric}_min"]
            summary_row[f"{metric}_max"] = stats[f"{metric}_max"]
        audit_profile_summary_rows.append(summary_row)

    audit_summary_by_profile = {
        row["prompt_profile"]: row for row in audit_profile_summary_rows
    }
    server_join_rows = []
    for phase_order in server_payload.get("phase_orders", []):
        for prompt_profile in server_payload.get("prompt_profiles", []):
            audit_row = audit_summary_by_profile.get(prompt_profile)
            throughput = server_summary_by_key.get(
                (
                    phase_order,
                    prompt_profile,
                    "cache_on_div_off_server_output_tokens_per_second",
                )
            )
            latency = server_summary_by_key.get(
                (
                    phase_order,
                    prompt_profile,
                    "cache_on_div_off_server_p95_latency_ms",
                )
            )
            if audit_row is None or throughput is None or latency is None:
                continue
            server_join_rows.append(
                {
                    "phase_order": phase_order,
                    "prompt_profile": prompt_profile,
                    "audit_common_prefix_full_blocks_mean": audit_row[
                        "common_prefix_full_blocks_mean"
                    ],
                    "audit_common_prefix_tokens_mean": audit_row[
                        "common_prefix_tokens_mean"
                    ],
                    "audit_estimated_reusable_block_tokens_mean": audit_row[
                        "estimated_reusable_block_tokens_mean"
                    ],
                    "audit_exact_duplicate_reusable_block_tokens_mean": audit_row[
                        "estimated_exact_duplicate_reusable_block_tokens_mean"
                    ],
                    "server_cache_on_div_off_throughput_ratio_mean": throughput[
                        "mean"
                    ],
                    "server_cache_on_div_off_throughput_ratio_p05": throughput[
                        "bootstrap_mean_p05"
                    ],
                    "server_cache_on_div_off_throughput_ratio_p95": throughput[
                        "bootstrap_mean_p95"
                    ],
                    "server_cache_on_div_off_p95_latency_ratio_mean": latency[
                        "mean"
                    ],
                    "server_cache_on_div_off_p95_latency_ratio_p05": latency[
                        "bootstrap_mean_p05"
                    ],
                    "server_cache_on_div_off_p95_latency_ratio_p95": latency[
                        "bootstrap_mean_p95"
                    ],
                    "overlap_class": (
                        "minimal_leading_overlap"
                        if audit_row["common_prefix_full_blocks_mean"] <= 1
                        else "large_leading_overlap"
                    ),
                }
            )

    shared_profile = str(prompt_payloads[0]["summary"].get("shared_profile", ""))
    control_profile = str(prompt_payloads[0]["summary"].get("control_profile", ""))
    if not shared_profile or not control_profile:
        first_payload_profiles = prompt_payloads[0].get("prompt_profiles") or []
        shared_profile, control_profile = _infer_vllm_prefix_cache_prompt_audit_pair(
            [str(profile) for profile in first_payload_profiles]
        )

    profile_control_summary = {}
    for metric in (
        "shared_common_prefix_full_blocks",
        "control_common_prefix_full_blocks",
        "shared_minus_control_common_prefix_full_blocks",
        "shared_estimated_reusable_block_tokens",
        "control_estimated_reusable_block_tokens",
        "shared_minus_control_estimated_reusable_block_tokens",
        "control_estimated_exact_duplicate_reusable_block_tokens",
    ):
        stats = _metric_distribution(audit_profile_control_rows, metric)
        profile_control_summary[f"{metric}_mean"] = stats[f"{metric}_mean"]
        profile_control_summary[f"{metric}_min"] = stats[f"{metric}_min"]
        profile_control_summary[f"{metric}_max"] = stats[f"{metric}_max"]

    matched_unique_server_rows = [
        row
        for row in server_join_rows
        if row["prompt_profile"] == control_profile
    ]
    shared_server_rows = [
        row for row in server_join_rows if row["prompt_profile"] == shared_profile
    ]
    summary = {
        "audit_trial_count": len(prompt_audit_dirs),
        "shared_profile": shared_profile,
        "control_profile": control_profile,
        "shared_common_prefix_full_blocks_mean": profile_control_summary[
            "shared_common_prefix_full_blocks_mean"
        ],
        "control_common_prefix_full_blocks_mean": profile_control_summary[
            "control_common_prefix_full_blocks_mean"
        ],
        "shared_minus_control_common_prefix_full_blocks_mean": (
            profile_control_summary[
                "shared_minus_control_common_prefix_full_blocks_mean"
            ]
        ),
        "shared_minus_control_estimated_reusable_block_tokens_mean": (
            profile_control_summary[
                "shared_minus_control_estimated_reusable_block_tokens_mean"
            ]
        ),
        "control_estimated_exact_duplicate_reusable_block_tokens_mean": (
            profile_control_summary[
                "control_estimated_exact_duplicate_reusable_block_tokens_mean"
            ]
        ),
        "matched_unique_min_server_throughput_ratio_mean": min(
            row["server_cache_on_div_off_throughput_ratio_mean"]
            for row in matched_unique_server_rows
        ),
        "matched_unique_max_server_p95_latency_ratio_mean": max(
            row["server_cache_on_div_off_p95_latency_ratio_mean"]
            for row in matched_unique_server_rows
        ),
        "shared_min_server_throughput_ratio_mean": min(
            row["server_cache_on_div_off_throughput_ratio_mean"]
            for row in shared_server_rows
        ),
        "control_has_minimal_leading_overlap": (
            profile_control_summary["control_common_prefix_full_blocks_max"] <= 1
        ),
        "control_has_no_exact_duplicate_reuse": (
            profile_control_summary[
                "control_estimated_exact_duplicate_reusable_block_tokens_max"
            ]
            == 0
        ),
    }

    payload = {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-prompt-overlap-server-compare",
        "prompt_audit_dirs": [str(path) for path in prompt_audit_dirs],
        "server_absolute_dir": str(server_absolute_dir),
        "server_absolute_json": str(server_json),
        "audit_trial_count": len(prompt_audit_dirs),
        "audit_scenario_rows": audit_scenario_rows,
        "audit_profile_control_rows": audit_profile_control_rows,
        "audit_profile_summary_rows": audit_profile_summary_rows,
        "server_join_rows": server_join_rows,
        "profile_control_summary": profile_control_summary,
        "summary": summary,
        "source_prompt_audits": prompt_payloads,
    }
    payload["markdown"] = (
        _format_vllm_prefix_cache_prompt_overlap_server_compare_markdown(payload)
    )
    return payload


def _format_vllm_prefix_cache_prompt_overlap_server_compare_markdown(
    payload: dict[str, Any],
) -> str:
    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}{suffix}"
        return f"{value}{suffix}"

    summary = payload["summary"]
    lines = [
        "# Prompt Overlap Vs Server Cache-Control",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Audit trials | {summary['audit_trial_count']} |",
        (
            "| Shared common-prefix full blocks | "
            f"{fmt(summary['shared_common_prefix_full_blocks_mean'])} |"
        ),
        (
            "| Control common-prefix full blocks | "
            f"{fmt(summary['control_common_prefix_full_blocks_mean'])} |"
        ),
        (
            "| Shared-minus-control full-block delta | "
            f"{fmt(summary['shared_minus_control_common_prefix_full_blocks_mean'])} |"
        ),
        (
            "| Shared-minus-control reusable block-token delta | "
            f"{fmt(summary['shared_minus_control_estimated_reusable_block_tokens_mean'])} |"
        ),
        (
            "| Control exact-duplicate reusable block tokens | "
            f"{fmt(summary['control_estimated_exact_duplicate_reusable_block_tokens_mean'])} |"
        ),
        (
            "| Matched-unique min server cache-on/off throughput ratio | "
            f"{fmt(summary['matched_unique_min_server_throughput_ratio_mean'], 'x')} |"
        ),
        (
            "| Matched-unique max server cache-on/off p95 latency ratio | "
            f"{fmt(summary['matched_unique_max_server_p95_latency_ratio_mean'], 'x')} |"
        ),
        "",
        "## Joined Server View",
        "",
        (
            "| Phase Order | Profile | Overlap Class | Common Blocks | "
            "Server Throughput Ratio | Server p95 Latency Ratio |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["server_join_rows"]:
        lines.append(
            "| "
            f"`{row['phase_order']}` | "
            f"`{row['prompt_profile']}` | "
            f"`{row['overlap_class']}` | "
            f"{fmt(row['audit_common_prefix_full_blocks_mean'])} | "
            f"{fmt(row['server_cache_on_div_off_throughput_ratio_mean'], 'x')} | "
            f"{fmt(row['server_cache_on_div_off_p95_latency_ratio_mean'], 'x')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The prompt-token audit says the matched-unique control has only "
                "one common leading full block and no exact duplicate prompts. "
                "That is not enough leading-token reuse to explain the "
                "order-of-magnitude server cache-on/cache-off speedup by the "
                "intended shared-prefix mechanism alone."
            ),
            "",
            (
                "The next measurement should separate explicit user-visible "
                "shared prefixes from other server-path effects: vLLM prefix-cache "
                "semantics, scenario ordering, shared chat-template scaffolding, "
                "or generic cache-on server behavior under this workload shape."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _format_vllm_prefix_cache_prompt_audit_markdown(
    payload: dict[str, Any],
) -> str:
    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.3f}{suffix}"
        return f"{value}{suffix}"

    summary = payload["summary"]
    lines = [
        "# Prefix-Cache Prompt Token Audit",
        "",
        f"Model: `{payload['model_id']}`",
        f"Profiles: `{','.join(payload['prompt_profiles'])}`",
        f"Shared profile: `{payload['shared_profile']}`",
        f"Control profile: `{payload['control_profile']}`",
        f"Request counts: `{','.join(str(v) for v in payload['request_counts'])}`",
        f"Repeats: `{payload['repeats']}`",
        f"Scenario seed: `{payload['scenario_seed']}`",
        f"KV cache block size: `{payload['kv_cache_block_size']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (
            "| Mean shared common-prefix full blocks | "
            f"{fmt(summary['shared_common_prefix_full_blocks_mean'])} |"
        ),
        (
            "| Mean control common-prefix full blocks | "
            f"{fmt(summary['control_common_prefix_full_blocks_mean'])} |"
        ),
        (
            "| Mean shared-minus-control common-prefix blocks | "
            f"{fmt(summary['shared_minus_control_common_prefix_blocks_mean'])} |"
        ),
        (
            "| Mean shared-minus-control reusable block tokens | "
            f"{fmt(summary['shared_minus_control_reusable_block_tokens_mean'])} |"
        ),
        (
            "| Mean shared-minus-control reusable block fraction | "
            f"{fmt(summary['shared_minus_control_reusable_block_fraction_mean'])} |"
        ),
        (
            "| Mean shared exact-duplicate reusable block tokens | "
            f"{fmt(summary['shared_estimated_exact_duplicate_reusable_block_tokens_mean'])} |"
        ),
        (
            "| Mean control exact-duplicate reusable block tokens | "
            f"{fmt(summary['control_estimated_exact_duplicate_reusable_block_tokens_mean'])} |"
        ),
        (
            "| Mean shared-minus-control exact-duplicate reusable block tokens | "
            f"{fmt(summary['shared_minus_control_exact_duplicate_reusable_block_tokens_mean'])} |"
        ),
        (
            "| Mean control exact-duplicate reusable block fraction | "
            f"{fmt(summary['control_exact_duplicate_reusable_block_fraction_mean'])} |"
        ),
        "",
        "## Scenario Audit",
        "",
        (
            "| Profile | Repeat | Requests | Common Prefix Tokens | Full Blocks | "
            "Reusable Block Tokens | Reusable Fraction | Unique Prompts | "
            "Duplicate Reusable Tokens | Duplicate Fraction |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["scenario_rows"]:
        lines.append(
            f"| `{row['prompt_profile']}` | "
            f"{row['repeat_index']} | "
            f"{row['request_count']} | "
            f"{row['common_prefix_tokens']} | "
            f"{row['common_prefix_full_blocks']} | "
            f"{row['estimated_reusable_block_tokens']} | "
            f"{fmt(row['estimated_reusable_block_token_fraction_of_total_prompt'])} | "
            f"{row['unique_prompt_count']} | "
            f"{row['estimated_exact_duplicate_reusable_block_tokens']} | "
            f"{fmt(row['estimated_exact_duplicate_reusable_block_token_fraction_of_total_prompt'])} |"
        )

    lines.extend(
        [
            "",
            "## Shared Vs Control",
            "",
            (
                "| Repeat | Requests | Shared Blocks | Control Blocks | "
                "Block Delta | Reusable Token Delta | Shared Duplicate Tokens | "
                "Control Duplicate Tokens |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["profile_control_rows"]:
        lines.append(
            f"| {row['repeat_index']} | "
            f"{row['request_count']} | "
            f"{row['shared_common_prefix_full_blocks']} | "
            f"{row['control_common_prefix_full_blocks']} | "
            f"{fmt(row['shared_minus_control_common_prefix_full_blocks'])} | "
            f"{fmt(row['shared_minus_control_estimated_reusable_block_tokens'])} | "
            f"{fmt(row['shared_estimated_exact_duplicate_reusable_block_tokens'])} | "
            f"{fmt(row['control_estimated_exact_duplicate_reusable_block_tokens'])} |"
        )

    lines.extend(
        [
            "",
            (
                "Reusable block tokens estimate how many full leading-token "
                "blocks could be reused by requests after the first request in "
                "a batch. Exact-duplicate reusable tokens estimate repeated "
                "full-prompt token groups within the same scenario. This is a "
                "tokenizer/block audit, not a vLLM timing measurement."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _tokenize_prompt(tokenizer: Any, prompt: str) -> tuple[dict[str, Any], str]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ),
            "chat_template",
        )
    return tokenizer(prompt, return_tensors="pt"), "plain"


def _format_prompt_for_generation(tokenizer: Any, prompt: str) -> tuple[str, str]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            ),
            "chat_template",
        )
    return prompt, "plain"


def _manual_greedy_decode(
    model: Any,
    inputs: dict[str, Any],
    max_new_tokens: int,
    torch_module: Any,
) -> dict[str, Any]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch_module.ones_like(input_ids)

    generated_tokens = []
    with torch_module.inference_mode():
        _synchronize_if_cuda(torch_module)
        started = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        past_key_values = outputs.past_key_values
        _synchronize_if_cuda(torch_module)
        ttft_ms = (time.perf_counter() - started) * 1000

        decode_started = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            attention_mask = torch_module.cat(
                [
                    attention_mask,
                    torch_module.ones(
                        (attention_mask.shape[0], 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            past_key_values = outputs.past_key_values
        _synchronize_if_cuda(torch_module)
        decode_ms = (time.perf_counter() - decode_started) * 1000

    return {
        "generated_ids": torch_module.cat(generated_tokens, dim=-1),
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
    }


def _synchronize_if_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _serialize_vllm_metrics(output: Any) -> dict[str, float | None]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return {}
    metric_names = (
        "arrival_time",
        "first_scheduled_time",
        "first_token_time",
        "last_token_time",
        "finished_time",
        "scheduler_time",
        "model_forward_time",
        "model_execute_time",
    )
    result: dict[str, float | None] = {}
    for name in metric_names:
        value = getattr(metrics, name, None)
        result[name] = float(value) if isinstance(value, int | float) else None
    return result


def _duration_from_metrics_ms(
    metrics: dict[str, float | None],
    start_key: str,
    end_key: str,
) -> float | None:
    start = metrics.get(start_key)
    end = metrics.get(end_key)
    if start is None or end is None:
        return None
    return max(0.0, (end - start) * 1000)


def _percentile(values: list[float | None], percentile: int) -> float | None:
    clean_values = sorted(value for value in values if value is not None)
    if not clean_values:
        return None
    rank = max(1, int((percentile / 100) * len(clean_values) + 0.999999))
    return clean_values[min(rank - 1, len(clean_values) - 1)]


def _run_command(command: list[str]) -> str:
    import subprocess

    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as error:
        return error.output.strip()
