"""AI Hub authentication and profiling utilities; credentials are never logged."""

import getpass
import os
from typing import Any

import numpy as np


def hub_client():
    import qai_hub as hub

    token = os.environ.get("QAI_HUB_API_TOKEN")
    if token:
        return hub.Client(hub.ClientConfig(api_token=token))
    # Respect an existing AI Hub configuration without copying it into the repo.
    try:
        return hub.Client()
    except Exception:
        token = getpass.getpass("AI Hub API token: ")
        return hub.Client(hub.ClientConfig(api_token=token))


def profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    summary = profile.get("execution_summary", {})
    times = np.asarray(summary.get("all_inference_times", []), dtype=np.float64)
    placement: dict[str, int] = {}
    detail = profile.get("execution_detail", [])
    for operation in detail if isinstance(detail, list) else []:
        unit = str(operation.get("compute_unit", "unknown"))
        placement[unit] = placement.get(unit, 0) + 1
    result: dict[str, Any] = {
        "estimated_latency_ms": (
            float(summary["estimated_inference_time"]) / 1000.0
            if summary.get("estimated_inference_time") is not None
            else None
        ),
        "estimated_peak_memory_bytes": summary.get("estimated_inference_peak_memory"),
        "placement_operation_counts": placement,
        "iterations": int(times.size),
    }
    if times.size:
        result.update(
            {
                "latency_min_ms": float(times.min() / 1000.0),
                "latency_median_ms": float(np.median(times) / 1000.0),
                "latency_p90_ms": float(np.quantile(times, 0.90) / 1000.0),
                "latency_mean_ms": float(times.mean() / 1000.0),
            }
        )
    return result
