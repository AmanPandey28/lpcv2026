#!/usr/bin/env python3
"""
LPCVC 2026 Track 1 - MobileCLIP2-S4 Text Encoder
Conservative, competition-aware, compile-friendly text pipeline

What this script does
---------------------
1. Loads MobileCLIP2-S4 from your local checkpoint.
2. Applies export patches needed for the text branch:
      - disables PyTorch MHA fast-path
      - replaces scaled dot-product attention with explicit math
      - patches text global pooling to avoid fragile dynamic behavior
3. Reparameterizes the full model.
4. Wraps the text encoder so competition-tokenized inputs become closer to the
   token format the MobileCLIP/OpenCLIP text branch appears to expect.
5. Exports a self-contained ONNX text encoder with:
      input  name = "text"
      output name = "text_embedding"
6. Validates the ONNX locally.
7. Compiles the ONNX to QNN DLC on XR2 Gen 2 (Proxy) using:
      --target_runtime qnn_dlc --truncate_64bit_io
8. Profiles the compiled model.
9. Optionally runs one smoke-test inference using the exact competition tokenizer:
      openai/clip-vit-base-patch32

Why this version is different
-----------------------------
Your diagnosis showed two things:
1. The competition tokenizer does NOT exactly match the MobileCLIP/OpenCLIP tokenizer.
2. The key difference is that after the first EOS token:
      - OpenCLIP/MobileCLIP behavior padded with 0
      - competition tokenizer pads with repeated EOS tokens (49407)

This script fixes that *inside the model wrapper* by converting repeated EOS
tokens after the first EOS into zeros before calling encode_text().

This lets the exported ONNX stay competition-compliant at the interface level,
while making the internal text input distribution closer to what the model was
implicitly expecting.

Also:
- smoke inference is now non-fatal
- compile/profile success will no longer be masked by a final smoke-test failure

Before running
--------------
Set your AI Hub token:

    export QAI_HUB_API_TOKEN="YOUR_TOKEN_HERE"

Expected local files
--------------------
- /home/aman/dev/lpcv/anvil/ml-mobileclip
- /home/aman/dev/lpcv/anvil/mobileclip2_s4.pt
- /home/aman/dev/lpcv/anvil/ml-mobileclip/mobileclip2/model_configs/MobileCLIP-S4.json
"""

from __future__ import annotations

import json
import math
import os
import pprint
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

import qai_hub as hub
from transformers import CLIPTokenizer


# ============================================================
# User configuration
# ============================================================

ML_MOBILECLIP_REPO = Path("/home/aman/dev/lpcv/anvil/ml-mobileclip")
MODEL_CONFIG_PATH = Path(
    "/home/aman/dev/lpcv/anvil/ml-mobileclip/mobileclip2/model_configs/MobileCLIP-S4.json"
)
CHECKPOINT_PATH = Path("/home/aman/dev/lpcv/anvil/mobileclip2_s4.pt")

MODEL_NAME_CANDIDATES = [
    os.environ.get("MOBILECLIP_MODEL_NAME", "MobileCLIP2-S4"),
    "MobileCLIP-S4",
]

# Keep token in environment, not in source
QAI_HUB_API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN", "qzi55yh0tdzhzvevw4kt53z8igzql5j83yukgg36")

TARGET_DEVICE_NAME = "XR2 Gen 2 (Proxy)"

TEXT_INPUT_NAME = "text"
TEXT_OUTPUT_NAME = "text_embedding"
TEXT_INPUT_SHAPE = (1, 77)

ARTIFACT_DIR = Path("lpcvc_track1_text_baseline_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_TEXT_PATH = ARTIFACT_DIR / "pure_text_encoder.onnx"
JOB_RECORD_PATH = ARTIFACT_DIR / "text_job_record.json"

RUN_SMOKE_TEST_INFERENCE = True
PRINT_RAW_PROFILE_IF_UNKNOWN = True

PROFILE_OPTIONS = "--max_profiler_iterations 100"

SHARE_WITH_ORGANIZERS = False
ORGANIZER_EMAIL = "lowpowervision@gmail.com"

# Competition tokenizer contract
COMPETITION_TOKENIZER_NAME = "openai/clip-vit-base-patch32"

# Based on the competition examples and the diagnosis output
COMPETITION_EOS_TOKEN_ID = 49407


# ============================================================
# Basic helpers
# ============================================================

def fail_if_missing(path: Path, description: str) -> None:
    """Fail early if an expected local path is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_json(path: Path) -> dict:
    """Load JSON for logging / sanity checking."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hub_client(api_token: str) -> hub.Client:
    """Authenticate to Qualcomm AI Hub using the modern client API."""
    if not api_token:
        raise RuntimeError(
            "QAI_HUB_API_TOKEN is empty. Set it first, for example:\n"
            'export QAI_HUB_API_TOKEN="YOUR_TOKEN_HERE"'
        )

    client = hub.Client(hub.ClientConfig(api_token=api_token))
    devices = client.get_devices()
    if not devices:
        raise RuntimeError("AI Hub authentication failed: no devices returned.")
    print(f"[auth] AI Hub client ready. Visible devices: {len(devices)}")
    return client


def wait_success(job, name: str, fatal: bool = True) -> bool:
    """
    Wait for a Hub job to finish.

    Returns:
        True  if the job succeeded
        False if the job failed and fatal=False

    Notes:
    - Not every QAI Hub job type supports download_logs().
    - Inference smoke tests should not crash the whole script if they fail.
    """
    status = str(job.wait())
    print(f"[hub] {name} status: {status}")

    if "SUCCESS" in status:
        return True

    # Only some job types support log download
    if hasattr(job, "download_logs"):
        log_dir = ARTIFACT_DIR / f"{name}_logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            job.download_logs(str(log_dir))
            print(f"[hub] Downloaded failure logs to: {log_dir.resolve()}")
        except Exception as exc:
            print(f"[hub] Could not download logs for {name}: {exc}")

    if fatal:
        raise RuntimeError(f"{name} failed. Inspect job URL: {job.url}")

    print(f"[hub] Non-fatal failure for {name}. Inspect job URL: {job.url}")
    return False


def prepare_python_imports() -> None:
    """
    Add only the repo root to sys.path.
    """
    repo_str = str(ML_MOBILECLIP_REPO.resolve())

    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def import_mobileclip_stack():
    """
    Import the local MobileCLIP/OpenCLIP stack after sys.path has been prepared.
    """
    import importlib
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model

    # Import the actual inner module: ml-mobileclip/mobileclip2/mobileclip2.py
    mobileclip2_module = importlib.import_module("mobileclip2.mobileclip2")

    return open_clip, reparameterize_model, mobileclip2_module


# ============================================================
# Export patches for the text branch
# ============================================================

def apply_text_export_patches(open_clip, mobileclip2_module) -> None:
    """
    Apply export-time patches that make the text branch more ONNX-friendly.

    Why these are needed:
    ---------------------
    - Disable PyTorch hidden MHA fast path
    - Expose SDPA as explicit matmul/softmax/matmul
    - Patch text pooling so export sees stable behavior
    """
    torch.backends.mha.set_fastpath_enabled(False)

    def explicit_math_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None):
        scale_factor = scale if scale is not None else (1.0 / math.sqrt(query.size(-1)))
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    F.scaled_dot_product_attention = explicit_math_sdpa

    # Patch LayerNormChannel if it exists in this module
    if hasattr(mobileclip2_module, "LayerNormChannel"):
        original_ln_init = mobileclip2_module.LayerNormChannel.__init__

        def patched_ln_init(self, *args, **kwargs):
            kwargs.pop("device", None)
            kwargs.pop("dtype", None)
            original_ln_init(self, *args, **kwargs)

        mobileclip2_module.LayerNormChannel.__init__ = patched_ln_init

    import open_clip.transformer
    original_text_global_pool = open_clip.transformer.text_global_pool

    def patched_text_global_pool(x, text, pool_type, eos_token_id=None):
        if pool_type == "argmax":
            # Batch size is effectively 1 in this competition encoder workflow
            seq_idx = text.argmax(dim=-1)[0]
            return x[0:1, seq_idx, :]
        return original_text_global_pool(x, text, pool_type, eos_token_id)

    open_clip.transformer.text_global_pool = patched_text_global_pool


# ============================================================
# Model wrappers
# ============================================================

class CompetitionCompatibleTextEncoder(nn.Module):
    """
    Text wrapper that makes the competition token format closer to the token
    format used by the MobileCLIP/OpenCLIP tokenizer behavior observed in diagnosis.

    Competition tokenizer behavior:
      [BOS, ..., EOS, EOS, EOS, EOS, ...]

    OpenCLIP/MobileCLIP tokenizer behavior seen in diagnosis:
      [BOS, ..., EOS,   0,   0,   0, ...]

    This wrapper preserves the first EOS token and converts repeated EOS padding
    *after* the first EOS into zeros before calling encode_text().
    """

    def __init__(self, clip_model: nn.Module, eos_token_id: int = COMPETITION_EOS_TOKEN_ID):
        super().__init__()
        self.model = clip_model
        self.eos_token_id = eos_token_id

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # Keep int64 to match export/compile expectations
        x = text.to(torch.int64)

        # Find EOS positions
        eos_mask = (x == self.eos_token_id).to(torch.int64)

        # Count EOS tokens cumulatively from left to right
        eos_cumsum = torch.cumsum(eos_mask, dim=1)

        # Repeated EOS tokens after the first one become padding zeros
        repeated_eos_mask = (x == self.eos_token_id) & (eos_cumsum > 1)
        x = torch.where(repeated_eos_mask, torch.zeros_like(x), x)

        return self.model.encode_text(x)


def load_reparameterized_text_encoder() -> Tuple[nn.Module, str]:
    """
    Load MobileCLIP2-S4, apply export patches, reparameterize, and return a
    competition-compatible text encoder wrapper.
    """
    prepare_python_imports()
    open_clip, reparameterize_model, mobileclip2_module = import_mobileclip_stack()

    apply_text_export_patches(open_clip, mobileclip2_module)

    last_err: Optional[Exception] = None

    for model_name in MODEL_NAME_CANDIDATES:
        try:
            print(f"[model] Trying model_name={model_name}")
            model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=str(CHECKPOINT_PATH),
            )
            model = model.eval()
            model = reparameterize_model(model).eval()

            text_encoder = CompetitionCompatibleTextEncoder(
                model, eos_token_id=COMPETITION_EOS_TOKEN_ID
            ).eval()

            return text_encoder, model_name
        except Exception as exc:
            last_err = exc
            print(f"[model] Failed with {model_name}: {exc}")

    raise RuntimeError("Could not load checkpoint with any model name candidate.") from last_err


# ============================================================
# ONNX export / validation
# ============================================================

def cleanup_old_export_files() -> None:
    """Remove stale export files from previous runs."""
    candidates = [
        ONNX_TEXT_PATH,
        ONNX_TEXT_PATH.with_suffix(".onnx.data"),
        ONNX_TEXT_PATH.parent / f"{ONNX_TEXT_PATH.name}.data",
        ONNX_TEXT_PATH.parent / f"{ONNX_TEXT_PATH.stem}.data",
    ]
    for p in candidates:
        if p.exists():
            p.unlink()


def export_text_encoder_to_onnx(text_encoder: nn.Module) -> None:
    """
    Export a self-contained ONNX text encoder.

    Design choices:
    ---------------
    - classic exporter path
    - opset 16
    - int64 dummy tokens to stay close to the natural PyTorch export path
    - conservative constant folding
    """
    cleanup_old_export_files()

    dummy_tokens = torch.randint(
        low=0,
        high=49408,
        size=TEXT_INPUT_SHAPE,
        dtype=torch.int64,
    )

    print(f"[onnx] Exporting to {ONNX_TEXT_PATH.resolve()}")
    torch.onnx.export(
        text_encoder,
        dummy_tokens,
        str(ONNX_TEXT_PATH),
        input_names=[TEXT_INPUT_NAME],
        output_names=[TEXT_OUTPUT_NAME],
        opset_version=16,
        do_constant_folding=False,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        external_data=False,
    )

    validate_onnx(ONNX_TEXT_PATH)


def validate_onnx(onnx_path: Path) -> None:
    """Validate the ONNX file locally and check IO names."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)

    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]

    if TEXT_INPUT_NAME not in input_names:
        raise RuntimeError(f"Expected ONNX input name '{TEXT_INPUT_NAME}', found {input_names}")
    if TEXT_OUTPUT_NAME not in output_names:
        raise RuntimeError(f"Expected ONNX output name '{TEXT_OUTPUT_NAME}', found {output_names}")

    print("[onnx] ONNX validated successfully.")
    print(f"[onnx] Inputs : {input_names}")
    print(f"[onnx] Outputs: {output_names}")


# ============================================================
# QAI Hub flow
# ============================================================

def compile_qnn_dlc(client: hub.Client, onnx_path: Path, device: hub.Device):
    """
    Compile the text ONNX to QNN DLC.

    Important:
    ----------
    The text encoder is exported with int64 input, but the device-side path can
    safely use 32-bit tokens internally. The truncate flag handles that.
    """
    print("[hub] Submitting compile job to QNN DLC...")
    job = client.submit_compile_job(
        model=str(onnx_path),
        device=device,
        input_specs={TEXT_INPUT_NAME: ((1, 77), "int64")},
        options="--target_runtime qnn_dlc --truncate_64bit_io",
        name="lpcvc_track1_mobileclip2s4_text_baseline_qnn_dlc",
    )
    print(f"[hub] compile job id: {job.job_id}")
    wait_success(job, "compile_qnn_dlc")

    compiled_model = job.get_target_model()
    if compiled_model is None:
        raise RuntimeError(f"compile_qnn_dlc returned no target model. Job URL: {job.url}")

    return job, compiled_model


def profile_model(client: hub.Client, compiled_model, device: hub.Device):
    """Profile the compiled text model on the target device."""
    print("[hub] Submitting profile job...")
    job = client.submit_profile_job(
        model=compiled_model,
        device=device,
        options=PROFILE_OPTIONS,
        name="lpcvc_track1_mobileclip2s4_text_baseline_profile",
    )
    print(f"[hub] profile job id: {job.job_id}")
    wait_success(job, "profile")
    return job


def extract_profile_metrics(profile: dict) -> tuple[Optional[float], Optional[object]]:
    """
    Extract latency and peak memory from the dict returned by download_profile().
    """
    if not isinstance(profile, dict):
        raise TypeError(f"Expected profile to be a dict, got {type(profile)}")

    if "execution_detail" in profile and isinstance(profile["execution_detail"], dict):
        exec_detail = profile["execution_detail"]
        latency_us = exec_detail.get("estimated_inference_time")
        peak_mem = exec_detail.get("estimated_peak_memory")
        latency_ms = latency_us / 1000.0 if latency_us is not None else None
        return latency_ms, peak_mem

    latency_us = profile.get("estimated_inference_time")
    peak_mem = profile.get("estimated_peak_memory")
    if latency_us is not None or peak_mem is not None:
        latency_ms = latency_us / 1000.0 if latency_us is not None else None
        return latency_ms, peak_mem

    if "metrics" in profile and isinstance(profile["metrics"], dict):
        metrics = profile["metrics"]
        latency_us = metrics.get("estimated_inference_time")
        peak_mem = metrics.get("estimated_peak_memory")
        latency_ms = latency_us / 1000.0 if latency_us is not None else None
        return latency_ms, peak_mem

    if PRINT_RAW_PROFILE_IF_UNKNOWN:
        print("\n[debug] Unrecognized profile schema returned by download_profile():")
        pprint.pprint(profile)

    return None, None


# ============================================================
# Competition tokenizer + optional smoke test
# ============================================================

def build_competition_tokenizer() -> CLIPTokenizer:
    """
    Build the exact tokenizer expected by the competition.

    Track 1 uses:
      openai/clip-vit-base-patch32
    and the sample flow adds:
      tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    """
    tokenizer = CLIPTokenizer.from_pretrained(COMPETITION_TOKENIZER_NAME)
    tokenizer.add_special_tokens({"cls_token": tokenizer.eos_token})
    return tokenizer


def tokenize_competition_text(text: str) -> np.ndarray:
    """
    Tokenize a string exactly according to the competition tokenizer contract.
    """
    tokenizer = build_competition_tokenizer()
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )

    # Keep as int64 because that matches the exported ONNX input type.
    return tokens["input_ids"].to(torch.int64).cpu().numpy()


def smoke_test_inference(client: hub.Client, compiled_model, device: hub.Device) -> bool:
    """
    Run one example prompt through the compiled text model to verify execution.

    This is a non-fatal sanity check. If it fails, we do NOT crash the whole script,
    because compile/profile success already means the main artifact was created successfully.
    """
    print("[hub] Running smoke-test inference...")

    sample_tokens = tokenize_competition_text("white soccer ball")

    inf_job = client.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs={TEXT_INPUT_NAME: [sample_tokens]},
        name="lpcvc_track1_mobileclip2s4_text_baseline_smoke",
    )
    print(f"[hub] inference job id: {inf_job.job_id}")

    ok = wait_success(inf_job, "smoke_inference", fatal=False)
    if not ok:
        return False

    output_data = inf_job.download_output_data()
    output_key = list(output_data.keys())[0]
    output_value = output_data[output_key][0]

    print(f"[smoke] compiled output key : {output_key}")
    print(f"[smoke] output shape        : {tuple(output_value.shape)}")
    print(f"[smoke] output dtype        : {output_value.dtype}")
    return True


# ============================================================
# Optional share
# ============================================================

def maybe_share_compile_job(compile_job) -> None:
    """Share the final compile job with organizers only when you are ready."""
    if not SHARE_WITH_ORGANIZERS:
        print("[submit] Sharing skipped.")
        return

    print(f"[submit] Sharing compile job with {ORGANIZER_EMAIL} ...")
    compile_job.modify_sharing(add_emails=[ORGANIZER_EMAIL])
    print("[submit] Sharing request sent.")


def save_job_record(record: dict) -> None:
    """Save all important run metadata for reproducibility."""
    JOB_RECORD_PATH.write_text(json.dumps(record, indent=2))
    print(f"[artifacts] Wrote job record to {JOB_RECORD_PATH.resolve()}")


# ============================================================
# Main
# ============================================================

def main() -> int:
    print("============================================================")
    print("LPCVC 2026 Track 1 - MobileCLIP2-S4 Text Baseline")
    print("============================================================")

    # Step 0: file checks
    fail_if_missing(ML_MOBILECLIP_REPO, "ml-mobileclip repo")
    fail_if_missing(MODEL_CONFIG_PATH, "model config JSON")
    fail_if_missing(CHECKPOINT_PATH, "checkpoint")

    config = load_json(MODEL_CONFIG_PATH)
    print(f"[config] Loaded config from {MODEL_CONFIG_PATH}")
    print(f"[config] Top-level keys: {list(config.keys())[:10]}")

    # Step 1: authenticate to AI Hub
    client = build_hub_client(QAI_HUB_API_TOKEN)
    target_device = hub.Device(TARGET_DEVICE_NAME)
    print(f"[hub] Target device: {target_device.name}")

    # Step 2: build patched + competition-compatible text encoder
    text_encoder, resolved_model_name = load_reparameterized_text_encoder()

    # Step 3: export ONNX
    export_text_encoder_to_onnx(text_encoder)

    # Step 4: compile to QNN DLC
    compile_job, compiled_model = compile_qnn_dlc(client, ONNX_TEXT_PATH, target_device)

    # Step 5: profile
    profile_job = profile_model(client, compiled_model, target_device)
    profile = profile_job.download_profile()

    latency_ms, peak_mem = extract_profile_metrics(profile)

    print("\n================================")
    print("TEXT ENCODER PROFILE RESULT")
    print("================================")
    if latency_ms is not None:
        print(f"Latency     : {latency_ms:.3f} ms")
    else:
        print("Latency     : not found in returned profile dict")

    if peak_mem is not None:
        print(f"Peak memory : {peak_mem}")
    else:
        print("Peak memory : not found in returned profile dict")
    print("================================")

    # Step 6: optional smoke test (non-fatal)
    smoke_ok = None
    if RUN_SMOKE_TEST_INFERENCE:
        smoke_ok = smoke_test_inference(client, compiled_model, target_device)

    # Step 7: optional share
    maybe_share_compile_job(compile_job)

    # Step 8: save metadata
    save_job_record(
        {
            "resolved_model_name": resolved_model_name,
            "model_config_path": str(MODEL_CONFIG_PATH),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "onnx_path": str(ONNX_TEXT_PATH),
            "device": TARGET_DEVICE_NAME,
            "compile_job_id": compile_job.job_id,
            "profile_job_id": profile_job.job_id,
            "text_latency_ms": latency_ms,
            "text_peak_memory": peak_mem,
            "raw_profile": profile,
            "tokenizer_name": COMPETITION_TOKENIZER_NAME,
            "eos_token_id": COMPETITION_EOS_TOKEN_ID,
            "smoke_inference_ok": smoke_ok,
        }
    )

    print("\n✅ Done.")
    print(f"[final] Text compile job id: {compile_job.job_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise
    except Exception as exc:
        print(f"\n❌ Pipeline failed: {exc}")
        raise
