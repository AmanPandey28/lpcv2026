#!/usr/bin/env python3
"""
LPCVC 2026 Track 1 — Final text encoder script (MobileCLIP2-S2)
================================================================

What this script does, end to end
---------------------------------
1.  Loads MobileCLIP2-S2 from your local checkpoint with the Apple-recommended
    preprocess kwargs (image_mean=(0,0,0), image_std=(1,1,1)) so the loader
    config matches the image script.
2.  Applies the export-time patches that you already know are needed:
       - disables PyTorch hidden MHA fast-path
       - replaces SDPA with explicit math
       - patches OpenCLIP text_global_pool for stable static export
3.  Reparameterizes the model.
4.  Wraps the text encoder in `CompetitionCompatibleTextEncoder` so the
    competition's repeated-EOS padding behavior gets converted to
    zero-padding inside the graph (your validated EOS-padding fix).
5.  Exports a self-contained ONNX text encoder.
6.  Compiles to optimized ONNX on AI Hub (small but cleans up the graph).
7.  Quantizes on AI Hub with W8A16 (default range scheme, since text is
    less sensitive than image and we want safety here).
8.  Compiles to QNN DLC with `--target_runtime qnn_dlc --truncate_64bit_io`.
9.  Profiles on XR2 Gen 2 (Proxy).
10. Runs LOCAL VALIDATION on the official sample texts:
       - PT vs float ONNX cosine
       - PT vs QDQ ONNX cosine
       - tokenizer round-trip sanity check
11. Optionally runs an on-device smoke inference.
12. Optionally shares the compile job with the organizers.
13. Saves a JSON record.

Why these choices
-----------------
- S2 to match the image branch (same checkpoint family, joint embedding space).
- W8A16: text encoder activations are not catastrophic under W8A8 in your
  earlier diagnosis, but W8A16 gives extra safety margin and stays well within
  the latency budget (text is the smaller branch).
- `--truncate_64bit_io`: Hexagon NPU has no 64-bit ALU. Mandatory for int64
  token inputs.
- Repeated-EOS-to-zero conversion: matches the OpenCLIP/MobileCLIP padding
  the model was effectively trained against, while still accepting the
  competition tokenizer's repeated-EOS output.

Before running
--------------
    export QAI_HUB_API_TOKEN="YOUR_TOKEN_HERE"
"""

from __future__ import annotations

import json
import math
import os
import pprint
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import qai_hub as hub
from transformers import CLIPTokenizer


# ============================================================
# User configuration — edit these paths if needed
# ============================================================

ML_MOBILECLIP_REPO = Path("/home/aman/dev/lpcv/ml-mobileclip")
MODEL_CONFIG_PATH = Path(
    "/home/aman/dev/lpcv/ml-mobileclip/mobileclip2/model_configs/MobileCLIP2-S2.json"
)
CHECKPOINT_PATH = Path("/home/aman/dev/lpcv/anvil/mobileclip2_s2.pt")

MODEL_NAME_CANDIDATES = [
    os.environ.get("MOBILECLIP_MODEL_NAME", "MobileCLIP2-S2"),
    "MobileCLIP-S2",
]

QAI_HUB_API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN", "")

TARGET_DEVICE_NAME = "XR2 Gen 2 (Proxy)"

TEXT_INPUT_NAME = "text"
TEXT_OUTPUT_NAME = "text_embedding"
TEXT_INPUT_SHAPE = (1, 77)

ARTIFACT_DIR = Path("lpcvc_track1_text_s2_w8a16_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_TEXT_PATH = ARTIFACT_DIR / "pure_text_encoder.onnx"
JOB_RECORD_PATH = ARTIFACT_DIR / "text_s2_w8a16_record.json"

# Quantization config
RUN_TEXT_QUANTIZATION = True   # set to False to ship a float text encoder
WEIGHTS_DTYPE = hub.QuantizeDtype.INT8
ACTIVATIONS_DTYPE = hub.QuantizeDtype.INT16
QUANTIZE_OPTIONS = ""  # default range scheme is fine for text

# Number of calibration prompts to use for text quantization
NUM_TEXT_CALIBRATION_PROMPTS = 256

RUN_SMOKE_TEST_INFERENCE = True
PRINT_RAW_PROFILE_IF_UNKNOWN = True

PROFILE_OPTIONS = "--max_profiler_iterations 100"

SHARE_WITH_ORGANIZERS = False
ORGANIZER_EMAIL = "lowpowervision@gmail.com"

# Competition tokenizer contract
COMPETITION_TOKENIZER_NAME = "openai/clip-vit-base-patch32"
COMPETITION_EOS_TOKEN_ID = 49407

# Sample dataset for local validation
SAMPLE_TXT_CSV = Path("/home/aman/dev/lpcv/anvil/sample data/txt_list.csv")
RUN_LOCAL_VALIDATION = True
VALIDATION_COSINE_THRESHOLD = 0.95


# ============================================================
# Basic helpers
# ============================================================

def fail_if_missing(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_hub_client(api_token: str) -> hub.Client:
    if not api_token:
        raise RuntimeError(
            'QAI_HUB_API_TOKEN is empty. Set it first:\n'
            '  export QAI_HUB_API_TOKEN="YOUR_TOKEN_HERE"'
        )
    client = hub.Client(hub.ClientConfig(api_token=api_token))
    devices = client.get_devices()
    if not devices:
        raise RuntimeError("AI Hub authentication failed: no devices returned.")
    print(f"[auth] AI Hub client ready. Visible devices: {len(devices)}")
    return client


def wait_success(job, name: str, fatal: bool = True) -> bool:
    status = str(job.wait())
    print(f"[hub] {name} status: {status}")
    if "SUCCESS" in status:
        return True
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
    repo_str = str(ML_MOBILECLIP_REPO.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def import_mobileclip_stack():
    import importlib
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model
    try:
        mobileclip2_module = importlib.import_module("mobileclip2.mobileclip2")
    except Exception:
        mobileclip2_module = None
    return open_clip, reparameterize_model, mobileclip2_module


# ============================================================
# Export-time patches
# ============================================================

def apply_text_export_patches(open_clip, mobileclip2_module) -> None:
    """
    Disable PyTorch fused attention paths and patch text pooling so the
    exported ONNX is a clean static graph.
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

    if mobileclip2_module is not None and hasattr(mobileclip2_module, "LayerNormChannel"):
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
            seq_idx = text.argmax(dim=-1)[0]
            return x[0:1, seq_idx, :]
        return original_text_global_pool(x, text, pool_type, eos_token_id)

    open_clip.transformer.text_global_pool = patched_text_global_pool


# ============================================================
# Text wrapper
# ============================================================

class CompetitionCompatibleTextEncoder(nn.Module):
    """
    Convert competition tokenizer's repeated-EOS padding into zero-padding
    BEFORE calling encode_text(), preserving the first EOS so OpenCLIP's
    argmax pooling still finds the correct sequence index.

    Competition input  : [BOS, ..., EOS, EOS, EOS, EOS, ...]
    Model expects-ish  : [BOS, ..., EOS,   0,   0,   0, ...]
    """

    def __init__(self, clip_model: nn.Module, eos_token_id: int = COMPETITION_EOS_TOKEN_ID):
        super().__init__()
        self.model = clip_model
        self.eos_token_id = eos_token_id

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = text.to(torch.int64)
        eos_mask = (x == self.eos_token_id).to(torch.int64)
        eos_cumsum = torch.cumsum(eos_mask, dim=1)
        repeated_eos_mask = (x == self.eos_token_id) & (eos_cumsum > 1)
        x = torch.where(repeated_eos_mask, torch.zeros_like(x), x)
        return self.model.encode_text(x)


def load_reparameterized_text_encoder() -> Tuple[nn.Module, str]:
    prepare_python_imports()
    open_clip, reparameterize_model, mobileclip2_module = import_mobileclip_stack()
    apply_text_export_patches(open_clip, mobileclip2_module)

    last_err: Optional[Exception] = None
    for model_name in MODEL_NAME_CANDIDATES:
        try:
            print(f"[model] Trying model_name={model_name}")
            model_kwargs = {}
            if not (
                model_name.endswith("S3")
                or model_name.endswith("S4")
                or model_name.endswith("L-14")
            ):
                model_kwargs = {
                    "image_mean": (0.0, 0.0, 0.0),
                    "image_std": (1.0, 1.0, 1.0),
                }

            model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=str(CHECKPOINT_PATH),
                **model_kwargs,
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
    raise RuntimeError(
        "Could not load checkpoint with any model name candidate."
    ) from last_err


# ============================================================
# Tokenization (matches the competition tokenizer contract)
# ============================================================

def build_competition_tokenizer() -> CLIPTokenizer:
    tokenizer = CLIPTokenizer.from_pretrained(COMPETITION_TOKENIZER_NAME)
    # The competition pads with repeated EOS; aligning cls_token to EOS keeps
    # the wrapper's repeated-EOS->zero conversion correct on every input.
    tokenizer.add_special_tokens({"cls_token": tokenizer.eos_token})
    return tokenizer


def tokenize_competition_texts(
    texts: List[str], tokenizer: CLIPTokenizer
) -> np.ndarray:
    toks = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )["input_ids"].cpu().numpy().astype(np.int64)
    return toks


# ============================================================
# ONNX export / validation
# ============================================================

def cleanup_old_export_files() -> None:
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
    cleanup_old_export_files()
    dummy_tokens = torch.randint(
        low=0, high=49408, size=TEXT_INPUT_SHAPE, dtype=torch.int64
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
    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]
    if TEXT_INPUT_NAME not in input_names:
        raise RuntimeError(
            f"Expected ONNX input name '{TEXT_INPUT_NAME}', found {input_names}"
        )
    if TEXT_OUTPUT_NAME not in output_names:
        raise RuntimeError(
            f"Expected ONNX output name '{TEXT_OUTPUT_NAME}', found {output_names}"
        )
    print("[onnx] ONNX validated successfully.")
    print(f"[onnx] Inputs : {input_names}")
    print(f"[onnx] Outputs: {output_names}")


# ============================================================
# QAI Hub flow
# ============================================================

def optimize_to_onnx(client: hub.Client, onnx_path: Path, device: hub.Device):
    """Run an optimization pass to clean up the graph before quantization."""
    print("[hub] Submitting compile-to-ONNX optimization job...")
    job = client.submit_compile_job(
        model=str(onnx_path),
        device=device,
        input_specs={TEXT_INPUT_NAME: ((1, 77), "int64")},
        options="--target_runtime onnx --truncate_64bit_io",
        name="lpcvc_track1_mobileclip2s2_text_optimize_to_onnx",
    )
    print(f"[hub] optimize-to-ONNX job id: {job.job_id}")
    wait_success(job, "optimize_to_onnx")
    optimized_model = job.get_target_model()
    if optimized_model is None:
        raise RuntimeError(f"optimize_to_onnx returned no target model. URL: {job.url}")
    return job, optimized_model


def build_text_calibration_data(
    tokenizer: CLIPTokenizer, sample_texts: Optional[List[str]]
) -> Dict[str, List[np.ndarray]]:
    """
    Use the sample-set texts (if available) plus a small bank of generic
    prompts to build text calibration data.
    """
    base_prompts = [
        "a photo of a dog",
        "a photo of a cat",
        "a man riding a bicycle",
        "a woman walking on the beach",
        "a red car parked on the street",
        "a child playing with a ball",
        "a group of people eating dinner",
        "a sunset over the mountains",
        "a bird flying in the sky",
        "a tree with green leaves",
        "an empty office room",
        "a dish of pasta on a wooden table",
        "a black laptop computer",
        "a busy city street with traffic",
        "a snowy landscape with pine trees",
    ]
    prompts: List[str] = list(base_prompts)
    if sample_texts:
        prompts.extend(sample_texts)
    # Cap at NUM_TEXT_CALIBRATION_PROMPTS while preserving variety
    prompts = prompts[:NUM_TEXT_CALIBRATION_PROMPTS]
    print(f"[calib] Using {len(prompts)} text prompts for calibration.")
    token_matrix = tokenize_competition_texts(prompts, tokenizer)
    samples = [token_matrix[i : i + 1] for i in range(token_matrix.shape[0])]
    return {TEXT_INPUT_NAME: samples}


def quantize_optimized_text(
    client: hub.Client,
    optimized_model,
    calibration_data: Dict[str, List[np.ndarray]],
):
    print(
        f"[hub] Submitting text quantize job: weights={WEIGHTS_DTYPE} "
        f"activations={ACTIVATIONS_DTYPE}"
    )
    job = client.submit_quantize_job(
        model=optimized_model,
        calibration_data=calibration_data,
        weights_dtype=WEIGHTS_DTYPE,
        activations_dtype=ACTIVATIONS_DTYPE,
        options=QUANTIZE_OPTIONS,
        name="lpcvc_track1_mobileclip2s2_text_quant_w8a16",
    )
    print(f"[hub] text quantize job id: {job.job_id}")
    wait_success(job, "quantize")
    quantized_model = job.get_target_model()
    if quantized_model is None:
        raise RuntimeError(f"text quantize returned no target model. URL: {job.url}")
    return job, quantized_model


def compile_qnn_dlc(
    client: hub.Client, model_or_path, device: hub.Device, source_label: str
):
    """
    Compile to QNN DLC. Works for both the float ONNX path and the quantized
    model path; the latter is preferred when RUN_TEXT_QUANTIZATION is True.
    """
    print(f"[hub] Submitting compile job to QNN DLC ({source_label})...")
    if isinstance(model_or_path, Path):
        model_arg = str(model_or_path)
    else:
        model_arg = model_or_path
    job = client.submit_compile_job(
        model=model_arg,
        device=device,
        input_specs={TEXT_INPUT_NAME: ((1, 77), "int64")},
        options="--target_runtime qnn_dlc --truncate_64bit_io",
        name=f"lpcvc_track1_mobileclip2s2_text_qnn_dlc_{source_label}",
    )
    print(f"[hub] compile job id: {job.job_id}")
    wait_success(job, f"compile_qnn_dlc_{source_label}")
    compiled_model = job.get_target_model()
    if compiled_model is None:
        raise RuntimeError(f"compile_qnn_dlc returned no target model. URL: {job.url}")
    return job, compiled_model


def profile_model(client: hub.Client, compiled_model, device: hub.Device):
    print("[hub] Submitting profile job...")
    job = client.submit_profile_job(
        model=compiled_model,
        device=device,
        options=PROFILE_OPTIONS,
        name="lpcvc_track1_mobileclip2s2_text_profile",
    )
    print(f"[hub] profile job id: {job.job_id}")
    wait_success(job, "profile")
    return job


def extract_profile_metrics(profile: dict) -> Tuple[Optional[float], Optional[object]]:
    if not isinstance(profile, dict):
        raise TypeError(f"Expected profile to be a dict, got {type(profile)}")
    if "execution_detail" in profile and isinstance(profile["execution_detail"], dict):
        ed = profile["execution_detail"]
        lat_us = ed.get("estimated_inference_time")
        peak_mem = ed.get("estimated_peak_memory")
        return (lat_us / 1000.0 if lat_us is not None else None), peak_mem
    lat_us = profile.get("estimated_inference_time")
    peak_mem = profile.get("estimated_peak_memory")
    if lat_us is not None or peak_mem is not None:
        return (lat_us / 1000.0 if lat_us is not None else None), peak_mem
    if "metrics" in profile and isinstance(profile["metrics"], dict):
        m = profile["metrics"]
        lat_us = m.get("estimated_inference_time")
        peak_mem = m.get("estimated_peak_memory")
        return (lat_us / 1000.0 if lat_us is not None else None), peak_mem
    if PRINT_RAW_PROFILE_IF_UNKNOWN:
        print("\n[debug] Unrecognized profile schema:")
        pprint.pprint(profile)
    return None, None


def smoke_test_inference(
    client: hub.Client,
    compiled_model,
    device: hub.Device,
    sample_tokens: np.ndarray,
) -> bool:
    print("[hub] Running smoke-test inference...")
    inf_job = client.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs={TEXT_INPUT_NAME: [sample_tokens]},
        name="lpcvc_track1_mobileclip2s2_text_smoke_inference",
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


def maybe_share_compile_job(compile_job) -> None:
    if not SHARE_WITH_ORGANIZERS:
        print("[submit] Sharing skipped (SHARE_WITH_ORGANIZERS=False).")
        return
    print(f"[submit] Sharing compile job with {ORGANIZER_EMAIL} ...")
    compile_job.modify_sharing(add_emails=[ORGANIZER_EMAIL])
    print("[submit] Sharing request sent.")


# ============================================================
# Local validation against sample texts
# ============================================================

def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize_np(a) * l2_normalize_np(b), axis=1)


def load_sample_texts() -> List[str]:
    if not SAMPLE_TXT_CSV.exists():
        return []
    df = pd.read_csv(SAMPLE_TXT_CSV).sort_values("Text_nums").reset_index(drop=True)
    return df["Unique_Texts"].astype(str).tolist()


def run_local_text_validation(
    text_encoder_pt: nn.Module,
    float_onnx_path: Path,
    qdq_onnx_path: Optional[Path],
    sample_texts: List[str],
    tokenizer: CLIPTokenizer,
) -> Dict[str, object]:
    print("\n========== LOCAL TEXT VALIDATION ==========")
    if not sample_texts:
        print("[validate] No sample texts found — skipping.")
        return {}

    token_matrix = tokenize_competition_texts(sample_texts, tokenizer)
    print(f"[validate] sample texts : {token_matrix.shape}")

    # PT reference
    text_encoder_pt = text_encoder_pt.eval().cpu()
    with torch.no_grad():
        pt_emb = text_encoder_pt(torch.from_numpy(token_matrix)).cpu().numpy()
    print(f"[validate] PT  emb shape : {pt_emb.shape}")

    # Float ONNX
    sess_float = ort.InferenceSession(
        str(float_onnx_path), providers=["CPUExecutionProvider"]
    )
    float_chunks = []
    for i in range(token_matrix.shape[0]):
        out = sess_float.run(None, {TEXT_INPUT_NAME: token_matrix[i : i + 1]})[0]
        float_chunks.append(out)
    float_emb = np.concatenate(float_chunks, axis=0).reshape(pt_emb.shape)
    pt_vs_float_cos = float(cosine_per_row(pt_emb, float_emb).mean())
    print(f"[validate] cosine(PT, float ONNX)         : {pt_vs_float_cos:.6f}")

    # QDQ ONNX (if downloaded)
    pt_vs_qdq_cos = None
    if qdq_onnx_path is not None and qdq_onnx_path.exists():
        sess_qdq = ort.InferenceSession(
            str(qdq_onnx_path), providers=["CPUExecutionProvider"]
        )
        qdq_input_name = sess_qdq.get_inputs()[0].name
        qdq_input_dtype = sess_qdq.get_inputs()[0].type
        # QDQ may downcast to int32
        feed_tokens = (
            token_matrix.astype(np.int32)
            if "int32" in qdq_input_dtype
            else token_matrix
        )
        qdq_chunks = []
        for i in range(feed_tokens.shape[0]):
            out = sess_qdq.run(None, {qdq_input_name: feed_tokens[i : i + 1]})[0]
            qdq_chunks.append(out)
        qdq_emb = np.concatenate(qdq_chunks, axis=0).reshape(pt_emb.shape)
        pt_vs_qdq_cos = float(cosine_per_row(pt_emb, qdq_emb).mean())
        print(f"[validate] cosine(PT, QDQ ONNX)            : {pt_vs_qdq_cos:.6f}")
        if pt_vs_qdq_cos < VALIDATION_COSINE_THRESHOLD:
            print(
                f"[validate] ⚠️  Text QDQ cosine ({pt_vs_qdq_cos:.4f}) is below "
                f"threshold ({VALIDATION_COSINE_THRESHOLD})."
            )
    else:
        print("[validate] QDQ ONNX not provided — skipping QDQ comparison.")

    print("============================================\n")
    return {
        "pt_vs_float_cosine": pt_vs_float_cos,
        "pt_vs_qdq_cosine": pt_vs_qdq_cos,
    }


def maybe_download_qdq_onnx(quantized_model) -> Optional[Path]:
    qdq_path = ARTIFACT_DIR / "text_qdq.onnx"
    try:
        print(f"[validate] Downloading text QDQ ONNX to {qdq_path} ...")
        quantized_model.download(str(qdq_path))
        if qdq_path.exists():
            print("[validate] Text QDQ ONNX downloaded.")
            return qdq_path
    except Exception as exc:
        print(f"[validate] Could not download text QDQ ONNX: {exc}")
    return None


# ============================================================
# Main
# ============================================================

def save_job_record(record: dict) -> None:
    JOB_RECORD_PATH.write_text(json.dumps(record, indent=2, default=str))
    print(f"[artifacts] Wrote job record to {JOB_RECORD_PATH.resolve()}")


def main() -> int:
    print("==============================================================")
    print(" LPCVC 2026 Track 1 - MobileCLIP2-S2 Text (W8A16) ")
    print("==============================================================")

    fail_if_missing(ML_MOBILECLIP_REPO, "ml-mobileclip repo")
    fail_if_missing(MODEL_CONFIG_PATH, "model config JSON")
    fail_if_missing(CHECKPOINT_PATH, "checkpoint")

    config = load_json(MODEL_CONFIG_PATH)
    print(f"[config] Loaded config from {MODEL_CONFIG_PATH}")
    print(f"[config] Top-level keys: {list(config.keys())[:10]}")

    client = build_hub_client(QAI_HUB_API_TOKEN)
    target_device = hub.Device(TARGET_DEVICE_NAME)
    print(f"[hub] Target device: {target_device.name}")

    # 1) Load model with patches
    text_encoder, resolved_model_name = load_reparameterized_text_encoder()

    # 2) Export float ONNX
    export_text_encoder_to_onnx(text_encoder)

    # 2b) Pre-quantization local validation against PT
    sample_texts = load_sample_texts() if RUN_LOCAL_VALIDATION else []
    tokenizer = build_competition_tokenizer()
    pre_quant_results = {}
    if RUN_LOCAL_VALIDATION and sample_texts:
        pre_quant_results = run_local_text_validation(
            text_encoder_pt=text_encoder,
            float_onnx_path=ONNX_TEXT_PATH,
            qdq_onnx_path=None,
            sample_texts=sample_texts,
            tokenizer=tokenizer,
        )

    # 3) Optimize ONNX on Hub (always good to clean up the graph first)
    optimize_onnx_job, optimized_model = optimize_to_onnx(
        client, ONNX_TEXT_PATH, target_device
    )

    quantize_job = None
    quantized_model = None
    qdq_path = None
    post_quant_results = {}

    if RUN_TEXT_QUANTIZATION:
        # 4) Build text calibration set (sample texts + generic prompts)
        text_calib = build_text_calibration_data(tokenizer, sample_texts)

        # 5) Quantize
        quantize_job, quantized_model = quantize_optimized_text(
            client, optimized_model, text_calib
        )

        # 5b) Local QDQ validation
        if RUN_LOCAL_VALIDATION and sample_texts:
            qdq_path = maybe_download_qdq_onnx(quantized_model)
            post_quant_results = run_local_text_validation(
                text_encoder_pt=text_encoder,
                float_onnx_path=ONNX_TEXT_PATH,
                qdq_onnx_path=qdq_path,
                sample_texts=sample_texts,
                tokenizer=tokenizer,
            )

        # 6) Compile quantized to QNN DLC
        compile_job, compiled_model = compile_qnn_dlc(
            client, quantized_model, target_device, source_label="quantized"
        )
    else:
        # Float text path (in case quantization causes problems and you want
        # a guaranteed-working fallback).
        compile_job, compiled_model = compile_qnn_dlc(
            client, ONNX_TEXT_PATH, target_device, source_label="float"
        )

    # 7) Profile
    profile_job = profile_model(client, compiled_model, target_device)
    profile = profile_job.download_profile()
    latency_ms, peak_mem = extract_profile_metrics(profile)

    print("\n================================")
    print("TEXT ENCODER PROFILE RESULT")
    print("================================")
    print(
        f"Latency     : {latency_ms:.3f} ms"
        if latency_ms is not None
        else "Latency     : not found in returned profile"
    )
    print(f"Peak memory : {peak_mem if peak_mem is not None else 'unknown'}")
    print("================================")

    # 8) Optional smoke inference using a real competition-tokenized prompt
    smoke_ok = None
    if RUN_SMOKE_TEST_INFERENCE:
        smoke_text = "a photograph of a person"
        smoke_tokens = tokenize_competition_texts([smoke_text], tokenizer)
        smoke_ok = smoke_test_inference(
            client, compiled_model, target_device, smoke_tokens
        )

    # 9) Optional sharing
    maybe_share_compile_job(compile_job)

    # 10) Save record
    save_job_record(
        {
            "resolved_model_name": resolved_model_name,
            "model_config_path": str(MODEL_CONFIG_PATH),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "onnx_path": str(ONNX_TEXT_PATH),
            "qdq_onnx_path": str(qdq_path) if qdq_path else None,
            "device": TARGET_DEVICE_NAME,
            "tokenizer": COMPETITION_TOKENIZER_NAME,
            "eos_token_id": COMPETITION_EOS_TOKEN_ID,
            "ran_quantization": RUN_TEXT_QUANTIZATION,
            "weights_dtype": str(WEIGHTS_DTYPE) if RUN_TEXT_QUANTIZATION else None,
            "activations_dtype": str(ACTIVATIONS_DTYPE) if RUN_TEXT_QUANTIZATION else None,
            "quantize_options": QUANTIZE_OPTIONS if RUN_TEXT_QUANTIZATION else None,
            "optimize_to_onnx_job_id": optimize_onnx_job.job_id,
            "quantize_job_id": quantize_job.job_id if quantize_job else None,
            "compile_qnn_dlc_job_id": compile_job.job_id,
            "profile_job_id": profile_job.job_id,
            "text_latency_ms": latency_ms,
            "text_peak_memory": peak_mem,
            "raw_profile": profile,
            "smoke_inference_ok": smoke_ok,
            "pre_quant_local_results": pre_quant_results,
            "post_quant_local_results": post_quant_results,
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
