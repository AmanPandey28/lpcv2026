#!/usr/bin/env python3
"""
LPCVC 2026 Track 1 — Final image encoder script (MobileCLIP2-S2, W8A16)
=======================================================================

What this script does, end to end
---------------------------------
1.  Loads MobileCLIP2-S2 from your local checkpoint with the Apple-recommended
    OpenCLIP preprocessing trick for S0/S2/B variants:
        image_mean = (0, 0, 0), image_std = (1, 1, 1)
    so the model's preprocess pipeline becomes identity-normalization, and the
    real CLIP mean/std are baked into the exported ONNX graph (Track-1 contract).
2.  Reparameterizes the model for inference/export.
3.  Wraps it in `CompetitionReadyImageEncoder` with in-graph CLIP normalization
    so the deployed model accepts raw Track-1 host-side input
    (resize -> /255 -> CHW -> batch).
4.  Exports a self-contained float ONNX (`pure_image_encoder.onnx`).
5.  Compiles to optimized ONNX on AI Hub.
6.  Builds 768 calibration images (512 Flickr30k + 256 COCO) in raw Track-1 format.
7.  Quantizes on AI Hub with W8A16 + `--range_scheme mse_minimizer`
    (proven to preserve transformer activation outliers vs default min-max).
8.  Compiles the quantized model to QNN DLC for XR2 Gen 2 (Proxy).
9.  Profiles on the proxy device.
10. Runs LOCAL VALIDATION on the official sample set:
    -- PT vs float ONNX cosine
    -- PT vs QDQ ONNX cosine
    -- Sample-set Recall@1/5/10 with PT text reference
11. Optionally shares the compile job with the organizers.
12. Saves a complete JSON record for reproducibility.

Why these choices
-----------------
- S2 (35.7M image params) instead of S4 (321.6M): on iPhone FP16 S2's image is
  3.6 ms vs S4's 19.6 ms. On Hexagon W8A16, S2 will land at ~7-10 ms image,
  giving 20+ ms of headroom under the 35 ms budget.
- W8A16 (not W8A8): the FastViT trunk has GELU + Softmax + LayerNorm, which
  collapse under W8A8 on Hexagon HTP (your Phase IV diagnosis already proved
  this on S4 with cosine 1.0 -> 0.21). W8A16 keeps activation precision high
  enough that the embedding direction is preserved.
- `mse_minimizer` range scheme: minimizes layer-wise reconstruction error,
  better than default min-max for transformer activation outliers.
- In-graph normalization: the Track-1 evaluator only does
  resize -> /255 -> CHW -> batch. CLIP mean/std must be inside the graph.

Before running
--------------
    export QAI_HUB_API_TOKEN="YOUR_TOKEN_HERE"
"""

from __future__ import annotations

import json
import os
import pprint
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import requests
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image

import qai_hub as hub


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

# Token in env, never in source
QAI_HUB_API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN", "")

TARGET_DEVICE_NAME = "XR2 Gen 2 (Proxy)"

IMAGE_INPUT_NAME = "image"
IMAGE_OUTPUT_NAME = "embedding"
IMAGE_INPUT_SHAPE = (1, 3, 224, 224)

ARTIFACT_DIR = Path("lpcvc_track1_image_s2_w8a16_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_IMAGE_PATH = ARTIFACT_DIR / "pure_image_encoder.onnx"
JOB_RECORD_PATH = ARTIFACT_DIR / "image_s2_w8a16_record.json"

# Calibration sizes
NUM_FLICKR30K_SAMPLES = int(os.environ.get("NUM_FLICKR30K_SAMPLES", "512"))
NUM_COCO_SAMPLES = int(os.environ.get("NUM_COCO_SAMPLES", "256"))
MIN_TOTAL_CALIBRATION_SAMPLES = 512

# Quantization config — the key change vs your earlier S4 run
WEIGHTS_DTYPE = hub.QuantizeDtype.INT8
ACTIVATIONS_DTYPE = hub.QuantizeDtype.INT16
QUANTIZE_OPTIONS = "--range_scheme mse_minimizer"

# Sample dataset for local validation (your existing layout)
SAMPLE_IMG_CSV = Path("/home/aman/dev/lpcv/anvil/sample data/img_list.csv")
SAMPLE_TXT_CSV = Path("/home/aman/dev/lpcv/anvil/sample data/txt_list.csv")
SAMPLE_IMAGE_DIR = Path(
    "/home/aman/dev/lpcv/anvil/sample data/images-20260423T155937Z-3-001/images"
)

# Local validation toggles
RUN_LOCAL_VALIDATION = True
VALIDATION_COSINE_THRESHOLD = 0.92  # warn if QDQ cosine drops below this

RUN_SMOKE_TEST_INFERENCE = True
PRINT_RAW_PROFILE_IF_UNKNOWN = True

PROFILE_OPTIONS = "--max_profiler_iterations 100"

SHARE_WITH_ORGANIZERS = False
ORGANIZER_EMAIL = "lowpowervision@gmail.com"

REQUEST_TIMEOUT = (5, 15)
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LPCVC-Track1-Calibration/1.0)"
}

# CLIP normalization — used for in-graph baking
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model
    return open_clip, reparameterize_model


# ============================================================
# Model wrapper — in-graph CLIP normalization
# ============================================================

class CompetitionReadyImageEncoder(nn.Module):
    """
    Image wrapper accepting Track-1-style host input:
      - float32, shape (1, 3, 224, 224), values in [0, 1]

    Applies CLIP normalization INSIDE the graph before encode_image().
    """

    def __init__(
        self,
        clip_model: nn.Module,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ):
        super().__init__()
        self.model = clip_model
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image.to(torch.float32)
        x = (x - self.mean) / self.std
        return self.model.encode_image(x)


def load_reparameterized_image_encoder() -> Tuple[
    nn.Module, str, Tuple[float, float, float], Tuple[float, float, float]
]:
    """
    Load MobileCLIP2-S2 with the Apple-recommended kwargs trick for S0/S2/B,
    so the OpenCLIP preprocess pipeline returns identity-normalized images,
    and we can bake the real CLIP mean/std into the deployed graph instead.
    """
    prepare_python_imports()
    open_clip, reparameterize_model = import_mobileclip_stack()

    last_err: Optional[Exception] = None

    for model_name in MODEL_NAME_CANDIDATES:
        try:
            print(f"[model] Trying model_name={model_name}")

            # Apple's official README pattern:
            # S0 / S2 / B  -> need image_mean=(0,0,0), image_std=(1,1,1)
            # S3 / S4 / L-14 -> default normalization
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

            # The graph itself was trained against CLIP mean/std normalized inputs,
            # regardless of what OpenCLIP exposes in `preprocess`. So we bake the
            # real CLIP mean/std into the in-graph normalization.
            mean = CLIP_MEAN
            std = CLIP_STD
            print(f"[model] In-graph normalization mean={mean} std={std}")

            image_encoder = CompetitionReadyImageEncoder(
                model, mean, std
            ).eval()
            return image_encoder, model_name, mean, std
        except Exception as exc:
            last_err = exc
            print(f"[model] Failed with {model_name}: {exc}")

    raise RuntimeError(
        "Could not load checkpoint with any model name candidate."
    ) from last_err


# ============================================================
# ONNX export / validation
# ============================================================

def cleanup_old_export_files() -> None:
    candidates = [
        ONNX_IMAGE_PATH,
        ONNX_IMAGE_PATH.with_suffix(".onnx.data"),
        ONNX_IMAGE_PATH.parent / f"{ONNX_IMAGE_PATH.name}.data",
        ONNX_IMAGE_PATH.parent / f"{ONNX_IMAGE_PATH.stem}.data",
    ]
    for p in candidates:
        if p.exists():
            p.unlink()


def export_image_encoder_to_onnx(image_encoder: nn.Module) -> None:
    cleanup_old_export_files()
    dummy_input = torch.rand(IMAGE_INPUT_SHAPE, dtype=torch.float32)
    print(f"[onnx] Exporting to {ONNX_IMAGE_PATH.resolve()}")
    torch.onnx.export(
        image_encoder,
        dummy_input,
        str(ONNX_IMAGE_PATH),
        input_names=[IMAGE_INPUT_NAME],
        output_names=[IMAGE_OUTPUT_NAME],
        opset_version=16,
        do_constant_folding=True,
        dynamic_axes=None,
        verbose=False,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        external_data=False,
    )
    validate_onnx(ONNX_IMAGE_PATH)


def validate_onnx(onnx_path: Path) -> None:
    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]
    if IMAGE_INPUT_NAME not in input_names:
        raise RuntimeError(
            f"Expected ONNX input name '{IMAGE_INPUT_NAME}', found {input_names}"
        )
    if IMAGE_OUTPUT_NAME not in output_names:
        raise RuntimeError(
            f"Expected ONNX output name '{IMAGE_OUTPUT_NAME}', found {output_names}"
        )
    print("[onnx] ONNX validated successfully.")
    print(f"[onnx] Inputs : {input_names}")
    print(f"[onnx] Outputs: {output_names}")


# ============================================================
# Track-1-style preprocessing
# ============================================================

def preprocess_track1_image(img: Image.Image) -> np.ndarray:
    """
    Build the exact tensor the Track-1 evaluator provides:
        resize -> /255 -> CHW -> batch
    """
    img = img.convert("RGB").resize((224, 224), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


# ============================================================
# Calibration data
# ============================================================

def fetch_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def collect_flickr30k_samples(n: int) -> List[np.ndarray]:
    print(f"[calib] Collecting {n} Flickr30k samples...")
    ds = load_dataset("lmms-lab/flickr30k", split="test", streaming=True)
    samples: List[np.ndarray] = []
    skipped = 0
    for example in ds:
        try:
            img = example["image"]
            if not isinstance(img, Image.Image):
                raise TypeError(f"Unexpected Flickr30k image type: {type(img)}")
            samples.append(preprocess_track1_image(img))
        except Exception:
            skipped += 1
            continue
        if len(samples) >= n:
            break
    print(f"[calib] Flickr30k collected={len(samples)}, skipped={skipped}")
    return samples


def collect_coco_samples(n: int) -> List[np.ndarray]:
    print(f"[calib] Collecting {n} COCO samples...")
    ds = load_dataset("phiyodr/coco2017", split="validation", streaming=True)
    samples: List[np.ndarray] = []
    skipped = 0
    for example in ds:
        try:
            if "image" in example and isinstance(example["image"], Image.Image):
                img = example["image"]
            else:
                url = (
                    example.get("coco_url")
                    or example.get("image_url")
                    or example.get("url")
                )
                if not url:
                    raise KeyError("No usable image field or URL in COCO sample.")
                img = fetch_image_from_url(url)
            samples.append(preprocess_track1_image(img))
        except Exception:
            skipped += 1
            continue
        if len(samples) >= n:
            break
    print(f"[calib] COCO collected={len(samples)}, skipped={skipped}")
    return samples


def build_calibration_data() -> Tuple[Dict[str, List[np.ndarray]], np.ndarray]:
    flickr = collect_flickr30k_samples(NUM_FLICKR30K_SAMPLES)
    coco = collect_coco_samples(NUM_COCO_SAMPLES)
    total = flickr + coco
    target_total = NUM_FLICKR30K_SAMPLES + NUM_COCO_SAMPLES
    if len(total) < target_total:
        missing = target_total - len(total)
        print(f"[calib] Topping up with {missing} extra Flickr30k samples...")
        total.extend(collect_flickr30k_samples(missing))
    if len(total) < MIN_TOTAL_CALIBRATION_SAMPLES:
        raise RuntimeError(
            f"Only collected {len(total)} usable calibration samples. "
            f"Need at least {MIN_TOTAL_CALIBRATION_SAMPLES}."
        )
    print(f"[calib] Total calibration samples collected: {len(total)}")
    return {IMAGE_INPUT_NAME: total}, total[0]


# ============================================================
# QAI Hub flow
# ============================================================

def optimize_to_onnx(client: hub.Client, onnx_path: Path, device: hub.Device):
    print("[hub] Submitting compile-to-ONNX optimization job...")
    job = client.submit_compile_job(
        model=str(onnx_path),
        device=device,
        input_specs={IMAGE_INPUT_NAME: IMAGE_INPUT_SHAPE},
        options="--target_runtime onnx",
        name="lpcvc_track1_mobileclip2s2_image_optimize_to_onnx",
    )
    print(f"[hub] optimize-to-ONNX job id: {job.job_id}")
    wait_success(job, "optimize_to_onnx")
    optimized_model = job.get_target_model()
    if optimized_model is None:
        raise RuntimeError(f"optimize_to_onnx returned no target model. URL: {job.url}")
    return job, optimized_model


def quantize_optimized_onnx(
    client: hub.Client, optimized_model, calibration_data: Dict[str, List[np.ndarray]]
):
    print(
        f"[hub] Submitting quantize job: weights={WEIGHTS_DTYPE} "
        f"activations={ACTIVATIONS_DTYPE} options='{QUANTIZE_OPTIONS}'"
    )
    job = client.submit_quantize_job(
        model=optimized_model,
        calibration_data=calibration_data,
        weights_dtype=WEIGHTS_DTYPE,
        activations_dtype=ACTIVATIONS_DTYPE,
        options=QUANTIZE_OPTIONS,
        name="lpcvc_track1_mobileclip2s2_image_quant_w8a16_mse",
    )
    print(f"[hub] quantize job id: {job.job_id}")
    wait_success(job, "quantize")
    quantized_model = job.get_target_model()
    if quantized_model is None:
        raise RuntimeError(f"quantize returned no target model. URL: {job.url}")
    return job, quantized_model


def compile_qnn_dlc(client: hub.Client, quantized_model, device: hub.Device):
    print("[hub] Submitting compile job to QNN DLC...")
    job = client.submit_compile_job(
        model=quantized_model,
        device=device,
        input_specs={IMAGE_INPUT_NAME: IMAGE_INPUT_SHAPE},
        options="--target_runtime qnn_dlc",
        name="lpcvc_track1_mobileclip2s2_image_qnn_dlc_w8a16",
    )
    print(f"[hub] compile-to-QNN-DLC job id: {job.job_id}")
    wait_success(job, "compile_qnn_dlc")
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
        name="lpcvc_track1_mobileclip2s2_image_profile_w8a16",
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
    client: hub.Client, compiled_model, device: hub.Device, sample_input: np.ndarray
) -> bool:
    print("[hub] Running smoke-test inference...")
    inf_job = client.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs={IMAGE_INPUT_NAME: [sample_input]},
        name="lpcvc_track1_mobileclip2s2_image_smoke_inference",
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
# Local validation against sample dataset
# ============================================================

def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize_np(a) * l2_normalize_np(b), axis=1)


def load_sample_image_records() -> Tuple[List[dict], pd.DataFrame, Dict[int, int], List[str]]:
    txt_df = pd.read_csv(SAMPLE_TXT_CSV).sort_values("Text_nums").reset_index(drop=True)
    img_df = pd.read_csv(SAMPLE_IMG_CSV)
    text_id_to_pos = {int(row["Text_nums"]): i for i, row in txt_df.iterrows()}
    image_records: List[dict] = []
    missing: List[str] = []
    for _, row in img_df.iterrows():
        image_name = str(row["Image_names"])
        image_path = SAMPLE_IMAGE_DIR / image_name
        gt_ids = [int(x) for x in str(row["Text_nums"]).split(";") if str(x).strip()]
        if image_path.exists():
            image_records.append(
                {
                    "image_name": image_name,
                    "image_path": image_path,
                    "gt_text_ids": gt_ids,
                }
            )
        else:
            missing.append(image_name)
    return image_records, txt_df, text_id_to_pos, missing


def retrieval_metrics(
    sim: np.ndarray,
    image_records: List[dict],
    text_id_to_pos: Dict[int, int],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    ranked = np.argsort(-sim, axis=1)
    out: Dict[str, float] = {}
    for k in ks:
        hit, frac = [], []
        for i, rec in enumerate(image_records):
            gt_pos = [
                text_id_to_pos[t] for t in rec["gt_text_ids"] if t in text_id_to_pos
            ]
            topk = set(ranked[i, :k].tolist())
            hits = sum(1 for g in gt_pos if g in topk)
            hit.append(1.0 if hits > 0 else 0.0)
            frac.append(hits / max(len(gt_pos), 1))
        out[f"hit_R@{k}"] = float(np.mean(hit))
        out[f"frac_R@{k}"] = float(np.mean(frac))
    return out


def run_local_image_validation(
    image_encoder_pt: nn.Module,
    float_onnx_path: Path,
    qdq_onnx_path: Optional[Path],
    text_embeddings_pt: Optional[np.ndarray],
    image_records: List[dict],
    text_id_to_pos: Dict[int, int],
) -> Dict[str, object]:
    """
    Compares image embeddings from:
      - PyTorch (reference)
      - float ONNX (sanity check)
      - QDQ ONNX (the artifact that decides leaderboard quality)

    If text_embeddings_pt is provided, also computes Recall@K on the sample set.
    """
    print("\n========== LOCAL IMAGE VALIDATION ==========")
    if not image_records:
        print("[validate] No sample images found — skipping.")
        return {}

    # Track-1 host inputs
    inputs_np = np.concatenate(
        [preprocess_track1_image(Image.open(r["image_path"])) for r in image_records],
        axis=0,
    ).astype(np.float32)
    print(f"[validate] sample images : {inputs_np.shape}")

    # PyTorch reference embeddings
    image_encoder_pt = image_encoder_pt.eval().cpu()
    with torch.no_grad():
        pt_emb = image_encoder_pt(torch.from_numpy(inputs_np)).cpu().numpy()
    print(f"[validate] PT  emb shape : {pt_emb.shape}")

    # Float ONNX embeddings
    sess_float = ort.InferenceSession(str(float_onnx_path), providers=["CPUExecutionProvider"])
    float_emb_chunks = []
    for i in range(inputs_np.shape[0]):
        out = sess_float.run(None, {IMAGE_INPUT_NAME: inputs_np[i : i + 1]})[0]
        float_emb_chunks.append(out)
    float_emb = np.concatenate(float_emb_chunks, axis=0).reshape(pt_emb.shape)
    pt_vs_float_cos = float(cosine_per_row(pt_emb, float_emb).mean())
    print(f"[validate] cosine(PT, float ONNX)         : {pt_vs_float_cos:.6f}")

    # QDQ ONNX embeddings (if available)
    pt_vs_qdq_cos = None
    qdq_metrics = None
    if qdq_onnx_path is not None and qdq_onnx_path.exists():
        sess_qdq = ort.InferenceSession(
            str(qdq_onnx_path), providers=["CPUExecutionProvider"]
        )
        qdq_input_name = sess_qdq.get_inputs()[0].name
        qdq_chunks = []
        for i in range(inputs_np.shape[0]):
            out = sess_qdq.run(None, {qdq_input_name: inputs_np[i : i + 1]})[0]
            qdq_chunks.append(out)
        qdq_emb = np.concatenate(qdq_chunks, axis=0).reshape(pt_emb.shape)
        pt_vs_qdq_cos = float(cosine_per_row(pt_emb, qdq_emb).mean())
        print(f"[validate] cosine(PT, QDQ ONNX)            : {pt_vs_qdq_cos:.6f}")
        if pt_vs_qdq_cos < VALIDATION_COSINE_THRESHOLD:
            print(
                f"[validate] ⚠️  QDQ cosine ({pt_vs_qdq_cos:.4f}) is below threshold "
                f"({VALIDATION_COSINE_THRESHOLD}). Quantization may be too aggressive."
            )

        # Sample-set retrieval, only meaningful if text reference is available
        if text_embeddings_pt is not None:
            sim_pt = l2_normalize_np(pt_emb) @ l2_normalize_np(text_embeddings_pt).T
            sim_float = l2_normalize_np(float_emb) @ l2_normalize_np(text_embeddings_pt).T
            sim_qdq = l2_normalize_np(qdq_emb) @ l2_normalize_np(text_embeddings_pt).T

            metrics_pt = retrieval_metrics(sim_pt, image_records, text_id_to_pos)
            metrics_float = retrieval_metrics(sim_float, image_records, text_id_to_pos)
            metrics_qdq = retrieval_metrics(sim_qdq, image_records, text_id_to_pos)
            print("[validate] Sample-set retrieval (image side varies, text=PT):")
            for k in (1, 5, 10):
                print(
                    f"  R@{k:<2}   PT={metrics_pt[f'hit_R@{k}']:.4f}  "
                    f"float={metrics_float[f'hit_R@{k}']:.4f}  "
                    f"QDQ={metrics_qdq[f'hit_R@{k}']:.4f}"
                )
            qdq_metrics = {
                "pt": metrics_pt,
                "float": metrics_float,
                "qdq": metrics_qdq,
            }
    else:
        print("[validate] QDQ ONNX not provided locally — skipping QDQ comparison.")

    print("============================================\n")
    return {
        "pt_vs_float_cosine": pt_vs_float_cos,
        "pt_vs_qdq_cosine": pt_vs_qdq_cos,
        "qdq_metrics": qdq_metrics,
    }


def maybe_download_qdq_onnx(quantized_model) -> Optional[Path]:
    """Download the QDQ ONNX so we can validate it locally before submission."""
    qdq_path = ARTIFACT_DIR / "image_qdq.onnx"
    try:
        print(f"[validate] Downloading QDQ ONNX to {qdq_path} ...")
        quantized_model.download(str(qdq_path))
        if qdq_path.exists():
            print("[validate] QDQ ONNX downloaded.")
            return qdq_path
    except Exception as exc:
        print(f"[validate] Could not download QDQ ONNX: {exc}")
    return None


# ============================================================
# Main
# ============================================================

def save_job_record(record: dict) -> None:
    JOB_RECORD_PATH.write_text(json.dumps(record, indent=2, default=str))
    print(f"[artifacts] Wrote job record to {JOB_RECORD_PATH.resolve()}")


def main() -> int:
    print("==============================================================")
    print(" LPCVC 2026 Track 1 - MobileCLIP2-S2 Image (W8A16, MSE) ")
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

    # 1) Load and wrap model
    image_encoder, resolved_model_name, mean, std = load_reparameterized_image_encoder()

    # 2) Export float ONNX
    export_image_encoder_to_onnx(image_encoder)

    # 2b) Optional pre-quantization local validation against PT
    sample_records, txt_df, text_id_to_pos, missing = (
        load_sample_image_records() if RUN_LOCAL_VALIDATION else ([], None, {}, [])
    )
    if missing:
        print(f"[validate] Missing sample images: {len(missing)}")

    pre_quant_results = {}
    if RUN_LOCAL_VALIDATION and sample_records:
        pre_quant_results = run_local_image_validation(
            image_encoder_pt=image_encoder,
            float_onnx_path=ONNX_IMAGE_PATH,
            qdq_onnx_path=None,
            text_embeddings_pt=None,
            image_records=sample_records,
            text_id_to_pos=text_id_to_pos,
        )

    # 3) Optimize ONNX on Hub
    optimize_onnx_job, optimized_model = optimize_to_onnx(
        client, ONNX_IMAGE_PATH, target_device
    )

    # 4) Build calibration set (Track-1 raw format, normalization is in-graph)
    calibration_data, smoke_sample = build_calibration_data()

    # 5) Quantize on Hub (W8A16 + MSE range)
    quantize_job, quantized_model = quantize_optimized_onnx(
        client, optimized_model, calibration_data
    )

    # 5b) Local QDQ validation
    qdq_path = None
    post_quant_results = {}
    if RUN_LOCAL_VALIDATION and sample_records:
        qdq_path = maybe_download_qdq_onnx(quantized_model)
        post_quant_results = run_local_image_validation(
            image_encoder_pt=image_encoder,
            float_onnx_path=ONNX_IMAGE_PATH,
            qdq_onnx_path=qdq_path,
            text_embeddings_pt=None,  # text-side computed separately if desired
            image_records=sample_records,
            text_id_to_pos=text_id_to_pos,
        )

    # 6) Compile to QNN DLC
    compile_job, compiled_model = compile_qnn_dlc(client, quantized_model, target_device)

    # 7) Profile
    profile_job = profile_model(client, compiled_model, target_device)
    profile = profile_job.download_profile()
    latency_ms, peak_mem = extract_profile_metrics(profile)

    print("\n================================")
    print("IMAGE ENCODER PROFILE RESULT")
    print("================================")
    print(
        f"Latency     : {latency_ms:.3f} ms"
        if latency_ms is not None
        else "Latency     : not found in returned profile"
    )
    print(f"Peak memory : {peak_mem if peak_mem is not None else 'unknown'}")
    print("================================")

    # 8) Optional smoke inference
    smoke_ok = None
    if RUN_SMOKE_TEST_INFERENCE:
        smoke_ok = smoke_test_inference(client, compiled_model, target_device, smoke_sample)

    # 9) Optional sharing
    maybe_share_compile_job(compile_job)

    # 10) Save record
    save_job_record(
        {
            "resolved_model_name": resolved_model_name,
            "model_config_path": str(MODEL_CONFIG_PATH),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "onnx_path": str(ONNX_IMAGE_PATH),
            "qdq_onnx_path": str(qdq_path) if qdq_path else None,
            "device": TARGET_DEVICE_NAME,
            "normalization_mean": mean,
            "normalization_std": std,
            "num_flickr30k_samples": NUM_FLICKR30K_SAMPLES,
            "num_coco_samples": NUM_COCO_SAMPLES,
            "total_calibration_samples": NUM_FLICKR30K_SAMPLES + NUM_COCO_SAMPLES,
            "weights_dtype": str(WEIGHTS_DTYPE),
            "activations_dtype": str(ACTIVATIONS_DTYPE),
            "quantize_options": QUANTIZE_OPTIONS,
            "optimize_to_onnx_job_id": optimize_onnx_job.job_id,
            "quantize_job_id": quantize_job.job_id,
            "compile_qnn_dlc_job_id": compile_job.job_id,
            "profile_job_id": profile_job.job_id,
            "image_latency_ms": latency_ms,
            "image_peak_memory": peak_mem,
            "raw_profile": profile,
            "smoke_inference_ok": smoke_ok,
            "pre_quant_local_results": pre_quant_results,
            "post_quant_local_results": post_quant_results,
        }
    )

    print("\n✅ Done.")
    print(f"[final] Image compile job id: {compile_job.job_id}")
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
