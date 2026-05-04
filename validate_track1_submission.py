#!/usr/bin/env python3
"""
LPCVC 2026 Track 1 — Joint sample-set retrieval validator
==========================================================

What this script does
---------------------
Given the compile job IDs (or AI Hub model IDs) for your image and text DLCs,
this script:

  1. Loads the official sample set (img_list.csv + txt_list.csv + images dir).
  2. Tokenizes texts using the competition tokenizer (openai/clip-vit-base-patch32).
  3. Runs the compiled image DLC on AI Hub against all sample images.
  4. Runs the compiled text DLC on AI Hub against all sample texts.
  5. Computes cosine similarity and Recall@1 / @5 / @10.
  6. Optionally compares against the pure PyTorch reference for sanity.
  7. Saves a JSON record of everything.

This is the script you run BEFORE submitting to the leaderboard, to know
roughly what your real leaderboard score will look like.

Usage
-----
Edit IMAGE_COMPILE_JOB_ID and TEXT_COMPILE_JOB_ID below (or set them via
env vars), then run:

    export QAI_HUB_API_TOKEN="YOUR_TOKEN_HERE"
    python validate_track1_submission.py

If COMPUTE_PT_REFERENCE is True, the script will also load the local
MobileCLIP2-S2 PyTorch model and compute reference embeddings, so you
can directly compare DLC vs PT on the same sample set.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import qai_hub as hub
from transformers import CLIPTokenizer


# ============================================================
# Configuration
# ============================================================

# AI Hub model identifiers — set these to your latest deployed DLCs.
# You can use either compile_job_id (it resolves to a target model) or
# the underlying model_id directly via client.get_model(model_id).
IMAGE_COMPILE_JOB_ID = os.environ.get("IMAGE_COMPILE_JOB_ID", "")
TEXT_COMPILE_JOB_ID = os.environ.get("TEXT_COMPILE_JOB_ID", "")

# Alternative: if you already know the model IDs (e.g. from AI Hub UI),
# you can fall back on these. The script will prefer compile_job_id if set.
IMAGE_DLC_MODEL_ID = os.environ.get("IMAGE_DLC_MODEL_ID", "")
TEXT_DLC_MODEL_ID = os.environ.get("TEXT_DLC_MODEL_ID", "")

QAI_HUB_API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN", "")

TARGET_DEVICE_NAME = "XR2 Gen 2 (Proxy)"

IMAGE_INPUT_NAME = "image"
TEXT_INPUT_NAME = "text"

# Sample-set layout
SAMPLE_IMG_CSV = Path("/home/aman/dev/lpcv/anvil/sample data/img_list.csv")
SAMPLE_TXT_CSV = Path("/home/aman/dev/lpcv/anvil/sample data/txt_list.csv")
SAMPLE_IMAGE_DIR = Path(
    "/home/aman/dev/lpcv/anvil/sample data/images-20260423T155937Z-3-001/images"
)

# Optional PyTorch reference comparison
COMPUTE_PT_REFERENCE = True
ML_MOBILECLIP_REPO = Path("/home/aman/dev/lpcv/ml-mobileclip")
CHECKPOINT_PATH = Path("/home/aman/dev/lpcv/anvil/mobileclip2_s2.pt")
MODEL_NAME_CANDIDATES = ["MobileCLIP2-S2", "MobileCLIP-S2"]

# Competition tokenizer contract
COMPETITION_TOKENIZER_NAME = "openai/clip-vit-base-patch32"
COMPETITION_EOS_TOKEN_ID = 49407

# CLIP normalization (for the optional PT reference path; for the DLC we feed
# raw Track-1 input because normalization is in-graph).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

ARTIFACT_DIR = Path("lpcvc_track1_validation_artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RECORD_PATH = ARTIFACT_DIR / "validation_record.json"


# ============================================================
# Helpers
# ============================================================

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
    return client


def wait_success(job, name: str) -> None:
    status = str(job.wait())
    print(f"[hub] {name} status: {status}")
    if "SUCCESS" not in status:
        raise RuntimeError(f"{name} failed. Inspect job URL: {job.url}")


def resolve_model(client: hub.Client, compile_job_id: str, model_id_fallback: str):
    if compile_job_id:
        print(f"[hub] Resolving model from compile job {compile_job_id} ...")
        compile_job = client.get_job(compile_job_id)
        model = compile_job.get_target_model()
        if model is None:
            raise RuntimeError(
                f"Compile job {compile_job_id} has no target model. Re-check status."
            )
        return model
    if model_id_fallback:
        print(f"[hub] Resolving model id {model_id_fallback} directly ...")
        return client.get_model(model_id_fallback)
    raise RuntimeError(
        "You must set either IMAGE_COMPILE_JOB_ID/TEXT_COMPILE_JOB_ID or "
        "IMAGE_DLC_MODEL_ID/TEXT_DLC_MODEL_ID."
    )


# ============================================================
# Sample dataset loading
# ============================================================

def load_sample_records():
    txt_df = pd.read_csv(SAMPLE_TXT_CSV).sort_values("Text_nums").reset_index(drop=True)
    img_df = pd.read_csv(SAMPLE_IMG_CSV)
    text_id_to_pos = {int(row["Text_nums"]): i for i, row in txt_df.iterrows()}
    image_records = []
    missing = []
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


# ============================================================
# Track-1-style preprocessing
# ============================================================

def preprocess_track1_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


# ============================================================
# Tokenization
# ============================================================

def build_competition_tokenizer() -> CLIPTokenizer:
    tokenizer = CLIPTokenizer.from_pretrained(COMPETITION_TOKENIZER_NAME)
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
# Optional PyTorch reference
# ============================================================

class CompetitionReadyImageEncoder(nn.Module):
    def __init__(self, clip_model, mean, std):
        super().__init__()
        self.model = clip_model
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, image):
        x = image.to(torch.float32)
        x = (x - self.mean) / self.std
        return self.model.encode_image(x)


class CompetitionCompatibleTextEncoder(nn.Module):
    def __init__(self, clip_model, eos_token_id=COMPETITION_EOS_TOKEN_ID):
        super().__init__()
        self.model = clip_model
        self.eos_token_id = eos_token_id

    def forward(self, text):
        x = text.to(torch.int64)
        eos_mask = (x == self.eos_token_id).to(torch.int64)
        eos_cumsum = torch.cumsum(eos_mask, dim=1)
        repeated_eos_mask = (x == self.eos_token_id) & (eos_cumsum > 1)
        x = torch.where(repeated_eos_mask, torch.zeros_like(x), x)
        return self.model.encode_text(x)


def load_pt_reference():
    """Optionally load PyTorch encoders for direct comparison."""
    repo_str = str(ML_MOBILECLIP_REPO.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # Apply text-side patches that match the export script
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

    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model

    last_err = None
    for model_name in MODEL_NAME_CANDIDATES:
        try:
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
                model_name, pretrained=str(CHECKPOINT_PATH), **model_kwargs
            )
            model = model.eval()
            model = reparameterize_model(model).eval()
            image_pt = CompetitionReadyImageEncoder(model, CLIP_MEAN, CLIP_STD).eval()
            text_pt = CompetitionCompatibleTextEncoder(
                model, COMPETITION_EOS_TOKEN_ID
            ).eval()
            return image_pt, text_pt
        except Exception as exc:
            last_err = exc
            print(f"[pt] Failed to load {model_name}: {exc}")
    raise RuntimeError("Could not load PT reference.") from last_err


# ============================================================
# Embedding inference helpers
# ============================================================

def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize_np(a) * l2_normalize_np(b), axis=1)


def similarity_matrix(img_embs: np.ndarray, txt_embs: np.ndarray) -> np.ndarray:
    return l2_normalize_np(img_embs) @ l2_normalize_np(txt_embs).T


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


def run_dlc_inference(
    client: hub.Client,
    model,
    device: hub.Device,
    inputs: List[np.ndarray],
    input_name: str,
    job_name: str,
) -> np.ndarray:
    print(f"[hub] Submitting inference job: {job_name} ({len(inputs)} samples)...")
    job = client.submit_inference_job(
        model=model,
        device=device,
        inputs={input_name: inputs},
        name=job_name,
    )
    print(f"[hub] inference job id: {job.job_id}")
    wait_success(job, job_name)
    output_data = job.download_output_data()
    output_key = list(output_data.keys())[0]
    embeddings = np.asarray(output_data[output_key], dtype=np.float32)
    if embeddings.ndim == 3 and embeddings.shape[1] == 1:
        embeddings = embeddings[:, 0, :]
    return embeddings


# ============================================================
# Main
# ============================================================

def main() -> int:
    print("==============================================================")
    print(" LPCVC 2026 Track 1 - Joint sample-set retrieval validator ")
    print("==============================================================")

    if not SAMPLE_IMG_CSV.exists() or not SAMPLE_TXT_CSV.exists():
        raise FileNotFoundError(
            f"Sample CSVs not found. Looked for:\n  {SAMPLE_IMG_CSV}\n  {SAMPLE_TXT_CSV}"
        )

    image_records, txt_df, text_id_to_pos, missing = load_sample_records()
    print(f"[data] sample images : {len(image_records)}")
    print(f"[data] sample texts  : {len(txt_df)}")
    if missing:
        print(f"[data] missing image files: {len(missing)}")

    if not image_records:
        raise RuntimeError("No sample images found — aborting.")

    # Build inputs
    image_inputs = [preprocess_track1_image(Image.open(r["image_path"])) for r in image_records]

    tokenizer = build_competition_tokenizer()
    text_strings = txt_df["Unique_Texts"].astype(str).tolist()
    token_matrix = tokenize_competition_texts(text_strings, tokenizer)
    text_inputs = [token_matrix[i : i + 1] for i in range(token_matrix.shape[0])]

    # Connect to Hub and resolve models
    client = build_hub_client(QAI_HUB_API_TOKEN)
    device = hub.Device(TARGET_DEVICE_NAME)
    image_model = resolve_model(client, IMAGE_COMPILE_JOB_ID, IMAGE_DLC_MODEL_ID)
    text_model = resolve_model(client, TEXT_COMPILE_JOB_ID, TEXT_DLC_MODEL_ID)

    # Run DLC inference for both branches
    img_embs_dlc = run_dlc_inference(
        client, image_model, device, image_inputs, IMAGE_INPUT_NAME,
        "lpcvc_track1_validate_image_dlc_inference",
    )
    txt_embs_dlc = run_dlc_inference(
        client, text_model, device, text_inputs, TEXT_INPUT_NAME,
        "lpcvc_track1_validate_text_dlc_inference",
    )
    print(f"[validate] image DLC emb shape : {img_embs_dlc.shape}")
    print(f"[validate] text  DLC emb shape : {txt_embs_dlc.shape}")

    # Retrieval metrics from DLC pair
    sim_dlc = similarity_matrix(img_embs_dlc, txt_embs_dlc)
    metrics_dlc = retrieval_metrics(sim_dlc, image_records, text_id_to_pos)
    print("\n========== DLC pair retrieval (deployed artifacts) ==========")
    for k, v in metrics_dlc.items():
        print(f"  {k:<10} {v:.6f}")

    # Optional PT reference comparison
    metrics_pt = None
    metrics_dlc_image_pt_text = None
    metrics_pt_image_dlc_text = None
    image_pt_vs_dlc_cos = None
    text_pt_vs_dlc_cos = None

    if COMPUTE_PT_REFERENCE:
        try:
            print("\n[pt] Loading PT reference for direct comparison ...")
            image_pt, text_pt = load_pt_reference()

            inputs_np = np.concatenate(image_inputs, axis=0).astype(np.float32)
            with torch.no_grad():
                pt_img_embs = image_pt.cpu()(torch.from_numpy(inputs_np)).cpu().numpy()
                pt_txt_embs = text_pt.cpu()(torch.from_numpy(token_matrix)).cpu().numpy()

            image_pt_vs_dlc_cos = float(cosine_per_row(pt_img_embs, img_embs_dlc).mean())
            text_pt_vs_dlc_cos = float(cosine_per_row(pt_txt_embs, txt_embs_dlc).mean())
            print(f"[pt] cosine(PT image, DLC image) : {image_pt_vs_dlc_cos:.6f}")
            print(f"[pt] cosine(PT text,  DLC text ) : {text_pt_vs_dlc_cos:.6f}")

            sim_pt = similarity_matrix(pt_img_embs, pt_txt_embs)
            metrics_pt = retrieval_metrics(sim_pt, image_records, text_id_to_pos)
            print("\n========== PT pair retrieval (upper bound) ==========")
            for k, v in metrics_pt.items():
                print(f"  {k:<10} {v:.6f}")

            sim_dlc_img_pt_txt = similarity_matrix(img_embs_dlc, pt_txt_embs)
            sim_pt_img_dlc_txt = similarity_matrix(pt_img_embs, txt_embs_dlc)
            metrics_dlc_image_pt_text = retrieval_metrics(
                sim_dlc_img_pt_txt, image_records, text_id_to_pos
            )
            metrics_pt_image_dlc_text = retrieval_metrics(
                sim_pt_img_dlc_txt, image_records, text_id_to_pos
            )

            print("\n========== Mixed retrieval (branch isolation) ==========")
            print("  image=DLC, text=PT (isolates image-branch quality):")
            for k, v in metrics_dlc_image_pt_text.items():
                print(f"    {k:<10} {v:.6f}")
            print("  image=PT,  text=DLC (isolates text-branch quality):")
            for k, v in metrics_pt_image_dlc_text.items():
                print(f"    {k:<10} {v:.6f}")
        except Exception as exc:
            print(f"[pt] Skipped PT reference comparison: {exc}")

    # Save record
    record = {
        "image_compile_job_id": IMAGE_COMPILE_JOB_ID,
        "text_compile_job_id": TEXT_COMPILE_JOB_ID,
        "image_dlc_model_id_fallback": IMAGE_DLC_MODEL_ID,
        "text_dlc_model_id_fallback": TEXT_DLC_MODEL_ID,
        "device": TARGET_DEVICE_NAME,
        "tokenizer": COMPETITION_TOKENIZER_NAME,
        "num_sample_images": len(image_records),
        "num_sample_texts": len(txt_df),
        "missing_images": len(missing),
        "metrics_dlc_pair": metrics_dlc,
        "metrics_pt_pair": metrics_pt,
        "metrics_dlc_image_pt_text": metrics_dlc_image_pt_text,
        "metrics_pt_image_dlc_text": metrics_pt_image_dlc_text,
        "cosine_pt_image_vs_dlc_image": image_pt_vs_dlc_cos,
        "cosine_pt_text_vs_dlc_text": text_pt_vs_dlc_cos,
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2, default=str))
    print(f"\n[artifacts] Wrote validation record to {RECORD_PATH.resolve()}")

    # Decision summary
    print("\n========== DECISION SUMMARY ==========")
    r10 = metrics_dlc["hit_R@10"]
    if metrics_pt is not None:
        r10_pt = metrics_pt["hit_R@10"]
        gap = r10_pt - r10
        print(f"  PT  R@10 (upper bound) : {r10_pt:.4f}")
        print(f"  DLC R@10               : {r10:.4f}")
        print(f"  Gap (quantization tax) : {gap:.4f}")
        if r10 >= 0.85 * r10_pt:
            print("  ✅ DLC retains ≥85% of PT R@10 — submit with confidence.")
        elif r10 >= 0.70 * r10_pt:
            print("  ⚠️  DLC retains 70-85% of PT R@10 — submit but consider further calibration.")
        else:
            print("  ❌ DLC retains <70% of PT R@10 — investigate before submitting.")
    else:
        print(f"  DLC R@10               : {r10:.4f}")
    print("=======================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise
    except Exception as exc:
        print(f"\n❌ Validation failed: {exc}")
        raise
