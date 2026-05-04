#!/usr/bin/env python3
"""
mobileclip2_image_light_qat_v3.py

Patch v3 for MobileCLIP2-S4 image-only light QAT.

Fixes relative to earlier cached-target version:
1. Preserve the INITIAL fake-quant model as a valid best checkpoint.
2. Activate KL + contrastive losses even with batch size 1 by sampling negative text embeddings.
3. Export on CPU and force AIMET encoding version 0.6.1.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import CLIPTokenizer

DEFAULT_REPO_ROOT = "/home/aman/dev/lpcv/anvil/ml-mobileclip"
DEFAULT_CHECKPOINT = "/home/aman/dev/lpcv/anvil/mobileclip2_s4.pt"
DEFAULT_MODEL_NAME = "MobileCLIP2-S4"

DEFAULT_IMG_CSV = "/home/aman/dev/lpcv/anvil/sample data/img_list.csv"
DEFAULT_TXT_CSV = "/home/aman/dev/lpcv/anvil/sample data/txt_list.csv"
DEFAULT_IMAGE_DIR = "/home/aman/dev/lpcv/anvil/sample data/images-20260423T155937Z-3-001/images"

DEFAULT_OUTPUT_DIR = "/home/aman/dev/lpcv/anvil/lpcvc_track1_qat_image_v3_artifacts"

COMPETITION_TOKENIZER_NAME = "openai/clip-vit-base-patch32"
COMPETITION_EOS_TOKEN_ID = 49407
IMAGE_SIZE = 224


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    rem = s - 60 * m
    return f"{m}m {rem:.1f}s"


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_corresponding_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = l2_normalize_np(a)
    b_n = l2_normalize_np(b)
    return np.sum(a_n * b_n, axis=1)


def similarity_matrix_np(img_embs: np.ndarray, txt_embs: np.ndarray) -> np.ndarray:
    return l2_normalize_np(img_embs) @ l2_normalize_np(txt_embs).T


def build_competition_tokenizer() -> CLIPTokenizer:
    tokenizer = CLIPTokenizer.from_pretrained(COMPETITION_TOKENIZER_NAME)
    tokenizer.add_special_tokens({"cls_token": tokenizer.eos_token})
    return tokenizer


def fix_repeated_eos_padding(tokens: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    eos_mask = (tokens == eos_token_id).to(torch.int64)
    eos_cumsum = torch.cumsum(eos_mask, dim=1)
    repeated_eos_mask = (tokens == eos_token_id) & (eos_cumsum > 1)
    return torch.where(repeated_eos_mask, torch.zeros_like(tokens), tokens)


def preprocess_track1_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


def prepare_python_imports(repo_root: Path) -> None:
    repo_str = str(repo_root.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def import_mobileclip_stack(repo_root: Path):
    prepare_python_imports(repo_root)
    import importlib
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model
    mobileclip2_module = importlib.import_module("mobileclip2.mobileclip2")
    return open_clip, reparameterize_model, mobileclip2_module


def apply_project_compat_patches(open_clip, mobileclip2_module) -> None:
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
            idx = text.argmax(dim=-1)
            batch_idx = torch.arange(x.shape[0], device=x.device)
            return x[batch_idx, idx, :]
        return original_text_global_pool(x, text, pool_type, eos_token_id)

    open_clip.transformer.text_global_pool = patched_text_global_pool


def extract_normalize_stats(preprocess) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    from torchvision import transforms as T
    transforms = getattr(preprocess, "transforms", None)
    if transforms is None:
        raise RuntimeError("Could not inspect preprocess.transforms.")
    for tr in transforms:
        if isinstance(tr, T.Normalize):
            mean = tuple(float(x) for x in tr.mean)
            std = tuple(float(x) for x in tr.std)
            return mean, std
    raise RuntimeError("No Normalize layer found in preprocess.")


class CompetitionReadyImageEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        super().__init__()
        self.model = clip_model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image.to(torch.float32)
        x = (x - self.mean) / self.std
        out = self.model.encode_image(x)
        if out.ndim == 3 and out.shape[1] == 1:
            out = out[:, 0, :]
        return out


class CompetitionCompatibleTextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module, eos_token_id: int = COMPETITION_EOS_TOKEN_ID):
        super().__init__()
        self.model = clip_model
        self.eos_token_id = eos_token_id

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = text.to(torch.int64)
        x = fix_repeated_eos_padding(x, self.eos_token_id)
        out = self.model.encode_text(x)
        if out.ndim == 3 and out.shape[1] == 1:
            out = out[:, 0, :]
        return out


def load_teacher_and_text_models(repo_root: Path, checkpoint: Path, model_name: str):
    open_clip, reparameterize_model, mobileclip2_module = import_mobileclip_stack(repo_root)
    apply_project_compat_patches(open_clip, mobileclip2_module)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=str(checkpoint),
    )
    model = model.eval()
    model = reparameterize_model(model).eval()

    mean, std = extract_normalize_stats(preprocess)

    image_model = CompetitionReadyImageEncoder(model, mean, std).eval()
    text_model = CompetitionCompatibleTextEncoder(model, eos_token_id=COMPETITION_EOS_TOKEN_ID).eval()
    return image_model, text_model, mean, std


def import_aimet_modules():
    aimet = {}

    from aimet_common.defs import QuantScheme
    aimet["QuantScheme"] = QuantScheme

    qsim_errs = []
    for path in ["aimet_torch.v2.quantsim", "aimet_torch.quantsim"]:
        try:
            module = __import__(path, fromlist=["QuantizationSimModel"])
            aimet["QuantizationSimModel"] = getattr(module, "QuantizationSimModel")
            break
        except Exception as exc:
            qsim_errs.append((path, str(exc)))
    if "QuantizationSimModel" not in aimet:
        raise ImportError(f"Could not import QuantizationSimModel from AIMET: {qsim_errs}")

    qm_errs = []
    for path in ["aimet_torch.v2.nn", "aimet_torch.nn"]:
        try:
            module = __import__(path, fromlist=["QuantizationMixin"])
            aimet["QuantizationMixin"] = getattr(module, "QuantizationMixin")
            break
        except Exception as exc:
            qm_errs.append((path, str(exc)))
    if "QuantizationMixin" not in aimet:
        raise ImportError(f"Could not import QuantizationMixin from AIMET: {qm_errs}")

    try:
        from aimet_torch.onnx import export as aimet_onnx_export
        aimet["aimet_onnx_export"] = aimet_onnx_export
    except Exception:
        aimet["aimet_onnx_export"] = None

    try:
        from aimet_torch.cross_layer_equalization import equalize_model
        aimet["equalize_model"] = equalize_model
    except Exception:
        aimet["equalize_model"] = None

    try:
        from aimet_torch.common import quantsim as common_quantsim
        aimet["common_quantsim"] = common_quantsim
    except Exception:
        aimet["common_quantsim"] = None

    return aimet


def register_or_ignore_unsupported_modules(aimet: dict) -> None:
    QuantizationMixin = aimet["QuantizationMixin"]
    unsupported = []

    try:
        from timm.models.fastvit import LayerScale2d
        unsupported.append(LayerScale2d)
    except Exception:
        pass
    try:
        import importlib
        mobileclip2_module = importlib.import_module("mobileclip2.mobileclip2")
        if hasattr(mobileclip2_module, "LayerNormChannel"):
            unsupported.append(mobileclip2_module.LayerNormChannel)
    except Exception:
        pass
    try:
        from timm.layers.activations import Sigmoid as TimmSigmoid
        unsupported.append(TimmSigmoid)
    except Exception:
        pass
    try:
        from open_clip.transformer import LayerNorm as OpenClipLayerNorm
        unsupported.append(OpenClipLayerNorm)
    except Exception:
        pass

    ignored_names = []
    for cls in unsupported:
        try:
            QuantizationMixin.ignore(cls)
            ignored_names.append(f"{cls.__module__}.{cls.__name__}")
        except Exception:
            pass

    if ignored_names:
        print("[aimet] Ignoring unsupported custom module types for QAT:")
        for name in ignored_names:
            print(f"         - {name}")


def build_quantsim(aimet: dict, model: nn.Module, dummy_input: torch.Tensor,
                   weight_bw: int, act_bw: int, quant_scheme_name: str, aimet_config: Optional[str]):
    QuantizationSimModel = aimet["QuantizationSimModel"]
    QuantScheme = aimet["QuantScheme"]
    quant_scheme = getattr(QuantScheme, quant_scheme_name)

    import inspect
    sig = inspect.signature(QuantizationSimModel)
    kwargs = {
        "model": model,
        "quant_scheme": quant_scheme,
        "default_output_bw": act_bw,
        "default_param_bw": weight_bw,
        "in_place": False,
    }
    if "dummy_input" in sig.parameters:
        kwargs["dummy_input"] = dummy_input.cpu()
    elif "input_shapes" in sig.parameters:
        kwargs["input_shapes"] = tuple(dummy_input.shape)
    if aimet_config:
        kwargs["config_file"] = aimet_config

    try:
        sim = QuantizationSimModel(**kwargs)
    except TypeError:
        kwargs.pop("in_place", None)
        sim = QuantizationSimModel(**kwargs)
    return sim


class CachedProxyDataset(Dataset):
    def __init__(self, images: np.ndarray, teacher_embs: np.ndarray, text_embs: np.ndarray):
        assert images.shape[0] == teacher_embs.shape[0] == text_embs.shape[0]
        self.images = images.astype(np.float32)
        self.teacher_embs = teacher_embs.astype(np.float32)
        self.text_embs = text_embs.astype(np.float32)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.tensor(idx, dtype=torch.long),
            torch.from_numpy(self.images[idx]),
            torch.from_numpy(self.teacher_embs[idx]),
            torch.from_numpy(self.text_embs[idx]),
        )


def build_flickr30k_proxy_samples(max_unique_samples: int, caption_index: int = 0, seed: int = 0):
    rng = random.Random(seed)
    ds = load_dataset("lmms-lab/flickr30k", split="test", streaming=True)

    samples = []
    for item in ds:
        img = item["image"]
        caps = item["caption"]
        if not isinstance(caps, (list, tuple)) or len(caps) == 0:
            continue
        cap = caps[min(caption_index, len(caps) - 1)]
        if not isinstance(cap, str) or not cap.strip():
            continue
        arr = preprocess_track1_image(img)
        samples.append((arr, cap))
        if len(samples) >= max_unique_samples:
            break

    if len(samples) == 0:
        raise RuntimeError("No proxy samples collected from Flickr30k.")
    rng.shuffle(samples)
    return samples


@torch.no_grad()
def precompute_proxy_targets(samples, teacher_image, frozen_text, tokenizer, batch_size, device):
    teacher_image.eval()
    frozen_text.eval()

    images_np = np.stack([s[0] for s in samples], axis=0).astype(np.float32)
    captions = [s[1] for s in samples]

    teacher_img_list = []
    text_emb_list = []

    for i in range(0, len(samples), batch_size):
        images = torch.from_numpy(images_np[i:i + batch_size]).to(device=device, dtype=torch.float32)
        toks = tokenizer(
            captions[i:i + batch_size],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"].to(device=device, dtype=torch.int64)

        teacher_img = teacher_image(images)
        text_emb = frozen_text(toks)

        teacher_img_list.append(teacher_img.cpu().numpy())
        text_emb_list.append(text_emb.cpu().numpy())

    teacher_img_np = np.concatenate(teacher_img_list, axis=0).astype(np.float32)
    text_emb_np = np.concatenate(text_emb_list, axis=0).astype(np.float32)
    return images_np, teacher_img_np, text_emb_np


def make_cached_proxy_loader(images_np, teacher_img_np, text_emb_np, batch_size, shuffle, num_workers):
    dataset = CachedProxyDataset(images_np, teacher_img_np, text_emb_np)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=False, drop_last=True)


def load_competition_metadata(img_csv: Path, txt_csv: Path, image_dir: Path):
    txt_df = pd.read_csv(txt_csv).sort_values("Text_nums").reset_index(drop=True)
    img_df = pd.read_csv(img_csv)
    text_id_to_pos = {int(row["Text_nums"]): i for i, row in txt_df.iterrows()}

    image_records = []
    missing = []
    for _, row in img_df.iterrows():
        image_name = str(row["Image_names"])
        image_path = image_dir / image_name
        gt_ids = [int(x) for x in str(row["Text_nums"]).split(";") if str(x).strip()]
        if image_path.exists():
            image_records.append({
                "image_name": image_name,
                "image_path": image_path,
                "gt_text_ids": gt_ids,
            })
        else:
            missing.append(image_name)

    return image_records, txt_df, text_id_to_pos, missing


@torch.no_grad()
def precompute_official_eval_cache(teacher_image, frozen_text, tokenizer, img_csv, txt_csv, image_dir, batch_size, device):
    image_records, txt_df, text_id_to_pos, missing = load_competition_metadata(img_csv, txt_csv, image_dir)
    if len(image_records) == 0:
        raise RuntimeError("No official sample images found.")
    if missing:
        warnings.warn(f"{len(missing)} official sample images are missing on disk.", RuntimeWarning)

    texts = txt_df["Unique_Texts"].astype(str).tolist()

    text_embs = []
    frozen_text.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        toks = tokenizer(batch, padding="max_length", truncation=True, max_length=77,
                         return_tensors="pt")["input_ids"].to(device=device, dtype=torch.int64)
        emb = frozen_text(toks)
        text_embs.append(emb.cpu().numpy())
    text_embs_np = np.concatenate(text_embs, axis=0).astype(np.float32)

    teacher_img_embs = []
    teacher_image.eval()
    for i in range(0, len(image_records), batch_size):
        batch = image_records[i:i + batch_size]
        imgs = [preprocess_track1_image(Image.open(rec["image_path"]).convert("RGB")) for rec in batch]
        x = torch.from_numpy(np.stack(imgs, axis=0)).to(device=device, dtype=torch.float32)
        emb = teacher_image(x)
        teacher_img_embs.append(emb.cpu().numpy())
    teacher_img_embs_np = np.concatenate(teacher_img_embs, axis=0).astype(np.float32)

    return {
        "image_records": image_records,
        "text_id_to_pos": text_id_to_pos,
        "texts": texts,
        "teacher_image_embs": teacher_img_embs_np,
        "text_embs": text_embs_np,
    }


@torch.no_grad()
def validate_with_cached_targets(student_model, eval_cache, batch_size, device):
    image_records = eval_cache["image_records"]
    teacher_img = eval_cache["teacher_image_embs"]
    text_embs = eval_cache["text_embs"]
    text_id_to_pos = eval_cache["text_id_to_pos"]

    student_model.eval()
    student_img_list = []

    for i in range(0, len(image_records), batch_size):
        batch = image_records[i:i + batch_size]
        imgs = [preprocess_track1_image(Image.open(rec["image_path"]).convert("RGB")) for rec in batch]
        x = torch.from_numpy(np.stack(imgs, axis=0)).to(device=device, dtype=torch.float32)
        emb = student_model(x)
        student_img_list.append(emb.cpu().numpy())

    student_img = np.concatenate(student_img_list, axis=0).astype(np.float32)

    cosine = float(cosine_corresponding_np(teacher_img, student_img).mean())
    sim_teacher = similarity_matrix_np(teacher_img, text_embs)
    sim_student = similarity_matrix_np(student_img, text_embs)
    corr = float(np.corrcoef(sim_teacher.reshape(-1), sim_student.reshape(-1))[0, 1])

    teacher_metrics = retrieval_metrics(sim_teacher, image_records, text_id_to_pos)
    student_metrics = retrieval_metrics(sim_student, image_records, text_id_to_pos)

    return {
        "num_images": len(image_records),
        "num_texts": len(eval_cache["texts"]),
        "mean_image_teacher_cosine": cosine,
        "similarity_matrix_corr_vs_teacher": corr,
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
    }


def retrieval_metrics(sim, image_records, text_id_to_pos, ks=(1, 5, 10)):
    ranked = np.argsort(-sim, axis=1)
    results = {}
    for k in ks:
        hit_scores = []
        frac_scores = []
        for i, rec in enumerate(image_records):
            gt_positions = [text_id_to_pos[t] for t in rec["gt_text_ids"] if t in text_id_to_pos]
            topk = set(ranked[i, :k].tolist())
            hits = sum(1 for g in gt_positions if g in topk)
            hit_scores.append(1.0 if hits > 0 else 0.0)
            frac_scores.append(hits / max(len(gt_positions), 1))
        results[f"hit_R@{k}"] = float(np.mean(hit_scores))
        results[f"frac_R@{k}"] = float(np.mean(frac_scores))
    return results


def candidate_set_contrastive_loss(student_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = torch.zeros(student_logits.shape[0], dtype=torch.long, device=student_logits.device)
    return F.cross_entropy(student_logits / temperature, labels)


def similarity_kl_distillation(student_sim: torch.Tensor, teacher_sim: torch.Tensor, temperature: float) -> torch.Tensor:
    teacher_probs = F.softmax(teacher_sim / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_sim / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def build_candidate_text_tensor(batch_indices: torch.Tensor, positive_text_embs: torch.Tensor,
                                full_text_pool_cpu: torch.Tensor, num_negatives: int,
                                rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    batch_indices_np = batch_indices.detach().cpu().numpy()
    positive_np = positive_text_embs.detach().cpu().numpy()
    pool_size = full_text_pool_cpu.shape[0]
    neg_k = min(num_negatives, max(pool_size - 1, 0))

    rows = []
    all_ids = np.arange(pool_size)
    for idx, pos in zip(batch_indices_np.tolist(), positive_np):
        if neg_k > 0:
            candidate_ids = all_ids[all_ids != idx]
            if len(candidate_ids) >= neg_k:
                neg_ids = rng.choice(candidate_ids, size=neg_k, replace=False)
            else:
                neg_ids = rng.choice(candidate_ids, size=neg_k, replace=True)
            neg = full_text_pool_cpu[torch.as_tensor(neg_ids, dtype=torch.long)]
            pos_t = torch.from_numpy(pos).unsqueeze(0)
            row = torch.cat([pos_t, neg], dim=0)
        else:
            row = torch.from_numpy(pos).unsqueeze(0)
        rows.append(row)

    return torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)


@dataclass
class TrainConfig:
    repo_root: str
    checkpoint: str
    model_name: str
    output_dir: str

    img_csv: str
    txt_csv: str
    image_dir: str

    seed: int
    device: str
    num_workers: int

    max_unique_train_samples: int
    proxy_cache_batch_size: int
    train_batch_size: int
    train_steps: int
    eval_every: int
    grad_clip_norm: float
    num_text_negatives: int

    lr: float
    weight_decay: float

    weight_bw: int
    act_bw: int
    quant_scheme: str
    aimet_config: str

    contrastive_temperature: float
    distill_temperature: float
    lambda_img_cos: float
    lambda_sim_kl: float
    lambda_clip: float

    calibration_batches: int
    sample_eval_batch_size: int

    apply_cle_if_available: bool
    export_opset: int
    reuse_caches_if_present: bool


def train_one_light_qat_run(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    out_dir = Path(cfg.output_dir)
    ensure_dir(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)
    cache_dir = out_dir / "cache"
    ensure_dir(cache_dir)
    export_dir = out_dir / "mobileclip_image_qat.aimet"
    ensure_dir(export_dir)

    print("=" * 88)
    print("MOBILECLIP2 IMAGE LIGHT QAT (PATCH V3)")
    print("=" * 88)
    print(json.dumps(asdict(cfg), indent=2))

    print("\n[1/9] Loading tokenizer...")
    tokenizer = build_competition_tokenizer()

    print("[2/9] Loading teacher image model + frozen text model...")
    teacher_image, frozen_text, mean, std = load_teacher_and_text_models(
        Path(cfg.repo_root), Path(cfg.checkpoint), cfg.model_name
    )
    teacher_image = teacher_image.to(device).eval()
    frozen_text = frozen_text.to(device).eval()
    for p in teacher_image.parameters():
        p.requires_grad = False
    for p in frozen_text.parameters():
        p.requires_grad = False

    print("[3/9] Building proxy adaptation dataset...")
    proxy_samples = build_flickr30k_proxy_samples(
        max_unique_samples=cfg.max_unique_train_samples,
        caption_index=0,
        seed=cfg.seed,
    )

    proxy_images_path = cache_dir / "proxy_images.npy"
    proxy_teacher_path = cache_dir / "proxy_teacher_image_embs.npy"
    proxy_text_path = cache_dir / "proxy_text_embs.npy"

    if cfg.reuse_caches_if_present and proxy_images_path.exists() and proxy_teacher_path.exists() and proxy_text_path.exists():
        print("[4/9] Reusing cached proxy targets...")
        proxy_images_np = np.load(proxy_images_path)
        proxy_teacher_np = np.load(proxy_teacher_path)
        proxy_text_np = np.load(proxy_text_path)
    else:
        print("[4/9] Precomputing cached proxy targets...")
        proxy_images_np, proxy_teacher_np, proxy_text_np = precompute_proxy_targets(
            samples=proxy_samples,
            teacher_image=teacher_image,
            frozen_text=frozen_text,
            tokenizer=tokenizer,
            batch_size=cfg.proxy_cache_batch_size,
            device=device,
        )
        np.save(proxy_images_path, proxy_images_np)
        np.save(proxy_teacher_path, proxy_teacher_np)
        np.save(proxy_text_path, proxy_text_np)

    train_loader = make_cached_proxy_loader(
        images_np=proxy_images_np,
        teacher_img_np=proxy_teacher_np,
        text_emb_np=proxy_text_np,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    train_iter = itertools.cycle(train_loader)
    full_text_pool_cpu = torch.from_numpy(proxy_text_np).cpu()

    eval_cache_path = cache_dir / "official_eval_cache.npz"
    if cfg.reuse_caches_if_present and eval_cache_path.exists():
        print("[5/9] Reusing cached official sample-set targets...")
        packed = np.load(eval_cache_path, allow_pickle=True)
        eval_cache = {
            "image_records": list(packed["image_records"]),
            "text_id_to_pos": dict(packed["text_id_to_pos"].item()),
            "texts": list(packed["texts"]),
            "teacher_image_embs": packed["teacher_image_embs"],
            "text_embs": packed["text_embs"],
        }
    else:
        print("[5/9] Precomputing official sample-set cache...")
        eval_cache = precompute_official_eval_cache(
            teacher_image=teacher_image,
            frozen_text=frozen_text,
            tokenizer=tokenizer,
            img_csv=Path(cfg.img_csv),
            txt_csv=Path(cfg.txt_csv),
            image_dir=Path(cfg.image_dir),
            batch_size=cfg.sample_eval_batch_size,
            device=device,
        )
        np.savez(
            eval_cache_path,
            image_records=np.array(eval_cache["image_records"], dtype=object),
            text_id_to_pos=np.array(eval_cache["text_id_to_pos"], dtype=object),
            texts=np.array(eval_cache["texts"], dtype=object),
            teacher_image_embs=eval_cache["teacher_image_embs"],
            text_embs=eval_cache["text_embs"],
        )

    print("[5b/9] Releasing teacher/text models from GPU before QAT...")
    teacher_image_cpu = teacher_image.cpu().eval()
    del teacher_image
    frozen_text_cpu = frozen_text.cpu().eval()
    del frozen_text
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[6/9] Creating student model copy and AIMET QuantizationSimModel...")
    student_image = copy.deepcopy(teacher_image_cpu).cpu().eval()
    aimet = import_aimet_modules()
    register_or_ignore_unsupported_modules(aimet)
    dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    if cfg.apply_cle_if_available and aimet.get("equalize_model") is not None:
        try:
            print("[6a/9] Applying AIMET Cross-Layer Equalization (optional)...")
            aimet["equalize_model"](student_image, tuple(dummy_input.shape))
        except Exception as exc:
            print(f"[warn] CLE failed; continuing without CLE. Error: {exc}")

    sim = build_quantsim(
        aimet=aimet,
        model=student_image,
        dummy_input=dummy_input,
        weight_bw=cfg.weight_bw,
        act_bw=cfg.act_bw,
        quant_scheme_name=cfg.quant_scheme,
        aimet_config=(cfg.aimet_config if cfg.aimet_config else None),
    )
    sim.model = sim.model.to(device)

    print("[7/9] Computing initial encodings from cached proxy images...")
    calib_seen = 0

    def calibration_callback(quantized_model: nn.Module, _unused_args=None):
        nonlocal calib_seen
        quantized_model.eval()
        with torch.no_grad():
            for _idx, images, _teacher, _text in train_loader:
                images = images.to(device=device, dtype=torch.float32)
                _ = quantized_model(images)
                calib_seen += 1
                if calib_seen >= cfg.calibration_batches:
                    break

    try:
        sim.compute_encodings(calibration_callback, None)
    except TypeError:
        sim.compute_encodings(calibration_callback)

    print(f"[info] calibration batches used: {calib_seen}")

    print("[7b/9] Initial validation before QAT...")
    initial_metrics = validate_with_cached_targets(
        student_model=sim.model,
        eval_cache=eval_cache,
        batch_size=cfg.sample_eval_batch_size,
        device=device,
    )
    print(json.dumps(initial_metrics, indent=2))

    print("[8/9] Starting patch-v3 cached-target light QAT...")
    sim.model.train()
    optimizer = torch.optim.AdamW(sim.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []

    best_score = float(initial_metrics["student_metrics"]["hit_R@10"])
    best_step = 0
    best_state = copy.deepcopy(sim.model.state_dict())
    torch.save(best_state, ckpt_dir / "best_student_qat_state.pt")
    with open(ckpt_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(initial_metrics, f, indent=2)

    t0 = time.time()

    for step in range(1, cfg.train_steps + 1):
        batch_indices, images, teacher_img_t, pos_text_emb_t = next(train_iter)
        batch_indices = batch_indices.to(dtype=torch.long)
        images = images.to(device=device, dtype=torch.float32)
        teacher_img_t = teacher_img_t.to(device=device, dtype=torch.float32)
        pos_text_emb_t = pos_text_emb_t.to(device=device, dtype=torch.float32)

        candidate_text = build_candidate_text_tensor(
            batch_indices=batch_indices,
            positive_text_embs=pos_text_emb_t,
            full_text_pool_cpu=full_text_pool_cpu,
            num_negatives=cfg.num_text_negatives,
            rng=rng,
            device=device,
        )

        student_img = sim.model(images)

        teacher_img_n = l2_normalize(teacher_img_t)
        student_img_n = l2_normalize(student_img)
        candidate_text_n = l2_normalize(candidate_text)

        loss_img_cos = (1.0 - (student_img_n * teacher_img_n).sum(dim=-1)).mean()
        teacher_sim = torch.einsum("bd,bkd->bk", teacher_img_n, candidate_text_n)
        student_sim = torch.einsum("bd,bkd->bk", student_img_n, candidate_text_n)

        loss_sim_kl = similarity_kl_distillation(student_sim, teacher_sim, cfg.distill_temperature)
        loss_clip = candidate_set_contrastive_loss(student_sim, cfg.contrastive_temperature)

        loss = (
            cfg.lambda_img_cos * loss_img_cos
            + cfg.lambda_sim_kl * loss_sim_kl
            + cfg.lambda_clip * loss_clip
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(sim.model.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()

        train_log = {
            "step": step,
            "loss_total": float(loss.detach().cpu()),
            "loss_img_cos": float(loss_img_cos.detach().cpu()),
            "loss_sim_kl": float(loss_sim_kl.detach().cpu()),
            "loss_clip": float(loss_clip.detach().cpu()),
        }

        if step % cfg.eval_every == 0 or step == 1 or step == cfg.train_steps:
            sim.model.eval()
            val_metrics = validate_with_cached_targets(
                student_model=sim.model,
                eval_cache=eval_cache,
                batch_size=cfg.sample_eval_batch_size,
                device=device,
            )
            sim.model.train()

            train_log["val"] = val_metrics
            score = val_metrics["student_metrics"]["hit_R@10"]

            print("-" * 88)
            print(
                f"[step {step:04d}] "
                f"loss={train_log['loss_total']:.6f} "
                f"img_cos={train_log['loss_img_cos']:.6f} "
                f"sim_kl={train_log['loss_sim_kl']:.6f} "
                f"clip={train_log['loss_clip']:.6f}"
            )
            print(
                f"          "
                f"teacher_cos={val_metrics['mean_image_teacher_cosine']:.6f} "
                f"sim_corr={val_metrics['similarity_matrix_corr_vs_teacher']:.6f} "
                f"hit_R@10={val_metrics['student_metrics']['hit_R@10']:.6f} "
                f"frac_R@10={val_metrics['student_metrics']['frac_R@10']:.6f}"
            )

            if score > best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(sim.model.state_dict())
                torch.save(best_state, ckpt_dir / "best_student_qat_state.pt")
                with open(ckpt_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(val_metrics, f, indent=2)

        history.append(train_log)

    total_time = time.time() - t0
    print(f"[done] training time: {format_seconds(total_time)}")

    print("[8b/9] Restoring best checkpoint before export...")
    sim.model.load_state_dict(best_state)
    sim.model.eval()

    final_metrics = validate_with_cached_targets(
        student_model=sim.model,
        eval_cache=eval_cache,
        batch_size=cfg.sample_eval_batch_size,
        device=device,
    )

    print("[9/9] Exporting AIMET artifact for AI Hub...")
    if aimet.get("common_quantsim") is not None:
        try:
            aimet["common_quantsim"].encoding_version = "0.6.1"
            print('[aimet] Set encoding_version = "0.6.1" for export.')
        except Exception as exc:
            print(f"[warn] Could not set AIMET encoding version: {exc}")

    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sim.model = sim.model.cpu().eval()
    dummy_cpu = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    export_prefix = "model"

    exported_encodings = False
    exported_onnx = False

    try:
        onnx_export_args = {
            "input_names": ["image"],
            "output_names": ["embedding"],
            "opset_version": cfg.export_opset,
            "dynamo": False,
        }
        try:
            sim.export(
                path=str(export_dir),
                filename_prefix=export_prefix,
                dummy_input=dummy_cpu,
                onnx_export_args=onnx_export_args,
                propagate_encodings=True,
                export_to_torchscript=False,
            )
        except TypeError:
            sim.export(str(export_dir), export_prefix, dummy_cpu, onnx_export_args=onnx_export_args)

        if list(export_dir.glob("*.encodings")):
            exported_encodings = True
        if list(export_dir.glob("*.onnx")):
            exported_onnx = True
    except Exception as exc:
        print(f"[warn] sim.export primary path failed: {exc}")

    if not exported_encodings:
        try:
            sim.export(str(export_dir), export_prefix, dummy_cpu)
            if list(export_dir.glob("*.encodings")):
                exported_encodings = True
            if list(export_dir.glob("*.onnx")):
                exported_onnx = True
        except Exception as exc:
            print(f"[warn] sim.export fallback for encodings failed: {exc}")

    if not exported_onnx and aimet.get("aimet_onnx_export") is not None:
        try:
            onnx_path = export_dir / f"{export_prefix}.onnx"
            aimet["aimet_onnx_export"](
                sim.model,
                dummy_cpu,
                f=str(onnx_path),
                input_names=["image"],
                output_names=["embedding"],
                opset_version=cfg.export_opset,
                dynamo=False,
            )
            exported_onnx = onnx_path.exists()
        except Exception as exc:
            print(f"[warn] AIMET ONNX export helper failed: {exc}")

    if not exported_encodings:
        raise RuntimeError(
            f"Could not export AIMET encodings to {export_dir}. "
            "Training may still be useful, but AI Hub compile expects .encodings."
        )
    if not exported_onnx:
        raise RuntimeError(
            f"Could not export AIMET ONNX to {export_dir}. "
            "Training may still be useful, but AI Hub compile expects .onnx + .encodings."
        )

    summary = {
        "config": asdict(cfg),
        "mean": mean,
        "std": std,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "best_step": best_step,
        "best_hit_R@10": best_score,
        "training_seconds": total_time,
        "export_dir": str(export_dir),
        "exported_files": [p.name for p in sorted(export_dir.iterdir())],
        "suggested_aihub_compile_snippet": [
            "import qai_hub as hub",
            "compile_job = hub.submit_compile_job(",
            f"    model=r'{str(export_dir)}',",
            "    device=hub.Device('XR2 Gen 2 (Proxy)'),",
            "    input_specs={'image': (1, 3, 224, 224)},",
            "    options='--target_runtime qnn_dlc',",
            ")",
        ],
    }

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 88)
    print("PATCH V3 COMPLETE")
    print("=" * 88)
    print(json.dumps(summary, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo-root", type=str, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--img-csv", type=str, default=DEFAULT_IMG_CSV)
    parser.add_argument("--txt-csv", type=str, default=DEFAULT_TXT_CSV)
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--max-unique-train-samples", type=int, default=128)
    parser.add_argument("--proxy-cache-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-text-negatives", type=int, default=31)

    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--weight-bw", type=int, default=8)
    parser.add_argument("--act-bw", type=int, default=8)
    parser.add_argument("--quant-scheme", type=str, default="training_range_learning_with_tf_init")
    parser.add_argument("--aimet-config", type=str, default="")

    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--distill-temperature", type=float, default=0.20)
    parser.add_argument("--lambda-img-cos", type=float, default=0.50)
    parser.add_argument("--lambda-sim-kl", type=float, default=0.30)
    parser.add_argument("--lambda-clip", type=float, default=0.20)

    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--sample-eval-batch-size", type=int, default=2)

    parser.add_argument("--apply-cle-if-available", action="store_true")
    parser.add_argument("--export-opset", type=int, default=17)
    parser.add_argument("--reuse-caches-if-present", action="store_true")

    args = parser.parse_args()
    return TrainConfig(
        repo_root=args.repo_root,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        output_dir=args.output_dir,
        img_csv=args.img_csv,
        txt_csv=args.txt_csv,
        image_dir=args.image_dir,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        max_unique_train_samples=args.max_unique_train_samples,
        proxy_cache_batch_size=args.proxy_cache_batch_size,
        train_batch_size=args.train_batch_size,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        grad_clip_norm=args.grad_clip_norm,
        num_text_negatives=args.num_text_negatives,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weight_bw=args.weight_bw,
        act_bw=args.act_bw,
        quant_scheme=args.quant_scheme,
        aimet_config=args.aimet_config,
        contrastive_temperature=args.contrastive_temperature,
        distill_temperature=args.distill_temperature,
        lambda_img_cos=args.lambda_img_cos,
        lambda_sim_kl=args.lambda_sim_kl,
        lambda_clip=args.lambda_clip,
        calibration_batches=args.calibration_batches,
        sample_eval_batch_size=args.sample_eval_batch_size,
        apply_cle_if_available=args.apply_cle_if_available,
        export_opset=args.export_opset,
        reuse_caches_if_present=args.reuse_caches_if_present,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_one_light_qat_run(cfg)
