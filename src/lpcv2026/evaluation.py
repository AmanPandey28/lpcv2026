#!/usr/bin/env python3
"""Reproducible local evaluator for LPCVC 2026 Track 1.

The organizer's sample implementation computes *fractional* Recall@K: for
each image, the number of relevant captions retrieved in the top K is divided
by the number of relevant captions for that image, and the result is averaged
over images.  We also report the more permissive hit Recall@K (at least one
relevant caption in the top K), but never conflate the two.

Image samples are loaded in img_list.csv row order.  This is intentional: the
organizer's upload example sorts directory entries independently, which can
misalign outputs and ground truth when filenames differ in case.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image


CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
EOS_TOKEN_ID = 49407
IMAGE_SIZE = 224


@dataclass(frozen=True)
class ImageRecord:
    name: str
    path: str
    positive_text_ids: tuple[int, ...]
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def manifest_digest(records: Sequence[ImageRecord], text_df: pd.DataFrame) -> str:
    """Hash dataset content and labels without checkout-specific paths."""
    payload = {
        "images": [
            {
                "name": record.name,
                "positive_text_ids": list(record.positive_text_ids),
                "sha256": record.sha256,
            }
            for record in records
        ],
        "texts": [
            {"id": int(row.Text_nums), "text": str(row.Unique_Texts)}
            for row in text_df.itertuples(index=False)
        ],
    }
    return canonical_digest(payload)


def load_track1_manifest(
    image_csv: Path, text_csv: Path, image_dir: Path
) -> tuple[list[ImageRecord], pd.DataFrame, dict[int, int], str]:
    image_df = pd.read_csv(image_csv, encoding="utf-8-sig")
    text_df = pd.read_csv(text_csv, encoding="utf-8-sig")
    expected_image_columns = {"Image_names", "Text_nums"}
    expected_text_columns = {"Text_nums", "Unique_Texts"}
    if not expected_image_columns.issubset(image_df.columns):
        raise ValueError(f"Unexpected image CSV columns: {list(image_df.columns)}")
    if not expected_text_columns.issubset(text_df.columns):
        raise ValueError(f"Unexpected text CSV columns: {list(text_df.columns)}")
    if image_df.empty or text_df.empty:
        raise ValueError("Image and text manifests must be nonempty")
    if image_df["Image_names"].duplicated().any():
        raise ValueError("Image filenames must be unique")
    text_df = text_df.sort_values("Text_nums").reset_index(drop=True)

    text_ids = text_df["Text_nums"].astype(int).tolist()
    if len(text_ids) != len(set(text_ids)):
        raise ValueError("Text IDs must be unique")
    text_id_to_position = {text_id: idx for idx, text_id in enumerate(text_ids)}

    records: list[ImageRecord] = []
    for row in image_df.itertuples(index=False):
        name = str(row.Image_names)
        path = image_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        positives = tuple(
            int(value) for value in str(row.Text_nums).split(";") if value
        )
        if not positives:
            raise ValueError(f"Image {name} has no positive text IDs")
        records.append(
            ImageRecord(
                name=name,
                path=str(path.resolve()),
                positive_text_ids=positives,
                sha256=sha256_file(path),
            )
        )

    return records, text_df, text_id_to_position, manifest_digest(records, text_df)


def preprocess_track1_image(path: Path) -> np.ndarray:
    """Match the organizer boundary: RGB, resize, /255, CHW, batch."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, eps)


def similarity_matrix(
    image_embeddings: np.ndarray, text_embeddings: np.ndarray
) -> np.ndarray:
    if (
        image_embeddings.ndim != 2
        or text_embeddings.ndim != 2
        or image_embeddings.shape[1] != text_embeddings.shape[1]
        or not image_embeddings.size
        or not text_embeddings.size
    ):
        raise ValueError(
            "Similarity requires nonempty matching-width embedding matrices"
        )
    if (
        not np.isfinite(image_embeddings).all()
        or not np.isfinite(text_embeddings).all()
    ):
        raise ValueError("Embedding matrices must be finite")
    return l2_normalize(image_embeddings) @ l2_normalize(text_embeddings).T


def per_query_retrieval(
    similarity: np.ndarray,
    records: Sequence[ImageRecord],
    text_id_to_position: dict[int, int],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fractional recall, hit recall, and first-positive rank per image."""
    if k <= 0 or similarity.ndim != 2 or not similarity.size:
        raise ValueError("Retrieval requires a nonempty matrix and positive K")
    if len(text_id_to_position) != similarity.shape[1] or set(
        text_id_to_position.values()
    ) != set(range(similarity.shape[1])):
        raise ValueError("Text IDs must map one-to-one onto similarity columns")
    if similarity.shape[0] != len(records):
        raise ValueError("Similarity/image-manifest row mismatch")
    k = min(k, similarity.shape[1])
    ranking = np.argsort(-similarity, axis=1)
    fractional: list[float] = []
    hit: list[float] = []
    first_ranks: list[int] = []
    for query_index, record in enumerate(records):
        # The official sample has one dangling reference (text ID 154).  Match
        # the organizer evaluator: it cannot be retrieved, but it remains in
        # the denominator of fractional recall.
        positive_positions = {
            text_id_to_position[text_id]
            for text_id in record.positive_text_ids
            if text_id in text_id_to_position
        }
        top_k = set(ranking[query_index, :k].tolist())
        matches = len(top_k & positive_positions)
        fractional.append(matches / len(record.positive_text_ids))
        hit.append(float(matches > 0))
        first_rank = next(
            (
                rank
                for rank, position in enumerate(ranking[query_index].tolist(), start=1)
                if position in positive_positions
            ),
            similarity.shape[1] + 1,
        )
        first_ranks.append(first_rank)
    return (
        np.asarray(fractional, dtype=np.float64),
        np.asarray(hit, dtype=np.float64),
        np.asarray(first_ranks, dtype=np.int64),
    )


def bootstrap_mean_ci(
    values: np.ndarray, seed: int = 42, iterations: int = 10_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[sample_indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def evaluate_stage(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    records: Sequence[ImageRecord],
    text_id_to_position: dict[int, int],
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, Any]:
    similarity = similarity_matrix(image_embeddings, text_embeddings)
    result: dict[str, Any] = {
        "num_images": int(image_embeddings.shape[0]),
        "num_texts": int(text_embeddings.shape[0]),
        "embedding_dim": int(image_embeddings.shape[1]),
    }
    for k in ks:
        fractional, hit, ranks = per_query_retrieval(
            similarity, records, text_id_to_position, k
        )
        result[f"fractional_recall_at_{k}"] = float(fractional.mean())
        result[f"fractional_recall_at_{k}_bootstrap_95ci"] = bootstrap_mean_ci(
            fractional, seed=42 + k
        )
        result[f"hit_recall_at_{k}"] = float(hit.mean())
        result[f"hit_recall_at_{k}_bootstrap_95ci"] = bootstrap_mean_ci(
            hit, seed=142 + k
        )
        if k == 10:
            reciprocal = np.where(ranks <= similarity.shape[1], 1.0 / ranks, 0.0)
            result["mean_reciprocal_rank"] = float(reciprocal.mean())
            result["median_first_positive_rank"] = float(np.median(ranks))
    return result


def dataset_integrity(
    records: Sequence[ImageRecord], text_id_to_position: dict[int, int], k: int = 10
) -> dict[str, Any]:
    dangling = sorted(
        {
            text_id
            for record in records
            for text_id in record.positive_text_ids
            if text_id not in text_id_to_position
        }
    )
    affected = [
        record.name
        for record in records
        if any(
            text_id not in text_id_to_position for text_id in record.positive_text_ids
        )
    ]
    per_image_ceiling = []
    for record in records:
        retrievable = sum(
            text_id in text_id_to_position for text_id in record.positive_text_ids
        )
        per_image_ceiling.append(min(k, retrievable) / len(record.positive_text_ids))
    return {
        "dangling_positive_text_ids": dangling,
        "affected_images": affected,
        f"fractional_recall_at_{k}_attainable_ceiling": float(
            np.mean(per_image_ceiling)
        ),
    }


def corresponding_fidelity(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.ndim != 2 or not reference.size:
        raise ValueError(
            "Fidelity requires nonempty two-dimensional embedding matrices"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Fidelity inputs must be finite")
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Fidelity shape mismatch: {reference.shape} vs {candidate.shape}"
        )
    ref_norm = l2_normalize(reference)
    cand_norm = l2_normalize(candidate)
    cosine = np.sum(ref_norm * cand_norm, axis=1)
    error = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        "cosine_p05": float(np.quantile(cosine, 0.05)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs": float(np.max(np.abs(error))),
        "norm_ratio_mean": float(
            np.mean(
                np.linalg.norm(candidate, axis=1)
                / np.maximum(np.linalg.norm(reference, axis=1), 1e-12)
            )
        ),
    }


def quantization_acceptance(
    reference_metrics: dict[str, Any],
    quantized_metrics: dict[str, Any],
    fidelity: dict[str, float],
    min_cosine: float,
    max_fractional_r10_drop: float,
) -> dict[str, Any]:
    cosine = float(fidelity["cosine_mean"])
    score_drop = float(
        reference_metrics["fractional_recall_at_10"]
        - quantized_metrics["fractional_recall_at_10"]
    )
    checks = {
        "image_cosine": {
            "value": cosine,
            "minimum": min_cosine,
            "passed": cosine >= min_cosine,
        },
        "fractional_recall_at_10_drop": {
            "value": score_drop,
            "maximum": max_fractional_r10_drop,
            "passed": score_drop <= max_fractional_r10_drop,
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {"status": "accepted" if passed else "rejected", "checks": checks}


def flat_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    if np.std(reference) < 1e-12 or np.std(candidate) < 1e-12:
        return float("nan")
    return float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1])


def build_token_matrix(texts: Sequence[str]) -> np.ndarray:
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32",
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    tokenizer.add_special_tokens({"cls_token": tokenizer.eos_token})
    return (
        tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )["input_ids"]
        .cpu()
        .numpy()
        .astype(np.int64)
    )


def load_mobileclip(
    repo_root: Path,
    checkpoint: Path,
    model_name: str,
    image_normalization: str,
):
    import importlib
    import torch
    import torch.nn as nn

    repo_string = str(repo_root.resolve())
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model

    # Register MobileCLIP2 model constructors with OpenCLIP.
    mobileclip2_module = importlib.import_module("mobileclip2.mobileclip2")
    if hasattr(mobileclip2_module, "LayerNormChannel"):
        layer_norm = mobileclip2_module.LayerNormChannel
        if not getattr(layer_norm, "_lpcv_compat_patched", False):
            original_init = layer_norm.__init__

            def compatible_init(self, *args, **kwargs):
                kwargs.pop("device", None)
                kwargs.pop("dtype", None)
                original_init(self, *args, **kwargs)

            layer_norm.__init__ = compatible_init
            layer_norm._lpcv_compat_patched = True

    model_kwargs: dict[str, Any] = {}
    if model_name.endswith(("S0", "S2", "B")):
        model_kwargs = {
            "image_mean": (0.0, 0.0, 0.0),
            "image_std": (1.0, 1.0, 1.0),
        }
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=str(checkpoint), **model_kwargs
    )
    model = reparameterize_model(model.eval()).eval()

    class ImageWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            mean = CLIP_MEAN if image_normalization == "clip" else (0.0, 0.0, 0.0)
            std = CLIP_STD if image_normalization == "clip" else (1.0, 1.0, 1.0)
            self.register_buffer(
                "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
            )

        def forward(self, image):
            image = image.to(torch.float32)
            return self.inner.encode_image((image - self.mean) / self.std)

    class TextWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, text):
            text = text.to(torch.int64)
            eos = text == EOS_TOKEN_ID
            repeated_eos = eos & (torch.cumsum(eos.to(torch.int64), dim=1) > 1)
            text = torch.where(repeated_eos, torch.zeros_like(text), text)
            return self.inner.encode_text(text)

    return ImageWrapper(model).eval(), TextWrapper(model).eval()


def run_pytorch_reference(
    repo_root: Path,
    checkpoint: Path,
    model_name: str,
    image_inputs: Sequence[np.ndarray],
    token_matrix: np.ndarray,
    text_batch_size: int,
    image_normalization: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    import torch

    started = time.perf_counter()
    image_model, text_model = load_mobileclip(
        repo_root, checkpoint, model_name, image_normalization
    )
    image_outputs: list[np.ndarray] = []
    text_outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for image in image_inputs:
            image_outputs.append(image_model(torch.from_numpy(image)).cpu().numpy())
        for start in range(0, len(token_matrix), text_batch_size):
            token_batch = torch.from_numpy(
                token_matrix[start : start + text_batch_size]
            )
            text_outputs.append(text_model(token_batch).cpu().numpy())
    elapsed = time.perf_counter() - started
    return (
        np.concatenate(image_outputs, axis=0).astype(np.float32),
        np.concatenate(text_outputs, axis=0).astype(np.float32),
        {"wall_seconds": elapsed},
    )


def resolve_onnx_artifact(path: Path, cache_dir: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    artifact_hash = sha256_file(path)[:16]
    extract_dir = cache_dir / f"{path.stem}-{artifact_hash}"
    marker = extract_dir / ".complete"
    if not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                target = (extract_dir / member.filename).resolve()
                if not target.is_relative_to(extract_dir.resolve()):
                    raise ValueError("Unsafe archive member path")
            archive.extractall(extract_dir)
        marker.write_text("ok\n", encoding="utf-8")
    candidates = sorted(extract_dir.rglob("*.onnx"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one ONNX in {path}, found {candidates}")
    return candidates[0]


def run_onnx_model(
    path: Path,
    inputs: Sequence[np.ndarray],
    threads: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_seconds = time.perf_counter() - started
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    outputs: list[np.ndarray] = []
    run_started = time.perf_counter()
    for value in inputs:
        feed = value
        if "int32" in input_meta.type:
            feed = value.astype(np.int32)
        elif "int64" in input_meta.type:
            feed = value.astype(np.int64)
        outputs.append(np.asarray(session.run(None, {input_name: feed})[0]))
    run_seconds = time.perf_counter() - run_started
    output = np.concatenate(outputs, axis=0)
    metadata = {
        "path": str(path.resolve()),
        "input_name": input_name,
        "input_type": input_meta.type,
        "output_names": [value.name for value in session.get_outputs()],
        "load_wall_seconds": load_seconds,
        "inference_wall_seconds": run_seconds,
    }
    del session
    gc.collect()
    return output.reshape(output.shape[0], -1).astype(np.float32), metadata


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = [
        "torch",
        "torchvision",
        "open-clip-torch",
        "onnx",
        "onnxruntime",
        "transformers",
        "numpy",
        "pandas",
        "pillow",
    ]
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="MobileCLIP2-S2")
    parser.add_argument("--image-csv", type=Path, required=True)
    parser.add_argument("--text-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--float-image-onnx", type=Path)
    parser.add_argument("--quant-image-onnx", type=Path)
    parser.add_argument("--float-text-onnx", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument(
        "--image-normalization",
        choices=("auto", "clip", "identity"),
        default="auto",
        help="Normalization baked into the PyTorch reference image wrapper.",
    )
    parser.add_argument("--min-quant-image-cosine", type=float, default=0.95)
    parser.add_argument("--max-quant-fractional-r10-drop", type=float, default=0.02)
    parser.add_argument(
        "--enforce-acceptance",
        action="store_true",
        help="Exit nonzero when a supplied quantized artifact fails its gates.",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use cached tokenizer files only."
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    if args.threads <= 0 or args.text_batch_size <= 0:
        raise ValueError("Thread count and text batch size must be positive")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    image_normalization = args.image_normalization
    if image_normalization == "auto":
        image_normalization = (
            "identity" if args.model_name.endswith(("S0", "S2", "B")) else "clip"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "artifact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records, text_df, text_id_to_position, manifest_hash = load_track1_manifest(
        args.image_csv, args.text_csv, args.image_dir
    )
    integrity = dataset_integrity(records, text_id_to_position)
    image_inputs = [preprocess_track1_image(Path(record.path)) for record in records]
    token_matrix = build_token_matrix(text_df["Unique_Texts"].astype(str).tolist())

    report: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "name": args.model_name,
            "pytorch_image_normalization": image_normalization,
        },
        "dataset": {
            "name": "LPCVC 2026 Track 1 official sample set",
            "num_images": len(records),
            "num_texts": len(text_df),
            "manifest_sha256": manifest_hash,
            "ordering": "img_list.csv row order",
            "primary_metric": "organizer fractional Recall@K",
            "integrity": integrity,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "artifacts": {
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": sha256_file(args.checkpoint),
            }
        },
        "stages": {},
    }

    print(f"[dataset] {len(records)} images, {len(text_df)} texts")
    print(f"[dataset] manifest sha256: {manifest_hash}")
    print("[pytorch] computing corrected image and text reference embeddings...")
    pt_images, pt_texts, pt_timing = run_pytorch_reference(
        args.model_repo,
        args.checkpoint,
        args.model_name,
        image_inputs,
        token_matrix,
        args.text_batch_size,
        image_normalization,
    )
    pt_similarity = similarity_matrix(pt_images, pt_texts)
    report["stages"]["pytorch_image__pytorch_text"] = {
        "metrics": evaluate_stage(pt_images, pt_texts, records, text_id_to_position),
        "timing": pt_timing,
    }
    np.savez_compressed(
        args.output_dir / "pytorch_embeddings.npz",
        image=pt_images,
        text=pt_texts,
        token_ids=token_matrix,
        manifest_sha256=np.asarray(manifest_hash),
    )
    print(
        "[pytorch] fractional R@10=",
        report["stages"]["pytorch_image__pytorch_text"]["metrics"][
            "fractional_recall_at_10"
        ],
    )

    stage_inputs: list[tuple[str, Path]] = []
    if args.float_image_onnx:
        stage_inputs.append(("float_onnx_image", args.float_image_onnx))
    if args.quant_image_onnx:
        stage_inputs.append(("quantized_onnx_image", args.quant_image_onnx))

    float_image_embeddings = None
    for stage_name, source_path in stage_inputs:
        resolved_path = resolve_onnx_artifact(source_path, cache_dir)
        print(f"[{stage_name}] running {resolved_path} ...")
        image_embeddings, runtime_meta = run_onnx_model(
            resolved_path, image_inputs, args.threads
        )
        if stage_name == "float_onnx_image":
            float_image_embeddings = image_embeddings
        stage_similarity = similarity_matrix(image_embeddings, pt_texts)
        report["artifacts"][stage_name] = {
            "source_path": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
            "resolved_path": str(resolved_path.resolve()),
        }
        report["stages"][f"{stage_name}__pytorch_text"] = {
            "metrics": evaluate_stage(
                image_embeddings, pt_texts, records, text_id_to_position
            ),
            "image_fidelity_vs_pytorch": corresponding_fidelity(
                pt_images, image_embeddings
            ),
            "similarity_matrix_correlation_vs_pytorch": flat_correlation(
                pt_similarity, stage_similarity
            ),
            "runtime": runtime_meta,
        }
        np.savez_compressed(
            args.output_dir / f"{stage_name}_embeddings.npz", image=image_embeddings
        )
        metrics = report["stages"][f"{stage_name}__pytorch_text"]["metrics"]
        fidelity = report["stages"][f"{stage_name}__pytorch_text"][
            "image_fidelity_vs_pytorch"
        ]
        print(
            f"[{stage_name}] fractional R@10={metrics['fractional_recall_at_10']:.6f} "
            f"hit R@10={metrics['hit_recall_at_10']:.6f} "
            f"cosine={fidelity['cosine_mean']:.6f}"
        )

    quantized_stage = report["stages"].get("quantized_onnx_image__pytorch_text")
    if quantized_stage is not None:
        report["acceptance"] = quantization_acceptance(
            report["stages"]["pytorch_image__pytorch_text"]["metrics"],
            quantized_stage["metrics"],
            quantized_stage["image_fidelity_vs_pytorch"],
            args.min_quant_image_cosine,
            args.max_quant_fractional_r10_drop,
        )
        print(f"[acceptance] {report['acceptance']['status']}")

    if args.float_text_onnx:
        resolved_text = resolve_onnx_artifact(args.float_text_onnx, cache_dir)
        text_inputs = [
            token_matrix[index : index + 1] for index in range(len(token_matrix))
        ]
        print(f"[float_onnx_text] running {resolved_text} ...")
        text_embeddings, runtime_meta = run_onnx_model(
            resolved_text, text_inputs, args.threads
        )
        report["artifacts"]["float_onnx_text"] = {
            "source_path": str(args.float_text_onnx.resolve()),
            "source_sha256": sha256_file(args.float_text_onnx),
            "resolved_path": str(resolved_text.resolve()),
        }
        report["stages"]["pytorch_image__float_onnx_text"] = {
            "metrics": evaluate_stage(
                pt_images, text_embeddings, records, text_id_to_position
            ),
            "text_fidelity_vs_pytorch": corresponding_fidelity(
                pt_texts, text_embeddings
            ),
            "runtime": runtime_meta,
        }
        if float_image_embeddings is not None:
            pair_metrics = evaluate_stage(
                float_image_embeddings, text_embeddings, records, text_id_to_position
            )
            report["stages"]["float_onnx_image__float_onnx_text"] = {
                "metrics": pair_metrics
            }
            float_checks = {
                "image_cosine": corresponding_fidelity(
                    pt_images, float_image_embeddings
                )["cosine_mean"]
                >= 0.99,
                "text_cosine": corresponding_fidelity(pt_texts, text_embeddings)[
                    "cosine_mean"
                ]
                >= 0.99,
                "fractional_r10_drop": report["stages"]["pytorch_image__pytorch_text"][
                    "metrics"
                ]["fractional_recall_at_10"]
                - pair_metrics["fractional_recall_at_10"]
                <= 0.005,
            }
            report["float_acceptance"] = {
                "status": "accepted" if all(float_checks.values()) else "rejected",
                "checks": float_checks,
            }

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path = args.output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_hash,
                "images": [asdict(record) for record in records],
                "texts": [
                    {"id": int(row.Text_nums), "text": str(row.Unique_Texts)}
                    for row in text_df.itertuples(index=False)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote {report_path}")
    if args.enforce_acceptance and any(
        report.get(key, {}).get("status") == "rejected"
        for key in ("acceptance", "float_acceptance")
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
