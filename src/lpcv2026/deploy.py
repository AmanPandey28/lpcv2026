#!/usr/bin/env python3
"""Build, validate, and optionally share an LPCVC Track 1 open submission.

The official open-submission target is Samsung Galaxy S22 (Family).  This
program compiles both encoders for that exact target, profiles them, evaluates
all frozen sample inputs using downloaded device outputs, applies fixed gates,
and shares the compile jobs only after all gates pass.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import (
    corresponding_fidelity,
    dataset_integrity,
    evaluate_stage,
    load_track1_manifest,
    preprocess_track1_image,
    sha256_file,
)
from .hub import hub_client, profile_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-model", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--reference-embeddings", type=Path, required=True)
    parser.add_argument("--image-csv", type=Path, required=True)
    parser.add_argument("--text-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="Samsung Galaxy S22 (Family)")
    parser.add_argument("--organizer-email", default="lowpowervision@gmail.com")
    parser.add_argument("--profile-iterations", type=int, default=100)
    parser.add_argument("--min-mean-cosine", type=float, default=0.99)
    parser.add_argument("--max-fractional-r10-drop", type=float, default=0.005)
    parser.add_argument("--max-sum-p90-ms", type=float, default=35.0)
    parser.add_argument(
        "--share-after-validation",
        action="store_true",
        help="Share both compile jobs with the organizer only when all gates pass.",
    )
    return parser.parse_args()


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def status_payload(job: Any) -> dict[str, Any]:
    status = job.get_status()
    code = getattr(status, "code", "unknown")
    if hasattr(code, "name"):
        code = code.name
    return {
        "code": str(code),
        "message": str(getattr(status, "message", "")),
        "failure": bool(status.failure),
    }


def wait_success(job: Any, label: str, record: dict[str, Any], output: Path) -> None:
    job.wait()
    record["jobs"][f"{label}_status"] = status_payload(job)
    write_record(output, record)
    if job.get_status().failure:
        raise RuntimeError(f"{label} failed: {job.url}")
    print(f"[{label}] SUCCESS")


def target_model(job: Any, label: str) -> Any:
    model = job.get_target_model()
    if model is None:
        raise RuntimeError(f"No target model from {label}: {job.url}")
    return model


def output_matrix(job: Any, expected_rows: int) -> tuple[str, np.ndarray]:
    data = job.download_output_data()
    if len(data) != 1:
        raise RuntimeError(f"Expected one output from {job.url}, got {list(data)}")
    name = next(iter(data))
    matrix = np.asarray(data[name], dtype=np.float32)
    if matrix.ndim == 3 and matrix.shape[1] == 1:
        matrix = matrix[:, 0, :]
    matrix = matrix.reshape(matrix.shape[0], -1)
    if matrix.shape != (expected_rows, 512):
        raise RuntimeError(
            f"Unexpected output shape {matrix.shape}; expected ({expected_rows}, 512)"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"Non-finite output values from {job.url}")
    return name, matrix


def check(value: float, threshold: float, relation: str) -> dict[str, Any]:
    if relation == "minimum":
        passed = value >= threshold
    elif relation == "maximum":
        passed = value <= threshold
    elif relation == "maximum_exclusive":
        passed = value < threshold
    else:
        raise ValueError(relation)
    return {"value": value, relation: threshold, "passed": passed}


def main() -> int:
    args = parse_args()
    if args.profile_iterations <= 0 or args.max_sum_p90_ms <= 0:
        raise ValueError("Profile iterations and latency budget must be positive")
    records, text_df, text_positions, manifest_hash = load_track1_manifest(
        args.image_csv, args.text_csv, args.image_dir
    )
    reference = np.load(args.reference_embeddings)
    reference_images = np.asarray(reference["image"], dtype=np.float32)
    reference_texts = np.asarray(reference["text"], dtype=np.float32)
    token_ids = np.asarray(reference["token_ids"], dtype=np.int32)
    if (
        "manifest_sha256" not in reference
        or str(reference["manifest_sha256"].item()) != manifest_hash
    ):
        raise ValueError(
            "Reference embeddings must be bound to this manifest; regenerate with lpcv-evaluate"
        )
    if reference_images.shape != (len(records), 512):
        raise ValueError(f"Reference image shape: {reference_images.shape}")
    if reference_texts.shape != (len(text_df), 512):
        raise ValueError(f"Reference text shape: {reference_texts.shape}")
    if token_ids.shape != (len(text_df), 77):
        raise ValueError(f"Token shape: {token_ids.shape}")
    if (
        not np.isfinite(reference_images).all()
        or not np.isfinite(reference_texts).all()
    ):
        raise ValueError("Reference embeddings must be finite")
    if token_ids.min() < 0 or token_ids.max() >= 49408:
        raise ValueError("Token IDs must be within the CLIP vocabulary")

    image_inputs = [preprocess_track1_image(Path(row.path)) for row in records]
    text_inputs = [token_ids[index : index + 1] for index in range(len(token_ids))]
    reference_metrics = evaluate_stage(
        reference_images, reference_texts, records, text_positions
    )

    image_compile_options = (
        "--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16"
    )
    text_compile_options = image_compile_options + " --truncate_64bit_io"
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "LPCVC 2026 Track 1 open submission",
        "device": args.device,
        "organizer_email": args.organizer_email,
        "models": {
            "image": {
                "path": str(args.image_model.resolve()),
                "sha256": sha256_file(args.image_model),
                "input_name": "image",
                "input_shape": [1, 3, 224, 224],
                "source_input_dtype": "float32",
                "compile_options": image_compile_options,
            },
            "text": {
                "path": str(args.text_model.resolve()),
                "sha256": sha256_file(args.text_model),
                "input_name": "text",
                "input_shape": [1, 77],
                "source_input_dtype": "int64",
                "compiled_input_dtype": "int32",
                "compile_options": text_compile_options,
            },
        },
        "dataset": {
            "num_images": len(records),
            "num_texts": len(text_df),
            "manifest_sha256": manifest_hash,
            "ordering": "img_list.csv row order",
            "integrity": dataset_integrity(records, text_positions),
        },
        "reference": {
            "embeddings_path": str(args.reference_embeddings.resolve()),
            "embeddings_sha256": sha256_file(args.reference_embeddings),
            "metrics": reference_metrics,
        },
        "jobs": {},
        "sharing": {
            "requested": bool(args.share_after_validation),
            "completed": False,
        },
    }
    write_record(args.output, record)

    import qai_hub as hub

    client = hub_client()

    available = client.get_devices(name=args.device)
    if not available:
        raise RuntimeError(f"Required device is not available: {args.device}")
    record["device_catalog_match"] = [str(device) for device in available]
    write_record(args.output, record)

    image_compile = client.submit_compile_job(
        model=str(args.image_model),
        device=hub.Device(args.device),
        input_specs={"image": ((1, 3, 224, 224), "float32")},
        options=image_compile_options,
        name="lpcvc_open_submission_mobileclip2s2_image_fp16_qnn",
    )
    text_compile = client.submit_compile_job(
        model=str(args.text_model),
        device=hub.Device(args.device),
        input_specs={"text": ((1, 77), "int64")},
        options=text_compile_options,
        name="lpcvc_open_submission_mobileclip2s2_text_fp16_qnn",
    )
    record["jobs"].update(
        {
            "image_compile_job_id": image_compile.job_id,
            "image_compile_job_url": image_compile.url,
            "text_compile_job_id": text_compile.job_id,
            "text_compile_job_url": text_compile.url,
        }
    )
    write_record(args.output, record)
    print(f"[image_compile] {image_compile.job_id}: {image_compile.url}")
    print(f"[text_compile]  {text_compile.job_id}: {text_compile.url}")
    wait_success(image_compile, "image_compile", record, args.output)
    wait_success(text_compile, "text_compile", record, args.output)
    image_model = target_model(image_compile, "image compile")
    text_model = target_model(text_compile, "text compile")

    profile_options = f"--max_profiler_iterations {args.profile_iterations}"
    image_profile = client.submit_profile_job(
        model=image_model,
        device=hub.Device(args.device),
        options=profile_options,
        name="lpcvc_open_submission_mobileclip2s2_image_profile",
    )
    text_profile = client.submit_profile_job(
        model=text_model,
        device=hub.Device(args.device),
        options=profile_options,
        name="lpcvc_open_submission_mobileclip2s2_text_profile",
    )
    record["jobs"].update(
        {
            "image_profile_job_id": image_profile.job_id,
            "image_profile_job_url": image_profile.url,
            "text_profile_job_id": text_profile.job_id,
            "text_profile_job_url": text_profile.url,
        }
    )
    write_record(args.output, record)
    print(f"[image_profile] {image_profile.job_id}: {image_profile.url}")
    print(f"[text_profile]  {text_profile.job_id}: {text_profile.url}")

    image_inference = client.submit_inference_job(
        model=image_model,
        device=hub.Device(args.device),
        inputs={"image": image_inputs},
        name="lpcvc_open_submission_mobileclip2s2_image_sample_validation",
    )
    text_inference = client.submit_inference_job(
        model=text_model,
        device=hub.Device(args.device),
        inputs={"text": text_inputs},
        name="lpcvc_open_submission_mobileclip2s2_text_sample_validation",
    )
    record["jobs"].update(
        {
            "image_inference_job_id": image_inference.job_id,
            "image_inference_job_url": image_inference.url,
            "text_inference_job_id": text_inference.job_id,
            "text_inference_job_url": text_inference.url,
        }
    )
    write_record(args.output, record)
    print(f"[image_inference] {image_inference.job_id}: {image_inference.url}")
    print(f"[text_inference]  {text_inference.job_id}: {text_inference.url}")

    wait_success(image_profile, "image_profile", record, args.output)
    wait_success(text_profile, "text_profile", record, args.output)
    wait_success(image_inference, "image_inference", record, args.output)
    wait_success(text_inference, "text_inference", record, args.output)

    image_profile_result = profile_summary(image_profile.download_profile())
    text_profile_result = profile_summary(text_profile.download_profile())
    image_output_name, device_images = output_matrix(image_inference, len(records))
    text_output_name, device_texts = output_matrix(text_inference, len(text_df))
    embeddings_path = args.output.with_name("device_embeddings.npz")
    np.savez_compressed(
        embeddings_path, image=device_images, text=device_texts, token_ids=token_ids
    )

    device_metrics = evaluate_stage(
        device_images, device_texts, records, text_positions
    )
    image_fidelity = corresponding_fidelity(reference_images, device_images)
    text_fidelity = corresponding_fidelity(reference_texts, device_texts)
    r10_drop = float(
        reference_metrics["fractional_recall_at_10"]
        - device_metrics["fractional_recall_at_10"]
    )
    combined_p90 = float(
        image_profile_result["latency_p90_ms"] + text_profile_result["latency_p90_ms"]
    )
    checks = {
        "image_mean_cosine": check(
            image_fidelity["cosine_mean"], args.min_mean_cosine, "minimum"
        ),
        "text_mean_cosine": check(
            text_fidelity["cosine_mean"], args.min_mean_cosine, "minimum"
        ),
        "fractional_recall_at_10_drop": check(
            r10_drop, args.max_fractional_r10_drop, "maximum"
        ),
        "sum_of_branch_p90_ms": check(
            combined_p90, args.max_sum_p90_ms, "maximum_exclusive"
        ),
        "image_all_recorded_ops_on_npu": {
            "value": image_profile_result["placement_operation_counts"],
            "passed": set(image_profile_result["placement_operation_counts"])
            == {"NPU"},
        },
        "text_all_recorded_ops_on_npu": {
            "value": text_profile_result["placement_operation_counts"],
            "passed": set(text_profile_result["placement_operation_counts"]) == {"NPU"},
        },
    }
    accepted = all(item["passed"] for item in checks.values())
    record.update(
        {
            "profiles": {
                "image": image_profile_result,
                "text": text_profile_result,
                "sum_of_branch_p90_ms": combined_p90,
            },
            "outputs": {
                "embeddings_path": str(embeddings_path.resolve()),
                "embeddings_sha256": sha256_file(embeddings_path),
                "image_output_name": image_output_name,
                "text_output_name": text_output_name,
                "image_shape": list(device_images.shape),
                "text_shape": list(device_texts.shape),
            },
            "device_metrics": device_metrics,
            "fidelity_vs_pytorch": {
                "image": image_fidelity,
                "text": text_fidelity,
            },
            "acceptance": {
                "status": "accepted" if accepted else "rejected",
                "checks": checks,
            },
        }
    )
    write_record(args.output, record)

    if accepted and args.share_after_validation:
        image_compile.modify_sharing(add_emails=[args.organizer_email])
        text_compile.modify_sharing(add_emails=[args.organizer_email])
        record["sharing"].update(
            {
                "completed": True,
                "shared_with": args.organizer_email,
                "image_compile_job_id": image_compile.job_id,
                "text_compile_job_id": text_compile.job_id,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_record(args.output, record)
        print(f"[sharing] compile jobs shared with {args.organizer_email}")

    print(
        "[result] "
        f"fractional R@10={device_metrics['fractional_recall_at_10']:.6f}; "
        f"image cosine={image_fidelity['cosine_mean']:.6f}; "
        f"text cosine={text_fidelity['cosine_mean']:.6f}; "
        f"sum P90={combined_p90:.4f} ms; "
        f"acceptance={record['acceptance']['status']}"
    )
    print(f"[done] {args.output}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
