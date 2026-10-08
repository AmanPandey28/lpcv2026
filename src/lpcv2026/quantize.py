"""Experimental W8A16 PTQ path. It is NOT the validated final submission.

Calibration arrays must already match the exported graph's public input contract.
This command never shares jobs; profiling is explicit because cloud jobs use resources.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import sha256_file
from .hub import hub_client, profile_summary


def calibration_samples(path, branch):
    values = np.load(path, allow_pickle=False)
    expected = (3, 224, 224) if branch == "image" else (77,)
    if (
        values.ndim != len(expected) + 1
        or values.shape[1:] != expected
        or not len(values)
    ):
        raise ValueError(
            f"Expected nonempty [N,{','.join(map(str, expected))}] calibration array"
        )
    if not np.isfinite(values).all():
        raise ValueError("Calibration must contain finite values")
    if branch == "image":
        if values.dtype != np.float32 or values.min() < 0 or values.max() > 1:
            raise ValueError(
                "Image calibration must be float32 RGB CHW in [0,1], without CLIP normalization"
            )
    elif values.dtype != np.int64 or values.min() < 0 or values.max() >= 49408:
        raise ValueError("Text calibration must be int64 CLIP tokens in [0,49408)")
    return [np.ascontiguousarray(values[i : i + 1]) for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--branch", choices=("image", "text"), required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="Samsung Galaxy S22 (Family)")
    parser.add_argument("--compile-profile", action="store_true")
    args = parser.parse_args()
    samples = calibration_samples(args.calibration, args.branch)
    import qai_hub as hub

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = hub_client()
    device = hub.Device(args.device)
    shape = (1, 3, 224, 224) if args.branch == "image" else (1, 77)
    dtype = "float32" if args.branch == "image" else "int64"
    quant_options = "--range_scheme mse_minimizer" if args.branch == "image" else ""
    record = {
        "schema_version": 1,
        "experiment": "W8A16 PTQ",
        "device": args.device,
        "model_sha256": sha256_file(args.model),
        "calibration_sha256": sha256_file(args.calibration),
        "samples": len(samples),
        "weights_dtype": "INT8",
        "activations_dtype": "INT16",
        "quantize_options": quant_options,
        "jobs": {},
        "warning": "Compilation success does not establish runtime validity or accuracy",
    }

    def save():
        (args.output_dir / "report.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )

    def wait(job, stage):
        record["jobs"][stage] = {"id": job.job_id, "url": job.url}
        save()
        job.wait()
        status = job.get_status()
        record["jobs"][stage].update(
            {"failure": bool(status.failure), "status": str(status)}
        )
        save()
        if status.failure:
            raise RuntimeError(f"{stage} failed; see {job.url}")
        model = job.get_target_model()
        if stage != "profile" and model is None:
            raise RuntimeError(f"No output model for {stage}")
        return model

    optimized = wait(
        client.submit_compile_job(
            model=str(args.model),
            device=device,
            input_specs={args.branch: (shape, dtype)},
            options="--target_runtime onnx",
            name=f"lpcv_s2_{args.branch}_optimize",
        ),
        "optimize",
    )
    quantized = wait(
        client.submit_quantize_job(
            model=optimized,
            calibration_data={args.branch: samples},
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            options=quant_options,
            name=f"lpcv_s2_{args.branch}_w8a16",
        ),
        "quantize",
    )
    # Hub can return an ONNX archive with external tensor data. Keep its returned path.
    artifact = quantized.download(str(args.output_dir / f"{args.branch}_qdq.onnx"))
    record["downloaded_artifact"] = str(artifact)
    save()
    if args.compile_profile:
        options = "--target_runtime qnn_dlc"
        if args.branch == "text":
            options += " --truncate_64bit_io"
        compiled = wait(
            client.submit_compile_job(
                model=quantized,
                device=device,
                input_specs={args.branch: (shape, dtype)},
                options=options,
                name=f"lpcv_s2_{args.branch}_w8a16_dlc",
            ),
            "compile",
        )
        profile = client.submit_profile_job(
            model=compiled, device=device, options="--max_profiler_iterations 100"
        )
        wait(profile, "profile")
        record["profile"] = profile_summary(profile.download_profile())
        save()
    print(
        f"Experiment recorded: {args.output_dir}; evaluate QDQ fidelity before promotion"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
