"""Export MobileCLIP2-S2 to checked, static, opset-16 ONNX graphs."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch

from .evaluation import (
    corresponding_fidelity,
    load_mobileclip,
    run_onnx_model,
    sha256_file,
)
from .models import export_rewrites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.model_repo.is_dir():
        parser.error(
            "Provide an existing Apple ml-mobileclip checkout and S2 checkpoint"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(8)
    image_model, text_model = load_mobileclip(
        args.model_repo, args.checkpoint, "MobileCLIP2-S2", "identity"
    )
    image = torch.rand(1, 3, 224, 224)
    text = torch.full((1, 77), 49407, dtype=torch.int64)
    text[0, :5] = torch.tensor([49406, 320, 1125, 539, 320])
    report = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "opset": 16,
        "note": "Synthetic export smoke test; use lpcv-evaluate for full dataset gates",
        "graphs": {},
    }
    for name, model, sample in (
        ("image", image_model, image),
        ("text", text_model, text),
    ):
        destination = args.output_dir / f"{name}_encoder.onnx"
        # The independent reference is computed BEFORE enabling graph rewrites.
        with torch.inference_mode():
            reference = model(sample).numpy()
        with torch.inference_mode(), export_rewrites():
            rewritten = model(sample).numpy()
            torch.onnx.export(
                model,
                sample,
                str(destination),
                input_names=[name],
                output_names=[f"{name}_embeddings"],
                opset_version=16,
                dynamo=False,
                export_params=True,
                do_constant_folding=True,
                dynamic_axes=None,
                training=torch.onnx.TrainingMode.EVAL,
                external_data=False,
            )
        graph = onnx.load(str(destination))
        onnx.checker.check_model(graph, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
        onnx.checker.check_model(inferred)
        outputs, _ = run_onnx_model(destination, [sample.numpy()], threads=8)
        np.testing.assert_allclose(rewritten, reference, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(outputs, reference, rtol=2e-3, atol=2e-4)
        report["graphs"][name] = {
            "path": destination.name,
            "sha256": sha256_file(destination),
            "input_shape": list(sample.shape),
            "input_dtype": str(sample.dtype),
            "output_shape": list(outputs.shape),
            "rewritten_vs_native": corresponding_fidelity(reference, rewritten),
            "onnx_vs_native": corresponding_fidelity(reference, outputs),
        }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"Export and synthetic parity checks passed: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
