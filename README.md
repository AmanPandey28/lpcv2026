# MobileCLIP2 on Qualcomm: LPCVC 2026 Track 1

[![CI](https://github.com/AmanPandey28/lpcv2026/actions/workflows/ci.yml/badge.svg)](https://github.com/AmanPandey28/lpcv2026/actions/workflows/ci.yml)

Export, validate, and deploy a compact image–text retrieval model on a Qualcomm
mobile accelerator. This project adapts Apple's **MobileCLIP2-S2** to the LPCVC
input contract, investigates integer quantization, and validates a working
**FP16 QNN DLC deployment on Samsung Galaxy S22 (Family)**.

The result is a model engineering workflow—not a new pretrained architecture or
a claim that every optimization improved performance.

## Goals and verified outcome

Rank candidate captions for an image using cosine similarity between separate
image and text embeddings. Preserve retrieval quality through export, meet the
combined encoder latency budget, and catch accuracy/runtime regressions before
submission.

| Recorded result | Value |
| --- | --- |
| Actual deployment target | Galaxy S22 (Family), Snapdragon 8 Gen 1 / SM8450 |
| Final precision / runtime | FP16 HTP execution / QNN DLC |
| Image / text P90, 100 iterations each | 14.5121 / 3.9856 ms |
| Sum of branch P90s | **18.4977 ms**, below the 35 ms budget |
| Sample fractional Recall@10, PyTorch → device | **0.897470 → 0.897470** |
| Mean embedding cosine vs. PyTorch, image / text | **0.999863 / 0.999998** |
| Recorded placement | 405 image + 389 text operations on NPU |

Historical measurements on **56 sample images and 211 candidate texts**—not
hidden-test accuracy or a new hardware run. Summed branch P90 is not a measured
joint/end-to-end P90. See [results and limitations](docs/RESULTS.md) and the
[machine-readable record](results/open_submission_fp16.json).

The [organizer's track page](https://lpcv.ai/2026LPCVC/tracks/track1/) specifies
Galaxy S22 (Family) for open submissions; XR2 Gen 2 (Proxy) was the original
competition target. These are not literal XR2 measurements.

## Implementation

```text
MobileCLIP2-S2 checkpoint
  → eval mode + upstream structural reparameterization
  → input-contract and export rewrites
  → static, batch-one, opset-16 ONNX
  → ONNX Runtime parity and retrieval checks
  → AI Hub QNN DLC compile with explicit FP16 HTP precision
  → device profiling + full sample inference
  → downloaded embeddings + local fidelity/accuracy gates
```

[Export adapters](src/lpcv2026/models.py) lower fused attention to matrix operations
and replace batch-one text pooling. [Reference wrappers](src/lpcv2026/evaluation.py)
preserve the first EOS token, zero repeated EOS padding, and use **identity image
normalization** for S2. Image input is RGB FP32 `[1,3,224,224]` in `[0,1]`;
text ONNX input is INT64 `[1,77]`, converted to INT32 for QNN.
Each output has 512 features; the evaluator applies L2 normalization.
See [architecture and rewrites](docs/ARCHITECTURE.md).

## What was tried

| Approach | Outcome | Decision |
| --- | --- | --- |
| S2 with generic CLIP mean/std | Sample fractional R@10 collapsed to 0.043155 | Fix checkpoint-specific preprocessing |
| Corrected S2 float ONNX | Preserved sample retrieval and embedding fidelity | Keep as baseline |
| W8A16 PTQ with image MSE ranges | Image cosine 0.810573; R@10 0.885714 | Reject under fidelity gate |
| Fresh S22 W8A16 image/text DLCs | Compile passed; runtime graph composition failed | Not a deployable result |
| Historical S4 / AIMET light QAT | Research code; no validated final deployment | Keep separate from final path |
| S2 FP16 QNN DLC | Both branches ran; sample R@10 unchanged; within budget | Final validated route |

[Experiment details](docs/EXPERIMENTS.md) cover calibration, QDQ, the graph error,
and what the diagnosis does and does not establish.

## Repository layout

```text
src/lpcv2026/       Portable export, evaluation, quantization, and deployment
experiments/       Labeled historical AIMET QAT research
tests/             Metrics, graph rewrites, profiling, and record checks
results/           Small, sanitized historical measurement records
docs/              Architecture, experiments, reproduction, and provenance
scripts/           Publication hygiene audit
.github/workflows/ CI without cloud credentials or model downloads
```

Model binaries, dataset images, raw embeddings, credentials, generated logs, and
personal documents are intentionally excluded.

## Quick start

Python 3.10+. Metric tests need only the lightweight base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/audit_public_repo.py
```

For model export:

```bash
python -m pip install -e '.[model]'
git clone https://github.com/apple/ml-mobileclip.git external/ml-mobileclip
# Obtain Apple's MobileCLIP2-S2 checkpoint and organizer sample data separately.
python -m lpcv2026.export \
  --model-repo external/ml-mobileclip \
  --checkpoint models/mobileclip2_s2.pt \
  --output-dir artifacts/export
```

[REPRODUCING.md](docs/REPRODUCING.md) covers dataset placement, full local
validation, AI Hub deployment, and optional W8A16 experiments.
Synthetic export checks are not a substitute for dataset validation.
AI Hub commands create remote jobs; sharing is opt-in.

## Scope and next steps

This project demonstrates adaptation, static graph rewrites, numerical validation,
and hardware profiling. It does **not** claim custom Triton kernels, torch.compile
deployment, LLM KV-cache/continuous batching, distributed serving, energy
measurements, or a solved W8A16 compiler defect.

Next useful work: larger disjoint retrieval validation, a minimized W8A16 runtime
reproducer, selective mixed precision, and measured end-to-end latency.
These are not reported as completed.

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[upstream/data provenance](docs/PROVENANCE.md).
