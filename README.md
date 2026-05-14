# LPCVC 2026 — Track 1: Latency-Constrained Image-to-Text Retrieval

Solution scripts for the [2026 IEEE Low-Power Computer Vision Challenge (LPCVC) Track 1](https://lpcv.ai/2026LPCVC/tracks/track1/) — open-world image-to-text retrieval on the **Qualcomm XR2 Gen 2 (Proxy)** NPU under a strict **35 ms** combined latency budget.

This repository deploys [Apple's MobileCLIP2](https://github.com/apple/ml-mobileclip) family of dual-encoder image–text models to Qualcomm's Hexagon NPU through the Qualcomm AI Hub, using a hardware-aware quantization pipeline. Submissions are ranked by **Recall@10** among entries that pass the latency filter.

---

## Table of contents

- [Competition overview](#competition-overview)
- [Solution at a glance](#solution-at-a-glance)
- [Repository contents](#repository-contents)
- [Pipeline architecture](#pipeline-architecture)
- [Key technical decisions](#key-technical-decisions)
- [Requirements](#requirements)
- [Directory layout assumed by the scripts](#directory-layout-assumed-by-the-scripts)
- [End-to-end usage](#end-to-end-usage)
- [Local validation against the sample dataset](#local-validation-against-the-sample-dataset)
- [Configuration reference](#configuration-reference)
- [Submission checklist](#submission-checklist)
- [Known issues and design notes](#known-issues-and-design-notes)
- [References](#references)

---

## Competition overview

| Item | Value |
|---|---|
| Task | Open-world image-to-text retrieval |
| Image input | `float32` tensor, shape `(1, 3, 224, 224)`, values in `[0, 1]` after `resize → /255 → CHW → batch` |
| Text input | `int64` token IDs, shape `(1, 77)`, OpenAI CLIP tokenizer (`openai/clip-vit-base-patch32`) |
| Target device | Qualcomm XR2 Gen 2 (Proxy) on AI Hub |
| Validity filter | Combined image + text latency **< 35 ms** |
| Ranking metric | **Recall@10** |
| Submission artifacts | Two AI Hub **compile job IDs** (image DLC + text DLC), shared with `lowpowervision@gmail.com` |

The evaluator performs **no normalization** outside the model, so any required CLIP mean/std normalization must be baked **inside** the deployed image graph.

---

## Solution at a glance

- **Model:** [MobileCLIP2-S2](https://huggingface.co/apple/MobileCLIP2-S2) — 35.7 M (image) + 63.4 M (text) parameters. Reparameterized for inference before export.
- **Image quantization:** Server-side PTQ on Qualcomm AI Hub with **W8A16 + MSE-minimizer range scheme**.
- **Text quantization:** Server-side PTQ with **W8A16** and default range scheme.
- **Deployment runtime:** **QNN DLC** (bare-metal, bypasses TensorFlow Lite delegate), with `--truncate_64bit_io` on the text branch to satisfy Hexagon's lack of 64-bit ALUs.
- **Calibration set:** ~768 raw Track-1-format images (512 from Flickr30k + 256 from COCO val2017) for the image branch; a mix of sample-set texts and generic prompts for the text branch.
- **Graph surgery:** In-graph CLIP normalization (image), repeated-EOS-to-zero padding conversion (text), SDPA replaced with explicit math for clean ONNX export, OpenCLIP `text_global_pool` patched to be statically traceable.

---

## Repository contents

| File | Role | Notes |
|---|---|---|
| **`image_encoder_s2_w8a16.py`** | End-to-end image deployment | Loads MobileCLIP2-S2 → reparameterizes → wraps with in-graph CLIP normalization → exports ONNX → Hub-optimize → Hub-quantize (W8A16+MSE) → compile to QNN DLC → profile → optional smoke inference. Includes pre/post-quantization local cosine validation. |
| **`text_encoder_s2_w8a16.py`** | End-to-end text deployment | Applies SDPA/`text_global_pool` export patches → reparameterizes → wraps with EOS-padding fix → exports ONNX → Hub-optimize → Hub-quantize (W8A16) → compile to QNN DLC (`--truncate_64bit_io`) → profile. Falls back to a float text path if `RUN_TEXT_QUANTIZATION=False`. |
| **`validate_track1_submission.py`** | Joint deployment validator | Given the two compile job IDs, runs the deployed DLCs against the official 56-image / 211-text sample set and reports Recall@1/5/10 four ways: DLC pair, PT pair (upper bound), DLC-image+PT-text, PT-image+DLC-text. |
| **`mobileclip2_image_light_qat_v3.py`** | Image-only QAT experiment | AIMET-based fake-quant calibration plus optional minimal QAT with cached teacher targets and sampled negative text embeddings. Useful when PTQ alone does not preserve retrieval fidelity. |
| **`text_encoder2.py`** | Earlier text encoder script | Earlier S4-era text deployment script. Kept for reference; the S2/W8A16 path is the current production text script. |

---

## Pipeline architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                      MobileCLIP2-S2 checkpoint                   │
│            (Apple HuggingFace: apple/MobileCLIP2-S2)             │
└─────────────────┬─────────────────────────┬──────────────────────┘
                  │                         │
        Reparameterize (MobileOne fusion)   │
                  │                         │
       ┌──────────▼──────────┐    ┌─────────▼──────────────┐
       │  Image branch       │    │  Text branch           │
       │  (FastViT trunk)    │    │  (CLIP-style transf.)  │
       └──────────┬──────────┘    └─────────┬──────────────┘
                  │                         │
   ┌──────────────▼──────────────┐   ┌──────▼─────────────────────┐
   │ CompetitionReadyImageEncoder│   │ CompetitionCompatibleText- │
   │ + in-graph CLIP normalize   │   │ Encoder + EOS-pad → zero   │
   │ (mean/std baked in)         │   │ + explicit-math SDPA       │
   └──────────────┬──────────────┘   └──────────────┬─────────────┘
                  │                                 │
        torch.onnx.export (opset 16)    torch.onnx.export (opset 16)
                  │                                 │
                  ▼                                 ▼
       pure_image_encoder.onnx          pure_text_encoder.onnx
                  │                                 │
       Hub: compile → optimized ONNX    Hub: compile → optimized ONNX
                  │                                 │
       Hub: submit_quantize_job          Hub: submit_quantize_job
       W8A16 + MSE range scheme          W8A16 (default range)
       768 Flickr30k+COCO calib          15 generic + sample prompts
                  │                                 │
                  ▼                                 ▼
       quantized image (QDQ ONNX)        quantized text (QDQ ONNX)
                  │                                 │
       Hub: compile --target_runtime    Hub: compile --target_runtime
            qnn_dlc                          qnn_dlc --truncate_64bit_io
                  │                                 │
                  ▼                                 ▼
            image.dlc                          text.dlc
                  │                                 │
                  └────────┬────────────────────────┘
                           ▼
                Hub: submit_profile_job
            (XR2 Gen 2 (Proxy), 100 iters)
                           │
                           ▼
              validate_track1_submission.py
        (sample-set Recall@1/5/10, PT vs DLC)
                           │
                           ▼
              Share compile jobs +
         Submit job IDs to lpcv.ai form
```

---

## Key technical decisions

### Backbone selection — MobileCLIP2-S2

S2 sits at the sweet spot for this competition. Its iPhone-native FP16 image/text latencies are **3.6 ms / 3.3 ms** (~7 ms combined), so even after Hexagon W8A16 quantization the combined budget should land well under 35 ms with comfortable headroom. Larger variants (S3, S4) leave little margin for quantization overhead and run into more aggressive operator-legality issues on the HTP backend.

### W8A16 over W8A8 for the image branch

The MobileCLIP image trunk is a hybrid FastViT architecture with GELU activations, Softmax, and LayerNorm in its transformer blocks. W8A8 PTQ on Hexagon HTP collapses these activations — earlier diagnostics on this codebase showed a clean PyTorch-vs-QDQ cosine drop from 1.00 → 0.21. **W8A16** keeps activations at int16, preserving the embedding direction while still halving weight memory bandwidth.

### `mse_minimizer` range scheme for image quantization

Default min-max calibration is sensitive to activation outliers, which transformer-style architectures produce heavily after LayerNorm. The MSE-minimizer scheme picks per-tensor scale factors that minimize layer-wise reconstruction error, which is markedly better for vision transformers and FastViT hybrids.

### In-graph CLIP normalization

The Track-1 evaluator only performs `resize → /255 → CHW → batch`. The model was trained on inputs normalized by CLIP mean/std, so this normalization must live inside the deployed graph. `CompetitionReadyImageEncoder` registers `mean` and `std` as buffers and applies them inside `forward()` before calling `encode_image()`. CLIP's standard `mean = (0.48145466, 0.4578275, 0.40821073)` and `std = (0.26862954, 0.26130258, 0.27577711)` are used — even though Apple's OpenCLIP loader exposes identity normalization for S0/S2/B (Apple's official README), the model's *learned weights* expect CLIP-normalized inputs.

### EOS-padding fix for the text branch

The competition tokenizer (`openai/clip-vit-base-patch32`) pads short sequences with **repeated EOS tokens (49407)** after the first EOS. The MobileCLIP/OpenCLIP text encoder, however, is implicitly aligned with **zero-padding** after the first EOS. `CompetitionCompatibleTextEncoder` preserves the first EOS (for OpenCLIP's argmax pooling) and converts all subsequent EOS tokens to zeros inside the graph. Without this fix, retrieval Recall@10 collapses to near zero even on a healthy compiled model.

### Bare-metal QNN DLC over TFLite

`--target_runtime qnn_dlc` produces a Qualcomm-native deep learning container, bypassing the TFLite delegate layer. This gives the Hexagon compiler full visibility to fuse the structurally reparameterized FastViT blocks and lets W8A16 land cleanly.

### `--truncate_64bit_io` for the text branch

PyTorch exports tokens as `int64`. Hexagon NPUs have no 64-bit ALU, so the compile flag instructs the MLIR stack to safely downcast tokens to `int32` before lowering.

### Explicit-math SDPA and static `text_global_pool`

Modern PyTorch routes `F.scaled_dot_product_attention` to fused C++ kernels that are opaque to ONNX tracing. `apply_text_export_patches()` disables MHA fast-path and monkey-patches SDPA to its explicit `Q@K^T → scale → softmax → @V` form. OpenCLIP's `text_global_pool` uses a dynamic `argmax` for the EOS position; the same patch replaces it with a static slice keyed on the input's argmax index, producing a deterministic export-friendly graph.

---

## Requirements

### System

- Python 3.10
- Linux x86_64 (development), the Hexagon NPU work happens on Qualcomm-hosted cloud devices via AI Hub
- A registered [Qualcomm AI Hub](https://app.aihub.qualcomm.com/) account and an API token

### Python packages

```bash
pip install \
    qai-hub \
    torch torchvision \
    onnx onnxruntime \
    numpy pandas pillow requests \
    datasets transformers
```

You also need a local clone of Apple's [ml-mobileclip](https://github.com/apple/ml-mobileclip) repository with OpenCLIP integrated (see Apple's README for the `git apply` setup), and the **MobileCLIP2-S2** pretrained checkpoint (`mobileclip2_s2.pt`) downloaded from [Hugging Face](https://huggingface.co/apple/MobileCLIP2-S2).

If you plan to use the QAT script, also install AIMET:

```bash
pip install aimet-torch aimet-onnx
```

### Environment variables

```bash
export QAI_HUB_API_TOKEN="your_ai_hub_api_token"
# Optional overrides
export MOBILECLIP_MODEL_NAME="MobileCLIP2-S2"
export NUM_FLICKR30K_SAMPLES=512
export NUM_COCO_SAMPLES=256
```

---

## Directory layout assumed by the scripts

The scripts use the following paths by default. Edit the `*_PATH` constants near the top of each file if your layout differs.

```
/home/aman/dev/lpcv/
├── ml-mobileclip/                            # cloned ml-mobileclip repo
│   └── mobileclip2/model_configs/
│       └── MobileCLIP2-S2.json
├── anvil/
│   ├── mobileclip2_s2.pt                     # MobileCLIP2-S2 checkpoint
│   └── sample data/
│       ├── img_list.csv                      # sample-set image metadata
│       ├── txt_list.csv                      # sample-set text metadata
│       └── images-.../images/                # sample-set image files
└── (your working directory where scripts run)
    ├── lpcvc_track1_image_s2_w8a16_artifacts/
    ├── lpcvc_track1_text_s2_w8a16_artifacts/
    └── lpcvc_track1_validation_artifacts/
```

The expected sample-set CSV format:

- **`img_list.csv`**: columns `Image_names`, `Text_nums` (semicolon-separated list of ground-truth text IDs).
- **`txt_list.csv`**: columns `Text_nums`, `Unique_Texts`.

---

## End-to-end usage

### Step 1 — Deploy the image encoder

```bash
export QAI_HUB_API_TOKEN="your_token"
python image_encoder_s2_w8a16.py
```

This will:

1. Load MobileCLIP2-S2, reparameterize, and wrap with in-graph CLIP normalization.
2. Export `pure_image_encoder.onnx`.
3. Run a local PT-vs-float-ONNX cosine sanity check (target: ≈1.0).
4. Compile to optimized ONNX on AI Hub.
5. Build the calibration set (~768 raw Track-1-format images from Flickr30k + COCO).
6. Run server-side W8A16 quantization with `--range_scheme mse_minimizer`.
7. Download the QDQ ONNX and run a local PT-vs-QDQ cosine check (warns if < 0.92).
8. Compile the quantized model to QNN DLC.
9. Profile on XR2 Gen 2 (Proxy) and run an optional smoke inference.
10. Write `image_s2_w8a16_record.json` with all job IDs and metrics.

Note the **image compile job ID** from the final printout — you will need it for the validator and for submission.

### Step 2 — Deploy the text encoder

```bash
python text_encoder_s2_w8a16.py
```

This will:

1. Apply the SDPA / `text_global_pool` / `LayerNormChannel` export patches.
2. Load MobileCLIP2-S2, reparameterize, and wrap with the EOS-padding fix.
3. Export `pure_text_encoder.onnx`.
4. Optimize on AI Hub, quantize to W8A16, validate locally.
5. Compile to QNN DLC with `--truncate_64bit_io`.
6. Profile, smoke-test, and save `text_s2_w8a16_record.json`.

Note the **text compile job ID**.

### Step 3 — Validate the joint submission against the sample set

```bash
export IMAGE_COMPILE_JOB_ID="j_xxx_from_step_1"
export TEXT_COMPILE_JOB_ID="j_xxx_from_step_2"
python validate_track1_submission.py
```

Output:

```
========== DLC pair retrieval (deployed artifacts) ==========
  hit_R@1    0.xxxxxx
  hit_R@5    0.xxxxxx
  hit_R@10   0.xxxxxx
========== PT pair retrieval (upper bound) ==========
  hit_R@10   0.9821xx  (typical for a healthy MobileCLIP2-S2)
========== Mixed retrieval (branch isolation) ==========
  image=DLC, text=PT  ...
  image=PT,  text=DLC ...

========== DECISION SUMMARY ==========
  PT  R@10 (upper bound) : 0.9821
  DLC R@10               : 0.xxxx
  Gap (quantization tax) : 0.xxxx
  ✅ DLC retains ≥85% of PT R@10 — submit with confidence.
```

The **branch isolation** rows are the most useful diagnostic: if `image=DLC, text=PT` matches the PT upper bound but `image=PT, text=DLC` collapses, your text DLC is broken. And vice versa. This is faster than retraining or re-quantizing blindly.

### Step 4 — Share and submit

Once you're confident in the validator output:

1. In `image_encoder_s2_w8a16.py` and `text_encoder_s2_w8a16.py`, set `SHARE_WITH_ORGANIZERS = True` and re-run just the sharing step (or use `compile_job.modify_sharing(...)` manually from a Python shell).
2. Submit the **image compile job ID** and **text compile job ID** through the [LPCVC submission form](https://lpcv.ai/2026LPCVC/submission/track1).

---

## Local validation against the sample dataset

`validate_track1_submission.py` is the single most important script in this repo, because it predicts your leaderboard score before you commit a submission. It computes four sets of Recall@K metrics:

| Metric set | Image source | Text source | Purpose |
|---|---|---|---|
| DLC pair | DLC | DLC | Predicts leaderboard score |
| PT pair | PyTorch | PyTorch | Upper bound (no quantization loss) |
| DLC-image + PT-text | DLC | PyTorch | Isolates image-branch quality |
| PT-image + DLC-text | PyTorch | DLC | Isolates text-branch quality |

Use the latter two rows as a binary check: if the DLC-only-on-one-side metric is close to the PT upper bound, that branch is healthy.

It also computes:

- **`cosine(PT image, DLC image)`** — should be ≥ 0.95 for W8A16; below 0.85 indicates quantization is hurting the image embedding.
- **`cosine(PT text, DLC text)`** — should be ≥ 0.98 for a properly quantized text branch.

---

## Configuration reference

All major knobs sit near the top of each script.

### `image_encoder_s2_w8a16.py`

| Variable | Default | Notes |
|---|---|---|
| `MODEL_CONFIG_PATH` | `MobileCLIP2-S2.json` | Apple's model config |
| `CHECKPOINT_PATH` | `mobileclip2_s2.pt` | MobileCLIP2-S2 checkpoint |
| `WEIGHTS_DTYPE` | `INT8` | Weights precision |
| `ACTIVATIONS_DTYPE` | `INT16` | Activations precision (W8A16) |
| `QUANTIZE_OPTIONS` | `"--range_scheme mse_minimizer"` | Calibration range scheme |
| `NUM_FLICKR30K_SAMPLES` | 512 | Calibration mix |
| `NUM_COCO_SAMPLES` | 256 | Calibration mix |
| `VALIDATION_COSINE_THRESHOLD` | 0.92 | Warns below this |
| `RUN_SMOKE_TEST_INFERENCE` | True | On-device sanity check |
| `SHARE_WITH_ORGANIZERS` | False | Flip to True when ready to submit |

### `text_encoder_s2_w8a16.py`

| Variable | Default | Notes |
|---|---|---|
| `RUN_TEXT_QUANTIZATION` | True | Set False for a guaranteed-working float text fallback |
| `WEIGHTS_DTYPE` | `INT8` | |
| `ACTIVATIONS_DTYPE` | `INT16` | |
| `COMPETITION_TOKENIZER_NAME` | `openai/clip-vit-base-patch32` | Fixed by the competition |
| `COMPETITION_EOS_TOKEN_ID` | 49407 | Fixed by the tokenizer |
| `NUM_TEXT_CALIBRATION_PROMPTS` | 256 | Calibration set size cap |

### `validate_track1_submission.py`

| Env var | Default | Notes |
|---|---|---|
| `IMAGE_COMPILE_JOB_ID` | empty | Set to your image compile job ID |
| `TEXT_COMPILE_JOB_ID` | empty | Set to your text compile job ID |
| `IMAGE_DLC_MODEL_ID` | empty | Alternative: resolve directly by model ID |
| `TEXT_DLC_MODEL_ID` | empty | Alternative: resolve directly by model ID |
| `COMPUTE_PT_REFERENCE` | True | Also runs the PT upper bound for direct comparison |

---

## Submission checklist

Before you share compile jobs and click submit:

- [ ] Image input is named `image`, shape `(1, 3, 224, 224)`, `float32`
- [ ] Text input is named `text`, shape `(1, 77)`, `int64` (truncated to `int32` at runtime via `--truncate_64bit_io`)
- [ ] In-graph CLIP normalization is present in the image encoder
- [ ] EOS-padding fix is present in the text encoder
- [ ] Profile job latency: image + text < **35 ms** combined
- [ ] Local PT-vs-DLC cosine ≥ 0.92 (image) and ≥ 0.98 (text)
- [ ] Sample-set DLC Recall@10 ≥ 85% of PT Recall@10
- [ ] Both compile jobs shared with `lowpowervision@gmail.com`
- [ ] Form submitted with both compile job IDs

---

## Known issues and design notes

### MobileCLIP2 S0 / S2 / B vs. S3 / S4 / L-14 preprocessing

Per Apple's [official README](https://github.com/apple/ml-mobileclip), S0, S2, and B variants require `image_mean=(0,0,0)` and `image_std=(1,1,1)` to be passed to `open_clip.create_model_and_transforms` so that the OpenCLIP preprocess pipeline returns identity-normalized images. The deployed graph then bakes the *real* CLIP `mean`/`std` into its own normalization. S3, S4, and L-14 use the default OpenCLIP normalization. The image script's `load_reparameterized_image_encoder()` handles this automatically based on the model name.

### Why W8A16 instead of full INT8 for the image branch

Earlier W8A8 PTQ runs on this architecture family produced clean ONNX, clean QDQ, and clean QNN DLC artifacts that nonetheless collapsed retrieval quality (PT-vs-QDQ image cosine dropped to ~0.21, sample-set Recall@10 dropped from ~0.98 to ~0.16). The collapse is concentrated at the QDQ stage and originates from FastViT's transformer attention blocks, where Hexagon HTP currently has incomplete native INT8 support for mixed-precision `MatMul` patterns involving FLOAT_16 × UFIXED_POINT_8 → FLOAT_16. W8A16 keeps activations at int16 precision and avoids the failing pattern entirely.

### When to fall back to a float text branch

If text-side W8A16 produces unexpected accuracy issues, set `RUN_TEXT_QUANTIZATION = False` in `text_encoder_s2_w8a16.py`. The float text branch is small enough that it will still profile well under budget — the text encoder is roughly 6–7 ms on iPhone FP16 and similar order on Hexagon FP16. This is documented as a defensive option, not the recommended path.

### When to reach for the QAT script

`mobileclip2_image_light_qat_v3.py` is preserved as a deeper-recovery option for cases where:

- PTQ alone cannot retain ≥ 70% of the PT Recall@10 baseline, **and**
- the dominant residual loss is in the image branch (confirmed via the validator's branch-isolation metrics).

In practice, an AIMET fake-quant **calibration-only** state (the "step-0" checkpoint, before any gradient updates) often outperforms longer QAT training in this setting; longer training can drift the image embedding away from the joint embedding space if the text branch is frozen. The script is configured with cached teacher targets, frozen text embeddings, and sampled negative text embeddings to keep the contrastive loss meaningful even at small batch sizes.

### `text_encoder2.py`

The earlier S4-era text deployment script. It contains the SDPA / `text_global_pool` / EOS-padding patches that were validated on the S4 path; the S2/W8A16 production script is structurally the same with the model swap. Kept for reference and diff'ing.

---

## References

- **Competition:** [LPCVC 2026 Track 1 official page](https://lpcv.ai/2026LPCVC/tracks/track1/), [submission form](https://lpcv.ai/2026LPCVC/submission/track1), [official sample solution](https://github.com/lpcvai/26LPCVC_Track1_Sample_Solution)
- **Model:** [Apple ml-mobileclip](https://github.com/apple/ml-mobileclip), MobileCLIP2 paper ([arXiv:2508.20691](https://arxiv.org/abs/2508.20691)), MobileCLIP CVPR 2024 ([arXiv:2311.17049](https://arxiv.org/abs/2311.17049))
- **Hardware platform:** [Qualcomm AI Hub documentation](https://app.aihub.qualcomm.com/docs/hub/), [quantization guide](https://app.aihub.qualcomm.com/docs/hub/quantize_examples.html)
- **Quantization toolkit:** [AIMET (Qualcomm AI Model Efficiency Toolkit)](https://github.com/quic/aimet)
- **Prior-year insights:** [Evaluation of Winning Solutions of 2025 LPCVC](https://arxiv.org/abs/2604.19054) — particularly the Track-1-winner Layer Merging + Imitation technique (LabLVM, Ajou University) as a future direction for additional latency headroom

---

## License and acknowledgements

This repository contains scripts to deploy Apple's MobileCLIP2 models, which are distributed under Apple's [ML Research Model TOU](https://github.com/apple/ml-mobileclip/blob/main/LICENSE_MODELS). The Qualcomm AI Hub usage is subject to Qualcomm's terms of service. Calibration data is sourced from publicly available Flickr30k and COCO val2017 splits.

Built for the IEEE Low-Power Computer Vision Challenge 2026 — Track 1.
