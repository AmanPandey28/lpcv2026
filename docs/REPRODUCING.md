# Reproducing the workflow

## 1. Install and acquire assets

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model,hub]'
git clone https://github.com/apple/ml-mobileclip.git external/ml-mobileclip
git -C external/ml-mobileclip checkout c16bfe5a4feb424762d6bdf5245539120a4ce9ef
```

Obtain the MobileCLIP2-S2 checkpoint from Apple's upstream release instructions;
save it as `models/mobileclip2_s2.pt`. Download/extract organizer sample data from
the [track page](https://lpcv.ai/2026LPCVC/tracks/track1/) into:

```text
data/sample/img_list.csv
data/sample/txt_list.csv
data/sample/images/<image filenames>
```

Do not invent labels or mix independently sorted image lists with CSV labels.
Image CSV needs `Image_names,Text_nums` (semicolon-separated positives); text CSV
needs `Text_nums,Unique_Texts`. Duplicate text IDs or missing image files fail.
The tokenizer is downloaded/cached on first use; `--offline` requires that cache.
Model dependencies are pinned for the recorded implementation, not claimed
compatible with every future PyTorch/timm release. AIMET needs a separate env.

## 2. Export and validate locally

```bash
python -m lpcv2026.export \
  --model-repo external/ml-mobileclip \
  --checkpoint models/mobileclip2_s2.pt \
  --output-dir artifacts/export

python -m lpcv2026.evaluation \
  --model-repo external/ml-mobileclip \
  --checkpoint models/mobileclip2_s2.pt \
  --image-csv data/sample/img_list.csv \
  --text-csv data/sample/txt_list.csv \
  --image-dir data/sample/images \
  --float-image-onnx artifacts/export/image_encoder.onnx \
  --float-text-onnx artifacts/export/text_encoder.onnx \
  --output-dir artifacts/local_validation \
  --enforce-acceptance
```

Export writes two ONNX files and a synthetic parity report. Evaluation writes a
dataset manifest, PyTorch reference embeddings, branch-isolated and paired ONNX
metrics, fidelity statistics, and gates. The new reference NPZ includes its
manifest digest; deployment rejects a reference bound to different data.

The public float gates are mean cosine ≥0.99 for each branch and fractional
R@10 loss ≤0.005. They are application tolerances, not mathematical proof.
Use a broader disjoint test set before claiming general accuracy preservation.
Local ORT timings are not mobile-device benchmarks.

## 3. Compile, profile, and infer with AI Hub

Use an existing AI Hub SDK configuration, a temporary `QAI_HUB_API_TOKEN`
environment variable, or the masked prompt. Never commit a token/config file.
This command creates remote jobs and may consume cloud resources:

```bash
python -m lpcv2026.deploy \
  --image-model artifacts/export/image_encoder.onnx \
  --text-model artifacts/export/text_encoder.onnx \
  --reference-embeddings artifacts/local_validation/pytorch_embeddings.npz \
  --image-csv data/sample/img_list.csv \
  --text-csv data/sample/txt_list.csv \
  --image-dir data/sample/images \
  --device 'Samsung Galaxy S22 (Family)' \
  --output artifacts/device_validation/report.json
```

Both branches compile to FP16 HTP QNN DLC, then profile and infer all sample
inputs. The program downloads device embeddings, checks finite values and shape,
computes fractional recall and cosine, and verifies recorded NPU-only placement.
It exits nonzero on runtime failure or rejected gates. This is not a fresh
leaderboard submission unless you separately complete organizer requirements.

Sharing is disabled by default. Only add `--share-after-validation` if you intend
to grant the organizer access to the two compile jobs. Sharing requires all gates
to pass; the Google form remains a manual step. The default organizer address is
the public address in the competition instructions.

The historical implementation downloaded embeddings, not the final DLC binary.
If you separately download a target model, verify its identity and execute it
with a compatible standalone QNN runtime before claiming local DLC validation.

## 4. Optional W8A16 experiment

Prepare a representative, separate calibration file:

- Image `.npy`: FP32 `[N,3,224,224]`, RGB `[0,1]`, no CLIP mean/std.
- Text `.npy`: INT64 `[N,77]`, the same competition CLIP tokenizer and EOS padding.

Calibration is uploaded by the SDK for server-side range estimation. The graph
wrapper supplies EOS adaptation. No intermediate tensors need to be uploaded.
Keep dataset origin, sample count, hash, preprocessing, and validation separation
with every new calibration run.

```bash
python -m lpcv2026.quantize \
  --model artifacts/export/image_encoder.onnx \
  --branch image --calibration data/calibration/images.npy \
  --output-dir artifacts/ptq_image
```

For text, change `--branch`, model path, calibration, and output directory.
`--compile-profile` explicitly adds target compile/profile jobs. Historical S22
W8A16 runtime failed; this command is exploratory, not a known successful path.
It never shares jobs. Hub may download an ONNX zip with external tensor data;
use the actual returned path in the report:

```bash
python -m lpcv2026.evaluation \
  --model-repo external/ml-mobileclip \
  --checkpoint models/mobileclip2_s2.pt \
  --image-csv data/sample/img_list.csv \
  --text-csv data/sample/txt_list.csv \
  --image-dir data/sample/images \
  --quant-image-onnx artifacts/ptq_image/image_qdq.onnx \
  --output-dir artifacts/ptq_validation --enforce-acceptance
```

Replace the last model path with the actual download filename if it is an archive.
The evaluator safely extracts supported archives and preserves external data.
Image PTQ gates: mean cosine ≥0.95; fractional R@10 drop ≤0.02.
Passing these local gates still requires target execution and device fidelity.

Inspect the actual extracted ONNX graph (including its external tensor data):

```bash
python -m lpcv2026.inspect_onnx artifacts/ptq_validation/artifact_cache/path/to/model.onnx
```

Replace the example path with the extracted model location. The report lists
opsets, interfaces, operator counts, and QDQ dtype/granularity/zero-point groups.

## 5. Tests and public-file audit

```bash
python -m unittest discover -s tests -v
python scripts/audit_public_repo.py --include-untracked
```

CI uses synthetic fixtures and historical JSON consistency checks, without
credentials, model/data downloads, or cloud jobs. Full-model export and dataset
validation require the separately acquired assets. Historical AIMET QAT is not
included in CI and uses a different metric/configuration; see the experiment doc.
