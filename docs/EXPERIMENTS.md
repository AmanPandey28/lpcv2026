# Optimization decisions and unsuccessful paths

## 1. Establish the reference

Generic CLIP mean/std on S2 produced sample fractional R@10=0.0431548.
Correcting to identity normalization restored 0.8974702. This was an input
correctness fix, not a learned model accuracy improvement. Tokenizer adaptation,
static export, and reparameterization were checked before changing precision.

## 2. W8A16 post-training quantization

Float ONNX → AI Hub optimize-to-ONNX → calibration/PTQ → QDQ ONNX → QNN DLC.
Image used `--range_scheme mse_minimizer`; text used default range selection.
API requests specified INT8 weights and INT16 activations. Observed QDQ
encodings predominantly used symmetric per-output-channel INT8 weights and
scalar per-tensor affine 16-bit activations, including UINT16 tensors; bias
tensors can be INT32. Precision labels alone do not specify all tensor encodings.

QDQ means QuantizeLinear / DequantizeLinear edges:
`q = clamp(round(x/scale) + zero_point)` and
`x_hat = (q - zero_point) * scale`. A runtime can lower these into integer
kernels. The final FP16 model must not be called W8A16 because this experiment
occurred earlier.

The [retained image QDQ inspection](../results/historical_image_qdq_inspection.json)
shows 198 INT8 per-axis parameter quantizers with zero-centered encodings,
predominantly UINT16 per-tensor intermediates, and INT32 bias dequantization.
It also shows **opset 21** after Hub transformation, despite the source export
being opset 16. Do not assume optimization/quantization preserves the source
opset. This observation alone does not prove the runtime failure's cause.

### Calibration

Representative unlabeled inputs estimate activation/intermediate ranges. This
is not restricted to linear layers: convolution, attention, residual, and other
tensor boundaries can need ranges. Weight scales can be derived from weights.

The retained image cache contains 512 Flickr30k inputs in the raw Track 1 format.
An earlier builder also offered 768 Flickr30k+COCO images, but that option is not
proof that 768 were used. Text used 256 OpenCLIP ImageNet-template prompts,
excluding exact evaluation captions; this does not prove semantic disjointness.

The SDK received an input-name mapping to a list of batch-one NumPy arrays,
e.g. `{"image": [array_0, array_1, ...]}`, and uploaded them for server-side
quantization. This was not live on-device streaming or an upload of intermediate
activations. [quantize.py](../src/lpcv2026/quantize.py) records calibration count
and hash and accepts prepared arrays without redistributing datasets.

### Local numerical outcome

Image QDQ with unchanged PyTorch text:

- Fractional R@10 0.885714 vs. reference 0.897470.
- Mean image cosine 0.810573; similarity correlation approximately 0.932883.
- Gate: mean cosine ≥0.95, R@10 drop ≤0.02. Recall passed; fidelity failed.

A small recall change did not justify accepting a major embedding rotation.
Text QDQ cosine was approximately 0.999765, but local fidelity did not guarantee
target execution. Exact quantization claims require inspecting QDQ scales,
zero points, signedness, and axes, not reading only an API precision option.

## 3. Compile passed; runtime failed

Fresh S22 W8A16 image and text jobs both compiled, then failed during graph
composition: `QnnModel_composeGraphsFromDlc: MODEL_GRAPH_ERROR`.
No usable device outputs resulted from these runs.

The [lineage record](../results/w8a16_runtime_lineage.json) preserves job IDs.
Fresh target-specific builds rule out reused cross-target artifacts for those
runs; two failing branches rule out a solely image-specific topology.
Successful float runs establish that the target can execute the float graphs.

These controls support investigating a W8A16 QDQ/QNN DLC/S22-v69 compatibility
limitation. They do not identify an exact operator or prove a compiler defect.
No minimized reproducer or definitive vendor diagnosis was available. The
failed graph was not silently rescued by CPU execution. A QCS8550 profile is
not interchangeable with an S22 result or a remedy for poor image fidelity.

## 4. Historical S4 / AIMET QAT

[aimet_image_qat.py](../experiments/aimet_image_qat.py) retains image-only S4 work:
teacher image targets, frozen text targets, sampled negatives, image-cosine,
similarity-KL, and contrastive losses. The initial fake-quant state remains a
checkpoint candidate; more training is not assumed to be better.

Its default W8A8 range-learning setup differs from S2 server-side W8A16.
Historical checkpoint selection uses **hit** R@10, not canonical fractional
recall. The script requires a separate compatible AIMET environment and is not
CI-validated. No final QAT deployment recovery or measured QAT speedup is claimed.
The S4 reference JSON is a PyTorch model-selection record, not device evidence.

## 5. Select the validated precision

S2 FP16 DLCs executed, preserved sample retrieval, passed embedding gates, and
fit the budget. This is a precision fallback from integer PTQ, not a CPU fallback.
All 794 operations recorded in the final branch profiles were NPU-placed.
ONNX Runtime CPU evaluation is an intentional correctness check, not target
fallback. Heterogeneous partitioning claims require the actual target profile.

Next steps are larger disjoint data, sensitive-quantizer isolation/mixed
precision, an operator-level failure reproducer, and end-to-end profiling.
These are future work, not completed results.
