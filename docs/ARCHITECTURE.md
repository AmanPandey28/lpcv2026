# Architecture and graph adaptations

## Model boundary

MobileCLIP2 is a contrastively trained dual encoder. Image and text branches map
inputs to a shared 512-dimensional space; retrieval ranks rows of
`normalize(image) @ normalize(text).T`. This project deploys an existing S2
checkpoint, not a new architecture or Apple's original training method.

| Component | Upstream S2 configuration |
| --- | --- |
| Vision | FastViT MCi2, average pooling, 512-feature output |
| Vision stages | depths (4,12,24,4), widths (80,160,320,640) |
| Token mixers | RepMixer in the first three stages; attention in the last |
| Text | 12 transformer layers, width 512, 8 heads |
| Vocabulary / context | 49,408 / 77 |
| Text attention | Bidirectional: `no_causal_mask=true` |
| Native image configuration | 256 pixels; competition graph uses 224 |

Using 224 pixels is an explicit deployment-contract adaptation, not a claim
that the upstream checkpoint was configured at 224. Parameter totals depend on
implementation version and whether branches have been reparameterized.

## Reparameterization

The loader calls upstream `reparameterize_model(model.eval())`. Supported
training-time convolution/normalization branches are algebraically collapsed
into inference-time weights. For Conv→BatchNorm with frozen statistics:

```text
W' = W * gamma / sqrt(variance + epsilon)
b' = (b - mean) * gamma / sqrt(variance + epsilon) + beta
```

Compatible branch weights can then be added into a single convolution. This is
not quantization or gradient-based fine-tuning. No isolated hardware speedup
against the unreparameterized model was recorded.

## Input-contract fixes

1. **Image:** RGB, resize to 224×224, divide by 255, transpose to CHW, batch.
   S2 uses identity normalization: mean=(0,0,0), std=(1,1,1). Generic CLIP
   normalization was incorrect. The evaluator's `clip` mode is diagnostic.
2. **Text:** preserve the first EOS=49407 and zero subsequent EOS padding before
   the bidirectional encoder. Padding affects attention values, not only pooling.
3. **Integer interface:** ONNX takes INT64 token IDs; `--truncate_64bit_io`
   makes compiled QNN input INT32. Device feeds must be INT32 and in vocabulary.

See [evaluation.py](../src/lpcv2026/evaluation.py): `load_mobileclip`,
`preprocess_track1_image`, and `build_token_matrix`.

## Attention and pooling rewrites

[models.py](../src/lpcv2026/models.py) exposes inference SDPA as:

```python
scores = (query @ key.transpose(-2, -1)) / sqrt(head_dim)
scores = scores + additive_mask  # when present
output = softmax(scores, dim=-1) @ value
```

This makes standard ONNX operations visible instead of a fused PyTorch path.
Boolean/additive and causal masks are unit-tested, but S2 text is noncausal.
Nonzero dropout and GQA are rejected; this is not a general training replacement.

During export, fused MHA is disabled and OpenCLIP's batch-indexed argmax pooling
becomes a single-row gather valid **only for static batch one**. The patch context
restores original functions even after exceptions. Patches are process-global
while active; do not run unrelated model execution concurrently.

The CPU/FP32 constructor shim removes unsupported `device`/`dtype` kwargs from
the upstream `LayerNormChannel`. It is not a general device-aware constructor.

## ONNX configuration

**Opset 16** selects versioned operator semantics; it does not mean INT16
precision. Fixed inputs are `[1,3,224,224]` and `[1,77]`, with no dynamic batch,
resolution, or context length.

[export.py](../src/lpcv2026/export.py) explicitly uses the legacy exporter
(`dynamo=False`), evaluation mode, embedded parameters, constant folding,
named inputs/outputs, no dynamic axes, and no external weight file. Opset 16 is
the recorded backend choice, not a universal recommendation.

Checks include ONNX schema/shape validity and synthetic native-PyTorch → rewritten
PyTorch → ONNX Runtime parity. Dataset fidelity and retrieval are separate gates.
A valid ONNX file can still be unsupported or inaccurate on a target runtime.
The refactored exporter need not have historical graph byte hashes; revalidate
new artifacts locally and on-device before submission.
