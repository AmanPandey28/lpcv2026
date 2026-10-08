# Provenance and distribution scope

## Upstream dependencies

- [Apple MobileCLIP / MobileCLIP2](https://github.com/apple/ml-mobileclip):
  architecture, configs, pretrained weights, upstream reparameterization.
- [OpenCLIP](https://github.com/mlfoundations/open_clip): model construction/text.
- [timm](https://github.com/huggingface/pytorch-image-models): FastViT components.
- [LPCVC sample solution](https://github.com/lpcvai/26LPCVC_Track1_Sample_Solution):
  task input boundary and organizer-style evaluation.
- [Qualcomm AI Hub](https://app.aihub.qualcomm.com/docs/hub/): cloud compilation,
  quantization, profiling, and inference.
- [AIMET](https://github.com/quic/aimet): historical QAT tooling only.

Upstream sources, weights, and dataset images are not vendored. Review their
licenses and usage terms separately; this cleanup does not assert a new license
grant for those assets. The public implementation refactors project scripts;
it does not reproduce Apple's pretraining or Qualcomm's compiler.

## Measurement provenance

[results/provenance.json](../results/provenance.json) records original report
SHA-256 hashes. Public copies reduce absolute paths to basenames and add a note;
measured numbers and job identifiers remain unchanged. Measurements are historical,
chiefly August 12, 2026, not fresh device runs during cleanup. The leaderboard
row is attributed to the participant's supplied result.

## Reproduction environment

Retained validation used Python 3.10, torch 2.8.0, torchvision 0.23.0,
OpenCLIP 3.3.0, ONNX 1.18.0, ORT 1.22.1, Transformers 4.57.6,
NumPy 2.2.6, pandas 2.2.3, and Pillow 11.3.0.
The public refactor was checked with timm 1.0.25 and AI Hub SDK 0.46.0.

The inspected Apple revision is `c16bfe5a4feb424762d6bdf5245539120a4ce9ef`, a
reproduction reference, not proof of the original submission revision.
The original upstream/compiler/runtime versions were not fully captured;
service evolution can change behavior.

Recorded S2 checkpoint SHA-256:
`37c2d839a856491f2fcc82c40dc28672dbd0907235b4cd4c38dfff6457f0c09f`.
Refactored ONNX byte hashes can differ while passing numerical parity gates.
