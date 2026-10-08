# Results and measurement boundaries

## Final validated route

Source: [open_submission_fp16.json](../results/open_submission_fp16.json).
Recorded August 12, 2026, on Galaxy S22 (Family), SM8450 / Hexagon v69.
Both float ONNX graphs used
`--target_runtime qnn_dlc --qnn_options default_graph_htp_precision=FLOAT16`;
text also used `--truncate_64bit_io`.

| Branch | Minimum | Median | P90 | Mean | NPU ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image | 14.2390 ms | 14.4210 ms | 14.5121 ms | 14.40337 ms | 405 |
| Text | 3.8930 ms | 3.9480 ms | 3.9856 ms | 3.96429 ms | 389 |

Each branch has 100 profile iterations. The P90 sum is **18.4977 ms**, 47.15%
below 35 ms. Adding marginal P90s is not the measured P90 of their joint
distribution. No end-to-end tokenization, preprocessing, queueing, transfer,
or ranking latency is established. Power/energy reduction was not measured.

Per-job peak memory: image 236,736,512 bytes; text 121,016,320 bytes. Do not
sum these peaks and describe the result as simultaneous residency.

| Sample quality | PyTorch pair | Device pair |
| --- | ---: | ---: |
| Fractional R@1 | 0.259425 | 0.259425 |
| Fractional R@5 | 0.751587 | 0.751587 |
| Fractional R@10 | **0.897470** | **0.897470** |
| Hit R@10 | 0.982143 | 0.982143 |

Corresponding embedding cosine, mean/minimum:

- Image: 0.999863 / 0.997851.
- Text: 0.999998 / 0.999983.

Identical retrieval does not imply bit-exact outputs; RMSE, maximum absolute
error, norm ratios, and bootstrap intervals are retained in the report.

## Identity and provenance

| Stage | Image job | Text job |
| --- | --- | --- |
| Compile | jp1697w25 | jgd2k8qe5 |
| Profile | jp437mdv5 | jpxxq361p |
| Full sample inference | j5m87o6wp | jprwro295 |

The original ONNX hashes are in the JSON record. Hosted inference used compiled
model **handles**; device embeddings were downloaded and scored locally.
This does **not** establish a downloaded final DLC or standalone local QNN run.
QDQ ONNX archives from the integer experiment were downloaded and evaluated
locally. AI Hub links may require account permissions. Binaries and raw
embedding arrays are excluded from this repository.

## Dataset and metric

56 images; 211 captions. Fractional R@K averages
`retrieved positive captions / annotated positive captions` over images.
Hit R@K asks whether any positive is retrieved. One of five relevant captions
means 0.2 fractional recall but 1.0 hit recall.

Images follow CSV row order; texts are sorted by numeric ID. Independent filename
sorting can silently break embedding/label alignment with mixed-case names.
The path-independent manifest digest binds image hashes, captions, and labels.

Missing caption ID 154 remains in the denominator; attainable sample R@10 is
0.998214. This small sample was used during engineering, so it is not an unbiased
held-out test. Its recall is not a hidden-test forecast.

## Participant-supplied organizer result

| Rank then | Team | Timestamp supplied | Score | Total / image / text |
| --- | --- | --- | --- | --- |
| 121 | Lightning Inference | 8/12/2026 17:50:21 | 0.5827017494 | 18092 / 14206 / 3886 μs |

This is a historical row supplied by the participant, not a live leaderboard
query or independently verified rank. Masked suffixes match the compile IDs.
The sample result is **not** the organizer score. Different data/annotations
can change recall without a graph error. Device parity argues against a gross
sample-path conversion error, not against every unseen-input failure.

## Supporting records

Fresh cleanup validation of the public exporter also passed all 56 images and
211 captions: fractional R@10 remains 0.897470 and both mean ONNX/PyTorch cosines
round to 1.0. This is local correctness evidence, not a new device deployment.

- [Public-refactor full local validation](../results/refactor_local_validation.json).
- [Public-refactor synthetic export checks](../results/refactor_export_smoke.json).
- [Retained image QDQ encoding inspection](../results/historical_image_qdq_inspection.json).

- [Corrected S2 / image W8A16](../results/local_s2_w8a16.json).
- [Text float parity](../results/local_s2_text_float.json).
- [Wrong image normalization](../results/local_s2_wrong_normalization.json).
- [S4 PyTorch reference](../results/local_s4_reference.json).
- [W8A16 runtime failure lineage](../results/w8a16_runtime_lineage.json).
- [Original report hashes and sanitization](../results/provenance.json).
