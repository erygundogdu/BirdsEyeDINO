# Validation report

## Backbone switching update

`dino.backbone: normal` selects the original pretrained DINOv2; `satellite` selects DINOv3 ViT-L/16 SAT-493M via Transformers. Nine tests pass, including local DINOv3 save/load, satellite-specific normalization, rectangular inputs, register-token exclusion, all embedding modes, dynamic dimensions and invalid patch sizes. The DINOv3 test model has randomly initialized weights and one transformer layer, so it validates integration rather than satellite representation quality.

The actual normal pretrained backbone was rechecked at `[2,3,112,112]`, yielding `[2,64,768]` patch tokens and normalized `[2,768]` CLS embeddings. A 12-image normal-profile end-to-end run completed with 14 accepted candidates; artifacts are in `test_outputs/e2e_normal_switch/`.

Satellite pretrained weights could not be validated here: the official Hugging Face request returned HTTP 401 / GatedRepoError, and no Hugging Face credentials are available. Accept the model conditions and authenticate, or configure a local Hugging Face-format model folder, before the first real satellite run. The application checks/downloads satellite files before detector inference.

Validated on 2026-11-16 in the existing TRAIN environment: Python 3.10, PyTorch 2.10.0+cu128, CUDA available. All dependencies were already installed. No detector source, dataset or checkpoint changes were made.

## Completed checks

- Inspected actual `detect.py`, `models/experimental.py`, `models/common.py`, `utils/datasets.py` and `utils/general.py` before implementation.
- Loaded the specified checkpoint and confirmed three historical SAM layers contain `max_like`, `avg_w`, and 7x7 convolution weights. Unadapted inference reproduced `AttributeError: SAM has no attribute max_kernel`; the layout-specific in-memory mapping to existing `SpatialAttention1` resolved it.
- Verified original-to-letterbox-to-original coordinate round trips for 719x1281, 1000x333 and 112x112 images. Maximum coordinate error was below 0.00001 pixels.
- Ran the actual pretrained official DINOv2 model at `[2,3,112,112]`: patch tokens `[2,64,768]`, CLS `[2,768]`, mean `[2,768]`, combined `[2,1536]`. Every embedding mode had unit L2 norms within floating-point tolerance.
- Five synthetic unittest groups passed: fractional bbox containment, square/padded crops, self exclusion, rejection by another class, edge-only contact, invalid boxes, empty detections, optional crop saving, assignment identity, annotation/source preservation, galleries, PCA/t-SNE, insufficient samples, and identical embeddings. Two distinct samples also successfully generated both projections.
- End-to-end CUDA test on 24 actual dataset images selected across target-class label files completed successfully. Labels selected the test images only and were never supplied to feature extraction or clustering.
- Export audit confirmed all 28 accepted instance IDs match across manifest, embeddings, assignments, PCA, t-SNE and cluster crop files. Confirmed unit embedding norms, `[4,768]` centroids and 35 rejected-candidate debug images. Inspected a generated contact sheet visually.

## End-to-end results

| Measure | Result |
| --- | --- |
| Images | 24 |
| Post-NMS detections, all classes | 389 |
| Class-4 candidates | 63 |
| Accepted | 28 |
| Rejected for overlap | 35 |
| Cluster 0 / 1 / 2 / 3 counts | 8 / 9 / 7 / 4 |
| Embedding matrix | `[28,768]` |
| Effective t-SNE perplexity | 27 |
| Status | complete |

Outputs: `test_outputs/e2e_class4/`, including `summary.json` and all requested visualization/export types. Selected input copies and original-path mapping: `test_inputs/class4/`. Config: `config.smoke.yaml`.

An earlier bounded test on the first 100 sorted images found no target-class detections and correctly returned `clustering_skipped` with empty manifests/features. Its artifacts remain in `test_outputs/e2e_100/`.

## Limits and observations

- The full 4,920-image run has not been executed; `config.yaml` is ready for it and `dino_analysis/` remains available for its outputs.
- The checkpoint identifies class 4 as helicopters; it remains class 4 following explicit user confirmation. No aircraft subclass ground truth or semantic cluster names are used.
- The local pretrained DINOv2 weights/repository were used. Automatic download paths were not exercised because they were unnecessary.
- xFormers is absent; official DINOv2's standard PyTorch attention worked. The legacy detector emits a torch.meshgrid future warning.
- This Windows environment lacks the physical-core query executable used by joblib. It falls back to logical-core counts, emits a warning, and completes K-Means. That warning is retained in the test summary.
- Numeric cluster IDs describe this fitted run and can permute with different data or settings. These tests validate execution and traceability, not semantic aircraft-subclass separation.
