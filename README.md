# Lightweight Arabic Traffic Sign Recognition Using Layer-Wise Iterative Structured Pruning and Knowledge Distillation

> T. Nugroho, E. B. Ashary, M. Bilal, and M. J. Abdulaal, "Lightweight Arabic Traffic Sign Recognition Using Layer-Wise Iterative Structured Pruning and Knowledge Distillation," *Computation* (MDPI), under review.

This repository contains the training/evaluation pipeline and experiment logs supporting the paper above. A VGG16-BN model pre-trained on ImageNet is fine-tuned on the Arabic Traffic Signs (ArTS) dataset and kept as the teacher. Sparsity-inducing training is applied to the Batch Normalization scaling factors ($\gamma$), followed by layer-wise iterative structured pruning of convolutional channels and fully connected neurons, and finally knowledge distillation to recover the student model's accuracy.

## What is evaluated

| Configuration | Description |
|---|---|
| `teacher` | ImageNet-pretrained VGG16-BN, fine-tuned on ArTS (no pruning) |
| `pruned` (unaligned) | Teacher after sparsity-inducing training + layer-wise iterative structured pruning (target rate 58%), before distillation |
| `distilled` (unaligned, **main result**) | Pruned student after knowledge distillation from the preserved teacher |
| `pruned_aligned` / `distilled_aligned` | Same pipeline, but retained channel counts are rounded to multiples of $G=8$ (hardware-aware channel alignment, Eq. 5 in the paper) |

## Repository structure

```
.
├── data/                                  # Placeholder for the ArTS dataset (not redistributed)
├── figures/                                # Rendered diagram figures used in the paper
│   ├── layerwise_pruning.png               # Layer-wise thresholding illustration
│   └── kd_framework.png                    # Knowledge distillation framework diagram
├── logs/                                   # Console logs of the training runs
│   ├── final_augv2_log.txt                 # Runs 1-5 (final configuration)
│   └── final_augv2_continue_log.txt        # Runs 6-10 (final configuration)
├── results/                                 # Result CSVs produced by the scripts (git-ignored)
├── scripts/
│   ├── new_scripts_Pruning_Kd.py           # Core pipeline: model builder, sparse_train,
│   │                                        # iterative_prune, kd_train, evaluation utilities
│   ├── run_final_experiment_augv2.py       # Main entry point: full 10-run pipeline
│   │                                        # (teacher fine-tuning -> sparse training ->
│   │                                        # iterative pruning -> knowledge distillation)
│   │                                        # with the final locked configuration
│   │                                        # (augmentation v2, IMAGE_SIZE=64)
│   ├── run_channel_rounding_poc.py         # Hardware-aware channel-alignment experiment
│   │                                        # (rounds pruned channel counts to multiples of 8)
│   ├── measure_alignment_params_flops.py   # Mean/SD of parameter count and GFLOPs for the
│   │                                        # aligned vs. unaligned configurations
│   ├── verify_compression_10run.py         # Verifies parameter/FLOPs reduction percentages
│   │                                        # across all 10 runs of the final checkpoints
│   └── regenerate_figures_run10.py         # Regenerates pruning-trajectory, KD-loss, and
│                                             # keep_best selection figures from a run's log
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Tested environment: Python 3.10, PyTorch 2.5.1, CUDA 12.1, cuDNN 9.1, on both a Windows 11 workstation (RTX 3060, main experiments) and an NVIDIA Jetson TX2 edge device (CPU-only, FP32, edge-latency evaluation).

```bash
conda create -n arts-pruning-kd python=3.10
conda activate arts-pruning-kd
pip install -r requirements.txt
```

## Dataset

Experiments use the ArTS Arabic Traffic Sign Dataset (2,718 images, 24 classes). The development set (2,200 images) is split into 1,870 training / 330 validation images using a fixed random seed (42); the 518-image test set is held out and evaluated only once, at the end of each run. The dataset itself is not redistributed in this repository -- see the paper for the dataset source and preprocessing details. Update `TRAIN_DIR`, `TRAIN_CSV`, `TEST_DIR`, `TEST_CSV`, and `CHECKPOINT_DIR` in `scripts/new_scripts_Pruning_Kd.py` / the entry-point scripts to point to your local copy.

## Reproducing the experiments

Run the following from the repository root.

```bash
# 1. Full 10-run final pipeline: teacher fine-tuning -> sparse training ->
#    iterative pruning -> knowledge distillation (final locked configuration)
python scripts/run_final_experiment_augv2.py
```

```bash
# 2. Hardware-aware channel-alignment proof-of-concept (reuses existing
#    teacher checkpoints from step 1; rounds pruned channels to multiples of 8)
python scripts/run_channel_rounding_poc.py
```

## Tables, statistics and figures (no retraining)

```bash
# Verify parameter/FLOPs reduction across the 10 reported runs
python scripts/verify_compression_10run.py
```

```bash
# Mean +- SD of parameter count and GFLOPs for aligned vs. unaligned configurations
python scripts/measure_alignment_params_flops.py
```

```bash
# Regenerate the pruning-trajectory / KD-loss / keep_best figures from a console log
python scripts/regenerate_figures_run10.py
```

## Main results

| Model | Params | Checkpoint size | GFLOPs | Test accuracy | Macro-F1 |
|---|---|---|---|---|---|
| Teacher (VGG16-BN, uncompressed) | 134.4 M | 512.6 MB | 2.745 | - | - |
| Pruned + distilled (unaligned, **main**) | 23.7 M (-82.3%) | 90.6 MB | 0.491 (-82.1%) | 99.40% ± 0.42% | 99.37% ± 0.44% |

Averaged over ten independent runs; the best single run reaches 99.81% test accuracy. On an NVIDIA Jetson TX2 edge device (CPU-only), the pruned model achieves a 3.3-5.0x inference speedup over the original model.

## Citation

```bibtex
@article{nugroho2026arts,
  title   = {Lightweight {A}rabic Traffic Sign Recognition Using Layer-Wise Iterative Structured Pruning and Knowledge Distillation},
  author  = {Nugroho, Triawan and Ashary, Ehab Bahaudien and Bilal, Muhammad and Abdulaal, Mohammed J.},
  journal = {Computation},
  year    = {2026},
  note    = {Under review}
}
```

## License

This repository's code is released under the MIT License (see `LICENSE`). The ArTS dataset has its own license/terms and is not redistributed here.
