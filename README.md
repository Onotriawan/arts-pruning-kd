# Lightweight Arabic Traffic Sign Recognition Using Layer-Wise Iterative Structured Pruning and Knowledge Distillation

> T. Nugroho, E. B. Ashary, M. Bilal, and M. J. Abdulaal, "Lightweight Arabic Traffic Sign Recognition Using Layer-Wise Iterative Structured Pruning and Knowledge Distillation," *Computation* (MDPI), Manuscript submitted for publication.

This repository contains the training/evaluation pipeline and experiment logs supporting the paper above. A VGG16-BN model pre-trained on ImageNet is fine-tuned on the Arabic Traffic Signs (ArTS) dataset and kept as the teacher. Sparsity-inducing training is applied to the Batch Normalization scaling factors ($\gamma$), followed by layer-wise iterative structured pruning of convolutional channels and fully connected neurons, and finally knowledge distillation to recover the student model's accuracy.

## What is evaluated

| Configuration | Description |
|---|---|
| `teacher` | ImageNet-pretrained VGG16-BN, fine-tuned on ArTS (no pruning) |
| `pruned` | Teacher after sparsity-inducing training + layer-wise iterative structured pruning (target rate 58%), before distillation |
| `distilled` (**main result**) | Pruned student after knowledge distillation from the preserved teacher |

## Installation

Tested environment: Python 3.10, PyTorch 2.5.1, CUDA 12.1, cuDNN 9.1, on both a laptop (AMD Ryzen 7 5800H CPU, NVIDIA GeForce RTX 3060 Laptop GPU, main training and GPU inference experiments) and an NVIDIA Jetson TX2 development board (JetPack 4.6.5, PyTorch 1.10.0, FP32, both CPU and onboard Tegra X2 GPU, edge-latency evaluation).

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

## Tables, statistics and figures (no retraining)

```bash
# Verify parameter/FLOPs reduction across the 10 reported runs
python scripts/verify_compression_10run.py
```

```bash
# Regenerate the pruning-trajectory / KD-loss / keep_best figures from a console log
python scripts/regenerate_figures_run10.py
```

Per-image inference latency (mean $\pm$ SD over 10 repeats, with warm-up and `torch.cuda.synchronize()` on GPU, across batch sizes 1/16/32) was measured following the procedure described in the Inference Latency Measurement section of the paper.

## Main results

| Model | Params | Checkpoint size | GFLOPs | Test accuracy | Macro-F1 |
|---|---|---|---|---|---|
| Teacher (VGG16-BN, uncompressed) | 134.4 M | 512.6 MB | 2.745 | 99.00% ± 0.49% | 98.99% ± 0.50% |
| Pruned (before distillation) | 23.7 M (-82.3%) | 90.6 MB | 0.491 (-82.1%) | 96.54% ± 1.69% | 96.44% ± 1.85% |
| Pruned + distilled (**main**) | 23.7 M (-82.3%) | 90.6 MB | 0.491 (-82.1%) | 99.40% ± 0.42% | 99.37% ± 0.44% |

Averaged over ten independent runs; the best single run reaches 99.81% test accuracy.

### Inference latency (mean per-image, teacher vs. pruned)

Measured with warm-up and `torch.cuda.synchronize()` on GPU, mean ± SD over 10 repeats; see the paper for the full measurement procedure.

| Hardware | Batch size | Teacher (ms) | Pruned (ms) | Speedup |
|---|---|---|---|---|
| Laptop GPU (RTX 3060) | 1 | 4.45 ± 0.26 | 3.73 ± 0.30 | 1.19× |
| Laptop GPU (RTX 3060) | 16 | 0.57 ± 0.00 | 0.23 ± 0.01 | 2.48× |
| Laptop GPU (RTX 3060) | 32 | 0.44 ± 0.00 | 0.17 ± 0.00 | 2.59× |
| Laptop CPU (Ryzen 7 5800H) | 1 | 29.17 ± 0.40 | 7.70 ± 0.93 | 3.79× |
| Laptop CPU (Ryzen 7 5800H) | 16 | 10.39 ± 0.11 | 2.91 ± 0.25 | 3.57× |
| Laptop CPU (Ryzen 7 5800H) | 32 | 9.41 ± 0.20 | 2.91 ± 0.32 | 3.23× |
| Jetson TX2 GPU (Tegra X2) | 1 | 32.64 ± 0.75 | 13.62 ± 0.63 | 2.40× |
| Jetson TX2 GPU (Tegra X2) | 16 | 6.62 ± 0.01 | 1.87 ± 0.01 | 3.54× |
| Jetson TX2 GPU (Tegra X2) | 32 | 4.82 ± 0.01 | 1.56 ± 0.00 | 3.09× |
| Jetson TX2 CPU | 1 | 475.44 ± 4.87 | 149.56 ± 11.79 | 3.18× |
| Jetson TX2 CPU | 16 | 126.37 ± 4.51 | 37.50 ± 0.24 | 3.37× |
| Jetson TX2 CPU | 32 | 88.83 ± 2.99 | 23.01 ± 0.15 | 3.86× |

Across all four hardware configurations (laptop GPU, laptop CPU, Jetson TX2 GPU, and Jetson TX2 CPU) and three batch sizes (1, 16, and 32), the pruned model is never slower than the teacher model, with speedups ranging from approximately 1.2× to 3.9×.

## Citation

```bibtex
@article{nugroho2026arts,
  title   = {Lightweight {A}rabic Traffic Sign Recognition Using Layer-Wise Iterative Structured Pruning and Knowledge Distillation},
  author  = {Nugroho, Triawan and Ashary, Ehab Bahaudien and Bilal, Muhammad and Abdulaal, Mohammed J.},
  journal = {Computation},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## License

This repository's code is released under the MIT License (see `LICENSE`). The ArTS dataset has its own license/terms and is not redistributed here.
