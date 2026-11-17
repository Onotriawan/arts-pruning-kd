"""
FINAL STAGE -- konfigurasi terkunci: augmentasi v2 (translasi+zoom+cutout+
brightness/contrast diperlebar) + IMAGE_SIZE=64 + parameter lain SAMA
PERSIS seperti manuskrip (target_rate=0.58, KD_EPOCHS=30, dst).

Ini BUKAN eksplorasi lagi -- ini laporan hasil dari konfigurasi yang
SUDAH diputuskan (lihat riwayat: teacher-only augmentasi v2 = 99.52%+-
0.17%, full-pipeline validation 3x run = 99.39% konsisten). Karena itu,
evaluasi TEST SET SEKALI per run di sini legitimate -- bukan "mengintip"
untuk memilih konfigurasi (itu sudah selesai di tahap validation-only),
tapi melaporkan hasil akhir dari SATU konfigurasi final untuk mengukur
stabilitasnya (5 run independen, persis semangat yang diminta reviewer).

Checkpoint disimpan ke folder BARU (TMP_CHECKPOINT_DIR di bawah) --
TIDAK menyentuh folder checkpoint lama/path tetap pipeline Anda sama
sekali.

CARA PAKAI:
  1. Ganti PIPELINE_MODULE_NAME kalau beda.
  2. python run_final_experiment_augv2.py
  3. Hasil (teacher/pruned/distilled, accuracy+macro-F1, test set,
     SEKALI per run) tertulis ke final_results_augv2.csv, plus ringkasan
     mean+-std di akhir -- INI yang jadi angka final.
"""
import os
import csv
import copy
import random as _random
import statistics
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

# =============================================================================
# KONFIGURASI (TERKUNCI -- ini konfigurasi final yang diajukan)
# =============================================================================
PIPELINE_MODULE_NAME = "new_scripts_Pruning_Kd"  # <-- ganti kalau beda
IMAGE_SIZE = 64
N_RUNS = 5
TEACHER_EPOCHS = 30
TEACHER_LR = 1e-3
RESULTS_CSV = "final_results_augv2.csv"

MAX_ROTATION_DEG = 15
BRIGHTNESS_JITTER = 0.30
CONTRAST_JITTER = 0.30
MAX_TRANSLATE_FRAC = 0.12
ZOOM_RANGE = (0.85, 1.15)
CUTOUT_FRAC = 0.15
CUTOUT_PROB = 0.5

TMP_CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints_final_augv2")
# =============================================================================


def augmented_transform_v2(image_size):
    def transform(pil_image):
        scale = _random.uniform(*ZOOM_RANGE)
        scaled_size = max(8, int(round(image_size * scale)))
        scaled = pil_image.resize((scaled_size, scaled_size))

        canvas = Image.new("RGB", (image_size, image_size), (128, 128, 128))
        if scaled_size >= image_size:
            left = (scaled_size - image_size) // 2
            top = (scaled_size - image_size) // 2
            scaled = scaled.crop((left, top, left + image_size, top + image_size))
            canvas.paste(scaled, (0, 0))
        else:
            offset = (image_size - scaled_size) // 2
            canvas.paste(scaled, (offset, offset))
        pil_image = canvas

        angle = _random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
        pil_image = pil_image.rotate(angle, fillcolor=(128, 128, 128))

        max_dx = int(image_size * MAX_TRANSLATE_FRAC)
        max_dy = int(image_size * MAX_TRANSLATE_FRAC)
        dx = _random.randint(-max_dx, max_dx)
        dy = _random.randint(-max_dy, max_dy)
        canvas2 = Image.new("RGB", (image_size, image_size), (128, 128, 128))
        canvas2.paste(pil_image, (dx, dy))
        pil_image = canvas2

        if BRIGHTNESS_JITTER > 0:
            factor = 1.0 + _random.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER)
            pil_image = ImageEnhance.Brightness(pil_image).enhance(factor)
        if CONTRAST_JITTER > 0:
            factor = 1.0 + _random.uniform(-CONTRAST_JITTER, CONTRAST_JITTER)
            pil_image = ImageEnhance.Contrast(pil_image).enhance(factor)

        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        if CUTOUT_FRAC > 0 and _random.random() < CUTOUT_PROB:
            h, w = tensor.shape[1], tensor.shape[2]
            cut_h = max(1, int(h * CUTOUT_FRAC))
            cut_w = max(1, int(w * CUTOUT_FRAC))
            cy = _random.randint(0, h - cut_h)
            cx = _random.randint(0, w - cut_w)
            tensor[:, cy:cy + cut_h, cx:cx + cut_w] = 0.0

        return tensor

    return transform


def build_loaders(P):
    custom_transform = augmented_transform_v2(IMAGE_SIZE)

    ds_augmented = P.ArabicTrafficSignsDataset(
        P.TRAIN_DIR, P.TRAIN_CSV, image_size=IMAGE_SIZE, transform=custom_transform
    )
    ds_plain = P.ArabicTrafficSignsDataset(
        P.TRAIN_DIR, P.TRAIN_CSV, image_size=IMAGE_SIZE, augment=False
    )
    ds_plain.label_to_idx = ds_augmented.label_to_idx
    ds_plain.num_classes = ds_augmented.num_classes

    n_total = len(ds_augmented)
    n_val = max(1, int(round(n_total * P.VAL_SPLIT_RATIO)))
    n_train = n_total - n_val

    split_generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=split_generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_subset = P.Subset(ds_augmented, train_indices)
    val_subset = P.Subset(ds_plain, val_indices)

    train_loader = P.DataLoader(train_subset, batch_size=P.BATCH_SIZE, shuffle=True)
    val_loader = P.DataLoader(val_subset, batch_size=P.BATCH_SIZE, shuffle=False)

    test_ds = P.ArabicTrafficSignsDataset(P.TEST_DIR, P.TEST_CSV, image_size=IMAGE_SIZE, augment=False)
    test_ds.label_to_idx = ds_augmented.label_to_idx
    test_ds.num_classes = ds_augmented.num_classes
    test_loader = P.DataLoader(test_ds, batch_size=P.BATCH_SIZE, shuffle=False)

    print(f"Split train/val/test (augmentasi v2, final): {n_train} train, {n_val} val, "
          f"{len(test_ds)} test -- test SEKALI per run, di akhir tiap run.")
    return train_loader, val_loader, test_loader, ds_augmented.num_classes


def finetune_teacher_keep_best(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=0.005)
    best_acc, best_state, best_epoch = -1.0, None, None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0

        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- TERBAIK sejauh ini"
        print(f"  [teacher-ft] epoch {epoch + 1}/{epochs}  val_acc={val_acc:.4f}{marker}")

    model.load_state_dict(best_state)
    print(f"  [teacher-ft] bobot dari epoch terbaik ({best_epoch}, val_acc={best_acc:.4f})")
    return model


def run_one(run_idx, P, train_loader, val_loader, test_loader, num_classes, prune_fn):
    print(f"\n{'#' * 70}\n# FINAL (augmentasi v2, IMAGE_SIZE={IMAGE_SIZE}) -- RUN {run_idx}/{N_RUNS}\n{'#' * 70}\n")

    os.makedirs(TMP_CHECKPOINT_DIR, exist_ok=True)

    # --- Teacher ---
    teacher = P.build_vgg_classifier(num_classes=num_classes, arch="vgg16_bn", pretrained=True)
    teacher = finetune_teacher_keep_best(teacher, train_loader, val_loader,
                                          epochs=TEACHER_EPOCHS, lr=TEACHER_LR, device=P.DEVICE)
    teacher_for_kd = copy.deepcopy(teacher)

    teacher_path = os.path.join(TMP_CHECKPOINT_DIR, f"1_finetuned_augv2_run{run_idx}.pt")
    P.save_checkpoint(teacher, teacher_path, backbone_type="vgg", num_classes=num_classes,
                       vgg_arch="vgg16_bn")
    teacher_detailed = P.evaluate_detailed(teacher, test_loader, num_classes=num_classes, device=P.DEVICE)
    print(f"\n[run{run_idx}/teacher] TEST accuracy={teacher_detailed['accuracy']*100:.2f}%  "
          f"macro-F1={teacher_detailed['macro_f1']*100:.2f}%")

    # --- Sparse training ---
    print(f"\n--- Stage 1: sparse training ({P.SPARSE_EPOCHS} epoch, lambda={P.SPARSITY_LAMBDA}) ---")
    model, _ = P.sparse_train(teacher, train_loader, epochs=P.SPARSE_EPOCHS, lr=P.SPARSE_LR,
                               sparsity_lambda=P.SPARSITY_LAMBDA, device=P.DEVICE)

    # --- Iterative pruning ---
    print(f"\n--- Stage 2: ITERATIVE pruning (target={P.ITERATIVE_TARGET_RATE}, "
          f"{P.ITERATIVE_N_STEPS} langkah) ---")
    pruned_model, history = P.iterative_prune(
        model, train_loader, val_loader,
        target_rate=P.ITERATIVE_TARGET_RATE, n_steps=P.ITERATIVE_N_STEPS,
        finetune_epochs_per_step=P.ITERATIVE_FINETUNE_EPOCHS,
        sparsity_lambda=P.SPARSITY_LAMBDA, min_channels_per_layer=P.MIN_CHANNELS_PER_LAYER,
        device=P.DEVICE, prune_fn=prune_fn,
    )
    pruned_path = os.path.join(TMP_CHECKPOINT_DIR, f"2_pruned_augv2_run{run_idx}.pt")
    P.save_checkpoint(pruned_model, pruned_path, backbone_type="vgg", num_classes=num_classes)
    pruned_detailed = P.evaluate_detailed(pruned_model, test_loader, num_classes=num_classes, device=P.DEVICE)
    print(f"\n[run{run_idx}/pruned_before_kd] TEST accuracy={pruned_detailed['accuracy']*100:.2f}%  "
          f"macro-F1={pruned_detailed['macro_f1']*100:.2f}%  params={P.count_params(pruned_model):,}")

    # --- Knowledge Distillation ---
    print(f"\n--- Stage 3: knowledge distillation (mu={P.KD_MU}, T={P.KD_TEMPERATURE}, "
          f"{P.KD_EPOCHS} epoch) ---")
    distilled_model, kd_history = P.kd_train(
        pruned_model, teacher_for_kd, train_loader,
        epochs=P.KD_EPOCHS, lr=P.KD_LR, mu=P.KD_MU, temperature=P.KD_TEMPERATURE,
        device=P.DEVICE, test_loader=val_loader, keep_best=True,
    )
    distilled_path = os.path.join(TMP_CHECKPOINT_DIR, f"3_distilled_augv2_run{run_idx}.pt")
    P.save_checkpoint(distilled_model, distilled_path, backbone_type="vgg", num_classes=num_classes)

    # --- TEST SET -- SEKALI, di sini, akhir dari segalanya untuk run ini ---
    distilled_detailed = P.evaluate_detailed(distilled_model, test_loader, num_classes=num_classes, device=P.DEVICE)
    acc, f1 = distilled_detailed["accuracy"], distilled_detailed["macro_f1"]
    print(f"\n[run{run_idx}/distilled_final] TEST accuracy={acc*100:.2f}%  macro-F1={f1*100:.2f}%  "
          f"params={P.count_params(distilled_model):,}")

    print(f"\n>>> RUN {run_idx} RINGKASAN (TEST): teacher={teacher_detailed['accuracy']*100:.2f}%  "
          f"pruned={pruned_detailed['accuracy']*100:.2f}%  distilled={acc*100:.2f}%")

    return {
        "run": run_idx,
        "teacher_acc": teacher_detailed["accuracy"], "teacher_f1": teacher_detailed["macro_f1"],
        "pruned_acc": pruned_detailed["accuracy"], "pruned_f1": pruned_detailed["macro_f1"],
        "distilled_acc": acc, "distilled_f1": f1,
        "distilled_params": P.count_params(distilled_model),
    }


def append_csv(path, result):
    fieldnames = ["run", "teacher_acc", "teacher_f1", "pruned_acc", "pruned_f1",
                  "distilled_acc", "distilled_f1", "distilled_params"]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in result.items()})


def summarize(path, target=0.9961):
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append({k: (int(v) if k in ("run", "distilled_params") else float(v))
                         for k, v in row.items()})

    n = len(rows)
    print(f"\n{'=' * 78}\nHASIL FINAL -- {n} RUN, AUGMENTASI v2, TEST SET (SEKALI per run)\n{'=' * 78}")
    for r in rows:
        flag = "  >>> LAMPAUI 99.61%!" if r["distilled_acc"] > target else ""
        print(f"  Run {r['run']}: teacher={r['teacher_acc']*100:.2f}%  "
              f"pruned={r['pruned_acc']*100:.2f}%  distilled={r['distilled_acc']*100:.2f}%{flag}")

    def mean_std(key):
        vals = [r[key] for r in rows]
        return statistics.mean(vals), (statistics.stdev(vals) if n > 1 else 0.0)

    for key, label in [("teacher_acc", "Teacher"), ("teacher_f1", "Teacher macro-F1"),
                        ("pruned_acc", "Pruned (pre-KD)"), ("pruned_f1", "Pruned macro-F1"),
                        ("distilled_acc", "Distilled (FINAL)"), ("distilled_f1", "Distilled macro-F1")]:
        mean, std = mean_std(key)
        print(f"  {label:<22}: {mean*100:.2f}% +- {std*100:.2f}%")

    n_pass = sum(1 for r in rows if r["distilled_acc"] > target)
    print(f"\n  Run yang melampaui {target*100:.2f}% di TEST SET: {n_pass}/{n}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    P = importlib.import_module(PIPELINE_MODULE_NAME)
    train_loader, val_loader, test_loader, num_classes = build_loaders(P)
    prune_fn = P.prune_vgg_full

    for run_idx in range(1, N_RUNS + 1):
        try:
            result = run_one(run_idx, P, train_loader, val_loader, test_loader, num_classes, prune_fn)
            append_csv(RESULTS_CSV, result)
        except Exception as e:
            print(f"\n{'!' * 70}")
            print(f"RUN {run_idx} GAGAL: {type(e).__name__}: {e}")
            print(f"Hasil run sebelumnya SUDAH AMAN di {RESULTS_CSV}.")
            print(f"{'!' * 70}")
            raise

    summarize(RESULTS_CSV)
