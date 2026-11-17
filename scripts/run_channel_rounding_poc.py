"""
Komentar SPV #4: "Round Pruned Channels to Multiples of 8" (Eq. 4) --
awalnya proof-of-concept 1-2 run, lalu diperluas ke 3 run (hasil konsisten,
trade-off akurasi vs latency jelas), dan SEKARANG diperluas lagi menjadi
FULL 10-RUN supaya apple-to-apple dengan Table Model Progression Summary
(10-run) dan tidak perlu embel-embel "preliminary".

Yang DIHEMAT: tidak mengulang Stage 0 (teacher fine-tuning, 30 epoch) --
reuse checkpoint 1_finetuned_augv2_run{i}.pt yang sudah ada sebagai titik
awal.

Yang DIULANG (karena algoritma pruning-nya sendiri berubah, jadi checkpoint
pruned/distilled yang lama TIDAK bisa dipakai lagi):
  Stage 1: sparse_train         (30 epoch)
  Stage 2: iterative_prune      (10 langkah x 10 epoch fine-tune = 100 epoch)
            -- prune_fn = prune_vgg_full_aligned (BARU, bulatkan ke kelipatan 8)
  Stage 3: kd_train (distillation) (30 epoch)
  TOTAL   ~160 epoch training per run -- PERKIRAAN WAKTU hampir sama dengan
  1 run di pipeline run_final_experiment_augv2.py dikurangi bagian teacher
  fine-tuning saja. Kalau 1 run penuh di komputer Anda butuh beberapa jam,
  perkirakan proof-of-concept ini juga butuh waktu yang sama (bukan lebih
  singkat) -- satu-satunya penghematan adalah Stage 0 (teacher fine-tune).

DI AKHIR, script ini mengukur:
  1. Jumlah parameter (lama vs baru) -- untuk cek rounding tidak membengkakkan model.
  2. Akurasi test set (lama vs baru) -- untuk cek rounding tidak merusak akurasi.
  3. Latency GPU batch=1 (lama vs baru) -- INI METRIK UTAMA yang SPV minta
     dibuktikan, membandingkan terhadap temuan "pruned 1.19x lebih lambat
     dari original di batch=1" (Table~tab:computational_efficiency).

CARA PAKAI:
  1. Taruh file ini DAN prune_channel_rounding.py di folder yang sama
     dengan new_scripts_Pruning_Kd.py.
  2. Sesuaikan CHECKPOINT_DIR, RUNS_TO_DO (default: [1] -- cukup 1 run dulu
     untuk proof-of-concept, sesuai keputusan "1-2 run").
  3. python -u run_channel_rounding_poc.py > log_channel_rounding.txt 2>&1
     (selalu pakai -u supaya log tidak "macet" karena buffering, dan JANGAN
     Ctrl+C walau kelihatan diam -- training 160 epoch wajar makan waktu.)
"""
import os
import csv
import importlib

import torch

from prune_channel_rounding import prune_vgg_full_aligned

# =============================================================================
PIPELINE_MODULE_NAME = "new_scripts_Pruning_Kd"
IMAGE_SIZE = 64
RUNS_TO_DO = [9, 10]  # run 1-8 sudah selesai (laptop sempat mati di tengah run 9); run 9 diulang dari awal
CHECKPOINT_DIR = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Research_Paper\IEEE\checkpoints_final_augv2"
RESULTS_CSV = "channel_rounding_poc.csv"
ROUND_TO = 8  # kelipatan yang diminta SPV
# =============================================================================


def build_loaders(P):
    """Loader POLOS (tanpa augmentasi custom) -- cukup untuk proof-of-concept.
    Kalau mau pipeline data IDENTIK dengan 10-run yang ada, ganti bagian ini
    dengan augmented_transform_v2()/build_loaders() dari
    run_finetune_baseline_vs_kd.py. Untuk proof-of-concept 1-2 run yang
    tujuannya cek dampak LATENCY (bukan membandingkan akurasi head-to-head
    secara ketat), loader polos ini cukup representatif."""
    train_ds = P.ArabicTrafficSignsDataset(
        P.TRAIN_DIR, P.TRAIN_CSV, image_size=IMAGE_SIZE, augment=True
    )
    n_total = len(train_ds)
    n_val = max(1, int(round(n_total * P.VAL_SPLIT_RATIO)))
    n_train = n_total - n_val

    split_generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=split_generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_subset = P.Subset(train_ds, train_indices)
    val_subset = P.Subset(train_ds, val_indices)

    train_loader = P.DataLoader(train_subset, batch_size=P.BATCH_SIZE, shuffle=True)
    val_loader = P.DataLoader(val_subset, batch_size=P.BATCH_SIZE, shuffle=False)

    test_ds = P.ArabicTrafficSignsDataset(P.TEST_DIR, P.TEST_CSV, image_size=IMAGE_SIZE, augment=False)
    test_ds.label_to_idx = train_ds.label_to_idx
    test_ds.num_classes = train_ds.num_classes
    test_loader = P.DataLoader(test_ds, batch_size=P.BATCH_SIZE, shuffle=False)

    print(f"Split train/val/test (channel-rounding PoC): {n_train} train, {n_val} val, {len(test_ds)} test.")
    return train_loader, val_loader, test_loader, train_ds.num_classes


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_batch1_latency(P, model, device, iterations=50, warmup=10):
    """Pakai measure_inference_time() yang sudah ada di pipeline module.
    Fungsi itu mengembalikan waktu rata-rata dalam DETIK, jadi di sini
    dikonversi ke ms supaya konsisten dengan angka Table edge_latency."""
    model = model.to(device)
    model.eval()
    avg_seconds = P.measure_inference_time(
        model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device,
        n_warmup=warmup, n_runs=iterations,
    )
    return avg_seconds * 1000.0


def append_csv(path, result):
    fieldnames = ["run", "old_params", "new_params", "old_acc", "new_acc",
                  "old_latency_bs1_ms", "new_latency_bs1_ms"]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in result.items()})


def main():
    P = importlib.import_module(PIPELINE_MODULE_NAME)
    train_loader, val_loader, test_loader, num_classes = build_loaders(P)
    device = P.DEVICE

    for run_idx in RUNS_TO_DO:
        print(f"\n{'#' * 70}\n# RUN {run_idx} -- channel rounding to multiples of {ROUND_TO} (Eq. 4)\n{'#' * 70}\n")

        teacher_path = os.path.join(CHECKPOINT_DIR, f"1_finetuned_augv2_run{run_idx}.pt")
        print(f"[1/4] Load starting point (teacher, PRE sparse-training): {teacher_path}")
        model, _ = P.load_checkpoint(teacher_path, device=device)
        teacher_for_kd, _ = P.load_checkpoint(teacher_path, device=device)  # salinan terpisah utk KD nanti

        print(f"[2/4] Stage 1: sparse_train ({P.SPARSE_EPOCHS} epoch) ...")
        model, _ = P.sparse_train(
            model, train_loader,
            epochs=P.SPARSE_EPOCHS, lr=P.SPARSE_LR,
            sparsity_lambda=P.SPARSITY_LAMBDA, device=device,
        )

        print(f"[3/4] Stage 2: iterative_prune dengan prune_vgg_full_aligned "
              f"(round_to={ROUND_TO}, {P.ITERATIVE_N_STEPS} langkah x "
              f"{P.ITERATIVE_FINETUNE_EPOCHS} epoch) ...")

        # iterative_prune() memanggil prune_fn(model, pruning_rate=..., min_channels_per_layer=...)
        # -- jadi nama parameter di sini HARUS "pruning_rate", bukan "rate".
        def aligned_prune_fn(m, pruning_rate, min_channels_per_layer=8):
            return prune_vgg_full_aligned(
                m, pruning_rate, min_channels_per_layer=min_channels_per_layer,
                min_neurons=64, round_to=ROUND_TO,
            )

        pruned_model, _ = P.iterative_prune(
            model, train_loader, val_loader,
            target_rate=P.ITERATIVE_TARGET_RATE,
            n_steps=P.ITERATIVE_N_STEPS,
            finetune_epochs_per_step=P.ITERATIVE_FINETUNE_EPOCHS,
            sparsity_lambda=P.SPARSITY_LAMBDA,
            min_channels_per_layer=P.MIN_CHANNELS_PER_LAYER,
            device=device,
            prune_fn=aligned_prune_fn,
        )

        print(f"[4/4] Stage 3: distillation ({P.KD_EPOCHS} epoch) ...")
        distilled_model, _ = P.kd_train(
            pruned_model, teacher_for_kd, train_loader,
            epochs=P.KD_EPOCHS, lr=P.KD_LR, mu=P.KD_MU, temperature=P.KD_TEMPERATURE,
            device=device, test_loader=val_loader,
        )

        new_detailed = P.evaluate_detailed(distilled_model, test_loader, num_classes=num_classes, device=device)
        new_acc = new_detailed["accuracy"]
        new_params = count_params(distilled_model)

        # --- bandingkan dengan model distilled LAMA (non-aligned) utk run yang sama ---
        old_distilled_path = os.path.join(CHECKPOINT_DIR, f"3_distilled_augv2_run{run_idx}.pt")
        old_acc, old_params = None, None
        old_latency_bs1, new_latency_bs1 = None, None
        if os.path.exists(old_distilled_path):
            old_model, _ = P.load_checkpoint(old_distilled_path, device=device)
            old_detailed = P.evaluate_detailed(old_model, test_loader, num_classes=num_classes, device=device)
            old_acc = old_detailed["accuracy"]
            old_params = count_params(old_model)

            if device != "cpu" and torch.cuda.is_available():
                old_latency_bs1 = measure_batch1_latency(P, old_model, device)
                new_latency_bs1 = measure_batch1_latency(P, distilled_model, device)
                print(f"\n>>> LATENCY GPU batch=1: lama={old_latency_bs1:.3f} ms  "
                      f"baru(aligned)={new_latency_bs1:.3f} ms  "
                      f"{'LEBIH CEPAT' if new_latency_bs1 < old_latency_bs1 else 'TIDAK lebih cepat'} "
                      f"({old_latency_bs1 / new_latency_bs1:.2f}x)")
            else:
                print("\n>>> PERINGATAN: tidak ada GPU terdeteksi -- latency batch=1 TIDAK diukur "
                      "(perbandingan ini hanya relevan di GPU, sesuai Table~tab:computational_efficiency "
                      "yang diukur di GPU). Ulangi langkah ini di komputer dengan GPU (CUDA) aktif.")
        else:
            print(f"\n>>> PERINGATAN: {old_distilled_path} tidak ditemukan -- tidak bisa "
                  f"membandingkan langsung. Cek nama file checkpoint distilled lama Anda.")

        print(f"\n>>> RUN {run_idx} TEST SET: "
              f"acc lama={'n/a' if old_acc is None else f'{old_acc*100:.2f}%'}  "
              f"acc baru(aligned)={new_acc*100:.2f}%  "
              f"params lama={'n/a' if old_params is None else f'{old_params/1e6:.2f}M'}  "
              f"params baru={new_params/1e6:.2f}M")

        new_path = os.path.join(CHECKPOINT_DIR, f"5_distilled_aligned_run{run_idx}.pt")
        P.save_checkpoint(distilled_model, new_path, backbone_type="vgg", num_classes=num_classes)
        print(f"Checkpoint baru disimpan: {new_path}")

        append_csv(RESULTS_CSV, {
            "run": run_idx,
            "old_params": old_params or 0, "new_params": new_params,
            "old_acc": old_acc or 0, "new_acc": new_acc,
            "old_latency_bs1_ms": old_latency_bs1 or 0,
            "new_latency_bs1_ms": new_latency_bs1 or 0,
        })

    print(f"\nSelesai. Hasil tersimpan di {RESULTS_CSV}.")
    summarize(RESULTS_CSV)


def summarize(path):
    import csv as _csv
    import statistics
    rows = []
    with open(path, "r") as f:
        for row in _csv.DictReader(f):
            rows.append({k: (int(v) if k == "run" else float(v)) for k, v in row.items()})

    n = len(rows)
    print(f"\n{'=' * 78}")
    print(f"RINGKASAN {n} RUN -- channel-aligned (round_to={ROUND_TO}) vs unaligned (checkpoint lama)")
    print(f"{'=' * 78}")
    for r in rows:
        print(f"  Run {r['run']}: acc lama={r['old_acc']*100:.2f}%  acc baru={r['new_acc']*100:.2f}%  "
              f"delta={((r['new_acc']-r['old_acc'])*100):+.2f}pp  "
              f"latency lama={r['old_latency_bs1_ms']:.3f}ms  latency baru={r['new_latency_bs1_ms']:.3f}ms")

    def mean_std(key):
        vals = [r[key] for r in rows if r[key] != 0]
        if not vals:
            return 0.0, 0.0
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    for key, label, scale, unit in [
        ("old_acc", "Acc lama (unaligned)", 100, "%"),
        ("new_acc", "Acc baru (aligned)", 100, "%"),
        ("old_latency_bs1_ms", "Latency lama (bs=1)", 1, "ms"),
        ("new_latency_bs1_ms", "Latency baru (bs=1)", 1, "ms"),
    ]:
        mean, std = mean_std(key)
        print(f"  {label:<24}: {mean*scale:.2f}{unit} +- {std*scale:.2f}{unit}")

    print(f"{'=' * 78}")
    print("Bandingkan 'Acc baru (aligned)' mean+-SD di atas dengan Table Model Progression")
    print("Summary (Distilled student: 99.40% +- 0.42%) -- kalau rentangnya tumpang tindih,")
    print("berarti selisih akurasi kemungkinan cuma noise antar-run, bukan efek alignment.")


if __name__ == "__main__":
    main()
