"""
Verifikasi params & FLOPs reduction untuk 10 checkpoint FINAL (augmentasi v2
+ IMAGE_SIZE=64), supaya angka "82.3% param reduction, 82.1% FLOPs
reduction" di Table 2/Table 4 BENAR mewakili konfigurasi final yang
dilaporkan (10-run), bukan diwariskan begitu saja dari eksperimen
single-run yang lebih lama.

Untuk tiap run (1-10), memuat 3 checkpoint:
  - 1_finetuned_augv2_run{N}.pt  -> teacher (baseline, belum di-pruning)
  - 2_pruned_augv2_run{N}.pt     -> pruned (pre-KD)
  - 3_distilled_augv2_run{N}.pt  -> distilled (model FINAL yang dilaporkan)

Lalu menghitung params & FLOPs tiap model, dan reduction% relatif terhadap
teacher (uncompressed) di run yang SAMA -- supaya adil (bukan dibandingkan
ke teacher run lain).

ASUMSI (perlu dicek saat dijalankan):
  - P.load_checkpoint(path) mengembalikan model PyTorch yang SUDAH punya
    arsitektur benar (termasuk untuk model pruned yang channel-nya sudah
    berkurang) -- bukan cuma state_dict mentah.
  - P.count_params(model) mengembalikan jumlah parameter (int).
  - P.count_flops(model) menghitung FLOPs -- KALAU fungsi ini butuh
    argumen tambahan (mis. input_size atau device), akan muncul TypeError
    saat dijalankan; sesuaikan baris pemanggilannya sesuai signature asli
    di new_scripts_Pruning_Kd.py Anda.

CARA PAKAI:
  python verify_compression_10run.py
"""
import os
import csv
import statistics
import importlib

PIPELINE_MODULE_NAME = "new_scripts_Pruning_Kd"  # <-- ganti kalau beda
CHECKPOINT_DIR = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Research_Paper\IEEE\checkpoints_final_augv2"
N_RUNS = 10
RESULTS_CSV = "compression_verification_augv2.csv"


def main():
    P = importlib.import_module(PIPELINE_MODULE_NAME)

    rows = []
    for run in range(1, N_RUNS + 1):
        teacher_path = os.path.join(CHECKPOINT_DIR, f"1_finetuned_augv2_run{run}.pt")
        pruned_path = os.path.join(CHECKPOINT_DIR, f"2_pruned_augv2_run{run}.pt")
        distilled_path = os.path.join(CHECKPOINT_DIR, f"3_distilled_augv2_run{run}.pt")

        print(f"\n--- Run {run}/{N_RUNS} ---")

        def _unwrap(result):
            """P.load_checkpoint() mengembalikan tuple (model, ...metadata),
            bukan model langsung -- ambil elemen pertama yang punya method
            .parameters() (ciri khas nn.Module)."""
            if hasattr(result, "parameters"):
                return result
            if isinstance(result, (tuple, list)):
                for item in result:
                    if hasattr(item, "parameters"):
                        return item
            raise TypeError(
                f"Tidak menemukan objek model (dengan .parameters()) di dalam "
                f"hasil P.load_checkpoint(): {type(result)} -- beri tahu saya "
                f"struktur return value-nya supaya saya sesuaikan."
            )

        teacher_model = _unwrap(P.load_checkpoint(teacher_path))
        pruned_model = _unwrap(P.load_checkpoint(pruned_path))
        distilled_model = _unwrap(P.load_checkpoint(distilled_path))

        teacher_params = P.count_params(teacher_model)
        pruned_params = P.count_params(pruned_model)
        distilled_params = P.count_params(distilled_model)

        teacher_flops = P.count_flops(teacher_model)
        pruned_flops = P.count_flops(pruned_model)
        distilled_flops = P.count_flops(distilled_model)

        param_reduction_distilled = (1 - distilled_params / teacher_params) * 100
        flops_reduction_distilled = (1 - distilled_flops / teacher_flops) * 100
        param_reduction_pruned = (1 - pruned_params / teacher_params) * 100
        flops_reduction_pruned = (1 - pruned_flops / teacher_flops) * 100

        print(f"  Teacher   : {teacher_params:,} params, {teacher_flops:,} FLOPs")
        print(f"  Pruned    : {pruned_params:,} params ({param_reduction_pruned:.2f}% reduction), "
              f"{pruned_flops:,} FLOPs ({flops_reduction_pruned:.2f}% reduction)")
        print(f"  Distilled : {distilled_params:,} params ({param_reduction_distilled:.2f}% reduction), "
              f"{distilled_flops:,} FLOPs ({flops_reduction_distilled:.2f}% reduction)")

        rows.append({
            "run": run,
            "teacher_params": teacher_params,
            "pruned_params": pruned_params,
            "distilled_params": distilled_params,
            "teacher_flops": teacher_flops,
            "pruned_flops": pruned_flops,
            "distilled_flops": distilled_flops,
            "param_reduction_pruned_pct": param_reduction_pruned,
            "param_reduction_distilled_pct": param_reduction_distilled,
            "flops_reduction_pruned_pct": flops_reduction_pruned,
            "flops_reduction_distilled_pct": flops_reduction_distilled,
        })

    fieldnames = list(rows[0].keys())
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    def mean_std(key):
        vals = [r[key] for r in rows]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"\n{'=' * 78}")
    print(f"RINGKASAN {N_RUNS} RUN -- VERIFIKASI PARAM & FLOPS REDUCTION (augmentasi v2 FINAL)")
    print(f"{'=' * 78}")
    for key, label in [
        ("param_reduction_distilled_pct", "Param reduction (distilled, model FINAL)"),
        ("flops_reduction_distilled_pct", "FLOPs reduction (distilled, model FINAL)"),
        ("param_reduction_pruned_pct", "Param reduction (pruned, pre-KD)"),
        ("flops_reduction_pruned_pct", "FLOPs reduction (pruned, pre-KD)"),
    ]:
        mean, std = mean_std(key)
        print(f"  {label}: {mean:.2f}% +- {std:.2f}%")
    print(f"\n  Bandingkan 'Param reduction (distilled)' dan 'FLOPs reduction (distilled)'")
    print(f"  mean di atas terhadap angka lama di Table 2/Table 4: 82.3% / 82.1%.")
    print(f"  Kalau std mendekati 0.00%, berarti angka itu memang konstan/deterministik")
    print(f"  (aman dilaporkan sebagai angka tunggal, seperti sebelumnya). Kalau std")
    print(f"  cukup besar, laporkan sebagai mean +- std, konsisten dengan cara kita")
    print(f"  melaporkan akurasi.")
    print(f"\n  Detail lengkap tersimpan di {RESULTS_CSV}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
