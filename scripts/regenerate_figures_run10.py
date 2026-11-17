"""
Regenerasi Figure 7 (Iterative Pruning Trajectory), Figure 9 (KD Training
Loss), dan Figure 10 (keep_best Model Selection) menggunakan data RUN 10
dari eksperimen final (augmentasi v2 + IMAGE_SIZE=64) -- diekstrak langsung
dari final_augv2_continue_log.txt, bukan dari run lama/pre-augmentasi-v2.

Run 10 dipilih karena ini run TERAKHIR yang dijalankan (representatif dari
konfigurasi final) dan kebetulan juga run di mana teacher mencapai 100%
akurasi test -- menarik untuk ditampilkan.

CATATAN: Figure 6 (distribusi gamma) TIDAK bisa direkonstruksi dari log ini
-- itu perlu data mentah nilai gamma per-channel yang tidak dicetak ke
console. Figure 8 (channel counts) TIDAK PERLU diupdate karena sudah
terverifikasi deterministik 100% di semua 10 run.

CARA PAKAI (jalankan di komputer manapun dengan matplotlib, tidak perlu
pipeline/model/GPU -- semua data sudah di-hardcode dari log):
  python regenerate_figures_run10.py
"""
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# DATA RUN 10 (diekstrak dari final_augv2_continue_log.txt)
# =============================================================================

# --- Figure 7: Stage 2 Iterative Pruning Trajectory ---
# starting accuracy (Step 0, sebelum pruning sama sekali)
starting_acc = 0.9636

# "potong -> acc=" (X merah, SETELAH pruning step ini, SEBELUM fine-tune)
after_prune_steps = list(range(1, 11))
after_prune_acc = [0.9667, 0.9576, 0.9727, 0.9697, 0.9273, 0.8576, 0.9273, 0.9515, 0.9364, 0.9636]

# "setelah fine-tune 10 epoch -> acc=" (garis biru, SETELAH fine-tune step ini -- recovery)
recovery_steps = list(range(0, 11))
recovery_acc = [starting_acc, 0.9576, 0.9727, 0.9697, 0.9273, 0.8788, 0.9364, 0.9455, 0.9424, 0.9667, 0.9667]

# --- Figure 9: Stage 3 KD Training Loss (30 epoch) ---
kd_epochs = list(range(1, 31))
hard_loss = [0.0318, 0.0272, 0.0142, 0.0139, 0.0110, 0.0148, 0.0171, 0.0100, 0.0123, 0.0113,
             0.0191, 0.0083, 0.0097, 0.0131, 0.0068, 0.0128, 0.0047, 0.0165, 0.0083, 0.0103,
             0.0136, 0.0100, 0.0096, 0.0110, 0.0110, 0.0192, 0.0063, 0.0115, 0.0093, 0.0084]
soft_loss = [5.6880, 2.5125, 1.9818, 1.8188, 1.7771, 1.7996, 1.7624, 1.6599, 1.5994, 1.6223,
             1.5529, 1.5594, 1.5629, 1.4800, 1.4410, 1.4632, 1.4576, 1.4897, 1.4099, 1.3975,
             1.4218, 1.3773, 1.3698, 1.4000, 1.3114, 1.3362, 1.2964, 1.2519, 1.2879, 1.3500]

# --- Figure 10: Stage 3 keep_best Validation Accuracy per Epoch ---
kd_val_acc = [0.9939, 0.9909, 0.9909, 0.9879, 0.9909, 0.9939, 0.9879, 0.9909, 0.9848, 0.9879,
              0.9909, 0.9879, 0.9909, 0.9909, 0.9909, 0.9879, 0.9879, 0.9909, 0.9909, 0.9909,
              0.9909, 0.9879, 0.9939, 0.9939, 0.9909, 0.9909, 0.9879, 0.9879, 0.9879, 0.9879]
best_epoch = 1
best_acc = 0.9939
final_epoch = 30
final_acc = 0.9879

# =============================================================================
# FIGURE 7: Iterative Pruning Trajectory
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(recovery_steps, [a * 100 for a in recovery_acc], "o-", color="tab:blue",
        label="After fine-tuning (recovery)", zorder=3)
ax.scatter(after_prune_steps, [a * 100 for a in after_prune_acc], marker="x", color="red",
           s=70, label="After pruning (before fine-tuning)", zorder=4)
ax.axhline(starting_acc * 100, color="green", linestyle=":", label="Starting accuracy")
ax.set_xlabel("Iterative pruning step")
ax.set_ylabel("Validation accuracy (%)")
ax.set_title("Stage 2: Iterative Layer-wise Pruning Trajectory (target rate = 58%, n = 10 steps)\n[Run 10, final configuration: augmentation v2]")
ax.set_xticks(range(0, 11))
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figure7_pruning_trajectory_run10.png", dpi=200)
print("Tersimpan: figure7_pruning_trajectory_run10.png")

# =============================================================================
# FIGURE 9: KD Training Loss
# =============================================================================
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(kd_epochs, hard_loss, color="tab:blue", label="Hard Label Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Hard Label Loss", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(kd_epochs, soft_loss, color="tab:red", label="Soft Label Loss")
ax2.set_ylabel("Soft Label Loss", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.suptitle("Stage 3: Knowledge Distillation Training Loss (30 epochs)\n[Run 10, final configuration: augmentation v2]")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
fig.tight_layout()
fig.savefig("figure9_kd_loss_run10.png", dpi=200)
print("Tersimpan: figure9_kd_loss_run10.png")

# =============================================================================
# FIGURE 10: keep_best Model Selection
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(kd_epochs, [a * 100 for a in kd_val_acc], "o-", color="tab:blue", label="Validation accuracy")
ax.scatter([best_epoch], [best_acc * 100], marker="*", color="green", s=250, zorder=5,
           label=f"keep_best selected (epoch {best_epoch}, {best_acc * 100:.2f}%)")
ax.scatter([final_epoch], [final_acc * 100], marker="x", color="red", s=100, zorder=5,
           label=f"Final epoch (epoch {final_epoch}, {final_acc * 100:.2f}%)")
diff_pp = (best_acc - final_acc) * 100
ax.annotate(f"+{diff_pp:.2f}pp", xy=((best_epoch + final_epoch) / 2, (best_acc + final_acc) / 2 * 100),
            ha="center", fontsize=10)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy (%)")
ax.set_title("Stage 3: keep_best Model Selection During Distillation\n[Run 10, final configuration: augmentation v2]")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figure10_keep_best_run10.png", dpi=200)
print("Tersimpan: figure10_keep_best_run10.png")

print("\nSelesai. 3 figure baru tersimpan di folder kerja Anda, siap menggantikan")
print("Figure 7, 9, 10 lama -- semuanya dari data RUN 10 konfigurasi final")
print("(augmentasi v2 + IMAGE_SIZE=64), bukan lagi dari run lama.")
