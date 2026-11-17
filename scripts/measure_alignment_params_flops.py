from new_scripts_Pruning_Kd import load_checkpoint, count_flops, count_params
import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Menggunakan device: {DEVICE}")

CKPT_DIR = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Research_Paper\IEEE\checkpoints_final_augv2"

def measure_params_flops(path):
    model, _ = load_checkpoint(path, device=DEVICE)
    params = count_params(model)
    flops = count_flops(model, input_size=(1, 3, 64, 64), device=DEVICE)
    return params, flops / 1e9  # GFLOPs

configs = {
    "Unaligned (3_distilled)": [f"{CKPT_DIR}\\3_distilled_augv2_run{i}.pt" for i in range(1, 11)],
    "Aligned (5_distilled_aligned)": [f"{CKPT_DIR}\\5_distilled_aligned_run{i}.pt" for i in range(1, 11)],
}

print("\n" + "=" * 70)
for name, paths in configs.items():
    all_params, all_flops = [], []
    for p in paths:
        params, flops = measure_params_flops(p)
        all_params.append(params)
        all_flops.append(flops)

    params_arr = np.array(all_params)
    flops_arr = np.array(all_flops)

    print(f"\n{name} (n={len(paths)} runs):")
    print(f"  Params : mean={params_arr.mean():,.0f}  sd={params_arr.std():,.0f}")
    print(f"  GFLOPs : mean={flops_arr.mean():.4f}  sd={flops_arr.std():.4f}")
    print(f"  Per-run params: {all_params}")
    print(f"  Per-run GFLOPs: {[round(f,4) for f in all_flops]}")
print("=" * 70)