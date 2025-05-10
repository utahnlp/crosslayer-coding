import os
import torch
import numpy as np
import argparse


def main(log_dir: str):
    final_dir = os.path.join(log_dir, "final")
    sum_path = os.path.join(final_dir, "final_sum_min_selected_preact.pt")
    cnt_path = os.path.join(final_dir, "final_count_min_selected_preact.pt")

    if not os.path.exists(sum_path):
        raise FileNotFoundError(sum_path)
    if not os.path.exists(cnt_path):
        raise FileNotFoundError(cnt_path)

    sum_t = torch.load(sum_path, map_location="cpu").float()
    cnt_t = torch.load(cnt_path, map_location="cpu").float()

    theta = sum_t / cnt_t
    theta_np = theta.numpy()

    finite_mask = np.isfinite(theta_np)
    theta_finite = theta_np[finite_mask]

    print("Theta statistics (finite elements):")
    for name, func in [
        ("min", np.min),
        ("1%", lambda x: np.percentile(x, 1)),
        ("median", np.median),
        ("99%", lambda x: np.percentile(x, 99)),
        ("max", np.max),
    ]:
        print(f"  {name:>6}: {func(theta_finite):.6f}")

    cnt_np = cnt_t.numpy()
    print("\nCount statistics:")
    zero_cnt = (cnt_np == 0).sum()
    for name, func in [
        ("min", np.min),
        ("1%", lambda x: np.percentile(x, 1)),
        ("median", np.median),
        ("99%", lambda x: np.percentile(x, 99)),
        ("max", np.max),
    ]:
        print(f"  {name:>6}: {func(cnt_np):.2f}")
    print(f"  zero_count: {zero_cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_dir",
        help="Path to training log directory",
        nargs="?",
        default="tutorials/clt_training_logs/clt_pythia_batchtopk_train_1746839133",
    )
    args = parser.parse_args()
    main(args.log_dir)
