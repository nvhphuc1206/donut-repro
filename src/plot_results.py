"""Vẽ đường cong huấn luyện cho báo cáo.

    python -m src.plot_results --run_dir outputs/donut_cord_reproduce --out results/reproduce/reproduce_train_curves.png
"""
import argparse
import os

import pandas as pd

from src import utils


def cmd_curves(args):
    import matplotlib.pyplot as plt

    hist = pd.read_csv(os.path.join(args.run_dir, "history.csv"))
    steps_path = os.path.join(args.run_dir, "train_steps.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    if os.path.exists(steps_path):
        steps = pd.read_csv(steps_path)
        axes[0].plot(steps["global_step"], steps["loss"], label="train loss (trung bình trong epoch)")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if "val_ted_acc" in hist:
        axes[1].plot(hist["epoch"], hist["val_ted_acc"], "o-", label="val TED Acc")
        axes[1].plot(hist["epoch"], hist["val_f1"], "s-", label="val F1")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    name = os.path.basename(os.path.normpath(args.run_dir))
    fig.suptitle(f"Đường cong huấn luyện – {name}")
    out = args.out
    utils.ensure_dir(os.path.dirname(out))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Đã lưu {out}")


def main():
    ap = argparse.ArgumentParser(description="Vẽ đường cong huấn luyện (train loss + val TED/F1) từ thư mục output")
    ap.add_argument("--run_dir", default="outputs/donut_cord_reproduce")
    ap.add_argument("--out", default="results/reproduce/reproduce_train_curves.png")
    cmd_curves(ap.parse_args())


if __name__ == "__main__":
    main()
