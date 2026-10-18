"""Vẽ hình / bảng cho báo cáo.

    python -m src.plot_results curves   --run_dir outputs/B0_full_1280
    python -m src.plot_results tradeoff --prefix I1-official- --out results/figures/i1_tradeoff.png
    python -m src.plot_results table    --out results/runs_table.md
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
    out = args.out or f"results/figures/{name}_curves.png"
    utils.ensure_dir(os.path.dirname(out))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Đã lưu {out}")


def cmd_tradeoff(args):
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.runs_csv)
    df = df[df["id"].astype(str).str.startswith(args.prefix) & (df["stage"] == "eval")]
    df = df.drop_duplicates("id", keep="last").copy()
    if df.empty:
        raise SystemExit(f"Không có dòng nào có id bắt đầu bằng {args.prefix}")
    df["pixels"] = df["input_size"].map(lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))
    df = df.sort_values("pixels")
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(df["input_size"], df["ted_acc"], "o-", label="TED Acc")
    ax1.plot(df["input_size"], df["f1"], "s-", label="F1")
    ax1.set_xlabel("input size (H x W)")
    ax1.set_ylabel("điểm")
    ax2 = ax1.twinx()
    ax2.plot(df["input_size"], df["sec_per_img"], "^--", color="tab:red", label="s/ảnh")
    ax2.set_ylabel("latency (s/ảnh)", color="tab:red")
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.set_title("Trade-off độ phân giải ↔ độ chính xác ↔ tốc độ")
    utils.ensure_dir(os.path.dirname(args.out))
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(df[["id", "input_size", "f1", "ted_acc", "sec_per_img", "peak_vram_gb"]].to_string(index=False))
    print(f"Đã lưu {args.out}")


def cmd_table(args):
    df = pd.read_csv(args.runs_csv).drop_duplicates("id", keep="last")
    cols = [c for c in ["id", "stage", "input_size", "num_beams", "precision", "variant", "epochs",
                        "trainable_params_M", "effective_trainable_params_M", "peak_vram_gb", "train_hours",
                        "f1", "ted_acc", "val_f1", "val_ted_acc", "sec_per_img", "model_size_mb", "gpu", "notes"]
            if c in df.columns]
    df = df[cols].fillna("")
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        md += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    utils.ensure_dir(os.path.dirname(args.out))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"Đã lưu {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("--run_dir", required=True)
    c.add_argument("--out")
    t = sub.add_parser("tradeoff")
    t.add_argument("--prefix", default="I1-official-")
    t.add_argument("--runs_csv", default="results/runs.csv")
    t.add_argument("--out", default="results/figures/i1_tradeoff.png")
    tb = sub.add_parser("table")
    tb.add_argument("--runs_csv", default="results/runs.csv")
    tb.add_argument("--out", default="results/runs_table.md")
    args = ap.parse_args()
    {"curves": cmd_curves, "tradeoff": cmd_tradeoff, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
