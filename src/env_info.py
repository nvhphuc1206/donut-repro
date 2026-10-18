"""Ghi lại môi trường chạy (phiên bản thư viện, GPU) -> results/env.txt."""
import argparse
import platform
import subprocess
import sys

import torch

from src import utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/env.txt")
    args = ap.parse_args()
    import transformers

    lines = [
        f"time: {utils.now_str()}",
        f"python: {sys.version.split()[0]} | platform: {platform.platform()}",
        f"torch: {torch.__version__} | cuda: {torch.version.cuda} | GPU: {utils.gpu_name()}",
        f"transformers: {transformers.__version__}",
    ]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        lines.append(f"VRAM: {props.total_memory / 1024 ** 3:.1f} GB")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    utils.ensure_dir("results")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n# pip freeze\n" + freeze)
    print("\n".join(lines))
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
