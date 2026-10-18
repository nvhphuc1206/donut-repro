"""I3 – Tạo các tập test nhiễu cố định từ 100 ảnh test CORD (lưu bằng save_to_disk).

    python -m src.make_noisy_test --out_root data/cord_noisy
Sau đó đánh giá: python -m src.evaluate --dataset data/cord_noisy/blur --run_id I3-official-blur
"""
import argparse
import os
import random

from datasets import Dataset, Features, Image, Value

from src import utils
from src.augment import CORRUPTIONS, corrupt
from src.data import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_root", default="data/cord_noisy")
    ap.add_argument("--types", nargs="+", default=CORRUPTIONS + ["mixed"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", default="results/figures/noisy_preview.png")
    args = ap.parse_args()
    utils.start_log("results/logs/make_noisy_test.log")

    ds = load_split(args.dataset, args.split)
    features = Features({"image": Image(), "ground_truth": Value("string")})
    previews = []
    for k_i, kind in enumerate(args.types):
        rng = random.Random(args.seed * 100 + k_i)
        images, gts = [], []
        for i in range(len(ds)):
            s = ds[i]
            images.append(corrupt(s["image"], kind, rng))
            gts.append(s["ground_truth"])
        out = os.path.join(args.out_root, kind)
        Dataset.from_dict({"image": images, "ground_truth": gts}, features=features).save_to_disk(out)
        previews.append((kind, images[0]))
        print(f"[{kind}] {len(images)} ảnh -> {out}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(previews) + 1, figsize=(3 * (len(previews) + 1), 4.5))
    axes[0].imshow(ds[0]["image"].convert("RGB"))
    axes[0].set_title("clean")
    for ax, (kind, img) in zip(axes[1:], previews):
        ax.imshow(img)
        ax.set_title(kind)
    for ax in axes:
        ax.axis("off")
    utils.ensure_dir(os.path.dirname(args.preview))
    fig.tight_layout()
    fig.savefig(args.preview, dpi=120)
    print(f"Preview: {args.preview}")


if __name__ == "__main__":
    main()
