"""Kiểm tra chất lượng bước cắt vùng chữ (KHÔNG cần model, vài giây mỗi tập).

Dùng toạ độ từng từ trong ground truth CORD (valid_line[].words[].quad) để đo khách quan:
  - word_recall: tỉ lệ từ GT nằm trọn trong vùng cắt (1.0 = không cắt mất chữ nào; chỉ có ở CORD và CORD chụp xa);
  - cut: ảnh có word_recall < 1 (= "cắt thất bại" cần báo cáo);
  - fallback: ảnh giữ nguyên vì không tin được vùng chữ (fail_no_text) hoặc không đáng cắt (skip_full);
  - gain: hệ số phóng to chữ sau khi resize về input_size (>1 = chữ to hơn);
  - oracle: cắt theo hộp bao các từ GT (+2% lề) -> cận trên của gain.

    python -m src.preprocess_preview --tag test
    python -m src.preprocess_preview --dataset data/cord_far --tag far

Kết quả: results/preprocess/crop/<tag>_crop_stats.csv, _summary.json, _grid.png, _failures.png
"""
import argparse
import json
import os
from collections import Counter
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

from src import utils
from src.data import load_split
from src.preprocess import DocPreprocessor, effective_gain


def word_boxes(ground_truth: str) -> Optional[np.ndarray]:
    """[N, 4, 2] toạ độ 4 góc của từng từ trong GT, hoặc None nếu dataset không có valid_line."""
    gt = json.loads(ground_truth)
    quads = [w["quad"] for line in gt.get("valid_line", []) for w in line.get("words", [])]
    if not quads:
        return None
    return np.array([[[q[f"x{k}"], q[f"y{k}"]] for k in (1, 2, 3, 4)] for q in quads], dtype=np.float64)


def word_recall(quads: np.ndarray, info: dict, img_wh) -> float:
    """Tỉ lệ từ (nằm trong ảnh gốc) còn nằm trọn trong vùng cắt."""
    if not info.get("ok"):
        return 1.0
    w, h = img_wh
    pts = quads
    valid = (pts[..., 0].min(1) >= 0) & (pts[..., 1].min(1) >= 0) & (pts[..., 0].max(1) <= w) & (pts[..., 1].max(1) <= h)
    if not valid.any():
        return 1.0
    x0, y0, x1, y1 = info["box"]
    tol = 2
    inside = ((pts[..., 0].min(1) >= x0 - tol) & (pts[..., 1].min(1) >= y0 - tol)
              & (pts[..., 0].max(1) <= x1 + tol) & (pts[..., 1].max(1) <= y1 + tol))
    return float(inside[valid].mean())


def oracle_gain(quads: np.ndarray, img_wh, input_hw) -> float:
    w, h = img_wh
    x0, y0 = np.clip(quads[..., 0].min(), 0, w), np.clip(quads[..., 1].min(), 0, h)
    x1, y1 = np.clip(quads[..., 0].max(), 0, w), np.clip(quads[..., 1].max(), 0, h)
    mx, my = 0.02 * (x1 - x0), 0.02 * (y1 - y0)
    bw, bh = min(w, x1 + mx) - max(0, x0 - mx), min(h, y1 + my) - max(0, y0 - my)
    return effective_gain((w, h), (bw, bh), input_hw)


def tile(img: Image.Image, out: Image.Image, info: dict, quads, label: str, size=(220, 330)) -> Image.Image:
    """Ô minh họa: trái = ảnh gốc (xanh: từ GT, đỏ: vùng cắt), phải = ảnh sau tiền xử lý."""
    a = img.convert("RGB").copy()
    d = ImageDraw.Draw(a)
    lw = max(2, a.width // 250)
    if quads is not None:
        for q in quads:
            d.polygon([tuple(p) for p in q], outline=(0, 140, 255), width=lw)
    if info.get("box"):
        x0, y0, x1, y1 = info["box"]
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=lw * 2)
    a.thumbnail(size)
    b = out.copy()
    b.thumbnail(size)
    t = Image.new("RGB", (size[0] * 2 + 6, size[1] + 18), "white")
    t.paste(a, (0, 0))
    t.paste(b, (size[0] + 6, 0))
    ImageDraw.Draw(t).text((3, size[1] + 3), label, fill=(0, 0, 0))
    return t


def grid(tiles: List[Image.Image], cols: int, path: str):
    if not tiles:
        return
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    g = Image.new("RGB", (tw * cols, th * rows), (200, 200, 200))
    for k, t in enumerate(tiles):
        g.paste(t, ((k % cols) * tw, (k // cols) * th))
    utils.ensure_dir(os.path.dirname(path))
    g.save(path)
    print(f"Hình: {path}")


def main():
    ap = argparse.ArgumentParser(description="Kiểm tra chất lượng bước cắt vùng chữ (không cần model)")
    ap.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--input_size", type=int, nargs=2, default=[1280, 960], metavar=("H", "W"))
    ap.add_argument("--tag", default="test", help="Tiền tố tên file kết quả")
    ap.add_argument("--out_dir", default="results/preprocess/crop")
    ap.add_argument("--n_grid", type=int, default=24, help="Số ảnh minh họa trong _grid.png")
    args = ap.parse_args()
    utils.start_log("results/logs/preprocess_preview.log")

    prep = DocPreprocessor("crop", args.input_size)
    ds = load_split(args.dataset, args.split)
    stem = os.path.join(args.out_dir, f"{args.tag}_crop")
    step = max(1, len(ds) // max(args.n_grid, 1))

    rows, tiles, fail_tiles = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        img = s["image"].convert("RGB")
        out, info = prep(img)
        quads = word_boxes(s["ground_truth"])
        rec = word_recall(quads, info, img.size) if quads is not None else None
        orc = oracle_gain(quads, img.size, args.input_size) if quads is not None else None
        rows.append({"idx": i, "status": info["status"], "gain": info["gain"], "area_frac": info["area_frac"],
                     "word_recall": None if rec is None else round(rec, 4),
                     "oracle_gain": None if orc is None else round(orc, 4),
                     "orig_w": img.width, "orig_h": img.height, "new_w": out.width, "new_h": out.height})
        label = f"#{i} {info['status']} gain={info['gain']:.2f}"
        if rec is not None:
            label += f" recall={rec:.2f}"
        if i % step == 0 and len(tiles) < args.n_grid:
            tiles.append(tile(img, out, info, quads, label))
        if (rec is not None and rec < 1) or not info["ok"]:
            fail_tiles.append(tile(img, out, info, quads, label))

    gains = np.array([r["gain"] for r in rows])
    st = Counter(r["status"] for r in rows)
    recs = [r["word_recall"] for r in rows if r["word_recall"] is not None]
    cut = [r["idx"] for r in rows if r["word_recall"] is not None and r["word_recall"] < 1]
    orcs = [r["oracle_gain"] for r in rows if r["oracle_gain"] is not None]
    summary = {
        "dataset": args.dataset, "split": args.split, "mode": prep.mode, "n": len(rows),
        "status_counts": dict(st),
        "cropped": st.get("cropped", 0),
        "fallback": sum(v for k, v in st.items() if k.startswith(("fail", "skip"))),
        "gain_median": round(float(np.median(gains)), 3),
        "gain_mean": round(float(gains.mean()), 3),
        "n_gain_ge_1.2": int((gains >= 1.2).sum()),
        "n_gain_ge_1.5": int((gains >= 1.5).sum()),
        "word_recall_mean": round(float(np.mean(recs)), 4) if recs else None,
        "n_cut_images": len(cut) if recs else None,
        "cut_images": cut,
        "oracle_gain_median": round(float(np.median(orcs)), 3) if orcs else None,
        "time": utils.now_str(),
    }
    utils.ensure_dir(args.out_dir)
    import csv
    with open(stem + "_stats.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    utils.save_json(summary, stem + "_summary.json")
    grid(tiles, 6, stem + "_grid.png")
    grid(fail_tiles, 6, stem + "_failures.png")
    print("\n===== CHẤT LƯỢNG CẮT =====")
    for k, v in summary.items():
        print(f"{k:>20}: {v}")
    print(f"Đã lưu: {stem}_stats.csv, {stem}_summary.json")


if __name__ == "__main__":
    main()
