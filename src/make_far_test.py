"""Tạo tập test "CORD chụp xa" cố định từ 100 ảnh test CORD (lưu bằng save_to_disk).

Mô phỏng ảnh chụp hóa đơn từ xa trên mặt bàn: mỗi ảnh gốc được
  - đặt lên một nền tổng hợp (vân gỗ / nền tối / mặt bàn sáng / vải màu), chiếm `scale` ∈ [0.5, 0.7] mỗi chiều khung;
  - xoay ±[3°, 8°];
  - giữ nguyên số pixel của ảnh gốc (khung lớn hơn) -> sau khi Donut resize về 1280×960 chữ nhỏ đi ~1/scale lần;
  - nén JPEG chất lượng 85–92 như ảnh chụp điện thoại.
gt_parse giữ nguyên; toạ độ valid_line / roi / meta.image_size được biến đổi tương ứng
(để src.preprocess_preview đo được word_recall trên tập này).

    python -m src.make_far_test --out data/cord_far
    python -m src.evaluate --dataset data/cord_far --preprocess crop ...
"""
import argparse
import copy
import io
import json
import os

import numpy as np
from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

from src import utils
from src.data import load_split
from src.preprocess import _cv2

BACKGROUNDS = ["wood", "dark", "table", "fabric"]


def make_background(kind: str, w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Nền RGB uint8 [h, w, 3], sinh ở độ phân giải thấp rồi phóng lên cho nhanh."""
    cv2 = _cv2()
    sw, sh = max(8, w // 4), max(8, h // 4)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    if kind == "wood":
        base = np.array([rng.uniform(150, 205), rng.uniform(105, 150), rng.uniform(60, 100)], np.float32)
        warp = cv2.GaussianBlur(rng.normal(0, 1, (sh, sw)).astype(np.float32), (0, 0), 12) * 40
        freq = rng.uniform(0.015, 0.04)
        stripes = np.sin((xx + warp) * freq * 2 * np.pi) * 0.5 + 0.5
        shade = (0.85 + 0.15 * stripes)[..., None]
        img = base * shade
    elif kind == "dark":
        img = np.full((sh, sw, 3), rng.uniform(15, 55), np.float32) * np.array([1, rng.uniform(.9, 1.1), rng.uniform(.9, 1.2)])
    elif kind == "table":  # mặt bàn sáng, tương phản thấp với giấy -> ca khó cho bước cắt
        c = rng.uniform(170, 215)
        grad = (xx / sw * rng.uniform(-30, 30) + yy / sh * rng.uniform(-30, 30))[..., None]
        img = np.array([c, c * rng.uniform(.95, 1.0), c * rng.uniform(.88, .97)], np.float32) + grad
    else:  # fabric: màu bão hòa
        hue = rng.integers(0, 180)
        hsv = np.uint8([[[hue, rng.integers(120, 220), rng.integers(90, 200)]]])
        img = np.tile(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32), (sh, sw, 1))
    img = img + rng.normal(0, 4, img.shape)  # nhiễu ở độ phân giải thấp -> sau khi phóng lên thành nhiễu mịn
    return cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)


def transform_gt(ground_truth: str, M: np.ndarray, w: int, h: int) -> str:
    """Áp affine 2×3 lên mọi quad trong valid_line và roi; cập nhật meta.image_size."""
    gt = json.loads(ground_truth)
    gt = copy.deepcopy(gt)

    def tq(q):
        if not q or "x1" not in q:
            return q
        out = {}
        for k in (1, 2, 3, 4):
            x, y = q[f"x{k}"], q[f"y{k}"]
            out[f"x{k}"] = int(round(M[0, 0] * x + M[0, 1] * y + M[0, 2]))
            out[f"y{k}"] = int(round(M[1, 0] * x + M[1, 1] * y + M[1, 2]))
        return out

    for line in gt.get("valid_line", []):
        for word in line.get("words", []):
            word["quad"] = tq(word["quad"])
    if isinstance(gt.get("roi"), dict):
        gt["roi"] = tq(gt["roi"])
    if isinstance(gt.get("meta"), dict):
        gt["meta"]["image_size"] = {"width": w, "height": h}
    return json.dumps(gt, ensure_ascii=False)


def make_far(img: Image.Image, ground_truth: str, rng: np.random.Generator, scale_range, angle_range):
    cv2 = _cv2()
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = rng.uniform(*scale_range)
    angle = rng.uniform(*angle_range) * rng.choice([-1, 1])
    kind = BACKGROUNDS[rng.integers(len(BACKGROUNDS))]
    W, H = int(round(w / scale)), int(round(h / scale))

    # xoay ảnh gốc (nới khung) kèm mặt nạ alpha
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(R[0, 0]), abs(R[0, 1])
    rw, rh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    R[0, 2] += rw / 2 - w / 2
    R[1, 2] += rh / 2 - h / 2
    rot = cv2.warpAffine(rgb, R, (rw, rh), flags=cv2.INTER_CUBIC)
    alpha = cv2.warpAffine(np.full((h, w), 255, np.uint8), R, (rw, rh), flags=cv2.INTER_LINEAR)
    W, H = max(W, rw + 2), max(H, rh + 2)
    ox, oy = int(rng.integers(0, W - rw + 1)), int(rng.integers(0, H - rh + 1))

    canvas = make_background(kind, W, H, rng).astype(np.float32)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    region = canvas[oy:oy + rh, ox:ox + rw]
    canvas[oy:oy + rh, ox:ox + rw] = rot * a + region * (1 - a)
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    quality = int(rng.integers(85, 93))  # nén như ảnh chụp điện thoại
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=quality)
    out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    M = R.copy()
    M[0, 2] += ox
    M[1, 2] += oy
    meta = {"scale": round(float(scale), 3), "angle": round(float(angle), 2), "background": kind, "jpeg_q": quality}
    return out, transform_gt(ground_truth, M, W, H), meta


def main():
    ap = argparse.ArgumentParser(description="Tạo tập test CORD chụp xa (nền + thu nhỏ + xoay)")
    ap.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/cord_far")
    ap.add_argument("--scale", type=float, nargs=2, default=[0.5, 0.7], help="Tỉ lệ ảnh gốc / khung (mỗi chiều)")
    ap.add_argument("--angle", type=float, nargs=2, default=[3.0, 8.0], help="|góc xoay| (độ)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", default="results/preprocess/far_preview.png")
    args = ap.parse_args()
    utils.start_log("results/logs/make_far_test.log")
    if os.path.exists(args.out):
        print(f"[!] {args.out} đã tồn tại -> ghi đè (dữ liệu sinh lại được, cùng seed cho cùng kết quả)")

    ds = load_split(args.dataset, args.split)
    rng = np.random.default_rng(args.seed)
    images, gts, metas = [], [], []
    for i in range(len(ds)):
        s = ds[i]
        img, gt, meta = make_far(s["image"], s["ground_truth"], rng, args.scale, args.angle)
        images.append(img)
        gts.append(gt)
        metas.append({"idx": i, **meta})
    features = Features({"image": HFImage(), "ground_truth": Value("string")})
    Dataset.from_dict({"image": images, "ground_truth": gts}, features=features).save_to_disk(args.out)
    utils.save_json({"args": vars(args), "samples": metas}, os.path.join(args.out, "far_meta.json"))
    print(f"[far] {len(images)} ảnh -> {args.out}")

    import matplotlib.pyplot as plt

    k = 6
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 9))
    for j in range(k):
        idx = j * (len(ds) // k)
        axes[0, j].imshow(ds[idx]["image"].convert("RGB"))
        axes[0, j].set_title(f"#{idx} gốc")
        axes[1, j].imshow(images[idx])
        m = metas[idx]
        axes[1, j].set_title(f"{m['background']} s={m['scale']} {m['angle']}°")
    for ax in axes.ravel():
        ax.axis("off")
    utils.ensure_dir(os.path.dirname(args.preview))
    fig.tight_layout()
    fig.savefig(args.preview, dpi=100)
    print(f"Preview: {args.preview}")


if __name__ == "__main__":
    main()
