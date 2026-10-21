"""Cải tiến – tiền xử lý "crop": cắt ảnh sát vùng chữ (phát hiện bằng Canny) trước khi đưa vào encoder Donut.

Ý tưởng: Donut resize ảnh (giữ tỉ lệ + pad) về input_size 1280×960. Khi hóa đơn chỉ chiếm phần nhỏ của ảnh (chụp xa,
đặt trên bàn, scan cả trang giấy), nền và lề thừa chiếm mất pixel -> chữ quá nhỏ. Cắt sát vùng chữ = phóng to chữ mà
không đổi model, không train lại, không tăng chi phí encoder.

Phát hiện vùng chữ (ở 2 tỉ lệ: cạnh dài 1000 px và 1600 px, hợp mặt nạ – bắt cả chữ to lẫn chữ nhỏ khi chụp xa):
  1. Xám -> CLAHE (chỉ để dò cạnh, bắt cả chữ mờ) -> Gaussian 3×3 -> Canny, ngưỡng tự động (1 ± 0.33)·median.
  2. Dilate ngang (7×3) để nối nét chữ thành "khối từ" -> connected components.
  3. Giữ các khối có hình học giống chữ: cao 0.4–5% cạnh dài ảnh, rộng < 60% ảnh, mật độ cạnh >= 12%,
     độ lệch chuẩn ảnh xám gốc trong khối >= 8 -> loại vân gỗ, mép bàn, ngón tay (đường dài, mảnh, thưa) và nhiễu nền.
  4. Hộp bao các khối chữ (bỏ 1% điểm ngoại lai mỗi phía) + lề an toàn (8% ngang, 6% dọc).
An toàn (giữ nguyên ảnh gốc, ghi lý do vào info["status"]):
  quá ít pixel chữ ("fail_no_text"); < 6 cụm chữ ("fail_few_text"); hộp > 92% ảnh ("skip_full").
  Hộp được nới quanh tâm để hệ số phóng to chữ <= 2.5.

Chất lượng cắt trên 100 ảnh test CORD (đo bằng toạ độ từ trong ground truth, src/preprocess_preview.py):
99.7% số từ nằm trong vùng cắt, hệ số phóng to chữ trung vị ×1.26.

    from src.preprocess import DocPreprocessor
    prep = DocPreprocessor("crop")
    img2, info = prep(img)    # info: status, ok, box, gain, ...
"""
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

DETECT_SIDES = (1000, 1600)  # cạnh dài (px) của các tỉ lệ phát hiện; hợp mặt nạ -> bắt cả chữ to lẫn chữ nhỏ
MIN_BLOB_STD = 8.0           # độ lệch chuẩn tối thiểu của ảnh xám gốc trong khối -> loại nhiễu bị CLAHE khuếch đại
MIN_TEXT_PIXELS = 200        # ít hơn -> không tin được vùng chữ
MIN_TEXT_BLOBS = 6           # ít hơn 6 cụm chữ -> chỉ thấy vài dòng chữ to/đậm, dễ cắt mất phần chữ mờ -> giữ ảnh gốc
MAX_GAIN = 2.5               # nới hộp quanh tâm tới khi hệ số phóng to <= 2.5
MAX_AREA_FRAC = 0.92         # hộp cắt > 92% ảnh -> không đáng cắt
MODES = ("none", "crop")


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("Cần OpenCV cho tiền xử lý: pip install opencv-python-headless") from e
    return cv2


def effective_gain(orig_wh: Sequence[int], new_wh: Sequence[int], input_hw: Sequence[int] = (1280, 960)) -> float:
    """Hệ số phóng to chữ sau resize-giữ-tỉ-lệ về input_hw (H, W): scale(ảnh mới) / scale(ảnh gốc)."""
    H, W = input_hw

    def scale(wh):
        return min(H / max(wh[1], 1), W / max(wh[0], 1))

    return scale(new_wh) / scale(orig_wh)


# ----------------------------------------------------------------------------
# Phát hiện vùng chữ bằng Canny
# ----------------------------------------------------------------------------
def _text_mask_at(rgb: np.ndarray, side: int, sigma: float = 0.33, min_h: float = 0.004, max_h: float = 0.05,
                  max_w: float = 0.6, min_density: float = 0.12, min_std: float = MIN_BLOB_STD) -> np.ndarray:
    """Mặt nạ khối giống chữ khi ảnh được đưa về cạnh dài `side` px."""
    cv2 = _cv2()
    h, w = rgb.shape[:2]
    r = side / max(h, w)  # chuẩn hóa cả ảnh nhỏ (phóng lên) -> ngưỡng hình học ổn định theo kích thước
    small = cv2.resize(rgb, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray0 = cv2.GaussianBlur(gray, (3, 3), 0)  # để đo độ tương phản thật (trước CLAHE)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    v = float(np.median(gray))
    lo, hi = int(max(0, (1.0 - sigma) * v)), int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lo, max(hi, lo + 1))

    blobs = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blobs)
    L = max(sh, sw)
    keep = []
    for j in range(1, n):
        x, y, bw, bh, _ = stats[j]
        if bh < min_h * L or bh > max_h * L or bw > max_w * sw:
            continue
        sub = lab[y:y + bh, x:x + bw] == j
        if (edges[y:y + bh, x:x + bw][sub] > 0).sum() < min_density * bw * bh:
            continue
        if gray0[y:y + bh, x:x + bw].std() < min_std:
            continue
        keep.append(j)
    return np.isin(lab, keep)


def text_mask(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Hợp mặt nạ chữ ở các tỉ lệ DETECT_SIDES, trên lưới của tỉ lệ đầu tiên.
    Trả về (mask bool [h', w'], tỉ lệ thu nhỏ r = w'/w)."""
    cv2 = _cv2()
    masks = [_text_mask_at(rgb, s) for s in DETECT_SIDES]
    base = masks[0]
    for m in masks[1:]:
        base = base | (cv2.resize(m.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA) > 0)
    return base, base.shape[1] / rgb.shape[1]


def text_box(mask: np.ndarray, r: float, pct: float = 1.0, margin_x: float = 0.08,
             margin_y: float = 0.06) -> Optional[Tuple[float, float, float, float]]:
    """Hộp bao vùng chữ (toạ độ ảnh gốc) hoặc None nếu quá ít pixel chữ."""
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_TEXT_PIXELS:
        return None
    sh, sw = mask.shape
    x0, x1 = np.percentile(xs, [pct, 100 - pct])
    y0, y1 = np.percentile(ys, [pct, 100 - pct])
    ax, ay = margin_x * (x1 - x0) + 8, margin_y * (y1 - y0) + 8
    x0, x1 = max(0.0, x0 - ax), min(float(sw), x1 + ax)
    y0, y1 = max(0.0, y0 - ay), min(float(sh), y1 + ay)
    return x0 / r, y0 / r, x1 / r, y1 / r


def count_text_blobs(mask: np.ndarray) -> int:
    """Số cụm chữ (khối từ gần nhau gộp lại) – thước đo độ tin cậy của vùng chữ phát hiện được."""
    cv2 = _cv2()
    return cv2.connectedComponents(cv2.dilate(mask.astype(np.uint8), np.ones((3, 7), np.uint8)))[0] - 1


def cap_gain(box, orig_wh, img_wh, input_hw, max_gain: float = MAX_GAIN):
    """Nới hộp quanh tâm (mỗi bước 5%, trong phạm vi ảnh) tới khi effective_gain <= max_gain."""
    x0, y0, x1, y1 = box
    W, H = img_wh
    for _ in range(100):
        if effective_gain(orig_wh, (x1 - x0, y1 - y0), input_hw) <= max_gain:
            break
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 1.05, (y1 - y0) * 1.05
        x0, x1 = max(0.0, cx - bw / 2), min(float(W), cx + bw / 2)
        y0, y1 = max(0.0, cy - bh / 2), min(float(H), cy + bh / 2)
    return x0, y0, x1, y1


class DocPreprocessor:
    """Callable: PIL.Image -> (PIL.Image, info). mode = "crop" (cắt vùng chữ) hoặc "none" (giữ nguyên)."""

    def __init__(self, mode: Optional[str] = "crop", input_size: Sequence[int] = (1280, 960)):
        mode = mode or "none"
        if mode not in MODES:
            raise ValueError(f"mode không hợp lệ: {mode} (hợp lệ: {MODES})")
        self.mode = mode
        self.input_size = tuple(int(x) for x in input_size)

    def __bool__(self):
        return self.mode != "none"

    def __call__(self, image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        info: Dict[str, Any] = {"mode": self.mode, "status": "none", "ok": False, "orig_wh": [w, h],
                                "box": None, "area_frac": 1.0}
        out = rgb
        if self.mode == "crop":
            mask, r = text_mask(rgb)
            box = text_box(mask, r)
            info["text_blobs"] = count_text_blobs(mask)
            if box is None:
                info["status"] = "fail_no_text"
            elif info["text_blobs"] < MIN_TEXT_BLOBS:
                info["status"] = "fail_few_text"
            else:
                box = cap_gain(box, (w, h), (w, h), self.input_size)
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                info["box"] = [x0, y0, x1, y1]
                info["area_frac"] = round((x1 - x0) * (y1 - y0) / (w * h), 4)
                if info["area_frac"] > MAX_AREA_FRAC:
                    info["status"] = "skip_full"
                else:
                    info["status"], info["ok"] = "cropped", True
                    out = rgb[y0:y1, x0:x1]
        nh, nw = out.shape[:2]
        info["new_wh"] = [nw, nh]
        info["gain"] = round(effective_gain((w, h), (nw, nh), self.input_size), 4)
        return Image.fromarray(np.ascontiguousarray(out)), info
