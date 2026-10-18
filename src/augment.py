"""Augmentation khi train (I3) và tạo ảnh nhiễu cố định cho tập test bền vững.

Chỉ dùng PIL + torchvision (có sẵn trên Colab/Kaggle), không cần albumentations.
"""
import copy
import io
import random
from typing import Any, Dict, Optional

from PIL import Image, ImageEnhance, ImageFilter

WHITE = (255, 255, 255)


def rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=WHITE)


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def color(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    return ImageEnhance.Contrast(img).enhance(contrast)


def perspective(img: Image.Image, distortion: float, rng: random.Random) -> Image.Image:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    w, h = img.size
    dx, dy = distortion * w, distortion * h

    def j(d):
        return int(rng.uniform(0, d))

    start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    end = [[j(dx), j(dy)], [w - 1 - j(dx), j(dy)], [w - 1 - j(dx), h - 1 - j(dy)], [j(dx), h - 1 - j(dy)]]
    return TF.perspective(img, start, end, interpolation=InterpolationMode.BICUBIC, fill=[255, 255, 255])


# ----------------------------------------------------------------------------
# Augmentation ngẫu nhiên khi train
# ----------------------------------------------------------------------------
DEFAULT_TRAIN_AUG: Dict[str, Dict[str, Any]] = {
    "rotate": {"p": 0.3, "max_angle": 5.0},
    "blur": {"p": 0.2, "radius": [0.3, 1.2]},
    "jpeg": {"p": 0.3, "quality": [20, 60]},
    "color": {"p": 0.3, "brightness": [0.7, 1.3], "contrast": [0.7, 1.3]},
    "perspective": {"p": 0.2, "distortion": 0.06},
}


class TrainAugment:
    """Áp từng phép biến đổi với xác suất p (cấu hình trong YAML: augment: {...})."""

    def __init__(self, cfg: Optional[Dict[str, Dict[str, Any]]] = None):
        self.cfg = copy.deepcopy(DEFAULT_TRAIN_AUG)
        for k, v in (cfg or {}).items():
            if isinstance(v, dict) and k in self.cfg:
                self.cfg[k].update(v)

    def __call__(self, img: Image.Image) -> Image.Image:
        c = self.cfg
        if random.random() < c["perspective"]["p"]:
            img = perspective(img, c["perspective"]["distortion"], random)
        if random.random() < c["rotate"]["p"]:
            a = c["rotate"]["max_angle"]
            img = rotate(img, random.uniform(-a, a))
        if random.random() < c["color"]["p"]:
            img = color(img, random.uniform(*c["color"]["brightness"]), random.uniform(*c["color"]["contrast"]))
        if random.random() < c["blur"]["p"]:
            img = blur(img, random.uniform(*c["blur"]["radius"]))
        if random.random() < c["jpeg"]["p"]:
            img = jpeg(img, random.randint(*c["jpeg"]["quality"]))
        return img


# ----------------------------------------------------------------------------
# Nhiễu cố định cho tập test (mức nặng hơn lúc train)
# ----------------------------------------------------------------------------
CORRUPTIONS = ["rotate", "blur", "jpeg", "color", "perspective"]


def corrupt(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    if kind == "rotate":
        return rotate(img, rng.choice([-1, 1]) * rng.uniform(3.0, 5.0))
    if kind == "blur":
        return blur(img, 1.5)
    if kind == "jpeg":
        return jpeg(img, 10)
    if kind == "color":
        return color(img, rng.choice([0.6, 1.4]), 0.6)
    if kind == "perspective":
        return perspective(img, 0.08, rng)
    if kind == "mixed":  # 2 phép ngẫu nhiên
        for k in rng.sample(CORRUPTIONS, 2):
            img = corrupt(img, k, rng)
        return img
    raise ValueError(f"Không biết loại nhiễu: {kind}")
