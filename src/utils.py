"""Tiện ích chung: seed, config, ghi log CSV, đo VRAM, đếm tham số."""
import csv
import datetime
import io
import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml


class _Tee:
    """Ghi đồng thời ra console và file log. Thanh tiến trình tqdm (dùng '\\r') chỉ lưu trạng thái cuối."""

    def __init__(self, stream, fh):
        self.stream, self.fh, self.buf = stream, fh, ""

    def write(self, s):
        self.stream.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.fh.write(line.split("\r")[-1] + "\n")
        if "\r" in self.buf:
            self.buf = self.buf.split("\r")[-1]
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log(path: str) -> str:
    """Sao chép toàn bộ stdout/stderr (print, warning, tqdm, traceback) vào file log (chế độ append)."""
    ensure_dir(os.path.dirname(path))
    fh = open(path, "a", encoding="utf-8")
    fh.write(f"\n{'=' * 80}\n[{now_str()}] {' '.join(sys.argv)}\n{'=' * 80}\n")
    old_err = sys.stderr
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    # logger của transformers/datasets đã giữ tham chiếu tới stderr cũ -> trỏ lại
    loggers = [logging.getLogger()] + [l for l in logging.Logger.manager.loggerDict.values() if isinstance(l, logging.Logger)]
    for lg in loggers:
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (old_err, sys.__stderr__):
                h.setStream(sys.stderr)
    print(f"[log] {path}")
    return path


def set_seed(seed: int = 42):
    from transformers import set_seed as hf_set_seed

    hf_set_seed(seed)


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Đọc YAML, rồi ghi đè bằng các cặp key=value (hỗ trợ key lồng: lora.r=16)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def save_yaml(obj: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def save_json(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_csv(path: str, row: Dict[str, Any]):
    """Thêm một dòng vào CSV; tự mở rộng header nếu có cột mới."""
    ensure_dir(os.path.dirname(path))
    rows: List[Dict[str, Any]] = []
    header: List[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            rows = list(reader)
    new_cols = [k for k in row.keys() if k not in header]
    if new_cols or not header:
        header = header + new_cols
        rows.append(row)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)


def log_issue(message: str, path: str = "results/issues.md"):
    """Ghi lỗi/cách khắc phục vào results/issues.md (phục vụ phần thảo luận)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- [{now_str()}] {message}\n")


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: torch.nn.Module) -> float:
    """Kích thước state_dict khi serialize (đúng cả với model đã lượng tử hóa int8)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return round(buf.tell() / 1024 ** 2, 1)


def size_str(input_size) -> str:
    return f"{input_size[0]}x{input_size[1]}" if input_size else ""
