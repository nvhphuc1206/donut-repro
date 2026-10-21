"""Chấm điểm model Donut-CORD trên bộ dữ liệu NGOÀI CORD – SROIE, WildReceipt (zero-shot, src/make_external_test.py).

Nhãn của bộ ngoài chỉ ánh xạ được một phần sang schema CORD, nên trước khi tính F1 / TED:
  1. Lọc dự đoán theo schema của CHÍNH tập đó (hợp mọi đường dẫn key xuất hiện trong GT của tập, không theo
     từng ảnh) -> model không bị phạt vì sinh các key CORD mà bộ ngoài không gán nhãn (vd. total.cashprice),
     nhưng vẫn bị phạt nếu sinh thừa/thiếu các key có trong schema.
  2. Chuẩn hóa giá trị ở CẢ dự đoán và GT: bỏ mọi khoảng trắng (bản WildReceipt trên HF đã xóa khoảng trắng
     trong nhãn, vd. "HAM&EGGBURRITO"), bỏ ký hiệu/mã tiền tệ ở đầu ($, RM, Rp, USD, MYR).
  3. total_acc: tỉ lệ ảnh đọc đúng total.total_price (sau chuẩn hóa) – thước đo dễ hiểu nhất.
Không thay đổi gì khi đánh giá CORD (chỉ dùng khi bật --ext_eval).
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def schema_from_gts(gts: Iterable[dict]) -> Tuple[Set[str], Set[str]]:
    """(tập đường dẫn key "menu.nm", tập key lá) xuất hiện trong các ground truth."""
    paths, leaf_keys = set(), set()

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else k
                paths.add(p)
                if isinstance(v, (dict, list)) and not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                    walk(v, p)
                else:
                    leaf_keys.add(k)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, prefix)

    for g in gts:
        walk(g)
    return paths, leaf_keys


def filter_schema(obj: Any, paths: Set[str], prefix: str = "") -> Any:
    """Bỏ các key không thuộc schema (theo đường dẫn đầy đủ)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if p in paths:
                v = filter_schema(v, paths, p)
                if v not in ({}, [], None):
                    out[k] = v
        return out
    if isinstance(obj, list):
        items = [filter_schema(x, paths, prefix) for x in obj]
        return [x for x in items if x not in ({}, [], None)]
    return obj


_CURRENCY_RE = re.compile(r"^(?:US\$|USD|MYR|RM|RP|IDR|\$|€|£)+", re.IGNORECASE)


def normalize_ext_value(v: Any) -> str:
    v = re.sub(r"\s+", "", str(v))
    return _CURRENCY_RE.sub("", v)


def normalize_ext_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {k: normalize_ext_tree(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v not in ("", {}, [], None)}
    if isinstance(obj, list):
        items = [normalize_ext_tree(x) for x in obj]
        return [x for x in items if x not in ("", {}, [], None)]
    return normalize_ext_value(obj)


def get_total(obj: Any) -> Optional[str]:
    """Giá trị total.total_price đầu tiên (đã chuẩn hóa) hoặc None."""
    t = obj.get("total") if isinstance(obj, dict) else None
    if isinstance(t, list):
        t = t[0] if t else None
    v = t.get("total_price") if isinstance(t, dict) else None
    if isinstance(v, list):
        v = v[0] if v else None
    return None if v in (None, "") else normalize_ext_value(v)


def apply_ext_eval(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sửa tại chỗ records (pred/gt -> bản đã lọc + chuẩn hóa; giữ pred_raw/gt_raw) và trả về thông số tóm tắt."""
    paths, _ = schema_from_gts([r["gt"] for r in records])
    n_total = n_total_ok = 0
    for r in records:
        r["pred_raw"], r["gt_raw"] = r["pred"], r["gt"]
        r["pred"] = normalize_ext_tree(filter_schema(r["pred"], paths))
        r["gt"] = normalize_ext_tree(r["gt"])
        gt_total = get_total(r["gt"])
        if gt_total is not None:
            r["total_ok"] = get_total(r["pred"]) == gt_total
            n_total += 1
            n_total_ok += r["total_ok"]
    return {"ext_schema": ",".join(sorted(paths)),
            "total_acc": round(n_total_ok / n_total, 4) if n_total else None,
            "n_total": n_total}
