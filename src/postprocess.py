"""I4 – Hậu xử lý chuỗi sinh ra: sửa cấu trúc thẻ, lọc key ngoài schema, chuẩn hóa giá trị số."""
import re
from typing import Any, Iterable, Set, Tuple


TAG_RE = re.compile(r"<(/?)s_([^>]+)>")


def schema_from_gts(gts: Iterable[dict]) -> Tuple[Set[str], Set[str]]:
    """Trả về (tập đường dẫn key "menu.nm", tập key lá) thu được từ ground truth train."""
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


def fix_structure(seq: str, leaf_keys: Set[str] = frozenset()) -> str:
    """Tự đóng thẻ thiếu / bỏ thẻ đóng thừa.

    - Gặp thẻ mở mới khi thẻ lá (leaf) đang mở -> đóng thẻ lá trước.
    - Gặp thẻ đóng không khớp: nếu key nằm trong stack -> đóng hết các thẻ phía trên; nếu không -> bỏ.
    - Cuối chuỗi: đóng toàn bộ thẻ còn mở.
    """
    out, stack, pos = [], [], 0
    for m in TAG_RE.finditer(seq):
        out.append(seq[pos:m.start()])
        pos = m.end()
        closing, key = m.group(1) == "/", m.group(2)
        if not closing:
            while stack and stack[-1] in leaf_keys:
                out.append(f"</s_{stack.pop()}>")
            stack.append(key)
            out.append(m.group(0))
        elif key in stack:
            while stack:
                k = stack.pop()
                out.append(f"</s_{k}>")
                if k == key:
                    break
    out.append(seq[pos:])
    while stack:
        out.append(f"</s_{stack.pop()}>")
    return "".join(out)


def filter_schema(obj: Any, paths: Set[str], prefix: str = "") -> Any:
    """Loại các key không xuất hiện trong schema CORD (theo đường dẫn đầy đủ)."""
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


_NUM_RE = re.compile(r"(-?)\s*(\d[\d\s.,]*)")


def normalize_value(v: str) -> str:
    """Chuẩn hóa nhẹ: gộp khoảng trắng, "x 1"->"x1", số tiền "25.000"/"25,000"/"25 000"->"25000"."""
    v = re.sub(r"\s+", " ", str(v)).strip()
    v = re.sub(r"(?i)\b(x)\s+(\d)", r"\1\2", v)
    m = _NUM_RE.fullmatch(v)
    if m:
        v = m.group(1) + re.sub(r"[\s.,]", "", m.group(2))
    return v


def normalize_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_tree(x) for x in obj]
    return normalize_value(obj)

