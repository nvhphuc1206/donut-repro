"""Tạo tập test NGOÀI CORD, định dạng giống CORD (image + ground_truth={"gt_parse": ...}), lưu bằng save_to_disk.

  wildreceipt : ảnh hóa đơn CHỤP BẰNG ĐIỆN THOẠI ngoài thực tế (có nền, nghiêng) – HF mychen76/wildreceipts_ocr_v1.
                Ánh xạ sang schema CORD: line_items -> menu[{nm, cnt, price}], subtotal -> sub_total.subtotal_price,
                tax -> sub_total.tax_price, total -> total.total_price. (store_name, date, ... không có trong CORD.)
                Lấy toàn bộ test (99) + valid (20), rồi bổ sung từ train cho đủ --max_samples (model CORD chưa
                từng thấy WildReceipt nên dùng train để đánh giá là hợp lệ).
  sroie       : hóa đơn SCAN bằng máy quét (gần như không có nền) – HF Voxel51/scanned_receipts (ICDAR SROIE).
                Chỉ trường total ánh xạ được -> total.total_price. Dùng để kiểm tra "tiền xử lý không gây hại".

    python -m src.make_external_test --which wildreceipt sroie --max_samples 300
    python -m src.evaluate --dataset data/ext_wildreceipt --ext_eval ...
"""
import argparse
import json
import os
import random
from typing import Dict, List, Optional

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

from src import utils
from src.data import load_split


def _clean(v) -> str:
    return str(v).strip() if v is not None else ""


def wildreceipt_to_cord(parsed_data: str) -> Optional[Dict]:
    d = json.loads(json.loads(parsed_data)["json"])
    menu = []
    for it in d.get("line_items") or []:
        item = {k_cord: _clean(it.get(k_wr)) for k_cord, k_wr in
                (("nm", "item_name"), ("cnt", "item_quantity"), ("price", "item_value"))}
        item = {k: v for k, v in item.items() if v}
        if item:
            menu.append(item)
    gt: Dict = {}
    if menu:
        gt["menu"] = menu
    sub = {k: _clean(d.get(src)) for k, src in (("subtotal_price", "subtotal"), ("tax_price", "tax"))}
    sub = {k: v for k, v in sub.items() if v}
    if sub:
        gt["sub_total"] = sub
    if _clean(d.get("total")):
        gt["total"] = {"total_price": _clean(d.get("total"))}
    return gt or None


def build_wildreceipt(max_samples: int, seed: int):
    images, gts = [], []
    order = ["test", "valid", "train"]
    skipped = 0
    for split in order:
        ds = load_split("mychen76/wildreceipts_ocr_v1", split)
        idxs = list(range(len(ds)))
        if split == "train":
            random.Random(seed).shuffle(idxs)
        for i in idxs:
            if len(images) >= max_samples:
                break
            s = ds[i]
            gt = wildreceipt_to_cord(s["parsed_data"])
            if gt is None:
                skipped += 1
                continue
            images.append(s["image"].convert("RGB"))
            gts.append(json.dumps({"gt_parse": gt, "meta": {"source": "wildreceipt", "split": split, "id": s["id"]}},
                                  ensure_ascii=False))
        print(f"[wildreceipt] sau split {split}: {len(images)} ảnh (bỏ {skipped} ảnh không có trường nào)")
    return images, gts


def build_sroie(max_samples: int, seed: int):
    from huggingface_hub import hf_hub_download

    repo = "Voxel51/scanned_receipts"
    samples = json.load(open(hf_hub_download(repo, "samples.json", repo_type="dataset"), encoding="utf-8"))["samples"]
    samples = [s for s in samples if _clean(s.get("total"))]
    samples.sort(key=lambda s: s["filepath"])
    random.Random(seed).shuffle(samples)
    samples = samples[:max_samples]
    images, gts = [], []
    for k, s in enumerate(samples):
        path = hf_hub_download(repo, s["filepath"], repo_type="dataset")
        images.append(Image.open(path).convert("RGB"))
        gts.append(json.dumps({"gt_parse": {"total": {"total_price": _clean(s["total"])}},
                               "meta": {"source": "sroie", "file": os.path.basename(s["filepath"])}},
                              ensure_ascii=False))
        if (k + 1) % 50 == 0:
            print(f"[sroie] {k + 1}/{len(samples)}")
    return images, gts


BUILDERS = {"wildreceipt": build_wildreceipt, "sroie": build_sroie}


def main():
    ap = argparse.ArgumentParser(description="Tạo tập test ngoài CORD (WildReceipt, SROIE) theo schema CORD")
    ap.add_argument("--which", nargs="+", default=["wildreceipt", "sroie"], choices=list(BUILDERS))
    ap.add_argument("--max_samples", type=int, default=300, help="Số ảnh tối đa mỗi tập")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    utils.start_log("results/logs/make_external_test.log")

    features = Features({"image": HFImage(), "ground_truth": Value("string")})
    for name in args.which:
        out = os.path.join(args.out_root, f"ext_{name}")
        images, gts = BUILDERS[name](args.max_samples, args.seed)
        Dataset.from_dict({"image": images, "ground_truth": gts}, features=features).save_to_disk(out)
        keys = sorted({k for g in gts for k in json.loads(g)["gt_parse"]})
        print(f"[{name}] {len(images)} ảnh -> {out} | key cấp 1: {keys}")


if __name__ == "__main__":
    main()
