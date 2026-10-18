"""Task 2.3 – Phân tích lỗi từ file dự đoán của evaluate.py.

    python -m src.error_analysis --pred results/p1_official_cord.json --top_k 10 --save_images
Output: results/error_analysis.md, results/error_analysis.csv, results/errors/*.png|json
"""
import argparse
import os
import re
from collections import Counter

from nltk import edit_distance

from src import utils
from src.data import load_split
from src.metrics import JSONParseEvaluator

EV = JSONParseEvaluator()
CATS = ["Sai cấu trúc JSON", "Thiếu field", "Thừa field", "Đọc sai ký tự (≤2)", "Sai giá trị (>2 ký tự)"]


def unbalanced_tags(seq: str) -> bool:
    opens = Counter(re.findall(r"<s_([^>]+)>", seq))
    closes = Counter(re.findall(r"</s_([^>]+)>", seq))
    return opens != closes


def analyse(rec):
    gt = EV.flatten(EV.normalize_dict(rec["gt"]))
    pr = EV.flatten(EV.normalize_dict(rec["pred"]))
    # bỏ các cặp khớp hoàn toàn
    pr_left = list(pr)
    gt_left = []
    for f in gt:
        if f in pr_left:
            pr_left.remove(f)
        else:
            gt_left.append(f)
    char_err, value_err, missing = 0, 0, 0
    for key, val in gt_left:
        cands = [v for k, v in pr_left if k == key]
        if not cands:
            missing += 1
            continue
        best = min(cands, key=lambda v: edit_distance(str(v), str(val)))
        pr_left.remove((key, best))
        if edit_distance(str(best), str(val)) <= 2:
            char_err += 1
        else:
            value_err += 1
    extra = len(pr_left)
    struct = "text_sequence" in rec["pred"] or unbalanced_tags(rec.get("raw_sequence", ""))
    counts = dict(zip(CATS, [int(struct), missing, extra, char_err, value_err]))
    return counts, len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/p1_official_cord.json")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_md", default="results/error_analysis.md")
    ap.add_argument("--out_csv", default="results/error_analysis.csv")
    ap.add_argument("--save_images", action="store_true", help="Lưu ảnh + gt/pred của top_k mẫu tệ nhất")
    ap.add_argument("--images_dir", default="results/errors")
    args = ap.parse_args()
    utils.start_log("results/logs/error_analysis.log")

    data = utils.load_json(args.pred)
    records = data["records"]
    rows, total = [], Counter()
    for r in records:
        counts, n_gt = analyse(r)
        total.update({k: v for k, v in counts.items()})
        labels = [k for k, v in counts.items() if v]
        rows.append({"idx": r["idx"], "ted_acc": round(r["ted_acc"], 4), "f1": round(r["f1"], 4),
                     "n_gt_fields": n_gt, **counts, "auto_label": "; ".join(labels) or "Đúng hoàn toàn"})
    rows.sort(key=lambda x: x["ted_acc"])
    worst = rows[: args.top_k]
    # ghi CSV đầy đủ (ghi đè)
    if os.path.exists(args.out_csv):
        os.remove(args.out_csv)
    for row in rows:
        utils.append_csv(args.out_csv, row)

    n_with = Counter()
    for row in rows:
        for c in CATS:
            n_with[c] += int(row[c] > 0)
    lines = [
        f"# Phân tích lỗi – `{args.pred}`",
        "",
        f"Mô hình: `{data['summary'].get('model')}` | n = {len(rows)} | "
        f"F1 = {data['summary'].get('f1')} | TED Acc = {data['summary'].get('ted_acc')}",
        "",
        "## Thống kê loại lỗi trên toàn tập",
        "",
        "| Loại lỗi | Số mẫu mắc | Tổng số lần |",
        "|---|---|---|",
    ]
    lines += [f"| {c} | {n_with[c]} | {total[c]} |" for c in CATS]
    lines += [
        "",
        f"## {args.top_k} mẫu có TED Acc thấp nhất",
        "",
        "> Cột *Ghi chú thủ công*: tự xem ảnh để bổ sung nguyên nhân (ảnh nghiêng/mờ, chữ nhỏ, hóa đơn dài…).",
        "",
        "| idx | TED Acc | F1 | #field GT | " + " | ".join(CATS) + " | Nhãn tự động | Ghi chú thủ công |",
        "|" + "---|" * (len(CATS) + 6),
    ]
    for row in worst:
        lines.append(f"| {row['idx']} | {row['ted_acc']} | {row['f1']} | {row['n_gt_fields']} | "
                     + " | ".join(str(row[c]) for c in CATS) + f" | {row['auto_label']} |  |")
    utils.ensure_dir(os.path.dirname(args.out_md))
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if args.save_images:
        s = data["summary"]
        ds = load_split(s.get("dataset", "naver-clova-ix/cord-v2"), s.get("split", "test"))
        by_idx = {r["idx"]: r for r in records}
        utils.ensure_dir(args.images_dir)
        for row in worst:
            r = by_idx[row["idx"]]
            ds[r["idx"]]["image"].convert("RGB").save(os.path.join(args.images_dir, f"{r['idx']:03d}.png"))
            utils.save_json({"ted_acc": r["ted_acc"], "f1": r["f1"], "raw_sequence": r["raw_sequence"],
                             "pred": r["pred"], "gt": r["gt"]},
                            os.path.join(args.images_dir, f"{r['idx']:03d}.json"))
        print(f"Đã lưu ảnh + gt/pred vào {args.images_dir}")


if __name__ == "__main__":
    main()
