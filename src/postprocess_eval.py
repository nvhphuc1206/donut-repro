"""I4 – Đánh giá lại các biến thể hậu xử lý trên file dự đoán có sẵn (không cần GPU).

    python -m src.postprocess_eval --pred results/p1_official_cord.json --run_prefix I4-greedy
Biến thể:
    raw          : token2json như gốc
    fix          : + tự sửa cấu trúc thẻ
    fix+schema   : + loại key ngoài schema CORD (schema lấy từ split train)
    fix+schema+norm : + chuẩn hóa giá trị số / khoảng trắng ở CẢ pred và GT (thay đổi metric -> báo cáo riêng)
"""
import argparse

from transformers import DonutProcessor

from src import utils
from src.data import get_gt_parses, load_split
from src.json_tokens import token2json
from src.metrics import score_predictions
from src.postprocess import filter_schema, fix_structure, normalize_tree, schema_from_gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/p1_official_cord.json")
    ap.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--run_prefix", default="I4")
    ap.add_argument("--out_csv", default="results/i4_postprocess.csv")
    ap.add_argument("--runs_csv", default="results/runs.csv")
    args = ap.parse_args()
    utils.start_log(f"results/logs/{args.run_prefix}-postprocess.log")

    data = utils.load_json(args.pred)
    src_summary, records = data["summary"], data["records"]
    added_vocab = DonutProcessor.from_pretrained(src_summary["model"]).tokenizer.get_added_vocab()

    train = load_split(args.dataset, args.train_split)
    paths, leaf_keys = schema_from_gts(p for g in train["ground_truth"] for p in get_gt_parses(g))
    print(f"Schema: {len(paths)} đường dẫn key, {len(leaf_keys)} key lá")

    gts = [r["gt"] for r in records]
    raws = [r["raw_sequence"] for r in records]
    fixed = [token2json(fix_structure(s, leaf_keys), added_vocab) for s in raws]
    schema = [filter_schema(p, paths) for p in fixed]
    variants = {
        "raw": ([token2json(s, added_vocab) for s in raws], gts),
        "fix": (fixed, gts),
        "fix+schema": (schema, gts),
        "fix+schema+norm": ([normalize_tree(p) for p in schema], [normalize_tree(g) for g in gts]),
    }

    print(f"\nNguồn: {args.pred} (beam={src_summary.get('num_beams')}, {src_summary.get('sec_per_img')} s/ảnh)")
    print(f"{'variant':<18}{'F1':>8}{'TED Acc':>10}")
    for name, (preds, golds) in variants.items():
        sc = score_predictions(preds, golds)
        row = {
            "id": f"{args.run_prefix}-{name}",
            "stage": "postprocess",
            "source": args.pred,
            "model": src_summary.get("model"),
            "input_size": src_summary.get("input_size"),
            "num_beams": src_summary.get("num_beams"),
            "variant": name,
            "n": len(records),
            "f1": round(sc["f1"], 4),
            "ted_acc": round(sc["ted_acc"], 4),
            "sec_per_img": src_summary.get("sec_per_img"),
            "time": utils.now_str(),
            "notes": "norm áp dụng cho cả GT" if name.endswith("norm") else "",
        }
        utils.append_csv(args.out_csv, row)
        utils.append_csv(args.runs_csv, row)
        print(f"{name:<18}{row['f1']:>8.4f}{row['ted_acc']:>10.4f}")


if __name__ == "__main__":
    main()
