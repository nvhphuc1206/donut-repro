"""Task 1.3 (tùy chọn) – Minh họa tính đa nhiệm: checkpoint chính thức RVL-CDIP / DocVQA trên TẬP CON.

    python -m src.eval_multitask --task rvlcdip --limit 500
    python -m src.eval_multitask --task docvqa --limit 500

Lưu ý: đây là subset -> KHÔNG so sánh trực tiếp với số liệu full test của paper.
Tên dataset/cột trên HF Hub có thể thay đổi: kiểm tra và truyền --dataset/--config/--split/--*_col nếu cần.
"""
import argparse
import time

import torch
from tqdm import tqdm

from src import utils
from src.data import load_split
from src.evaluate import generate_sequences, load_model_and_processor
from src.json_tokens import token2json
from src.metrics import anls

DEFAULTS = {
    "rvlcdip": dict(model="naver-clova-ix/donut-base-finetuned-rvlcdip", dataset="aharley/rvl_cdip",
                    config=None, split="test"),
    "docvqa": dict(model="naver-clova-ix/donut-base-finetuned-docvqa", dataset="lmms-lab/DocVQA",
                   config="DocVQA", split="validation"),
}


def norm_label(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def detect_columns(ds, image_col, label_col):
    feats = getattr(ds, "features", None) or {}
    if image_col is None:
        image_col = next((k for k, v in feats.items() if type(v).__name__ == "Image"), "image")
    if label_col is None:
        label_col = next((k for k, v in feats.items() if type(v).__name__ == "ClassLabel"), "label")
    names = feats[label_col].names if label_col in feats and hasattr(feats[label_col], "names") else None
    return image_col, label_col, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rvlcdip", "docvqa"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--dataset")
    ap.add_argument("--config")
    ap.add_argument("--split")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--no_streaming", action="store_true", help="Tải toàn bộ split thay vì streaming")
    ap.add_argument("--image_col")
    ap.add_argument("--label_col")
    ap.add_argument("--question_col", default="question")
    ap.add_argument("--answers_col", default="answers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs_csv", default="results/runs.csv")
    args = ap.parse_args()
    utils.start_log(f"results/logs/P1-{args.task}.log")
    d = DEFAULTS[args.task]
    model_id = args.model or d["model"]
    dataset = args.dataset or d["dataset"]
    config = args.config or d["config"]
    split = args.split or d["split"]

    utils.set_seed(42)
    model, processor, device = load_model_and_processor(model_id, device=args.device)
    max_length = model.decoder.config.max_position_embeddings
    ds = load_split(dataset, split, config_name=config, streaming=not args.no_streaming)
    image_col, label_col, names = detect_columns(ds, args.image_col, args.label_col)
    added = processor.tokenizer.get_added_vocab()

    records, times = [], []
    it = ds.take(args.limit) if hasattr(ds, "take") else ds.select(range(min(args.limit, len(ds))))
    for i, s in enumerate(tqdm(it, total=args.limit, desc=args.task)):
        image = s[image_col].convert("RGB")
        pv = processor.image_processor(image, return_tensors="pt").pixel_values.to(device)
        if args.task == "rvlcdip":
            prompt = "<s_rvlcdip>"
        else:
            prompt = f"<s_docvqa><s_question>{s[args.question_col]}</s_question><s_answer>"
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        seqs, _ = generate_sequences(model, processor, pv, prompt, max_length)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        pred = token2json(seqs[0], added)
        if args.task == "rvlcdip":
            gt = s[label_col]
            gt = names[gt] if names and isinstance(gt, int) else gt
            p = pred.get("class", "") if isinstance(pred, dict) else ""
            score = float(norm_label(p) == norm_label(gt))
        else:
            gt = list(s[args.answers_col])
            p = pred.get("answer", "") if isinstance(pred, dict) else ""
            if isinstance(p, list):
                p = p[0] if p else ""
            score = anls(p, gt)
        records.append({"i": i, "pred": p, "gt": gt, "raw_sequence": seqs[0], "score": score})

    metric = "accuracy" if args.task == "rvlcdip" else "anls"
    value = sum(r["score"] for r in records) / max(len(records), 1)
    summary = {
        "id": f"P1-{args.task}-subset{len(records)}", "stage": "eval", "model": model_id,
        "dataset": dataset, "split": split, "n": len(records), metric: round(value, 4),
        "sec_per_img": round(sum(times) / max(len(times), 1), 4), "gpu": utils.gpu_name(),
        "time": utils.now_str(), "notes": "subset – không so sánh trực tiếp với full test",
    }
    utils.save_json({"summary": summary, "records": records}, f"results/p1_multitask_{args.task}.json")
    utils.append_csv(args.runs_csv, summary)
    print(summary)


if __name__ == "__main__":
    main()
