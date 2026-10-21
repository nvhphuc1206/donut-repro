"""Phần 2 – Cải tiến: cắt vùng chữ (Canny) trước encoder, đánh giá trên dữ liệu có viền và dữ liệu chụp sát.

    python -m src.run_preprocess                      # mọi bước (bỏ qua bước đã có kết quả)
    python -m src.run_preprocess --stage eval compare
    python -m src.run_preprocess --stage eval --limit 5 --out_dir results/preprocess_smoke   # chạy thử
    python -m src.run_preprocess --dry_run

Các bước:
  data      tạo tập test: data/cord_far (CORD chụp xa), data/ext_sroie, data/ext_wildreceipt (nếu chưa có)
  preview   chất lượng bước cắt (không cần GPU): % từ GT nằm trong vùng cắt, số ảnh giữ nguyên, hệ số phóng to chữ
  eval      2 model × 4 tập × {none, crop}; bỏ qua file đã có
  compare   bảng so sánh crop vs none + paired bootstrap -> report.md
  examples  ảnh minh họa (5 SROIE + 5 CORD chụp xa cải thiện rõ nhất)

Model: official = checkpoint chính thức; reproduce = model tự train ở phần 1 (outputs/donut_cord_reproduce/best).
Tập:   test = CORD-v2 test; far = CORD chụp xa; sroie, wildreceipt = bộ ngoài CORD (zero-shot, chấm bằng --ext_eval).
Kết quả: results/preprocess/
"""
import argparse
import os
import time

from src.runner import Runner

MODELS = {"official": "naver-clova-ix/donut-base-finetuned-cord-v2",
          "reproduce": "outputs/donut_cord_reproduce/best"}
DATASETS = {"test": "naver-clova-ix/cord-v2", "far": "data/cord_far",
            "sroie": "data/ext_sroie", "wildreceipt": "data/ext_wildreceipt"}
EXTERNAL = {"sroie", "wildreceipt"}
VARIANTS = ["none", "crop"]
STAGES = ["data", "preview", "eval", "compare", "examples"]


def main():
    ap = argparse.ArgumentParser(description="Cải tiến: cắt vùng chữ trước encoder",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--stage", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="Chỉ N ảnh đầu (chạy thử)")
    ap.add_argument("--out_dir", default="results/preprocess")
    ap.add_argument("--ext_max_samples", type=int, default=300, help="Số ảnh mỗi bộ ngoài CORD khi tạo dữ liệu")
    ap.add_argument("--keep_going", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    run, t0, out = Runner(args.dry_run, args.keep_going), time.time(), args.out_dir
    st = set(args.stage)

    if "data" in st:
        run("src.make_far_test", "--out", "data/cord_far", "--preview", f"{out}/far_preview.png",
            done_if=["data/cord_far/dataset_info.json"])
        todo = [d for d in ("wildreceipt", "sroie") if not os.path.isdir(DATASETS[d])]
        if todo:
            run("src.make_external_test", "--which", *todo, "--max_samples", str(args.ext_max_samples))
    if "preview" in st:
        for d in args.datasets:
            run("src.preprocess_preview", "--dataset", DATASETS[d], "--tag", d, "--out_dir", f"{out}/crop",
                done_if=[f"{out}/crop/{d}_crop_summary.json"])
    if "eval" in st:
        for m in args.models:
            if m == "reproduce" and not os.path.isdir(MODELS[m]) and not args.dry_run:
                print(f"[!] Không thấy {MODELS[m]} -> bỏ qua (chạy src.run_reproduce --stage train trước)")
                continue
            for d in args.datasets:
                for v in VARIANTS:
                    f = f"{out}/eval/{m}__{d}__{v}.json"
                    extra = (["--limit", str(args.limit)] if args.limit else []) + (["--ext_eval"] if d in EXTERNAL else [])
                    # chuẩn hóa ImageNet + fp16 = đúng như test.py gốc
                    run("src.evaluate", "--model", MODELS[m], "--dataset", DATASETS[d], "--split", "test",
                        "--image_norm", "imagenet", "--precision", "fp16", "--max_length", "768", "--preprocess", v,
                        *extra, "--run_id", f"preprocess-{m}-{d}-{v}", "--out", f, "--runs_csv", f"{out}/runs.csv",
                        done_if=[f])
    if "compare" in st:
        run("src.preprocess_compare", "--eval_dir", f"{out}/eval", "--out_dir", out)
    if "examples" in st:
        run("src.preprocess_examples", "--eval_dir", f"{out}/eval", "--out_dir", f"{out}/examples")
    run.finish(t0, out)


if __name__ == "__main__":
    main()
