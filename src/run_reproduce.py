"""Phần 1 – Tái lập Donut trên CORD-v2 + phân tích "Donut hoạt động thế nào".

    python -m src.run_reproduce                       # mọi bước (bỏ qua bước đã có kết quả)
    python -m src.run_reproduce --stage official      # chỉ đánh giá checkpoint chính thức
    python -m src.run_reproduce --stage train eval curves
    python -m src.run_reproduce --dry_run             # chỉ in lệnh

Các bước:
  env       ghi môi trường -> results/env.txt; selftest logic
  official  checkpoint chính thức naver-clova-ix/donut-base-finetuned-cord-v2 trên CORD-v2 test:
            chuẩn hóa ImageNet như code gốc (fp16 = đúng test.py gốc; fp32) và chuẩn hóa 0.5 của HF (test + val)
  train     fine-tune từ donut-base theo công thức gốc (configs/train_cord_reproduce.yaml, ~3.4 giờ trên RTX 5060)
  eval      đánh giá model vừa train trên test (fp16 như test.py gốc; fp32)
  curves    đường cong huấn luyện
  analysis  kiến trúc + số tham số, phân tích lỗi, cross-attention (checkpoint chính thức)
Kết quả: results/reproduce/, results/analysis/, outputs/donut_cord_reproduce/
"""
import argparse
import time

from src import utils
from src.runner import Runner

OFFICIAL = "naver-clova-ix/donut-base-finetuned-cord-v2"
OUT = "outputs/donut_cord_reproduce"
R = "results/reproduce"
A = "results/analysis"
STAGES = ["env", "official", "train", "eval", "curves", "analysis"]


def ev(run, run_id, out, model, *extra):
    run("src.evaluate", "--model", model, "--run_id", run_id, "--out", out, *extra, done_if=[out])


def main():
    ap = argparse.ArgumentParser(description="Tái lập Donut trên CORD-v2 + phân tích",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--stage", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--keep_going", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    run, t0 = Runner(args.dry_run, args.keep_going), time.time()
    st = set(args.stage)

    if "env" in st:
        run("src.env_info", done_if=["results/env.txt"])
        run("src.selftest")
    if "official" in st:
        ev(run, "official-imagenet-fp16", f"{R}/official_imagenet_fp16.json", OFFICIAL,
           "--image_norm", "imagenet", "--precision", "fp16", "--max_length", "768", "--notes", "giống test.py gốc")
        ev(run, "official-imagenet-fp32", f"{R}/official_imagenet_fp32.json", OFFICIAL,
           "--image_norm", "imagenet", "--max_length", "768")
        ev(run, "official-hfnorm-fp32", f"{R}/official_hfnorm_fp32.json", OFFICIAL, "--image_norm", "hf")
        ev(run, "official-hfnorm-fp32-val", f"{R}/official_hfnorm_fp32_val.json", OFFICIAL,
           "--image_norm", "hf", "--split", "validation")
    if "train" in st:
        run("src.train", "--config", "configs/train_cord_reproduce.yaml",
            done_if=[f"{OUT}/train_summary.json", f"{OUT}/best/config.json"])
    if "eval" in st:
        ev(run, "reproduce-test-fp16", f"{R}/reproduce_test_fp16.json", f"{OUT}/best",
           "--image_norm", "imagenet", "--precision", "fp16", "--max_length", "768")
        ev(run, "reproduce-test-fp32", f"{R}/reproduce_test_fp32.json", f"{OUT}/best",
           "--image_norm", "imagenet", "--max_length", "768")
    if "curves" in st:
        run("src.plot_results", "--run_dir", OUT, "--out", f"{R}/reproduce_train_curves.png",
            done_if=[f"{R}/reproduce_train_curves.png"])
    if "analysis" in st:
        run("src.model_info", "--model", OFFICIAL, "--out_dir", A, done_if=[f"{A}/model_info.json"])
        run("src.error_analysis", "--pred", f"{R}/official_hfnorm_fp32.json", "--top_k", "10", "--save_images",
            done_if=[f"{A}/error_analysis.md"])
        worst = []
        if not args.dry_run:
            recs = utils.load_json(f"{R}/official_hfnorm_fp32.json")["records"]
            worst = [str(r["idx"]) for r in sorted(recs, key=lambda r: r["ted_acc"])[:2]]
        run("src.visualize_attention", "--model", OFFICIAL, "--indices", "0", "1", "2", *worst, "--layer", "last",
            done_if=[f"{A}/attention/sample0_summary.png"])
        run("src.visualize_attention", "--model", OFFICIAL, "--indices", "0", "--layer", "mean",
            "--out_dir", f"{A}/attention_meanlayers", done_if=[f"{A}/attention_meanlayers/sample0_summary.png"])
    run.finish(t0, f"{R}/, {A}/")


if __name__ == "__main__":
    main()
