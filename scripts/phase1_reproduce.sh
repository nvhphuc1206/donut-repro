#!/usr/bin/env bash
# Phase 1 – Tái lặp kết quả.
#   Task 1.1: inference checkpoint chính thức trên CORD-v2 test (~15 phút T4)
#   Task 1.2: fine-tune từ donut-base (3–5 giờ T4) -> chạy riêng bằng phase1_finetune.sh
set -euo pipefail
OFFICIAL=naver-clova-ix/donut-base-finetuned-cord-v2

python -m src.evaluate --model "$OFFICIAL" --split test \
  --run_id P1-official --out results/p1_official_cord.json --summary_csv results/p1_summary.csv \
  --notes "checkpoint chính thức, greedy, fp32"

# (khuyến nghị) báo cáo thêm trên validation
python -m src.evaluate --model "$OFFICIAL" --split validation \
  --run_id P1-official-val --out results/p1_official_cord_val.json --summary_csv results/p1_summary.csv
