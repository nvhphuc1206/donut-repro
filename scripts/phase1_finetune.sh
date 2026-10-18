#!/usr/bin/env bash
# Task 1.2 – Fine-tune B0 rồi đánh giá checkpoint tốt nhất trên test bằng đúng script Task 1.1.
# Dùng: bash scripts/phase1_finetune.sh [OUTPUT_DIR]
#   Colab: bash scripts/phase1_finetune.sh /content/drive/MyDrive/donut/outputs/B0_full_1280
# Bị ngắt phiên? Chạy lại đúng lệnh này -> tự resume từ last.pt.
set -euo pipefail
OUT=${1:-outputs/B0_full_1280}

python -m src.train --config configs/train_cord_full.yaml --set "output_dir=$OUT"

python -m src.evaluate --model "$OUT/best" --split test \
  --run_id B0-test --out results/b0_test.json --summary_csv results/p1_summary.csv \
  --notes "fine-tune nhóm, best theo val TED"
python -m src.plot_results curves --run_dir "$OUT" --out results/figures/B0_curves.png
