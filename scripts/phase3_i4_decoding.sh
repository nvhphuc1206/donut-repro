#!/usr/bin/env bash
# I4 – Beam search, hậu xử lý, giải mã có ràng buộc (không cần train).
set -euo pipefail
MODEL=${1:-naver-clova-ix/donut-base-finetuned-cord-v2}

for B in 1 3 5; do
  python -m src.evaluate --model "$MODEL" --num_beams $B --run_id "I4-beam$B" --out "results/i4/beam$B.json"
done
python -m src.evaluate --model "$MODEL" --constrained --run_id I4-greedy-constrained --out results/i4/greedy_constrained.json

# hậu xử lý trên các file dự đoán (CPU, vài giây)
python -m src.postprocess_eval --pred results/i4/beam1.json --run_prefix I4-beam1
python -m src.postprocess_eval --pred results/i4/beam5.json --run_prefix I4-beam5
