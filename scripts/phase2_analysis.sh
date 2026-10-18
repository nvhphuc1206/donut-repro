#!/usr/bin/env bash
# Phase 2 – "How does Donut work?": kiến trúc, cross-attention, phân tích lỗi.
# Cần results/p1_official_cord.json từ phase1_reproduce.sh.
set -euo pipefail
MODEL=${1:-naver-clova-ix/donut-base-finetuned-cord-v2}

python -m src.model_info --model "$MODEL"                                   # Task 2.1
python -m src.error_analysis --pred results/p1_official_cord.json --top_k 10 --save_images   # Task 2.3

# Task 2.2: vài mẫu tốt + 2 mẫu tệ nhất (lấy idx từ results/error_analysis.md)
WORST=$(python - <<'PY'
import json
r = json.load(open("results/p1_official_cord.json", encoding="utf-8"))["records"]
print(" ".join(str(x["idx"]) for x in sorted(r, key=lambda x: x["ted_acc"])[:2]))
PY
)
python -m src.visualize_attention --model "$MODEL" --indices 0 1 2 $WORST --layer last
python -m src.visualize_attention --model "$MODEL" --indices 0 --layer mean --out_dir results/attention_meanlayers
