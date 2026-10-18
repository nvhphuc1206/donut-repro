#!/usr/bin/env bash
# I5 – fp16 (GPU) và dynamic int8 (CPU): latency, kích thước, độ chính xác.
set -euo pipefail
MODEL=${1:-naver-clova-ix/donut-base-finetuned-cord-v2}
LIMIT=${2:-30}   # CPU chậm -> chỉ chạy N ảnh đầu cho cặp so sánh CPU

python -m src.evaluate --model "$MODEL" --precision fp32 --run_id I5-gpu-fp32 --out results/i5/gpu_fp32.json
python -m src.evaluate --model "$MODEL" --precision fp16 --run_id I5-gpu-fp16 --out results/i5/gpu_fp16.json
python -m src.evaluate --model "$MODEL" --precision fp32 --device cpu --limit $LIMIT --warmup 1 \
  --run_id "I5-cpu-fp32-n$LIMIT" --out results/i5/cpu_fp32.json
python -m src.evaluate --model "$MODEL" --precision int8 --limit $LIMIT --warmup 1 \
  --run_id "I5-cpu-int8-n$LIMIT" --out results/i5/cpu_int8.json
