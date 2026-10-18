#!/usr/bin/env bash
# I2 – LoRA fine-tune, so sánh với full FT (B0): tham số train, VRAM đỉnh, thời gian/epoch, F1, TED.
set -euo pipefail
OUT=${1:-outputs/I2_lora_r8}
python -m src.train --config configs/train_cord_lora.yaml --set "output_dir=$OUT"
python -m src.evaluate --model "$OUT/best" --run_id I2-test --out results/i2_lora_test.json \
  --notes "LoRA r=8 (đã merge)"
python -m src.plot_results curves --run_dir "$OUT" --out results/figures/I2_curves.png
# Thử r=16: python -m src.train --config configs/train_cord_lora.yaml --set run_id=I2_lora_r16 lora.r=16 lora.alpha=32 output_dir=outputs/I2_lora_r16
