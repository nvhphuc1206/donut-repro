#!/usr/bin/env bash
# I1 – Ablation độ phân giải khi inference với checkpoint chính thức (train ở 1280x960).
# Đây là kiểm tra độ bền theo resolution, không phải train lại.
set -euo pipefail
OFFICIAL=naver-clova-ix/donut-base-finetuned-cord-v2
for HW in "640 480" "960 720" "1280 960" "1600 1200"; do
  set -- $HW
  python -m src.evaluate --model "$OFFICIAL" --input_size $1 $2 \
    --run_id "I1-official-${1}x${2}" --out "results/i1/official_${1}x${2}.json" \
    --notes "inference-only, resolution khác lúc train" || echo "[!] ${1}x${2} lỗi (OOM?) – ghi vào results/issues.md"
done
python -m src.plot_results tradeoff --prefix I1-official- --out results/figures/i1_tradeoff.png

# (tùy chọn) I1-b: fine-tune ở 960x720 rồi so với B0
# python -m src.train --config configs/train_cord_full_960.yaml --set output_dir=outputs/I1b_full_960
# python -m src.evaluate --model outputs/I1b_full_960/best --input_size 960 720 --run_id I1b-test --out results/i1b_test.json
