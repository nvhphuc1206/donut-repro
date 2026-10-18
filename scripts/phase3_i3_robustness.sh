#!/usr/bin/env bash
# I3 – Tập test nhiễu + fine-tune có augmentation.
# Dùng: bash scripts/phase3_i3_robustness.sh [B0_BEST_DIR] [OUT_ROOT]
set -euo pipefail
B0_BEST=${1:-outputs/B0_full_1280/best}
OUT_ROOT=${2:-outputs}
OFFICIAL=naver-clova-ix/donut-base-finetuned-cord-v2
TYPES="rotate blur jpeg color perspective mixed"

# 1) tạo tập nhiễu
python -m src.make_noisy_test --out_root data/cord_noisy --types $TYPES

# 2) checkpoint chính thức trên tập nhiễu -> mức sụt giảm
for T in $TYPES; do
  python -m src.evaluate --model "$OFFICIAL" --dataset "data/cord_noisy/$T" \
    --run_id "I3-official-$T" --out "results/i3/official_$T.json"
done

# 3) fine-tune thêm từ B0: có augmentation và đối chứng không augmentation
python -m src.train --config configs/train_cord_aug.yaml \
  --set "model_name_or_path=$B0_BEST" "output_dir=$OUT_ROOT/I3_aug_from_B0"
python -m src.train --config configs/train_cord_noaug_control.yaml \
  --set "model_name_or_path=$B0_BEST" "output_dir=$OUT_ROOT/I3_noaug_from_B0"

# 4) đánh giá B0, B0+aug, B0+noaug trên test sạch + nhiễu
for NAME in B0:"$B0_BEST" AUG:"$OUT_ROOT/I3_aug_from_B0/best" NOAUG:"$OUT_ROOT/I3_noaug_from_B0/best"; do
  TAG=${NAME%%:*}; M=${NAME#*:}
  python -m src.evaluate --model "$M" --run_id "I3-$TAG-clean" --out "results/i3/${TAG}_clean.json"
  for T in $TYPES; do
    python -m src.evaluate --model "$M" --dataset "data/cord_noisy/$T" \
      --run_id "I3-$TAG-$T" --out "results/i3/${TAG}_$T.json"
  done
done
