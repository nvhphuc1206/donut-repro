#!/usr/bin/env bash
# Phase 0 – Cài đặt + ghi môi trường + kiểm tra nhanh các module (chạy từ thư mục gốc dự án).
set -euo pipefail
pip install -q -r requirements.txt
python -m src.env_info --out results/env.txt
python -m src.selftest
