#!/usr/bin/env bash
# Tổng hợp mọi dòng trong results/runs.csv thành bảng Markdown cho báo cáo.
set -euo pipefail
python -m src.plot_results table --out results/runs_table.md
