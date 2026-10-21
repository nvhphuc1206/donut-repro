# Donut trên CORD-v2: tái lập, phân tích và cải tiến bằng cắt vùng chữ

Dự án gồm 3 phần, dùng HuggingFace `transformers`, chạy trên 1 GPU 8 GB (RTX 5060):

1. **Tái lập** Donut ([Kim et al., 2022](https://arxiv.org/abs/2111.15664), repo `clovaai/donut`) trên CORD-v2. Gồm chạy lại checkpoint chính thức và tự fine-tune từ `donut-base` theo đúng công thức gốc.
2. **Phân tích Donut hoạt động thế nào**: kiến trúc và số tham số, cross-attention, phân tích lỗi.
3. **Cải tiến**: cắt ảnh sát vùng chữ (phát hiện bằng Canny) trước khi đưa vào encoder. Không đổi model, không train lại.

Hàm chấm điểm (`JSONParseEvaluator`: F1 field-level, TED Acc) được port nguyên vẹn từ repo gốc (MIT).

## Kết quả chính

**Tái lập** (CORD-v2 test, 100 ảnh)

| Mô hình | F1 | TED Acc |
|---|---|---|
| Donut công bố (README repo gốc) | ~0.841 | 0.913 |
| Checkpoint chính thức, chạy lại bằng pipeline của nhóm | 0.8373 | 0.9103 |
| Nhóm tự fine-tune theo công thức gốc (30 epoch, 3.35 giờ) | 0.8326 | 0.8934 |

**Cải tiến**: dùng checkpoint chính thức, chỉ thêm bước cắt vùng chữ trước encoder

| Dữ liệu | Không cắt | Có cắt |
|---|---|---|
| CORD chụp xa (100 ảnh): F1 | 0.667 | **0.804** |
| SROIE, ảnh có nhiều lề trắng (47 ảnh): đọc đúng tổng tiền | 8.5% | **36.2%** |
| CORD-v2 test gốc (ảnh chụp sát): F1 / TED | 0.837 / 0.910 | 0.845 / 0.907 (không khác biệt có ý nghĩa) |

Chi tiết, bảng đầy đủ kèm khoảng tin cậy và hình minh họa: [bao_cao_ket_qua/](bao_cao_ket_qua/README.md).

## Cấu trúc thư mục

```
donut/
├── README.md                       # (file này)
├── RUN_GUIDE.md                    # cài đặt + lệnh chạy từng phần
├── requirements.txt
├── configs/
│   └── train_cord_reproduce.yaml   # công thức train gốc (có chú thích giá trị gốc từng dòng)
├── src/
│   ├── data.py, json_tokens.py, metrics.py, utils.py   # dữ liệu CORD, JSON <-> token, F1/TED (port repo gốc)
│   ├── train.py                    # fine-tune Donut (resume theo epoch)
│   ├── evaluate.py                 # đánh giá F1/TED/NED/latency; --preprocess crop; --ext_eval cho bộ ngoài CORD
│   ├── plot_results.py             # đường cong huấn luyện
│   ├── model_info.py, visualize_attention.py, error_analysis.py   # phần 2 – phân tích
│   ├── preprocess.py               # phần 3 – thuật toán cắt vùng chữ (Canny)
│   ├── make_far_test.py            # tạo tập "CORD chụp xa"
│   ├── make_external_test.py       # tạo tập SROIE, WildReceipt theo schema CORD
│   ├── ext_eval.py                 # chấm điểm bộ ngoài CORD (lọc schema, chuẩn hóa giá trị, total_acc)
│   ├── preprocess_preview.py, preprocess_compare.py, preprocess_examples.py   # chất lượng cắt, so sánh + bootstrap, ví dụ
│   ├── run_reproduce.py            # chạy phần 1 + 2
│   ├── run_preprocess.py           # chạy phần 3
│   ├── build_report_folder.py      # gom kết quả chính -> bao_cao_ket_qua/
│   ├── runner.py, env_info.py, selftest.py
├── outputs/
│   └── donut_cord_reproduce/       # model tự train (best/, last.pt, history.csv, train.log, ...)
├── results/
│   ├── reproduce/                  # đánh giá checkpoint chính thức + model tự train, đường cong train
│   ├── analysis/                   # kiến trúc, cross-attention, phân tích lỗi
│   ├── preprocess/                 # eval/ (2 model × 4 tập × none/crop), crop/, examples/, report.md, compare.csv
│   ├── logs/, runs.csv, env.txt, issues.md
├── data/                           # cord_far/, ext_sroie/, ext_wildreceipt/ (sinh lại được, không lên git)
├── bao_cao_ket_qua/                # thư mục báo cáo: 1_tai_lap_donut/, 2_phan_tich_donut/, 3_cai_tien_preprocess/
└── _archive/                       # thí nghiệm cũ không còn dùng (không lên git)
```

## Chạy nhanh

```bash
conda activate donut
python -m src.selftest                 # kiểm tra logic (CPU, vài giây)
python -m src.run_reproduce            # phần 1 + 2 (bỏ qua bước đã có kết quả)
python -m src.run_preprocess           # phần 3
python -m src.build_report_folder      # cập nhật bao_cao_ket_qua/
```

Mọi lệnh chạy từ thư mục gốc dưới dạng `python -m src.<module>`, và mỗi module đều có `--help`. Chi tiết xem [RUN_GUIDE.md](RUN_GUIDE.md).
