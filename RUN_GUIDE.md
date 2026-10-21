# Hướng dẫn chạy

Mọi lệnh chạy **từ thư mục gốc dự án** (`donut/`), trong Anaconda Prompt, PowerShell hoặc terminal Linux.
Các runner **tự bỏ qua bước đã có kết quả**; chạy lại cùng lệnh để tiếp tục khi bị ngắt (train cũng tự resume từ `last.pt`).

## 0. Cài đặt

```bash
conda create -n donut python=3.11 -y
conda activate donut
# PyTorch có CUDA: chọn lệnh phù hợp với GPU tại https://pytorch.org (dự án dùng torch 2.11 + cu128)
pip install -r requirements.txt
python -m src.selftest          # kiểm tra json<->token, metric, chấm bộ ngoài, bước cắt (CPU)
```

- Windows: các runner tự đặt `HF_HUB_DISABLE_XET=1`. Nếu tự gọi `python -m src.evaluate` mà tải dữ liệu bị treo ở 0%, hãy đặt biến này trước.
- Môi trường đã dùng: `results/env.txt` (Windows 11, Python 3.11, torch 2.11 + CUDA 12.8, transformers 4.57.6, RTX 5060 8 GB).

## 1–2. Tái lập + phân tích: `python -m src.run_reproduce`

| Bước (`--stage`) | Việc làm | Thời gian (RTX 5060) | Kết quả |
|---|---|---|---|
| `env` | ghi môi trường, selftest | < 1 phút | `results/env.txt` |
| `official` | checkpoint chính thức trên CORD-v2 test: chuẩn hóa ImageNet (fp16 = đúng `test.py` gốc, và fp32); chuẩn hóa 0.5 của HF (test + val) | ~4 × 1 phút | `results/reproduce/official_*.json` |
| `train` | fine-tune từ `donut-base` theo `configs/train_cord_reproduce.yaml` | ~3.4 giờ | `outputs/donut_cord_reproduce/` |
| `eval` | model vừa train trên test (fp16, fp32) | ~2 × 1 phút | `results/reproduce/reproduce_test_*.json` |
| `curves` | đường cong huấn luyện | vài giây | `results/reproduce/reproduce_train_curves.png` |
| `analysis` | kiến trúc + số tham số, phân tích lỗi, cross-attention | ~5 phút | `results/analysis/` |

```bash
python -m src.run_reproduce                         # tất cả
python -m src.run_reproduce --stage official        # một phần
python -m src.run_reproduce --dry_run               # xem lệnh sẽ chạy
# chạy thử pipeline train vài phút (ghi ra thư mục riêng, không ảnh hưởng kết quả):
python -m src.train --config configs/train_cord_reproduce.yaml --set output_dir=outputs/_smoke runs_csv=outputs/_smoke/runs.csv train_limit=8 epochs=1 val_limit=4 val_every=1
```

## 3. Cải tiến (cắt vùng chữ): `python -m src.run_preprocess`

| Bước (`--stage`) | Việc làm | Thời gian | Kết quả |
|---|---|---|---|
| `data` | tạo `data/cord_far` (CORD chụp xa) và `data/ext_sroie`, `data/ext_wildreceipt` (300 ảnh mỗi bộ, theo schema CORD) | ~15 phút (tải SROIE) | `data/` |
| `preview` | chất lượng bước cắt, không cần GPU (% từ GT nằm trong vùng cắt, số ảnh giữ nguyên, hệ số phóng to chữ) | ~1 phút | `results/preprocess/crop/` |
| `eval` | 2 model (`official`, `reproduce`) × 4 tập (`test`, `far`, `sroie`, `wildreceipt`) × {`none`, `crop`} | ~40–60 phút | `results/preprocess/eval/` |
| `compare` | so sánh crop với none, paired bootstrap 10 000 lần | vài giây | `results/preprocess/report.md`, `compare.csv` |
| `examples` | ảnh minh họa: 5 SROIE + 5 CORD chụp xa | ~30 giây | `results/preprocess/examples/` |

```bash
python -m src.run_preprocess
python -m src.run_preprocess --stage eval compare --models official --datasets far sroie
python -m src.run_preprocess --stage eval --limit 5 --out_dir results/preprocess_smoke   # chạy thử
```

- `--preprocess crop` có thể dùng riêng với mọi lệnh `python -m src.evaluate`.
- SROIE và WildReceipt là đánh giá zero-shot: nhãn chỉ ánh xạ một phần sang schema CORD, nên phải chấm bằng `--ext_eval` (runner tự thêm).

## Thư mục báo cáo

```bash
python -m src.build_report_folder      # -> bao_cao_ket_qua/ (chỉ đọc results/ và outputs/)
```

## Lưu ý kỹ thuật (đã xử lý trong code)

| Vấn đề | Cách xử lý |
|---|---|
| transformers ≥ 4.46: `VisionEncoderDecoderModel(labels=...)` dịch nhãn thêm 1 vị trí → loss giảm nhưng sinh chuỗi sai | `train.py` tự tính cross-entropy (`seq2seq_loss`) |
| `preprocessor_config.json` trên HF ghi mean = std = 0.5, còn code gốc dùng mean/std ImageNet (TED 0.9048 so với 0.9104) | `--image_norm imagenet` (mặc định trong mọi runner) |
| nltk ≥ 3.9 giới hạn `edit_distance` ≤ 2000 ký tự → dừng validation | bỏ giới hạn trong `metrics.py` (code gốc không có giới hạn này) |
| Windows khóa file checkpoint lớn (`WinError 32`) | `atomic_save` thử lại tối đa 30 × 10 giây |
| `hf_xet` treo khi tải dữ liệu trên Windows | `HF_HUB_DISABLE_XET=1`; chỉ tải file của split cần dùng |

Nhật ký sự cố đầy đủ: `results/issues.md`.
