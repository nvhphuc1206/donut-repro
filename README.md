# Donut – Tái lặp & Cải tiến (OCR-free Document Understanding Transformer)

Mã nguồn hiện thực [DONUT_PROJECT_PLAN.md](DONUT_PROJECT_PLAN.md) bằng HuggingFace `transformers` (phương án B).
Evaluator (`JSONParseEvaluator`) được port nguyên vẹn từ repo gốc `clovaai/donut` (MIT) và là **hàm đánh giá duy nhất** cho mọi thí nghiệm.

---

## 1. Cấu trúc thư mục

```
donut/
├── DONUT_PROJECT_PLAN.md
├── README.md                    # (file này) hướng dẫn chạy
├── requirements.txt
├── configs/                     # YAML cho từng thí nghiệm train
│   ├── smoke_test.yaml          # chạy thử 3–5 phút
│   ├── train_cord_full.yaml     # B0  – Task 1.2 full fine-tune 1280x960
│   ├── train_cord_full_960.yaml # I1-b – full fine-tune 960x720
│   ├── train_cord_lora.yaml     # I2  – LoRA
│   ├── train_cord_aug.yaml      # I3  – B0 + augmentation
│   └── train_cord_noaug_control.yaml  # I3 – đối chứng
├── src/
│   ├── json_tokens.py           # json2token / token2json
│   ├── metrics.py               # F1, TED Acc (port repo gốc), ANLS
│   ├── data.py                  # Dataset CORD, special tokens, resolution
│   ├── augment.py               # augmentation train + nhiễu cho test
│   ├── evaluate.py              # Task 1.1 / I1 / I3 / I4 / I5 – đánh giá + latency
│   ├── train.py                 # Task 1.2 / I1-b / I2 / I3 – fine-tune, resume
│   ├── model_info.py            # Task 2.1 – shape tensor, số tham số, sơ đồ kiến trúc
│   ├── visualize_attention.py   # Task 2.2 – heatmap cross-attention
│   ├── error_analysis.py        # Task 2.3 – phân loại lỗi
│   ├── postprocess.py           # I4 – sửa cấu trúc / lọc schema / chuẩn hóa số
│   ├── postprocess_eval.py      # I4 – chấm lại các biến thể hậu xử lý (CPU)
│   ├── make_noisy_test.py       # I3 – tạo tập test nhiễu
│   ├── eval_multitask.py        # Task 1.3 – RVL-CDIP / DocVQA (subset)
│   ├── plot_results.py          # đường cong train, trade-off, bảng Markdown
│   ├── env_info.py              # ghi results/env.txt
│   └── selftest.py              # kiểm tra logic nhanh trên CPU
├── scripts/                     # bash: gom lệnh theo từng phase
├── notebooks/donut_colab.ipynb  # notebook chạy trên Colab/Kaggle
├── results/                     # JSON dự đoán, CSV, hình (tự sinh)
└── report/report_template.md    # khung báo cáo
```

**Quy ước đầu ra** (tự động, theo mục 9 của kế hoạch):

| File | Nội dung |
|------|----------|
| `results/runs.csv` | Mỗi thí nghiệm 1 dòng: id, config, thời gian, VRAM, F1, TED Acc, latency… |
| `results/<run_id>.json` | `summary` + `records` (raw sequence, pred, gt, TED/F1 từng ảnh) |
| `outputs/<run_id>/` | `best/` (checkpoint HF tốt nhất theo val TED), `last.pt` (resume), `history.csv`, `train_steps.csv`, `config.yaml` |
| `results/logs/<run_id>.log` | Toàn bộ output console của evaluate / phân tích (append theo mỗi lần chạy) |
| `outputs/<run_id>/train.log` | Toàn bộ output console khi train (kể cả các lần resume) |
| `results/issues.md` | Ghi lỗi & cách khắc phục (tự ghi tay khi gặp sự cố) |

> Mọi lệnh đều chạy **từ thư mục gốc dự án** dưới dạng module: `python -m src.<tên_script> ...`
> Mỗi script đều có `--help`.

---

## 2. Chuẩn bị môi trường

### 2.1 Google Colab (khuyến nghị, GPU T4)
1. Nén thư mục `donut/` (bỏ `outputs/`, `data/`) và upload lên Google Drive, ví dụ `MyDrive/donut/`.
2. Mở `notebooks/donut_colab.ipynb` trên Colab → *Runtime → Change runtime type → T4 GPU*.
3. Chạy cell Setup (mount Drive, `cd`, cài thư viện). Hoặc tự gõ:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   %cd /content/drive/MyDrive/donut
   !pip install -q -r requirements.txt
   ```
4. **Checkpoint train phải nằm trên Drive** để không mất khi phiên bị ngắt: luôn truyền
   `output_dir=/content/drive/MyDrive/donut/outputs/<RUN>` (các script `phase*.sh` nhận tham số này).

### 2.2 Kaggle (GPU T4 x2 / P100)
- Upload thư mục dự án thành một *Kaggle Dataset* rồi `cp -r /kaggle/input/<tên>/ /kaggle/working/donut && cd /kaggle/working/donut`.
- Bật *Internet* trong Settings (để tải model/dataset từ HF Hub).
- Output trong `/kaggle/working` được giữ lại sau khi *Save Version*; để resume, thêm output phiên trước làm input rồi copy `last.pt` về `output_dir`.
- Nếu P100 + bản torch mới báo lỗi CUDA capability, chọn T4.

### 2.3 Máy local (Windows, chỉ để đọc code/chạy phần CPU)
```powershell
pip install -r requirements.txt   # cần cài torch + torchvision riêng theo pytorch.org
python -m src.selftest
```
Trên Windows đặt `num_workers=0` khi train: `--set num_workers=0`.

### 2.4 Bước 0 – kiểm tra & ghi môi trường (~1 phút)
```bash
bash scripts/phase0_setup.sh
# = pip install + python -m src.env_info + python -m src.selftest
```
Kỳ vọng: `Tất cả kiểm tra: OK` và file `results/env.txt` (phiên bản thư viện + GPU, đưa vào báo cáo).

---

## 3. PHASE 1 – Tái lặp kết quả (G1)

### Bước 1.1 – Inference checkpoint chính thức trên CORD-v2 test (~15 phút T4)
```bash
bash scripts/phase1_reproduce.sh
```
Lệnh chính bên trong:
```bash
python -m src.evaluate --model naver-clova-ix/donut-base-finetuned-cord-v2 --split test \
  --run_id P1-official --out results/p1_official_cord.json --summary_csv results/p1_summary.csv
```
- Prompt `<s_cord-v2>`, greedy (`num_beams=1`), `max_length=768`, `bad_words_ids=[[unk]]`, fp32, batch 1, có 2 lượt warm-up không tính giờ.
- Kết quả in ra màn hình và lưu ở `results/p1_official_cord.json`, `results/p1_summary.csv`, `results/runs.csv`.
- **DoD:** TED Acc ≈ 0.913 (±0.01), F1 ≈ 0.84 (±0.015). Nếu lệch: kiểm tra dòng log `input_size(H,W)=[1280, 960]`, prompt token, phiên bản `transformers`.

### Bước 1.2 – Smoke test pipeline train (~3–5 phút) — **nên chạy trước B0**
```bash
python -m src.train --config configs/smoke_test.yaml
```
Chỉ để chắc chắn không lỗi/OOM (40 ảnh train, 8 ảnh val). Điểm số không có ý nghĩa. Xóa `outputs/SMOKE` sau khi xong.

### Bước 1.3 – Fine-tune B0 từ `donut-base` (3–5 giờ T4)
```bash
# Colab: lưu checkpoint lên Drive
bash scripts/phase1_finetune.sh /content/drive/MyDrive/donut/outputs/B0_full_1280
```
Bên trong:
```bash
python -m src.train --config configs/train_cord_full.yaml --set output_dir=<OUT>
python -m src.evaluate --model <OUT>/best --run_id B0-test --out results/b0_test.json --summary_csv results/p1_summary.csv
python -m src.plot_results curves --run_dir <OUT> --out results/figures/B0_curves.png
```
- Cấu hình (theo mục Task 1.2 của kế hoạch): 1280×960, batch 1 × grad-accum 8, lr 3e-5, warmup 300, cosine, fp16, gradient checkpointing, 12 epoch, max_length 768.
- Special tokens được sinh tự động từ `gt_parse` của tập train; `decoder_start_token_id = <s_cord-v2>`; prompt không bị tính loss.
- Mỗi epoch: log loss → `train_steps.csv`; chạy val (100 ảnh) → `history.csv`; nếu val TED tốt hơn → lưu `best/`; luôn ghi `last.pt`.
- **Phiên bị ngắt?** Chạy lại *đúng lệnh cũ* → tự resume từ epoch cuối đã hoàn thành.
- **OOM?** Theo thứ tự: giữ `batch_size=1`; `--set input_size=[960,720]`; giảm `val_batch_size=1`. Ghi lại thay đổi vào `results/issues.md` và báo cáo.
- Ghi đè bất kỳ tham số nào: `--set epochs=10 lr=5e-5 run_id=B0_e10`.

### Bước 1.4 (tùy chọn) – Đa nhiệm trên tập con
```bash
python -m src.eval_multitask --task rvlcdip --limit 500
python -m src.eval_multitask --task docvqa --limit 500
```
- Dùng streaming, chỉ lấy N mẫu đầu → **ghi rõ là subset**, không so sánh với full test.
- Tên dataset mặc định: `aharley/rvl_cdip` (test) và `lmms-lab/DocVQA` (config `DocVQA`, split `validation`). Nếu Hub đổi tên/cột, truyền `--dataset --config --split --image_col --label_col --question_col --answers_col`.
- Lưu ý: kiểm tra model card xem checkpoint DocVQA có dùng tập validation khi train không (nếu có, điểm sẽ lạc quan).

### Bảng Phase 1 cần điền
Lấy số từ `results/p1_summary.csv` và `outputs/B0_full_1280/train_summary.json`:

| Thiết lập | Resolution | Epoch | F1 | TED Acc | s/ảnh | GPU |
|-----------|-----------|-------|----|---------|-------|-----|
| Paper / README | 1280 | 30 | ~84.1 | 91.3 | (đối chiếu paper) | A100 |
| Checkpoint chính thức – tái lặp (`P1-official`) | 1280 | – | | | | T4 |
| Fine-tune của nhóm (`B0-test`) | 1280 | 12 | | | | T4 |

---

## 4. PHASE 2 – "How does Donut work?" (G2)

```bash
bash scripts/phase2_analysis.sh            # mặc định dùng checkpoint chính thức
```

| Bước | Lệnh | Đầu ra |
|------|------|--------|
| 2.1 Kiến trúc | `python -m src.model_info` | `results/model_info.json` (shape `pixel_values`, các stage Swin, `last_hidden_state`, depths, window, số layer decoder, vocab, số tham số), `results/architecture.mmd` (Mermaid), `results/figures/architecture.png` |
| 2.3 Phân tích lỗi | `python -m src.error_analysis --pred results/p1_official_cord.json --top_k 10 --save_images` | `results/error_analysis.md` (thống kê + bảng 10 mẫu tệ nhất), `results/error_analysis.csv`, `results/errors/<idx>.png/json` |
| 2.2 Cross-attention | `python -m src.visualize_attention --indices 0 1 2 <idx_tệ> --keys nm price cnt total_price --layer last` | `results/attention/sample<idx>_summary.png` + từng heatmap |

Ghi chú:
- **Chạy 2.3 trước 2.2** để biết idx của mẫu thất bại (script phase2 tự lấy 2 mẫu có TED thấp nhất).
- Heatmap = cross-attention trung bình theo head (lớp cuối `--layer last`, hoặc `--layer mean` / `--layer 0..3`), gộp trung bình các token thuộc cùng một giá trị field, reshape về lưới `(H/32, W/32)` (40×30 cho 1280×960), chồng lên ảnh đã qua processor.
- Cột *Ghi chú thủ công* trong `error_analysis.md` cần tự điền sau khi xem ảnh (nghiêng/mờ/chữ nhỏ/hóa đơn dài…).
- Số tham số in ra cần **đối chiếu với paper** trước khi đưa vào báo cáo.

---

## 5. PHASE 3 – Cải tiến (G3)

Chọn 2–3 hạng mục; tất cả dùng chung `evaluate.py`, seed 42. Thứ tự ưu tiên ★ như kế hoạch.

### I1 ★★★ – Độ phân giải × tốc độ (~0.5 giờ, chỉ inference)
```bash
bash scripts/phase3_i1_resolution.sh
```
- Chạy checkpoint chính thức ở 640×480, 960×720, 1280×960, 1600×1200 (`--input_size H W`); run_id `I1-official-HxW`.
- Vẽ `results/figures/i1_tradeoff.png` (TED/F1 & latency theo resolution).
- Ghi rõ: model train ở 1280×960 → đây là **kiểm tra độ bền theo resolution**.
- (Tốt hơn) fine-tune ở 960×720: `python -m src.train --config configs/train_cord_full_960.yaml` rồi so với B0.

### I2 ★★★ – LoRA (2–3 giờ)
```bash
bash scripts/phase3_i2_lora.sh /content/drive/MyDrive/donut/outputs/I2_lora_r8
```
- LoRA r=8, α=16, dropout 0.05 trên `q_proj, v_proj` của decoder (self + cross attention) và `query, value` của Swin stage 3–4.
- Embedding: chỉ học các hàng của special token mới (`train_embeddings: new_rows`) — tương đương `modules_to_save` nhưng không phá weight-tying `embed_tokens ↔ lm_head`.
- Cuối train, adapter tốt nhất được **merge** vào model và lưu `best/` → đánh giá bằng đúng `evaluate.py`.
- So sánh với B0 trong `results/runs.csv` (dòng `stage=train`): `trainable_params_M`, `effective_trainable_params_M`, `peak_vram_gb`, `sec_per_epoch`, và dòng `I2-test` (F1, TED).
- Thử thêm: `--set run_id=I2_lora_r16 lora.r=16 lora.alpha=32` hoặc chỉ decoder: `lora.target_encoder_stages=[]`.

### I3 ★★ – Augmentation + độ bền (2–3 giờ, cần B0 xong trước)
```bash
bash scripts/phase3_i3_robustness.sh /content/drive/MyDrive/donut/outputs/B0_full_1280/best /content/drive/MyDrive/donut/outputs
```
1. `make_noisy_test` tạo `data/cord_noisy/{rotate,blur,jpeg,color,perspective,mixed}` (cố định seed) + `results/figures/noisy_preview.png`.
2. Đánh giá checkpoint chính thức trên từng tập nhiễu → mức sụt giảm.
3. Fine-tune thêm 4 epoch từ B0 **có** augmentation (`I3_aug_from_B0`) và **không** augmentation (`I3_noaug_from_B0`, đối chứng).
4. Đánh giá B0 / AUG / NOAUG trên test sạch + 6 tập nhiễu (run_id `I3-<TAG>-<type>`).

### I4 ★★ – Beam search & hậu xử lý (~0.5 giờ, không train)
```bash
bash scripts/phase3_i4_decoding.sh
```
- Beam 1/3/5 (`--num_beams`), giải mã có ràng buộc (`--constrained`: chỉ cho phép thẻ đóng khớp thẻ mở gần nhất, cấm EOS khi còn thẻ mở).
- `postprocess_eval` chấm lại 4 biến thể trên file dự đoán: `raw`, `fix` (tự đóng thẻ), `fix+schema` (lọc key ngoài schema CORD lấy từ train), `fix+schema+norm` (chuẩn hóa số/khoảng trắng ở **cả pred và GT** → thay đổi metric, phải báo cáo cả trước/sau). Kết quả: `results/i4_postprocess.csv`.

### I5 ★ – fp16 / int8 (~0.5 giờ)
```bash
bash scripts/phase3_i5_compress.sh naver-clova-ix/donut-base-finetuned-cord-v2 30
```
GPU fp32 vs fp16 trên toàn bộ test; CPU fp32 vs dynamic int8 trên 30 ảnh đầu (CPU rất chậm). So sánh `sec_per_img`, `model_size_mb`, F1, TED.

### I6 – Hóa đơn tiếng Việt (mở rộng)
Chưa có script vì cần kiểm tra giấy phép/khả năng truy cập MC-OCR 2021 trước. Khi có dữ liệu, chỉ cần chuyển về định dạng `{"image", "ground_truth": "{\"gt_parse\": {...}}"}`, `save_to_disk` thành `DatasetDict` có các split `train/validation/test`, rồi chạy `train.py` với `dataset=<thư mục>` và `task_token=<s_mcocr>`.

### Bảng Phase 3
```bash
bash scripts/make_tables.sh      # -> results/runs_table.md (toàn bộ runs.csv dạng Markdown)
```

| ID | Thiết lập | Lấy từ dòng `runs.csv` |
|----|-----------|------------------------|
| B0 | Baseline full FT | `B0_full_1280` (train) + `B0-test` |
| I1-a | Res 960×720 (inference) | `I1-official-960x720` |
| I2 | LoRA r=8 | `I2_lora_r8` (train) + `I2-test` |
| I3 | +Augmentation (sạch / nhiễu) | `I3-AUG-clean`, `I3-AUG-mixed`… so với `I3-NOAUG-*`, `I3-B0-*` |
| I4 | Beam=5 + hậu xử lý | `I4-beam5-fix+schema` |

---

## 6. Thứ tự chạy đề xuất & ngân sách GPU (T4)

| # | Lệnh | GPU-giờ | Phụ thuộc |
|---|------|---------|-----------|
| 0 | `bash scripts/phase0_setup.sh` | ~0 | – |
| 1 | `bash scripts/phase1_reproduce.sh` | 0.3 | – |
| 2 | `bash scripts/phase2_analysis.sh` | 0.3 | 1 |
| 3 | `bash scripts/phase3_i1_resolution.sh` | 0.5 | – |
| 4 | `bash scripts/phase3_i4_decoding.sh` | 0.5 | – |
| 5 | `python -m src.train --config configs/smoke_test.yaml` | 0.1 | – |
| 6 | `bash scripts/phase1_finetune.sh <DRIVE>/B0_full_1280` | 3–5 | 5 |
| 7 | `bash scripts/phase3_i2_lora.sh <DRIVE>/I2_lora_r8` | 2–3 | 5 |
| 8 | `bash scripts/phase3_i3_robustness.sh <DRIVE>/B0_full_1280/best <DRIVE>` | 2–3 | 6 |
| 9 | `bash scripts/phase3_i5_compress.sh` (tùy chọn) | 0.5 | – |
| 10 | `bash scripts/make_tables.sh` | 0 | tất cả |

Các bước 1–4 chỉ cần inference → làm trước để có số liệu sớm. Bước 6–8 nên chạy ở các phiên Colab khác nhau, output trên Drive.

---

## 7. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|-------------|-----------|
| `CUDA out of memory` khi train | `--set input_size=[960,720]` / `val_batch_size=1`; giữ `gradient_checkpointing: true`, `fp16: true` |
| Điểm checkpoint chính thức lệch nhiều so với 91.3 | Kiểm tra `input_size(H,W)` in ra là `[1280, 960]`, prompt `<s_cord-v2>`, `transformers<5` |
| Model sinh lặp/không dừng | Đã đặt `eos_token_id`, `max_length=768`, `bad_words_ids=[[unk]]`; thử `--constrained` |
| `visualize_attention` báo không có `cross_attentions` | Model phải load với `attn_implementation="eager"` (đã tự làm); nâng/hạ `transformers` nếu vẫn lỗi |
| Lỗi `peft` khi train LoRA | `pip install -U peft`; kiểm tra tên module bằng `print(model)` nếu bản transformers đổi tên lớp |
| Dataset Task 1.3 không tải được | Dataset Hub đổi tên/định dạng → tìm bản parquet khác, truyền `--dataset`, `--*_col` |
| Windows: DataLoader treo | `--set num_workers=0` |

Mỗi khi gặp lỗi và sửa được, ghi một dòng vào `results/issues.md` (dùng cho phần thảo luận).

---

## 8. Ghi chú về tính trung thực số liệu
- Không điền số paper từ trí nhớ: đối chiếu trực tiếp bảng trong arXiv 2111.15664.
- Mọi số tái lặp phải lấy từ `results/runs.csv` / file JSON tương ứng.
- Test CORD chỉ 100 ảnh → báo cáo thêm điểm trên validation; nếu còn GPU, chạy B0 với 2–3 seed (`--set seed=1 run_id=B0_s1`) và báo mean ± std.
