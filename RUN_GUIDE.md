# Hướng dẫn chạy từng bước – Dự án Donut

> Checklist thực hành: làm lần lượt từ trên xuống, đánh dấu `[x]` khi xong.
> Giải thích chi tiết từng script: [README.md](README.md). Kế hoạch gốc: [DONUT_PROJECT_PLAN.md](DONUT_PROJECT_PLAN.md).

Hướng dẫn có **hai cách chạy**:

| | Cách A – Máy local (conda, Windows) | Cách B – Colab / Kaggle |
|---|---|---|
| GPU | RTX 5060 **8 GB** | T4 **16 GB** |
| Gõ lệnh ở | Anaconda Prompt / PowerShell | Cell notebook (thêm `!` trước lệnh) |
| Chạy lệnh | `python -m src.<script> ...` (PowerShell) hoặc `bash scripts/*.sh` (Git Bash) | `bash scripts/*.sh` |
| Phù hợp | Bước 0–5, 7, 9; bước 6, 8 nếu đủ VRAM | Mọi bước, nhất là train dài (6, 8) |

**Quy ước chung**
- Mọi lệnh chạy **từ thư mục gốc dự án** (`donut/`), sau khi đã `conda activate donut` (cách A).
- Lệnh trong file này viết **một dòng** nên dán được vào cả PowerShell lẫn bash.
- Sau mỗi bước, xem bảng **"Kiểm tra"** trước khi sang bước tiếp theo.
- Mọi kết quả tự ghi vào `results/runs.csv`; sự cố tự ghi tay vào `results/issues.md`.
- Toàn bộ output console (print, warning, traceback) tự lưu vào **file log**: `results/logs/<run_id>.log` cho đánh giá/phân tích, `outputs/<run>/train.log` cho train. Chạy lại cùng `run_id` sẽ **nối tiếp** vào file cũ.

---

## ⚡ Chạy tự động toàn bộ (khuyến nghị)

Thay vì gõ từng lệnh ở các bước bên dưới, có thể chạy **một lệnh** – script tự bỏ qua bước đã có kết quả, chạy lần lượt phần còn lại:

```powershell
conda activate donut
python -m src.run_all --list     # xem 61 bước + bước nào đã xong ([x])
python -m src.run_all            # chạy tất cả bước chưa xong
```

**Theo dõi** (mở trong VS Code, tự cập nhật ~5 giây/lần):

| File | Nội dung |
|---|---|
| `results/pipeline_status.md` | Bước hiện tại, % hoàn thành (theo số bước và theo thời gian ước tính), thanh tiến trình đang chạy, bảng trạng thái mọi bước + link log |
| `results/logs/pipeline.log` | Nhật ký bắt đầu / xong / lỗi của từng bước |
| `results/logs/pipeline/NN_<bước>.log` | Toàn bộ output của từng bước |
| `results/issues.md` | Bước lỗi được ghi tự động (lệnh, dòng lỗi cuối, đường dẫn log) |

**Hành vi**
- **Dừng giữa chừng** (Ctrl+C, tắt máy): chạy lại `python -m src.run_all` → bỏ qua bước đã xong; bước train tự resume từ `last.pt`.
- **Bước lỗi** không dừng cả pipeline: các bước phụ thuộc vào nó bị đánh dấu ⛔ `blocked`, các bước độc lập vẫn chạy. Sửa lỗi xong chạy lại là được. (Muốn dừng ngay khi lỗi: `--stop_on_error`.)
- **Resolution train tự chọn**: bước `smoke` thử train ở 1280×960; nếu lỗi (OOM trên GPU 8 GB) → B0, LoRA, I3 tự dùng 960×720. Lựa chọn được ghi trong `results/pipeline_state.json`. Ép cố định: `--train_size 1280x960` hoặc `--train_size 960x720`.
- Tự đặt `HF_HUB_DISABLE_XET=1`, `num_workers=0` (Windows) cho các bước con.

**Chọn phần muốn chạy**
```powershell
python -m src.run_all --only_groups P2 I1 I4      # chỉ phần inference (~3 giờ)
python -m src.run_all --skip_groups I5 I3         # bỏ I5 và I3
python -m src.run_all --only_steps i4_beam5       # một bước
python -m src.run_all --only_steps p2_attn --force   # chạy lại bước đã xong
python -m src.run_all --with_multitask            # thêm Task 1.3 (RVL-CDIP, DocVQA)
```
Nhóm: `P0 P1 P2 I1 I4 I5 TRAIN B0 I2 I3 TABLE`.

Thứ tự trong pipeline: P0 → P1 → P2 → I1 → I4 → I5 → smoke → B0 → I2 → I3 → bảng tổng hợp. Toàn bộ khoảng **15–20 giờ GPU** (phần lớn là 4 lần train), nên có thể chạy qua đêm hoặc chia thành nhiều lần bằng `--only_groups`.

Các mục bên dưới mô tả chi tiết từng bước (để chạy tay hoặc hiểu kết quả).

---

## Tổng quan

| Bước | Nội dung | Thời gian (T4) | Cần bước trước | Máy 8 GB |
|------|----------|----------------|----------------|----------|
| 0 | Cài đặt, kiểm tra môi trường | ~10 phút | – | ✅ |
| 1 | Chạy checkpoint chính thức trên CORD (Task 1.1) | ~15 phút | 0 | ✅ |
| 2 | Phân tích kiến trúc, lỗi, attention (Phase 2) | ~20 phút | 1 | ✅ |
| 3 | I1 – độ phân giải × tốc độ | ~30 phút | 0 | ✅ (1600×1200 có thể OOM) |
| 4 | I4 – beam search, hậu xử lý | ~30 phút | 0 | ✅ |
| 5 | Smoke test pipeline train + đo VRAM | ~5 phút | 0 | ✅ |
| 6 | Fine-tune B0 (Task 1.2) | 3–5 giờ | 5 | ⚠️ xem kết quả bước 5 |
| 7 | I2 – LoRA | 2–3 giờ | 5 | ✅ nhiều khả năng vừa |
| 8 | I3 – augmentation + độ bền | 2–3 giờ | 6 | ⚠️ như bước 6 |
| 9 | I5 – fp16 / int8 (tùy chọn) | ~30 phút | 0 | ✅ |
| 10 | Tổng hợp bảng kết quả | ~0 | tất cả | ✅ |

Các bước 1–4 chỉ là inference → làm trước để có số liệu sớm.

---

## Bước 0 – Chuẩn bị môi trường

### Cách A – Conda trên Windows (máy local)

- [ ] Mở **Anaconda Prompt** (hoặc PowerShell đã `conda init`) và tạo môi trường:
  ```powershell
  conda create -n donut python=3.11 -y
  conda activate donut
  ```
  > Dùng **Python 3.11**, không dùng 3.13 (một số gói như `zss`, `sentencepiece` có thể chưa có wheel).

- [ ] Cài PyTorch bản **CUDA 12.8** (bắt buộc cho RTX 50xx – kiến trúc Blackwell):
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  ```

- [ ] Cài các thư viện còn lại:
  ```powershell
  cd D:\Study\Data_science_K35\MDL21_Advanced-Computer-Vision\final_project\donut
  pip install -r requirements.txt
  ```

- [ ] Kiểm tra:
  ```powershell
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python -m src.env_info
  python -m src.selftest
  ```

**Lưu ý riêng cho Windows**
- Khi train **luôn thêm** `--set num_workers=0` (DataLoader nhiều worker hay treo trên Windows).
- Muốn chạy các file `scripts/*.sh`: mở **Git Bash**, chạy `conda activate donut` (nếu chưa được, chạy `source ~/anaconda3/etc/profile.d/conda.sh` trước), rồi `bash scripts/<tên>.sh`.
- Lần đầu tải model/dataset từ HuggingFace có thể báo cảnh báo symlink trên Windows – bỏ qua được.

### Cách B – Colab / Kaggle

- [ ] **Colab:** upload thư mục `donut/` (không cần `outputs/`, `data/`) lên Google Drive, ví dụ `MyDrive/donut/`. Mở `notebooks/donut_colab.ipynb` → *Runtime → Change runtime type → T4 GPU*, rồi chạy:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  %cd /content/drive/MyDrive/donut
  ```
  ```bash
  !nvidia-smi
  !bash scripts/phase0_setup.sh
  ```
- [ ] **Kaggle:** upload dự án thành Dataset, bật *Internet* + *GPU T4*, rồi `cp -r /kaggle/input/<ten-dataset>/ /kaggle/working/donut && cd /kaggle/working/donut`.
- [ ] Nếu Colab yêu cầu restart runtime sau khi pip hạ cấp `transformers` → restart rồi chạy lại cell `%cd`.
- [ ] Thư mục checkpoint **phải nằm trên Drive**: thay `outputs/...` trong các lệnh train bằng `/content/drive/MyDrive/donut/outputs/...`.

| Kiểm tra | |
|---|---|
| `torch.cuda.is_available()` = `True`, thấy tên GPU | [ ] |
| `python -m src.selftest` in `Tất cả kiểm tra: OK` | [ ] |
| Có file `results/env.txt` | [ ] |

---

## Bước 1 – Tái lặp với checkpoint chính thức (Task 1.1)

```powershell
python -m src.evaluate --split test --run_id P1-official --out results/p1_official_cord.json --summary_csv results/p1_summary.csv --notes "checkpoint chinh thuc, greedy, fp32"
python -m src.evaluate --split validation --run_id P1-official-val --out results/p1_official_cord_val.json --summary_csv results/p1_summary.csv
```
(Git Bash / Colab: `bash scripts/phase1_reproduce.sh`)

Mặc định `--model` là `naver-clova-ix/donut-base-finetuned-cord-v2`; lần đầu sẽ tải model (~800 MB) và dataset CORD.

| Kiểm tra | |
|---|---|
| Log in `input_size(H,W)=[1280, 960]` | [ ] |
| TED Acc test ≈ **0.913** (chấp nhận ±0.01) | [ ] |
| F1 test ≈ **0.84** (chấp nhận ±0.015) | [ ] |
| Có `results/p1_official_cord.json`, `results/p1_summary.csv` | [ ] |

Nếu lệch nhiều: kiểm tra phiên bản `transformers` trong `results/env.txt` (phải < 5), prompt `<s_cord-v2>`, resolution. Ghi vào `results/issues.md`.

**Ghi số liệu:** dòng *"Checkpoint chính thức – tái lặp"* trong bảng Phase 1 (F1, TED Acc, s/ảnh, tên GPU).

---

## Bước 2 – Phân tích "How does Donut work?" (Phase 2)

> Cần `results/p1_official_cord.json` từ Bước 1. Chạy **2.1 → 2.3 → 2.2**.

```powershell
# 2.1 Kiến trúc: shape tensor, số tham số, sơ đồ
python -m src.model_info

# 2.3 Phân tích lỗi: 10 mẫu TED thấp nhất
python -m src.error_analysis --pred results/p1_official_cord.json --top_k 10 --save_images
```

Mở `results/error_analysis.md`, lấy **idx của 2 mẫu đầu bảng** (tệ nhất), thay vào `<idx1> <idx2>`:

```powershell
# 2.2 Cross-attention: 3 mẫu thường + 2 mẫu thất bại
python -m src.visualize_attention --indices 0 1 2 <idx1> <idx2> --layer last
python -m src.visualize_attention --indices 0 --layer mean --out_dir results/attention_meanlayers
```
(Git Bash / Colab: `bash scripts/phase2_analysis.sh` – tự lấy idx tệ nhất)

| Kiểm tra | Đầu ra |
|---|---|
| [ ] Sơ đồ kiến trúc có shape thật | `results/figures/architecture.png`, `results/architecture.mmd` |
| [ ] Số tham số, depths, window, số layer decoder | `results/model_info.json` (đối chiếu với paper) |
| [ ] Bảng 10 mẫu tệ nhất + thống kê loại lỗi | `results/error_analysis.md` |
| [ ] Ảnh + gt/pred của mẫu tệ | `results/errors/` |
| [ ] Heatmap cross-attention | `results/attention/sample*_summary.png` |

**Việc làm tay:**
- [ ] Xem ảnh trong `results/errors/`, điền cột *Ghi chú thủ công* trong `error_analysis.md` (nghiêng, mờ, chữ nhỏ, hóa đơn dài…).
- [ ] Chọn ≥1 heatmap thành công (tập trung đúng dòng chữ) và 1 ví dụ thất bại cho báo cáo.

---

## Bước 3 – I1: Độ phân giải × tốc độ

```powershell
python -m src.evaluate --input_size 640 480 --run_id I1-official-640x480 --out results/i1/official_640x480.json
python -m src.evaluate --input_size 960 720 --run_id I1-official-960x720 --out results/i1/official_960x720.json
python -m src.evaluate --input_size 1280 960 --run_id I1-official-1280x960 --out results/i1/official_1280x960.json
python -m src.evaluate --input_size 1600 1200 --run_id I1-official-1600x1200 --out results/i1/official_1600x1200.json
python -m src.plot_results tradeoff --prefix I1-official- --out results/figures/i1_tradeoff.png
```
(Git Bash / Colab: `bash scripts/phase3_i1_resolution.sh`)

| Kiểm tra | |
|---|---|
| Có các dòng `I1-official-*` trong `results/runs.csv` (nếu 1600×1200 OOM → ghi vào `issues.md`, bỏ qua) | [ ] |
| Có biểu đồ `results/figures/i1_tradeoff.png` | [ ] |

> Khi viết báo cáo: model được train ở 1280×960 → đây là kiểm tra **độ bền theo resolution**, không phải train lại.

---

## Bước 4 – I4: Beam search & hậu xử lý

```powershell
python -m src.evaluate --num_beams 1 --run_id I4-beam1 --out results/i4/beam1.json
python -m src.evaluate --num_beams 3 --run_id I4-beam3 --out results/i4/beam3.json
python -m src.evaluate --num_beams 5 --run_id I4-beam5 --out results/i4/beam5.json
python -m src.evaluate --constrained --run_id I4-greedy-constrained --out results/i4/greedy_constrained.json
python -m src.postprocess_eval --pred results/i4/beam1.json --run_prefix I4-beam1
python -m src.postprocess_eval --pred results/i4/beam5.json --run_prefix I4-beam5
```
(Git Bash / Colab: `bash scripts/phase3_i4_decoding.sh`)

| Kiểm tra | |
|---|---|
| Có `results/i4/beam1.json`, `beam3.json`, `beam5.json`, `greedy_constrained.json` | [ ] |
| Có `results/i4_postprocess.csv` (4 biến thể: `raw`, `fix`, `fix+schema`, `fix+schema+norm`) | [ ] |
| So sánh latency beam 5 vs greedy | [ ] |

> `fix+schema+norm` chuẩn hóa cả GT → **thay đổi metric**; báo cáo cả số trước và sau.

---

## Bước 5 – Smoke test pipeline train + đo VRAM

```powershell
# Kiểm tra pipeline (nhanh, 960x720)
python -m src.train --config configs/smoke_test.yaml --set num_workers=0

# Máy 8 GB: đo VRAM ở đúng độ phân giải của B0
python -m src.train --config configs/smoke_test.yaml --set num_workers=0 run_id=SMOKE_1280 "input_size=[1280,960]"
```
(Colab: bỏ `--set num_workers=0` cũng được)

| Kiểm tra | |
|---|---|
| Không lỗi; chạy hết 1 epoch + val 8 ảnh | [ ] |
| Xem cột `peak_vram_gb` trong `outputs/SMOKE_1280/history.csv` | [ ] |

**Quyết định cho bước 6 (máy 8 GB):**
- Lần chạy 1280×960 **không OOM** → chạy B0 trên máy local như bình thường.
- **OOM** → chọn một trong hai: (a) chạy B0 trên Colab; (b) chạy local ở 960×720 và ghi rõ trong báo cáo.

Điểm số ở bước này **không có ý nghĩa**. Xong thì xóa `outputs/SMOKE*` và `results/smoke_runs.csv`.

---

## Bước 6 – Fine-tune B0 từ `donut-base` (Task 1.2)

**Máy local (đủ VRAM):**
```powershell
python -m src.train --config configs/train_cord_full.yaml --set num_workers=0 output_dir=outputs/B0_full_1280
```

**Máy local, phải giảm resolution:**
```powershell
python -m src.train --config configs/train_cord_full.yaml --set num_workers=0 run_id=B0_full_960 output_dir=outputs/B0_full_960 "input_size=[960,720]" val_batch_size=1
```

**Colab:**
```bash
!bash scripts/phase1_finetune.sh /content/drive/MyDrive/donut/outputs/B0_full_1280
```

Sau khi train xong (local), đánh giá và vẽ đường cong (đổi `B0_full_1280` nếu dùng 960):
```powershell
python -m src.evaluate --model outputs/B0_full_1280/best --run_id B0-test --out results/b0_test.json --summary_csv results/p1_summary.csv --notes "fine-tune nhom, best theo val TED"
python -m src.plot_results curves --run_dir outputs/B0_full_1280 --out results/figures/B0_curves.png
```
> Nếu B0 train ở 960×720, thêm `--input_size 960 720` khi đánh giá (thực ra `best/` đã lưu sẵn kích thước này trong processor).

**Trong khi train:**
- Mỗi epoch in `val F1`, `TED` và lưu `last.pt`.
- **Bị ngắt** (tắt máy, Colab hết phiên)? Chạy lại **đúng lệnh train cũ** → tự resume từ epoch đã xong.
- Đổi cấu hình giữa chừng (vd. resolution) → dùng `output_dir` mới, đừng resume từ thư mục cũ.

| Kiểm tra | |
|---|---|
| `outputs/<run>/train_summary.json` (best_epoch, val TED, VRAM, giờ train) | [ ] |
| Dòng `B0-test` trong `results/p1_summary.csv` | [ ] |
| Đường cong `results/figures/B0_curves.png` | [ ] |

**Ghi số liệu:** dòng *"Fine-tune của nhóm"* (bảng Phase 1) + dòng B0 (bảng Phase 3). Giải thích chênh lệch với paper: 12 vs 30 epoch, batch hiệu dụng, fp16, GPU, resolution (nếu giảm).

---

## Bước 7 – I2: LoRA

```powershell
python -m src.train --config configs/train_cord_lora.yaml --set num_workers=0 output_dir=outputs/I2_lora_r8
python -m src.evaluate --model outputs/I2_lora_r8/best --run_id I2-test --out results/i2_lora_test.json --notes "LoRA r=8 (da merge)"
python -m src.plot_results curves --run_dir outputs/I2_lora_r8 --out results/figures/I2_curves.png
```
(Colab: `!bash scripts/phase3_i2_lora.sh /content/drive/MyDrive/donut/outputs/I2_lora_r8`)

> Để so sánh công bằng với B0, dùng **cùng resolution với B0** (nếu B0 chạy 960×720 thì thêm `"input_size=[960,720]"`).

Resume giống Bước 6. Cuối quá trình, adapter tốt nhất được merge và lưu vào `best/`.

| Kiểm tra | |
|---|---|
| Log `trainable params` nhỏ hơn nhiều so với B0 | [ ] |
| Dòng `I2_lora_r8` (stage=train) và `I2-test` trong `runs.csv` | [ ] |

**So sánh với B0:** `trainable_params_M`, `effective_trainable_params_M`, `peak_vram_gb`, `sec_per_epoch`, F1, TED Acc.

(Tùy chọn) r=16:
```powershell
python -m src.train --config configs/train_cord_lora.yaml --set num_workers=0 run_id=I2_lora_r16 lora.r=16 lora.alpha=32 output_dir=outputs/I2_lora_r16
```

---

## Bước 8 – I3: Augmentation + độ bền (cần B0)

`B0` dưới đây = thư mục `best/` của Bước 6 (vd. `outputs/B0_full_1280/best`).

**8.1 Tạo 6 tập test nhiễu** (rotate, blur, jpeg, color, perspective, mixed):
```powershell
python -m src.make_noisy_test --out_root data/cord_noisy
```

**8.2 Checkpoint chính thức trên tập nhiễu** (PowerShell):
```powershell
$types = "rotate","blur","jpeg","color","perspective","mixed"
foreach ($t in $types) { python -m src.evaluate --dataset "data/cord_noisy/$t" --run_id "I3-official-$t" --out "results/i3/official_$t.json" }
```

**8.3 Train thêm 4 epoch từ B0 – có và không augmentation:**
```powershell
python -m src.train --config configs/train_cord_aug.yaml --set num_workers=0 model_name_or_path=outputs/B0_full_1280/best output_dir=outputs/I3_aug_from_B0
python -m src.train --config configs/train_cord_noaug_control.yaml --set num_workers=0 model_name_or_path=outputs/B0_full_1280/best output_dir=outputs/I3_noaug_from_B0
```
> Nếu B0 train ở 960×720, thêm `"input_size=[960,720]"` vào cả hai lệnh.

**8.4 Đánh giá B0 / AUG / NOAUG trên test sạch + 6 tập nhiễu** (PowerShell):
```powershell
$models = @{ "B0" = "outputs/B0_full_1280/best"; "AUG" = "outputs/I3_aug_from_B0/best"; "NOAUG" = "outputs/I3_noaug_from_B0/best" }
foreach ($tag in $models.Keys) {
  $m = $models[$tag]
  python -m src.evaluate --model $m --run_id "I3-$tag-clean" --out "results/i3/${tag}_clean.json"
  foreach ($t in $types) { python -m src.evaluate --model $m --dataset "data/cord_noisy/$t" --run_id "I3-$tag-$t" --out "results/i3/${tag}_$t.json" }
}
```
(Git Bash / Colab: `bash scripts/phase3_i3_robustness.sh <B0_best> <thư_mục_outputs>` làm cả 8.1–8.4)

| Kiểm tra | |
|---|---|
| Ảnh minh họa `results/figures/noisy_preview.png` | [ ] |
| Các dòng `I3-official-*`, `I3-B0-*`, `I3-AUG-*`, `I3-NOAUG-*` trong `runs.csv` | [ ] |

---

## Bước 9 – I5: fp16 / int8 (tùy chọn)

```powershell
python -m src.evaluate --precision fp32 --run_id I5-gpu-fp32 --out results/i5/gpu_fp32.json
python -m src.evaluate --precision fp16 --run_id I5-gpu-fp16 --out results/i5/gpu_fp16.json
python -m src.evaluate --precision fp32 --device cpu --limit 30 --warmup 1 --run_id I5-cpu-fp32-n30 --out results/i5/cpu_fp32.json
python -m src.evaluate --precision int8 --limit 30 --warmup 1 --run_id I5-cpu-int8-n30 --out results/i5/cpu_int8.json
```
(Git Bash / Colab: `bash scripts/phase3_i5_compress.sh naver-clova-ix/donut-base-finetuned-cord-v2 30`)

| Kiểm tra | |
|---|---|
| 4 dòng `I5-*` trong `runs.csv` | [ ] |
| So sánh `sec_per_img`, `model_size_mb`, F1, TED | [ ] |

---

## Bước 10 – Tổng hợp kết quả

```powershell
python -m src.plot_results table --out results/runs_table.md
```

- [ ] Mở `results/runs_table.md` → chép số vào bảng Phase 1 và Phase 3 của `report/report_template.md`.
- [ ] Ghi rõ GPU đã dùng cho từng thí nghiệm (cột `gpu` trong `runs.csv`) – RTX 5060 và T4 có latency khác nhau, **không so sánh latency giữa hai GPU**.
- [ ] Đối chiếu **số liệu paper** trực tiếp với bảng trong arXiv 2111.15664 (không điền từ trí nhớ).
- [ ] Kiểm tra `results/issues.md` đã ghi đủ các sự cố gặp phải.
- [ ] Nếu chạy cả local lẫn Colab: gộp thư mục `results/` (nhất là `runs.csv`) về một chỗ.

---

## Phụ lục

### A. Lệnh lẻ hay dùng
```powershell
# Đánh giá một checkpoint bất kỳ
python -m src.evaluate --model <thư mục hoặc HF id> --run_id <ID> --out results/<ID>.json

# Đánh giá trên validation / chỉ N ảnh đầu
python -m src.evaluate --model <...> --split validation --limit 20 --run_id <ID>

# Ghi đè tham số train bất kỳ (nhiều seed để báo mean ± std)
python -m src.train --config configs/train_cord_full.yaml --set num_workers=0 seed=1 run_id=B0_s1 output_dir=outputs/B0_s1

# Xem tham số của script
python -m src.evaluate --help
```

> PowerShell: giá trị có dấu `[ ]` như `input_size=[960,720]` cần đặt trong ngoặc kép: `"input_size=[960,720]"`.

### B. (Tùy chọn) Task 1.3 – đa nhiệm trên tập con
```powershell
python -m src.eval_multitask --task rvlcdip --limit 500
python -m src.eval_multitask --task docvqa --limit 500
```
Ghi rõ là **subset**. Nếu dataset không tải được, truyền `--dataset`, `--config`, `--split` và tên cột (`--image_col`, `--label_col`, `--question_col`, `--answers_col`).

### C. Sự cố thường gặp
| Triệu chứng | Xử lý |
|---|---|
| `no kernel image is available for execution on the device` | PyTorch không hỗ trợ RTX 50xx → cài lại từ index `cu128` (Bước 0) |
| `torch.cuda.is_available()` = `False` | Chưa cài bản CUDA của torch, hoặc driver NVIDIA cũ → cập nhật driver, cài lại torch |
| `CUDA out of memory` | `"input_size=[960,720]"`, `val_batch_size=1`; hoặc chuyển sang Colab |
| Train trên Windows bị treo ở epoch đầu | Thiếu `--set num_workers=0` |
| Cảnh báo *symlinks* trên Windows | Vô hại (cache HF copy file thay vì symlink, tốn thêm dung lượng). Đã tắt sẵn trong code; muốn hết hẳn thì bật *Developer Mode* của Windows |
| `Xet Storage is enabled ... 'hf_xet' package is not installed` | Vô hại, chỉ tải chậm hơn → `pip install hf_xet` |
| Tải dataset/model đứng yên ở 0% nhiều phút | Ctrl+C, tắt Xet rồi chạy lại: PowerShell `$env:HF_HUB_DISABLE_XET="1"` (Git Bash: `export HF_HUB_DISABLE_XET=1`). File đã tải xong được giữ trong cache |
| File log đứng yên khi đang tải/chạy | Bình thường: log chỉ ghi trạng thái **cuối** của thanh tiến trình; xem tiến độ trên terminal |
| `Using a slow image processor...` | Vô hại; giữ processor "slow" để giống checkpoint gốc |
| Lỗi cài `zss` / `sentencepiece` | Đang dùng Python 3.13 → tạo lại env với `python=3.11` |
| TED Acc chính thức lệch xa 0.913 | Kiểm tra `input_size(H,W)=[1280, 960]`, `transformers<5` |
| Model sinh lặp / không dừng | Thử `--constrained` |
| Lỗi khi train LoRA | `pip install -U peft`, gửi log để sửa tên module |
| `conda activate` không chạy trong Git Bash | `source ~/anaconda3/etc/profile.d/conda.sh` (hoặc `~/miniconda3/...`) rồi activate lại |
