# Dự án: Tái lặp & Cải tiến Donut (OCR-free Document Understanding Transformer)

> **Tài liệu này dành cho AI agent / người thực hiện.** Đọc toàn bộ trước khi bắt đầu. Mỗi task có *Input*, *Các bước*, *Output* và *Tiêu chí hoàn thành (DoD)*. Thực hiện tuần tự theo Phase. Khi gặp lỗi môi trường, xem mục 7 (Rủi ro & cách xử lý) trước khi tự sửa.

- **Bài báo:** Kim et al., *OCR-free Document Understanding Transformer*, ECCV 2022 — https://arxiv.org/abs/2111.15664
- **Code chính thức:** https://github.com/clovaai/donut (MIT license)
- **Bản HuggingFace:** `transformers` → `VisionEncoderDecoderModel` + `DonutProcessor`
- **Bối cảnh:** Bài tập môn Thị giác máy tính (cao học). Tài nguyên giả định: **1 GPU 16 GB (Colab T4 / Kaggle P100/T4)**, tổng thời gian GPU ≤ ~15 giờ.

---

## 0. Mục tiêu bài tập

| # | Mục tiêu | Sản phẩm cuối |
|---|----------|---------------|
| G1 | Tái lặp kết quả của bài báo (ít nhất trên CORD) | Bảng so sánh số liệu paper vs. tái lặp + log/notebook |
| G2 | Giải thích ý chính, tập trung vào **kiến trúc** và câu hỏi **"How does Donut work?"** | Phần báo cáo + hình minh họa (sơ đồ kiến trúc, cross-attention map) |
| G3 | Đề xuất & thực hiện 1–3 cải tiến nhẹ, có kết quả định lượng | Bảng ablation + phân tích |

---

## 1. Tóm tắt bài báo (kiến thức nền cho agent)

### 1.1 Vấn đề
Các hệ thống Visual Document Understanding (VDU) truyền thống (LayoutLM, BROS, …) là pipeline **OCR → model hiểu văn bản**. Nhược điểm mà paper chỉ ra:
1. **Chi phí tính toán cao** – OCR engine nặng, phải chạy riêng.
2. **Thiếu linh hoạt** về ngôn ngữ / loại tài liệu – OCR phải được huấn luyện lại cho từng domain.
3. **Lỗi OCR lan truyền** (error propagation) sang bước sau, đặc biệt với ngôn ngữ có bộ ký tự phức tạp (tiếng Hàn, Trung, Nhật).

### 1.2 Ý tưởng cốt lõi
**Donut = mô hình Transformer end-to-end, ảnh vào → chuỗi token (JSON) ra, không dùng OCR.** Mọi tác vụ (phân loại, trích xuất thông tin, VQA) được quy về **một bài toán duy nhất: sinh chuỗi có cấu trúc** (image-to-sequence).

### 1.3 Kiến trúc (trọng tâm G2)

```
Ảnh tài liệu x ∈ R^{H×W×C}
        │
        ▼
┌─────────────────────────────┐
│ ENCODER: Swin Transformer   │  patch 4x4 → patch merging qua 4 stage
│ layers {2, 2, 14, 2}        │  shifted window attention (window = 10)
│ input 2560×1920 (pretrain)  │
└─────────────────────────────┘
        │  chuỗi embedding {z_i} ∈ R^{n×d}  (n = số patch cuối, d = hidden dim)
        ▼  (cross-attention)
┌─────────────────────────────┐
│ DECODER: BART (4 layers)    │  khởi tạo từ mBART đa ngôn ngữ
│ auto-regressive             │  nhận prompt token: <s_cord-v2>, <s_docvqa>...
└─────────────────────────────┘
        │  chuỗi token y_1..y_m
        ▼
Token → JSON  (<s_key> … </s_key> ↔ {"key": …})
```

**Các điểm cần giải thích trong báo cáo:**

1. **Encoder – Swin Transformer**
   - Vì sao Swin, không phải ViT: tài liệu có độ phân giải cao (chữ nhỏ) → ViT attention toàn cục O(n²) quá tốn; Swin dùng attention cục bộ theo cửa sổ + shifted window → chi phí tuyến tính theo số patch, vẫn có liên kết giữa các cửa sổ.
   - Paper có ablation so sánh backbone (ResNet, EfficientNet, ViT, Swin) → Swin cho kết quả tốt nhất. Agent cần đọc Section 3 + phần phân tích (Section 4.x) của paper để lấy đúng số liệu.
   - **Độ phân giải đầu vào là siêu tham số quan trọng**: tăng resolution → tăng độ chính xác (đặc biệt DocVQA) nhưng tăng latency. Đây là cơ sở cho cải tiến I1.

2. **Decoder – BART**
   - Sinh token tự hồi quy, cross-attend vào {z_i}.
   - Chỉ 4 layer (nhẹ), khởi tạo từ mBART → có sẵn tri thức ngôn ngữ đa ngữ.
   - Từ điển được mở rộng bằng **special tokens** cho từng key JSON (`<s_menu>`, `<s_nm>`, `</s_nm>`, `<sep/>` cho phần tử danh sách…).

3. **Prompt / task token**
   - Decoder được "điều khiển" bởi token đầu: `<s_cord-v2>` → trích xuất hóa đơn; `<s_rvlcdip>` → phân loại; `<s_docvqa><s_question>…</s_question><s_answer>` → VQA (câu hỏi được đưa vào làm prompt, model sinh câu trả lời).
   - Tương tự cách GPT-3 dùng prompt → một kiến trúc cho mọi tác vụ.

4. **Output → JSON**
   - Chuỗi `<s_menu><s_nm>Lemon Tea</s_nm><s_price>25.000</s_price></s_menu>` được chuyển 1-1 thành `{"menu": {"nm": "Lemon Tea", "price": "25.000"}}`.
   - Nếu chuỗi sinh ra sai cấu trúc (thiếu thẻ đóng) → field đó bị coi là mất.

### 1.4 Huấn luyện – "How does Donut learn?"

| Giai đoạn | Mục tiêu | Dữ liệu | Hàm mất mát |
|-----------|----------|---------|-------------|
| **Pre-training** ("học đọc") | Đọc toàn bộ chữ trong ảnh theo thứ tự trên-trái → dưới-phải (*pseudo-OCR*) | IIT-CDIP (~11M ảnh tài liệu thật, nhãn chữ lấy từ OCR thương mại) + **SynthDoG** (0.5M × 4 ngôn ngữ: en, zh, ja, ko) | Cross-entropy dự đoán token kế tiếp |
| **Fine-tuning** ("học hiểu") | Sinh JSON đúng theo tác vụ | CORD, Ticket, RVL-CDIP, DocVQA… | Cross-entropy, teacher forcing |

- Inference: greedy decoding (hoặc beam search), không cần OCR.
- **SynthDoG** (Synthetic Document Generator): ghép nền (ảnh ImageNet) + giấy + văn bản (lấy từ Wikipedia) với layout ngẫu nhiên, font, nhiễu, biến đổi hình học → giúp mở rộng sang ngôn ngữ/domain mới mà không cần dữ liệu thật. Paper cho thấy chỉ pretrain bằng SynthDoG vẫn cho kết quả khá trên CORD.

> **Câu trả lời ngắn cho "How does Donut work?":** Swin Transformer mã hóa ảnh tài liệu độ phân giải cao thành chuỗi vector đặc trưng; một BART decoder nhỏ, được điều kiện hóa bằng prompt token của tác vụ, cross-attend vào các vector đó và sinh ra tuần tự một chuỗi token có cấu trúc, rồi chuỗi này được chuyển thành JSON. Khả năng "đọc" có được nhờ pre-training bằng tác vụ đọc chữ (pseudo-OCR) trên dữ liệu thật + tổng hợp; khả năng "hiểu" có được nhờ fine-tuning trên từng tác vụ. Cross-attention đóng vai trò như một cơ chế định vị chữ ngầm (implicit text localization) thay cho OCR.

### 1.5 Thước đo (metric)
- **Field-level F1**: so khớp chính xác từng cặp (key, value) – rất khắt khe (sai 1 ký tự = sai cả field).
- **TED-based Accuracy**: `max(0, 1 − TED(pred, gt) / TED(∅, gt))` với TED = Tree Edit Distance giữa hai cây JSON (thuật toán Zhang-Shasha). Phản ánh mức độ đúng của cả cấu trúc.
- **Accuracy** (RVL-CDIP), **ANLS** (DocVQA).

### 1.6 Kết quả tham chiếu (agent PHẢI đối chiếu lại với bảng trong paper trước khi đưa vào báo cáo)

| Tác vụ | Dataset | Metric | Kết quả công bố (README repo) |
|--------|---------|--------|------------------------------|
| Document parsing | CORD v2 | TED Acc | 91.3 (input 1280) |
| Document parsing | CORD v1 | TED Acc | 91.1 (1280) / 90.9 (2560) |
| Document parsing | CORD (paper) | Field F1 | ~84.1 (đối chiếu Table paper) |
| Document parsing | Train Ticket | TED Acc | 98.7 |
| Classification | RVL-CDIP | Accuracy | 95.3 |
| Document VQA | DocVQA Task1 | ANLS | 67.5 |

Ví dụ log `test.py` chính thức trên CORD-v2 (100 ảnh test): TED Acc ≈ 0.913, F1 ≈ 0.841.

---

## 2. Tài nguyên & môi trường

### 2.1 Hai lựa chọn code base

| Lựa chọn | Ưu | Nhược | Khuyến nghị |
|----------|----|-------|-------------|
| **A. Repo `clovaai/donut`** | Đúng 100% code paper, có `train.py`, `test.py`, `config/train_cord.yaml`, evaluator chính thức | Phụ thuộc phiên bản cũ (torch 1.11, transformers 4.11.3, timm 0.5.4, pytorch-lightning 1.6.4) → hay vỡ môi trường trên Colab mới | Dùng cho **evaluator** (`donut/util.py: JSONParseEvaluator`) và tham khảo config |
| **B. HuggingFace `transformers`** | Cài dễ, chạy ổn trên Colab/Kaggle, hỗ trợ fp16, PEFT/LoRA | Khác nhẹ về chi tiết so với code gốc | **Dùng cho train/infer chính** |

### 2.2 Cài đặt (phương án B)
```bash
pip install -U transformers datasets accelerate sentencepiece zss nltk peft
# zss: tree edit distance (Zhang-Shasha) cho TED accuracy
```
Nếu muốn dùng evaluator gốc mà không vỡ môi trường: **copy** class `JSONParseEvaluator` từ `donut/util.py` của repo vào `src/metrics.py` (giữ ghi chú nguồn + license MIT), thay vì `pip install donut-python`.

### 2.3 Checkpoint & dữ liệu (HuggingFace Hub)
- Backbone pretrain: `naver-clova-ix/donut-base`
- Đã fine-tune CORD: `naver-clova-ix/donut-base-finetuned-cord-v2`
- (Tùy chọn) `naver-clova-ix/donut-base-finetuned-rvlcdip`, `naver-clova-ix/donut-base-finetuned-docvqa`
- Dữ liệu: `naver-clova-ix/cord-v2` (800 train / 100 val / 100 test), `naver-clova-ix/synthdog-en`

### 2.4 Cấu trúc thư mục dự án
```
donut-project/
├── README.md
├── configs/                 # yaml cho từng thí nghiệm
├── src/
│   ├── data.py              # Dataset CORD → (pixel_values, labels)
│   ├── json_tokens.py       # json2token / token2json + add special tokens
│   ├── metrics.py           # F1 + TED accuracy (port từ repo gốc)
│   ├── train.py
│   ├── evaluate.py
│   └── visualize_attention.py
├── notebooks/               # bản Colab/Kaggle chạy từng phase
├── results/                 # json dự đoán, bảng số liệu, hình
└── report/                  # báo cáo cuối
```

---

## 3. PHASE 1 – Tái lặp kết quả (G1)

### Task 1.1 – Inference với checkpoint chính thức trên CORD-v2 (≈ 15 phút GPU)
**Mục đích:** kiểm tra pipeline đánh giá, có con số "tái lặp chắc chắn".

Các bước:
1. Load `DonutProcessor` và `VisionEncoderDecoderModel` từ `naver-clova-ix/donut-base-finetuned-cord-v2`.
2. Với mỗi ảnh trong split `test` (100 ảnh): prompt = `<s_cord-v2>`, `model.generate(..., max_length=768, num_beams=1, early_stopping=True, pad_token_id=..., eos_token_id=..., bad_words_ids=[[unk_token_id]])`.
3. Hậu xử lý: bỏ `eos`/`pad`, bỏ prompt token đầu, `processor.token2json(seq)`.
4. Ground truth: `json.loads(sample["ground_truth"])["gt_parse"]`.
5. Tính **F1** và **TED accuracy** bằng `metrics.py`; đo **thời gian/ảnh**.
6. Lưu `results/p1_official_cord.json` (từng dự đoán + điểm) và `results/p1_summary.csv`.

**DoD:** TED Acc nằm trong khoảng ±1 điểm so với 91.3; F1 trong ±1.5 so với ~84. Nếu lệch nhiều → kiểm tra lại: prompt token, cách chuẩn hóa ground truth, độ phân giải ảnh của processor (phải là 1280×960 cho checkpoint này).

### Task 1.2 – Fine-tune từ `donut-base` trên CORD với tài nguyên hạn chế (≈ 3–5 giờ T4)
**Mục đích:** tái lặp *quá trình* huấn luyện, không chỉ số liệu.

Cấu hình gốc (tham khảo `config/train_cord.yaml`): input 1280×960, max_length 768, lr 3e-5, ~30 epoch, batch 8 trên 1 A100.

Cấu hình rút gọn cho GPU 16 GB:
| Tham số | Giá trị đề xuất |
|---------|----------------|
| input_size | `[1280, 960]` (nếu OOM → `[960, 720]`, ghi rõ trong báo cáo) |
| batch size | 1–2, `gradient_accumulation_steps` = 4–8 (hiệu dụng ≈ 8) |
| precision | fp16 (T4 không hỗ trợ bf16) |
| gradient checkpointing | bật cho encoder nếu OOM |
| lr | 3e-5, warmup ~300 bước, cosine decay |
| epochs | 10–15 (thay vì 30), lưu checkpoint theo TED Acc trên val |
| max_length | 768 |

Các bước:
1. Thêm special tokens: duyệt toàn bộ `gt_parse` train → sinh `<s_key>`, `</s_key>`, cộng `<sep/>`, `<s_cord-v2>`; `tokenizer.add_special_tokens(...)`; `model.decoder.resize_token_embeddings(len(tokenizer))`.
2. `json2token(gt_parse)` → chuỗi đích; labels = token ids, thay `pad` bằng `-100`, và **mask prompt token** (không tính loss cho prompt).
3. Cấu hình `model.config.decoder_start_token_id = id(<s_cord-v2>)`, `pad_token_id`.
4. Huấn luyện, log loss + TED Acc trên val sau mỗi epoch.
5. Đánh giá checkpoint tốt nhất trên test bằng đúng script Task 1.1.

**DoD:** có đường cong loss/val-acc; TED Acc test được báo cáo kèm giải thích chênh lệch với paper (ít epoch hơn, batch nhỏ hơn, resolution có thể thấp hơn…).

### Task 1.3 (tùy chọn, nếu còn thời gian) – Inference RVL-CDIP / DocVQA
Chạy checkpoint chính thức trên **một tập con** (ví dụ 500–1000 mẫu) để minh họa tính đa nhiệm của prompt. Ghi rõ là subset, không so sánh trực tiếp với số liệu full test.

### Bảng kết quả Phase 1 (mẫu cần điền)
| Thiết lập | Resolution | Epoch | F1 | TED Acc | s/ảnh | GPU |
|-----------|-----------|-------|----|---------|-------|-----|
| Paper / README | 1280 | 30 | ~84.1 | 91.3 | 0.7 (A100) | A100 |
| Checkpoint chính thức – tái lặp | 1280 | – | | | | T4 |
| Fine-tune của nhóm | | | | | | T4 |

---

## 4. PHASE 2 – Phân tích "How does Donut work?" (G2)

### Task 2.1 – Sơ đồ kiến trúc
Vẽ lại sơ đồ (draw.io / Mermaid / matplotlib) theo mục 1.3, có ghi kích thước tensor thực tế. Lấy số liệu bằng cách in shape:
```python
enc_out = model.encoder(pixel_values)          # last_hidden_state: [B, n, d]
print(pixel_values.shape, enc_out.last_hidden_state.shape)
print(model.config.encoder.depths, model.config.encoder.window_size)
print(model.config.decoder.decoder_layers, len(processor.tokenizer))
sum(p.numel() for p in model.parameters())     # kỳ vọng ~143M (đối chiếu paper)
```

### Task 2.2 – Trực quan hóa cross-attention (minh chứng "định vị chữ ngầm")
1. `model.generate(..., output_attentions=True, return_dict_in_generate=True)`.
2. Với mỗi token sinh ra, lấy `cross_attentions` (trung bình theo head, lớp cuối hoặc trung bình các lớp) → vector độ dài `n` → reshape về lưới `(H/32, W/32)` → upsample → chồng lên ảnh.
3. Chọn 3–5 token tiêu biểu (tên món, giá, tổng tiền) để minh họa model "nhìn" đúng vùng chữ.
4. Lưu `results/attention/*.png`.

**DoD:** ít nhất 1 hình có heatmap tập trung đúng dòng chữ tương ứng với token; 1 ví dụ thất bại + phân tích.

### Task 2.3 – Phân tích lỗi
Từ `p1_official_cord.json`: liệt kê 10 mẫu có TED Acc thấp nhất, phân loại lỗi (đọc sai ký tự / sai cấu trúc JSON / thiếu field / ảnh nghiêng-mờ). Đây là đầu vào để chọn cải tiến ở Phase 3.

### Nội dung báo cáo phần G2 (checklist)
- [ ] Động cơ: hạn chế của pipeline OCR-based
- [ ] Kiến trúc encoder (Swin, vì sao) + decoder (BART, prompt, special tokens)
- [ ] Biểu diễn đầu ra JSON ↔ token
- [ ] Pre-training (pseudo-OCR, IIT-CDIP, SynthDoG) vs fine-tuning
- [ ] Metric F1 vs TED Acc
- [ ] Hình cross-attention + phân tích lỗi
- [ ] Ưu/nhược: không cần OCR, nhanh, đa ngôn ngữ ↔ cần resolution cao, chuỗi dài, khó với tài liệu nhiều chữ, lỗi cấu trúc JSON

---

## 5. PHASE 3 – Cải tiến nhẹ (G3)

> Chọn **2–3** hạng mục. Tất cả đều dùng CORD, cùng một script đánh giá, cùng seed. Ưu tiên theo thứ tự ★.

### I1 ★★★ – Ablation độ phân giải đầu vào × tốc độ (chi phí thấp, kết quả rõ ràng)
- Chỉ inference với checkpoint chính thức ở nhiều resolution: 640×480, 960×720, 1280×960, (1600×1200 nếu đủ bộ nhớ). Lưu ý: thay đổi resolution khi inference khác với lúc train → ghi rõ đây là kiểm tra độ bền (robustness) theo resolution.
- (Tốt hơn, nếu có thời gian) fine-tune ngắn ở 960×720 rồi so với 1280×960.
- Output: biểu đồ TED Acc & F1 vs. resolution, và latency vs. resolution → **đường cong trade-off**.

### I2 ★★★ – Fine-tune hiệu quả tham số bằng LoRA
- Áp `peft.LoraConfig` (r=8–16, alpha=16–32, dropout 0.05) vào các ma trận `q, v` của decoder và (tùy chọn) attention của Swin stage 3–4; vẫn train đầy đủ embedding vì có special token mới (`modules_to_save=["embed_tokens", "lm_head"]` hoặc tương đương).
- So sánh với full fine-tune (Task 1.2): **số tham số train, VRAM đỉnh, thời gian/epoch, F1, TED Acc**.
- Giả thuyết: giảm mạnh VRAM/tham số, độ chính xác giảm nhẹ.

### I3 ★★ – Tăng cường dữ liệu (augmentation) + đánh giá độ bền
- Tạo **tập test nhiễu** từ 100 ảnh test CORD: xoay ±5°, Gaussian blur, JPEG nén mạnh, thay đổi độ sáng/tương phản, biến dạng phối cảnh nhẹ.
- Đo checkpoint gốc trên tập nhiễu → thấy mức sụt giảm.
- Fine-tune thêm (vài epoch) với các augmentation trên (albumentations / torchvision), đo lại trên test sạch + test nhiễu.

### I4 ★★ – Hậu xử lý / giải mã có ràng buộc (không cần train)
- **Beam search** (num_beams = 3, 5) vs greedy: đo F1/TED/latency.
- **Sửa cấu trúc JSON**: tự động đóng thẻ thiếu, loại key không có trong schema CORD.
- **Chuẩn hóa giá trị số**: chuẩn hóa định dạng số tiền (dấu `.`/`,`), khoảng trắng thừa (`x 1` → `x1`) – lưu ý chỉ áp dụng nếu chuẩn hóa tương ứng phía ground truth là hợp lý; báo cáo cả hai số (trước/sau) để minh bạch.
- (Nâng cao) Constrained decoding: dùng `prefix_allowed_tokens_fn` để chỉ cho phép thẻ mở/đóng hợp lệ theo schema.

### I5 ★ – Nén mô hình cho suy luận nhanh
- Dynamic int8 quantization (CPU) hoặc fp16 (GPU): so sánh latency, kích thước, độ chính xác.

### I6 ★ (mở rộng, nếu muốn điểm sáng tạo) – Chuyển sang hóa đơn tiếng Việt
- Dataset gợi ý: MC-OCR 2021 (hóa đơn tiếng Việt) – agent cần kiểm tra lại giấy phép/khả năng truy cập trước khi dùng.
- Thử nghiệm: fine-tune `donut-base` trên vài trăm ảnh, đánh giá theo từng field. Thảo luận khó khăn: tokenizer mBART với dấu tiếng Việt, ít dữ liệu, có thể sinh thêm dữ liệu tiếng Việt bằng SynthDoG (thư mục `synthdog/` trong repo).

### Bảng kết quả Phase 3 (mẫu)
| ID | Thiết lập | Tham số train | VRAM đỉnh | Thời gian | F1 | TED Acc | s/ảnh | Ghi chú |
|----|-----------|--------------|-----------|-----------|----|---------|-------|---------|
| B0 | Baseline (Task 1.2) | | | | | | | |
| I1-a | Res 960×720 | | | | | | | |
| I2 | LoRA r=8 | | | | | | | |
| I3 | +Augmentation (test sạch / nhiễu) | | | | | | | |
| I4 | Beam=5 + hậu xử lý | | | | | | | |

---

## 6. Ngân sách thời gian (ước lượng, GPU T4)

| Phase | Task | GPU-giờ |
|-------|------|---------|
| 1 | 1.1 Inference chính thức | 0.3 |
| 1 | 1.2 Fine-tune rút gọn | 3–5 |
| 2 | 2.1–2.3 Phân tích, attention | 0.5 |
| 3 | I1 (inference-only) | 0.5 |
| 3 | I2 LoRA | 2–3 |
| 3 | I3 Augmentation | 2–3 |
| 3 | I4 Beam/hậu xử lý | 0.5 |
| **Tổng** | | **~9–13 giờ** |

Colab giới hạn phiên → **lưu checkpoint lên Google Drive / Kaggle Output mỗi epoch** và hỗ trợ resume.

---

## 7. Rủi ro & cách xử lý

| Rủi ro | Cách xử lý |
|--------|-----------|
| Vỡ phụ thuộc khi cài repo gốc | Dùng phương án B (HF transformers); chỉ port evaluator |
| OOM trên 16 GB | batch 1 + grad accumulation, fp16, gradient checkpointing, giảm resolution, giảm max_length |
| Số liệu lệch paper | Kiểm tra prompt token, `decoder_start_token_id`, special tokens, resolution, cách tính TED (dùng đúng code repo) |
| Model sinh chuỗi lặp / không dừng | đặt `eos_token_id`, `max_length`, `no_repeat_ngram_size` thận trọng (có thể hại kết quả), `bad_words_ids=[[unk]]` |
| Phiên Colab bị ngắt | checkpoint mỗi epoch, script resume |
| Kết quả không có ý nghĩa thống kê (test chỉ 100 ảnh) | Báo cáo thêm trên val; nếu được, chạy 2–3 seed và báo mean ± std |

---

## 8. Sản phẩm bàn giao cuối

1. `notebooks/` – notebook chạy được từ đầu đến cuối cho từng Phase.
2. `results/` – JSON dự đoán, CSV bảng số liệu, hình (kiến trúc, attention, trade-off).
3. `report/report.md` (hoặc PDF) gồm:
   1. Giới thiệu & động cơ
   2. Phương pháp Donut (kiến trúc, huấn luyện, metric) – trả lời "How does Donut work?"
   3. Tái lặp: thiết lập, kết quả, so sánh với paper, giải thích chênh lệch
   4. Cải tiến: giả thuyết → thí nghiệm → kết quả → phân tích
   5. Hạn chế & hướng phát triển
   6. Tài liệu tham khảo (paper, repo, HF docs)
4. Slide trình bày (10–15 slide) tóm tắt các mục trên.

---

## 9. Ghi chú cho AI agent

- **Không bịa số liệu.** Mọi con số của paper phải được đối chiếu trực tiếp với bảng trong arXiv 2111.15664; mọi con số tái lặp phải đến từ log thực tế.
- Cố định seed (`transformers.set_seed(42)`), ghi lại phiên bản thư viện (`pip freeze > results/env.txt`) và loại GPU.
- Dùng **một** hàm đánh giá duy nhất cho mọi thí nghiệm.
- Mỗi thí nghiệm ghi vào `results/runs.csv`: id, config, thời gian, VRAM, F1, TED Acc, latency.
- Khi một bước thất bại, ghi lại lỗi và cách khắc phục vào `results/issues.md` (có ích cho phần thảo luận trong báo cáo).
