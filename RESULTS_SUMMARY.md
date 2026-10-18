# Tổng hợp kết quả – Tái lặp & Cải tiến Donut

> Tổng hợp toàn bộ 61 bước của pipeline (`python -m src.run_all`): **mục tiêu → thiết lập → kết quả → giải thích**.
> Mọi con số lấy từ `results/runs.csv`, `outputs/*/history.csv` và các file trong `results/` (không có số nào điền tay).
> Đánh giá: tập **test CORD-v2 (100 ảnh)**, greedy decoding, `max_length=768`, seed 42, trừ khi ghi khác.

**Môi trường** (`results/env.txt`): Windows 11, Python 3.11.17, PyTorch 2.11.0+cu128, transformers 4.57.6, GPU **NVIDIA RTX 5060 8 GB**.

---

## Mục lục
0. [Tóm tắt nhanh](#0-tóm-tắt-nhanh)
1. [Phase 1 – Tái lặp kết quả](#1-phase-1--tái-lặp-kết-quả-g1)
2. [Phase 2 – "How does Donut work?"](#2-phase-2--how-does-donut-work-g2)
3. [Phase 3 – Cải tiến I1–I5](#3-phase-3--cải-tiến-g3)
4. [Sự cố gặp phải & cách xử lý](#4-sự-cố-gặp-phải--cách-xử-lý)
5. [Hạn chế của các kết quả](#5-hạn-chế-của-các-kết-quả)
6. [Danh mục file kết quả](#6-danh-mục-file-kết-quả)

---

## 0. Tóm tắt nhanh

| Mục tiêu | Kết quả chính | Đạt? |
|---|---|---|
| **G1** – Tái lặp checkpoint chính thức | TED Acc **0.9048**, F1 **0.8356** (tham chiếu 0.913 / ~0.841) | ✅ trong ngưỡng DoD (±1 / ±1.5 điểm) |
| **G1** – Tự fine-tune từ `donut-base` (B0) | TED **0.8462**, F1 **0.7579** sau 12 epoch (1.1 giờ) | ✅ có đường cong; thấp hơn chính thức ~6 điểm TED |
| **G2** – Giải thích kiến trúc | 201.1M tham số; ảnh 1280×960 → 1200 vector × 1024 chiều; cross-attention định vị đúng dòng chữ | ✅ |
| **G3 / I1** – Độ phân giải | Tốt nhất ở đúng resolution lúc train (1280×960); 640×480 mất 25 điểm TED | ✅ |
| **G3 / I2** – LoRA | Giảm 330× tham số train hiệu dụng, −40% VRAM, −29% thời gian/epoch; **mất ~10 điểm TED** | ✅ |
| **G3 / I3** – Augmentation | Nhiễu làm mất tới 11.6 điểm TED; augmentation giúp **+0.7 điểm TED trung bình trên tập nhiễu** (nhỏ) | ✅ (hiệu quả khiêm tốn) |
| **G3 / I4** – Giải mã / hậu xử lý | Beam search **không** giúp; sửa cấu trúc +0.4 F1; chuẩn hóa số +3.5 F1 (thay đổi metric) | ✅ |
| **G3 / I5** – Nén mô hình | fp16: giữ nguyên độ chính xác, ½ bộ nhớ, nhanh hơn 18%; int8 CPU: nhanh hơn 29%, mất ~2.6 điểm TED | ✅ |

---

## 1. PHASE 1 – Tái lặp kết quả (G1)

### 1.1 Inference checkpoint chính thức trên CORD-v2

**Mục tiêu:** kiểm tra pipeline đánh giá và có một con số "tái lặp chắc chắn" để so với bài báo.
**Thiết lập:** `naver-clova-ix/donut-base-finetuned-cord-v2`, input 1280×960, prompt `<s_cord-v2>`, greedy, fp32.

| Thiết lập | Split | F1 | TED Acc | s/ảnh | VRAM đỉnh |
|---|---|---|---|---|---|
| Tham chiếu (README repo, theo kế hoạch) | test | ~0.841 | 0.913 | 0.7 (A100) | – |
| **Tái lặp (`P1-official`)** | test | **0.8356** | **0.9048** | 0.353 | 1.47 GB |
| Tái lặp (`P1-official-val`) | validation | 0.8792 | 0.9417 | 0.341 | 1.47 GB |

**Giải thích**
- Chênh lệch so với tham chiếu: TED −0.82 điểm, F1 −0.54 điểm → **nằm trong ngưỡng DoD** (±1 và ±1.5). Pipeline đánh giá (prompt, processor 1280×960, evaluator port từ repo gốc) được coi là đúng.
- Phần lệch nhỏ còn lại có thể do khác phiên bản thư viện (transformers 4.57 vs 4.11 của repo gốc: khác chi tiết resize/padding ảnh, thuật toán sinh) và do test chỉ có 100 ảnh (1 ảnh ≈ 1 điểm TED trung bình).
- Validation cao hơn test (~3.7 điểm TED) → test CORD khó hơn val một chút; với 100 ảnh, nên báo cáo cả hai.
- Số tham chiếu lấy theo README repo trong kế hoạch; **cần đối chiếu lại trực tiếp bảng trong arXiv 2111.15664** trước khi đưa vào báo cáo.

### 1.2 Fine-tune của nhóm từ `donut-base` (B0)

**Mục tiêu:** tái lặp *quá trình* huấn luyện (không chỉ con số) với tài nguyên hạn chế.
**Thiết lập:** full fine-tune, 1280×960, 12 epoch (gốc: 30), batch 1 × grad-accum 8, lr 3e-5, warmup 300 bước, cosine, fp16, gradient checkpointing. Bước `smoke` đo VRAM đỉnh 5.2 GB → giữ 1280×960 trên GPU 8 GB.

| Thiết lập | Res | Epoch | F1 test | TED test | F1 val | TED val | Thời gian train | VRAM đỉnh |
|---|---|---|---|---|---|---|---|---|
| Checkpoint chính thức | 1280 | 30 | 0.8356 | 0.9048 | 0.8792 | 0.9417 | – (A100) | – |
| **B0 (nhóm)** | 1280 | 12 (best = 11) | **0.7579** | **0.8462** | 0.8241 | 0.8888 | **1.09 giờ** (~5.4 phút/epoch) | 5.21 GB |
| B0 + 4 epoch (đối chứng I3) | 1280 | 16 | 0.7745 | 0.8699 | 0.8095 | 0.8986 | +0.44 giờ | 5.21 GB |

Đường cong huấn luyện: `results/figures/B0_curves.png`

![B0 curves](results/figures/B0_curves.png)

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Train loss | 9.41 | 5.35 | 1.94 | 0.96 | 0.51 | 0.31 | 0.21 | 0.16 | 0.13 | 0.11 | 0.10 | 0.10 |
| Val TED | 0.00 | 0.00 | 0.35 | 0.64 | 0.78 | 0.82 | 0.85 | 0.87 | 0.88 | 0.89 | **0.889** | 0.886 |

**Giải thích**
- **Epoch 1–2 TED = 0**: đang trong giai đoạn warmup (300 bước ≈ 3 epoch); model chưa học được cấu trúc thẻ và chưa biết khi nào dừng → sinh chuỗi lặp dài. Từ epoch 3 (lr đạt đỉnh 3e-5) TED tăng vọt.
- **B0 thấp hơn chính thức ~6 điểm TED test** do: (1) 12 epoch so với 30; (2) LR giảm về 0 rất sớm do cosine trên lịch ngắn; (3) batch hiệu dụng 8 nhưng batch thật 1 (khác thống kê so với batch 8 trên A100); (4) fp16 thay vì fp32/bf16; (5) chỉ 1 seed.
- **Bằng chứng model còn "thiếu train":** train thêm 4 epoch (I3-NOAUG) tăng TED test từ 0.846 → **0.870** (+2.4 điểm) dù không đổi gì khác. Train đủ 30 epoch nhiều khả năng thu hẹp khoảng cách thêm.
- Khoảng cách val – test (0.889 vs 0.846) lớn hơn ở checkpoint chính thức → model của nhóm khái quát hóa kém hơn một chút.
- Huấn luyện nhanh hơn nhiều so với ước tính cho T4 (1.1 giờ thay vì 3–5 giờ).

> ⚠️ Lần train B0 đầu tiên **không hợp lệ** do lỗi tính loss (mục 4) – kết quả ở trên là sau khi sửa. Kết quả cũ được lưu riêng ở `outputs/_invalid_lossbug/`.

### 1.3 RVL-CDIP / DocVQA (tùy chọn)
**Chưa chạy** (pipeline chạy không có `--with_multitask`). Có thể bổ sung bằng `python -m src.run_all --with_multitask`.

---

## 2. PHASE 2 – "How does Donut work?" (G2)

### 2.1 Kiến trúc & kích thước tensor thực tế

**Mục tiêu:** minh họa kiến trúc bằng số liệu đo trực tiếp từ model (`results/model_info.json`, `results/figures/architecture.png`).

![architecture](results/figures/architecture.png)

| Thành phần | Cấu hình đo được | Shape tensor (ảnh 1280×960) |
|---|---|---|
| Ảnh vào | – | `[1, 3, 1280, 960]` |
| Patch embedding 4×4 | embed_dim 128 | `[1, 128, 320, 240]` |
| Swin stage 1 → 2 → 3 → 4 | depths **{2, 2, 14, 2}**, heads {4, 8, 16, 32}, window **10** | `[1,256,160,120]` → `[1,512,80,60]` → `[1,1024,40,30]` |
| Đầu ra encoder {z_i} | – | **`[1, 1200, 1024]`** (lưới 40×30 = 1200 vector) |
| Decoder BART | **4 layer**, d_model 1024, 16 head, FFN 4096, max 768 vị trí | – |
| Từ điển | 57 580 token (gồm **63** special token thêm vào) | – |

| Tham số | Số lượng |
|---|---|
| Encoder (Swin) | 74.18 M |
| Decoder (BART) | 126.94 M (trong đó **embedding 58.96 M**) |
| **Tổng** | **201.12 M** |

**Giải thích**
- **Vì sao Swin:** mỗi ảnh tạo 320×240 = 76 800 patch ở stage 1. Self-attention toàn cục kiểu ViT cần ma trận 76 800² ≈ 5.9 tỷ phần tử – không khả thi. Swin chỉ tính attention trong cửa sổ 10×10 (+ shifted window để liên kết các cửa sổ) → chi phí tuyến tính theo số patch, đồng thời patch-merging giảm dần độ phân giải xuống lưới 40×30.
- Stage 3 chiếm 14/20 block → phần lớn năng lực encoder nằm ở độ phân giải trung gian (80×60), phù hợp với kích thước chữ trên hóa đơn.
- Decoder chỉ 4 layer nhưng chiếm 63% tham số, **gần một nửa trong đó là bảng embedding** của từ điển đa ngữ mBART (57k token × 1024).
- Kế hoạch ghi kỳ vọng ~143 M tham số; số đo là 201.1 M. Đáng chú ý: **201.12 − 58.96 (embedding) = 142.2 M ≈ 143 M** → có thể con số trong paper không tính bảng embedding token. Cần đối chiếu lại cách đếm trong paper trước khi kết luận.

### 2.2 Cross-attention – "định vị chữ ngầm"

**Mục tiêu:** chứng minh decoder tự "nhìn" đúng vùng chữ khi sinh từng giá trị, thay cho bước OCR.
**Cách làm:** lấy cross-attention lớp cuối, trung bình các head và các token thuộc cùng một giá trị field, reshape về lưới 40×30, chồng lên ảnh (`results/attention/`).

![attention sample 1](results/attention/sample1_summary.png)

| Mẫu | TED | Quan sát |
|---|---|---|
| **1** (thành công) | 1.000 | Mỗi giá trị `nm` (`J.STB PROMO`, `Y.B.BAT`, `Y.BASO PROM`) có heatmap nằm **đúng dòng tên món**; mỗi `price` (`17500`, `46000`, `27500`) tập trung **đúng ô số tiền cùng dòng**. Model lần lượt "dò" từ trên xuống, trái sang phải |
| 0, 2 | 0.738, 0.769 | Định vị đúng vùng; lỗi chủ yếu là đọc sai ký tự (`DOSMINE` thay vì `JASMINE`) |
| **94** (thất bại) | 0.333 | Model đọc **dòng chữ trang trí in dọc lề** ("VIETNAMESE", "PHO NOODLE") thành các món `N`, `O`, `E T T E`… → sinh nhiều món thừa trước khi tới món thật |
| **39** (thất bại) | 0.086 | Ảnh tối, giấy in nhiệt mờ, độ tương phản thấp, hóa đơn dài → model chỉ đọc được một phần tên món đầu tiên rồi sinh chuỗi sai cấu trúc và dừng |

**Giải thích:** cross-attention đóng vai trò như cơ chế **định vị chữ ngầm** – không có bounding box nào được cung cấp khi train, nhưng model tự học căn chỉnh token ↔ vùng ảnh. Các ca thất bại cho thấy giới hạn: model không có khái niệm "hướng chữ" (đọc cả chữ in dọc) và nhạy với chất lượng ảnh.

### 2.3 Phân tích lỗi (checkpoint chính thức, test)

**Mục tiêu:** phân loại lỗi để chọn hướng cải tiến (`results/error_analysis.md`).

| Loại lỗi | Số mẫu mắc (/100) | Tổng số lần |
|---|---|---|
| Đọc sai ký tự (≤ 2 ký tự) | **46** | 98 |
| Thiếu field | 29 | 86 |
| Thừa field | 27 | 76 |
| Sai giá trị (> 2 ký tự) | 18 | 34 |
| Sai cấu trúc JSON (thẻ không khớp) | 10 | 10 |

10 mẫu tệ nhất: 39, 94, 12, 98, 57, 26, 17, 51, 47, 60 (ảnh + gt/pred trong `results/errors/`).

**Giải thích**
- **Lỗi phổ biến nhất là đọc sai 1–2 ký tự** (dấu `.`/`,` trong số tiền, `0`/`O`…). Với F1 cấp field, sai 1 ký tự = sai cả field → đây là lý do F1 (0.836) thấp hơn nhiều so với TED (0.905), vốn cho điểm từng phần.
- **Thiếu/thừa field** thường đi cùng nhau: model gán giá trị vào key khác (vd. mẫu 12: sinh `price` thay vì `itemsubtotal`, gộp `num` vào `nm`) – lỗi "hiểu" schema chứ không phải lỗi "đọc".
- **Sai cấu trúc** chỉ 10% mẫu nhưng gây mất điểm nặng (mẫu 39: TED 0.09) vì cả nhánh JSON bị bỏ.
- → Gợi ý cải tiến: chuẩn hóa định dạng số (I4), ràng buộc cấu trúc khi giải mã (I4), tăng độ bền với ảnh mờ/tối (I3).

---

## 3. PHASE 3 – Cải tiến (G3)

> ⚠️ **Về latency:** cùng một cấu hình (1280×960, greedy, fp32) đo được 0.35 – 0.96 s/ảnh ở các thời điểm khác nhau (do tải nền của máy). Vì vậy **chỉ so sánh latency giữa các thí nghiệm chạy liền nhau trong cùng một nhóm**.

### I1 ★★★ – Độ phân giải đầu vào × tốc độ

**Mục tiêu:** đo đánh đổi độ chính xác ↔ tốc độ ↔ bộ nhớ khi thay độ phân giải lúc inference (model train ở 1280×960 → đây là kiểm tra độ bền).

| Resolution | F1 | TED Acc | s/ảnh | VRAM đỉnh |
|---|---|---|---|---|
| 640×480 | 0.5317 | 0.6566 | 0.849 | 0.96 GB |
| 960×720 | 0.7071 | 0.7773 | 0.602 | 1.19 GB |
| **1280×960 (lúc train)** | **0.8356** | **0.9048** | 0.592 | 1.47 GB |
| 1600×1200 | 0.7966 | 0.8499 | 0.680 | 1.86 GB |

![I1 tradeoff](results/figures/i1_tradeoff.png)

**Giải thích**
- **Đỉnh nằm đúng ở resolution lúc train**; giảm *hoặc* tăng đều làm giảm độ chính xác. Giảm xuống 640×480 mất 25 điểm TED (chữ quá nhỏ để đọc); tăng lên 1600×1200 cũng mất 5.5 điểm vì kích thước chữ/bố cục khác phân phối đã học (model chưa từng thấy chữ ở tỉ lệ đó).
- **Latency không giảm khi giảm resolution** (640×480 còn chậm nhất): thời gian bị chi phối bởi **decoder tự hồi quy** (số token sinh ra), không phải encoder. Ở resolution thấp model đọc sai → sinh chuỗi dài/lặp hơn → chậm hơn.
- VRAM tăng gần tuyến tính theo số pixel (0.96 → 1.86 GB).
- **Kết luận:** muốn dùng resolution thấp để tiết kiệm bộ nhớ thì phải **fine-tune lại** ở resolution đó (cấu hình `configs/train_cord_full_960.yaml` – chưa chạy).

### I2 ★★★ – LoRA

**Mục tiêu:** giảm chi phí fine-tune, so với full fine-tune (B0) trong cùng điều kiện (1280×960, 12 epoch).
**Thiết lập:** LoRA r=8, α=16 trên `q_proj, v_proj` của decoder (self + cross attention) và `query, value` của Swin stage 3–4; chỉ học các hàng embedding của 63 special token mới; lr 1e-4.

| | Full FT (B0) | LoRA r=8 (I2) | Thay đổi |
|---|---|---|---|
| Tham số train hiệu dụng | 201.91 M | **0.61 M** | ÷ 330 (0.3%) |
| VRAM đỉnh | 5.21 GB | **3.10 GB** | −40% |
| Thời gian / epoch | 327 s | **232 s** | −29% |
| Tổng thời gian train | 1.09 giờ | 0.77 giờ | −29% |
| Val TED / F1 (best) | 0.8888 / 0.8241 | 0.7655 / 0.6884 | −12.3 / −13.6 điểm |
| **Test TED / F1** | **0.8462 / 0.7579** | **0.7445 / 0.6451** | **−10.2 / −11.3 điểm** |

(`trainable_params_M` = 59.52 M cho LoRA vì PyTorch đếm cả ma trận embedding đặt `requires_grad`; thực tế chỉ 63 hàng được cập nhật nhờ mask gradient → 0.61 M hiệu dụng.)

![I2 curves](results/figures/I2_curves.png)

**Giải thích**
- **Đúng giả thuyết** về chi phí: giảm mạnh tham số, VRAM và thời gian.
- **Mất độ chính xác lớn hơn kỳ vọng** ("giảm nhẹ"): −10 điểm TED. Lý do có thể: `donut-base` chỉ được pretrain cho tác vụ *đọc chữ* (pseudo-OCR), còn CORD cần học *schema hoàn toàn mới* (63 thẻ, cấu trúc lồng nhau) – một thay đổi hành vi lớn mà adapter hạng 8 trên q/v khó biểu diễn hết; encoder stage 1–2 và FFN đóng băng.
- Train loss dừng ở 0.34 (B0: 0.10) → LoRA **underfit**, không phải overfit. Hướng thử: r=16/32, thêm LoRA vào FFN (`fc1/fc2`) và `k_proj/out_proj`, nhiều epoch hơn.
- Ý nghĩa thực tế: LoRA cho phép fine-tune trên GPU ~4 GB, đánh đổi độ chính xác.

### I3 ★★ – Augmentation + độ bền với nhiễu

**Mục tiêu:** (a) đo độ sụt giảm khi ảnh test bị nhiễu; (b) kiểm tra augmentation khi train có giúp bền hơn không.
**Thiết lập:** 6 tập test nhiễu cố định từ 100 ảnh test (`results/figures/noisy_preview.png`): xoay 3–5°, Gaussian blur r=1.5, JPEG q=10, độ sáng ×0.6/1.4 + tương phản ×0.6, phối cảnh 8%, mixed (2 phép ngẫu nhiên). Từ B0 train thêm 4 epoch **có** augmentation (AUG) và **không** augmentation (NOAUG – đối chứng để tách hiệu ứng "train thêm").

**TED Acc**

| Tập test | Chính thức | B0 | NOAUG (B0+4) | **AUG (B0+4)** | AUG − NOAUG |
|---|---|---|---|---|---|
| Sạch | 0.9048 | 0.8462 | 0.8699 | **0.8701** | +0.0 |
| rotate | 0.8963 | 0.8486 | 0.8538 | **0.8594** | +0.6 |
| blur | 0.8343 | 0.8018 | 0.8150 | **0.8262** | +1.1 |
| jpeg | 0.8438 | 0.7853 | 0.7902 | **0.7927** | +0.3 |
| color | 0.8295 | 0.8241 | 0.8362 | **0.8431** | +0.7 |
| perspective | 0.8989 | 0.8553 | 0.8568 | **0.8604** | +0.4 |
| mixed | 0.7888 | 0.7603 | 0.7679 | **0.7802** | +1.2 |
| **TB 6 tập nhiễu** | 0.8486 | 0.8126 | 0.8200 | **0.8270** | **+0.7** |

**F1**

| Tập test | Chính thức | B0 | NOAUG | AUG |
|---|---|---|---|---|
| Sạch | 0.8356 | 0.7579 | 0.7745 | 0.7839 |
| rotate | 0.8144 | 0.7607 | 0.7715 | 0.7848 |
| blur | 0.7459 | 0.6916 | 0.7083 | 0.7227 |
| jpeg | 0.7286 | 0.6784 | 0.6780 | 0.6766 |
| color | 0.7686 | 0.7349 | 0.7377 | 0.7529 |
| perspective | 0.8244 | 0.7531 | 0.7627 | 0.7657 |
| mixed | 0.7022 | 0.6644 | 0.6594 | 0.6789 |

**Giải thích**
- **(a) Độ bền:** xoay nhẹ và phối cảnh gần như không ảnh hưởng (−0.9 / −0.6 điểm TED với model chính thức) nhờ attention linh hoạt về vị trí. **Blur, JPEG nặng, đổi sáng/tương phản** làm mất 6–7.5 điểm; **mixed mất 11.6 điểm** – nhiễu làm hỏng nét chữ, đúng với loại lỗi "đọc sai ký tự" ở 2.3.
- **(b) Augmentation:** so với đối chứng NOAUG, AUG **tốt hơn trên cả 6 tập nhiễu về TED** (+0.3 → +1.2, trung bình +0.7), cải thiện rõ nhất ở blur và mixed – đúng những loại có trong augmentation; **không làm giảm** trên ảnh sạch (0.8701 vs 0.8699).
- **Phần lớn cải thiện so với B0 đến từ việc train thêm**, không phải augmentation: B0 → NOAUG +2.4 điểm TED trên ảnh sạch, trong khi AUG − NOAUG chỉ +0.7 trên ảnh nhiễu. Đây là lý do cần run đối chứng.
- **Thận trọng:** khác biệt +0.7 điểm trên 100 ảnh, 1 seed là nhỏ (≈ cỡ nhiễu thống kê); JPEG gần như không đổi và F1 JPEG còn giảm nhẹ. Kết luận an toàn: *augmentation cho cải thiện nhỏ nhưng nhất quán về TED trên ảnh nhiễu, không gây hại trên ảnh sạch*; cần thêm seed để khẳng định.

### I4 ★★ – Giải mã & hậu xử lý (không train)

**Mục tiêu:** cải thiện kết quả của checkpoint chính thức chỉ bằng cách đổi chiến lược giải mã / hậu xử lý.

**Giải mã**

| Thiết lập | F1 | TED Acc | s/ảnh (cùng đợt chạy) |
|---|---|---|---|
| Greedy (beam 1) | 0.8356 | **0.9048** | 0.63 |
| Beam 3 | 0.8326 | 0.9007 | 1.19 (×1.9) |
| Beam 5 | 0.8294 | 0.8993 | 1.69 (×2.7) |
| Greedy + ràng buộc cấu trúc | **0.8383** | 0.9044 | 1.38 (×2.2) |

**Hậu xử lý** (trên dự đoán greedy / beam 5)

| Biến thể | F1 (greedy) | TED (greedy) | F1 (beam 5) | TED (beam 5) |
|---|---|---|---|---|
| raw | 0.8356 | 0.9048 | 0.8294 | 0.8993 |
| + sửa cấu trúc thẻ (`fix`) | **0.8396** | 0.9045 | 0.8316 | 0.8984 |
| + lọc key ngoài schema | 0.8396 | 0.9045 | 0.8329 | 0.8967 |
| + chuẩn hóa số/khoảng trắng **ở cả pred và GT** | *0.8703* | *0.9066* | *0.8560* | *0.8985* |

**Giải thích**
- **Beam search làm giảm nhẹ độ chính xác và chậm gấp 2–3 lần.** Model đã rất tự tin (phân phối xác suất nhọn), beam lớn ưu tiên chuỗi ngắn/an toàn hơn → bỏ sót field. Greedy là lựa chọn tốt nhất – khớp với cách paper dùng.
- **Ràng buộc cấu trúc** (chỉ cho phép thẻ đóng khớp thẻ mở) tăng F1 +0.27 điểm nhờ cứu các field bị đóng sai thẻ, nhưng chậm gấp 2.2 lần do kiểm tra bằng Python ở mỗi bước sinh.
- **Tự sửa cấu trúc sau khi sinh** cho cùng lợi ích (+0.4 F1) mà **không tốn thời gian** → nên dùng thay cho ràng buộc khi giải mã.
- **Lọc schema không có tác dụng** với greedy: model chính thức không bao giờ sinh key ngoài schema (từ điển chỉ có thẻ của CORD).
- **Chuẩn hóa số (+3.5 F1)** cho thấy một phần đáng kể lỗi F1 chỉ là khác định dạng (`25.000` vs `25,000`, `x 1` vs `x1`). Tuy nhiên đây là **thay đổi metric** (áp dụng cả cho GT), *không* phải cải tiến model → phải báo cáo tách riêng, không so trực tiếp với paper. TED gần như không đổi vì TED vốn đã phạt nhẹ sai lệch vài ký tự.

### I5 ★ – Nén mô hình

**Mục tiêu:** đánh đổi tốc độ / kích thước / độ chính xác khi giảm độ chính xác số.

| Thiết lập | Thiết bị | n ảnh | F1 | TED Acc | s/ảnh | Kích thước | VRAM đỉnh |
|---|---|---|---|---|---|---|---|
| fp32 | GPU | 100 | 0.8356 | 0.9048 | 0.958 | 768.9 MB | 1.47 GB |
| **fp16** | GPU | 100 | **0.8356** | **0.9048** | **0.783** (−18%) | **385.3 MB** | **0.74 GB** |
| fp32 | CPU | 30 | 0.7925 | 0.8963 | 6.37 | 768.9 MB | – |
| dynamic int8 | CPU | 30 | 0.7883 | 0.8702 | **4.50** (−29%) | 421.8 MB | – |

**Giải thích**
- **fp16 trên GPU là "miễn phí":** kết quả trùng khớp hoàn toàn với fp32 (cùng chuỗi sinh), giảm một nửa kích thước và VRAM, nhanh hơn 18%. Nên dùng mặc định khi triển khai trên GPU.
- **int8 trên CPU** nhanh hơn 29%, nhỏ hơn 45%, nhưng mất 2.6 điểm TED (−0.4 F1): lượng tử hóa các lớp Linear làm sai lệch nhỏ tích lũy qua 768 bước sinh tự hồi quy. Phần lớn kích thước còn lại là embedding (không bị lượng tử hóa).
- CPU chậm hơn GPU khoảng 6.6 lần (6.37 vs 0.96 s/ảnh, fp32) → Donut vẫn cần GPU cho ứng dụng thời gian thực.
- Số CPU chỉ trên 30 ảnh đầu → không so trực tiếp với số trên 100 ảnh.

---

## 4. Sự cố gặp phải & cách xử lý

(Chi tiết trong `results/issues.md` – hữu ích cho phần thảo luận của báo cáo.)

| Sự cố | Nguyên nhân | Cách xử lý | Ảnh hưởng |
|---|---|---|---|
| **B0/LoRA train 12 epoch, loss giảm còn 0.36 nhưng val F1 = 0, TED ≈ 0.05** | (1) transformers 4.57: `VisionEncoderDecoderModel(labels=...)` dùng `ForCausalLMLoss` → **dịch nhãn thêm 1 vị trí**, cộng với shift-right decoder input → model học đoán token cách 2 vị trí. (2) `json2token` sắp key ngược bảng chữ cái, trong khi checkpoint chính thức giữ thứ tự gốc | Tự tính cross-entropy (`seq2seq_loss`); `sort_json_key=False`. Kiểm chứng trên checkpoint chính thức: loss 26.4 (cũ) → **0.10–0.87** (mới) | Train lại toàn bộ; kết quả cũ chuyển vào `outputs/_invalid_lossbug/` |
| Train dừng ở validation epoch 1 | nltk 3.10 giới hạn `edit_distance` ≤ 2000 ký tự; model chưa học dừng sinh chuỗi rất dài | Bỏ giới hạn (code gốc Donut không có) | Không đổi cách tính metric |
| `PermissionError [WinError 32]` khi lưu `last.pt` | File 2.4 GB bị Windows Defender/indexer khóa tạm | Thử lại tối đa 30 × 10 s | – |
| Tải dataset treo ở 0% | `hf_xet` treo trên Windows; `load_dataset` tải cả 2.3 GB mọi split | `HF_HUB_DISABLE_XET=1`; chỉ tải file của split cần dùng | – |
| Log pipeline trống | Windows xuống dòng `\r\n` bị xử lý như thanh tiến trình | Chuẩn hóa `\r\n` | – |

**Bài học:** loss giảm đều **không** chứng minh pipeline train đúng – phải luôn theo dõi metric sinh chuỗi (val TED) và kiểm tra loss trên một checkpoint đã biết là tốt.

---

## 5. Hạn chế của các kết quả

1. **Test chỉ 100 ảnh, 1 seed:** chênh lệch < 1 điểm (vd. AUG vs NOAUG, ràng buộc giải mã) chưa đủ ý nghĩa thống kê. Nên chạy thêm 2–3 seed (`--set seed=1 run_id=B0_s1`) và báo mean ± std.
2. **Latency dao động** theo tải máy (0.35–0.96 s/ảnh cho cùng cấu hình) → chỉ so trong cùng nhóm; GPU RTX 5060 khác GPU của paper (A100).
3. **B0 train 12 epoch thay vì 30** → chưa phản ánh đầy đủ khả năng tái lặp quá trình train.
4. **I1 chỉ là inference ở resolution khác lúc train**; chưa fine-tune ở 960×720 (I1-b).
5. **Chưa chạy** Task 1.3 (RVL-CDIP, DocVQA) và I6 (hóa đơn tiếng Việt).
6. Số liệu tham chiếu của paper (TED 91.3, F1 ~84.1, 143M tham số) lấy theo kế hoạch/README – **cần đối chiếu trực tiếp với bảng trong arXiv 2111.15664**.

---

## 6. Danh mục file kết quả

| Nội dung | File |
|---|---|
| Bảng tất cả thí nghiệm | `results/runs.csv`, `results/runs_table.md` |
| Dự đoán từng ảnh (raw sequence, pred, gt, TED/F1) | `results/p1_official_cord*.json`, `results/b0_test.json`, `results/i2_lora_test.json`, `results/i1/`, `results/i3/`, `results/i4/`, `results/i5/` |
| Kiến trúc | `results/model_info.json`, `results/architecture.mmd`, `results/figures/architecture.png` |
| Cross-attention | `results/attention/`, `results/attention_meanlayers/` |
| Phân tích lỗi | `results/error_analysis.md`, `results/error_analysis.csv`, `results/errors/` |
| Đường cong train | `results/figures/B0_curves.png`, `results/figures/I2_curves.png`, `outputs/*/history.csv` |
| Trade-off resolution | `results/figures/i1_tradeoff.png` |
| Ảnh minh họa nhiễu | `results/figures/noisy_preview.png` |
| Hậu xử lý | `results/i4_postprocess.csv` |
| Checkpoint | `outputs/B0_full_1280/best`, `outputs/I2_lora_r8/best`, `outputs/I3_aug_from_B0/best`, `outputs/I3_noaug_from_B0/best` |
| Log | `results/logs/`, `outputs/*/train.log`, `results/pipeline_status.md` |
| Sự cố | `results/issues.md` |
| Môi trường | `results/env.txt` |
