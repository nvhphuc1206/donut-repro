"""Gom các kết quả CHÍNH vào một thư mục báo cáo (chỉ đọc results/ và outputs/, không sửa gì ở đó).

    python -m src.build_report_folder            # -> bao_cao_ket_qua/

1_tai_lap_donut/        tái lập Donut trên CORD-v2: cấu hình train/test, đường cong train, kết quả test
2_phan_tich_donut/      Donut hoạt động thế nào: kiến trúc + số tham số, cross-attention, phân tích lỗi
3_cai_tien_preprocess/  cải tiến – cắt vùng chữ bằng Canny ("crop") trước encoder; kết quả trên dữ liệu có viền
"""
import argparse
import csv
import os
import shutil

import numpy as np

from src import utils
from src.preprocess_compare import load, micro_f1

N_BOOT = 10000
BORDER_GAIN = 1.3  # ảnh "nhiều viền": bước cắt phóng to chữ >= 1.3 lần


def subset(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]) if k != "summary" else v
            for k, v in d.items()}


def compare(base, var, idx, seed=0):
    """Số liệu none vs crop trên tập con idx + CI 95% paired bootstrap cho ΔF1, ΔTED, Δtotal_acc."""
    b, v = subset(base, idx), subset(var, idx)
    n = len(idx)
    rng = np.random.default_rng(seed)
    d_ted = v["ted"] - b["ted"]
    d_tot = v["total_ok"] - b["total_ok"]
    has_tot = not np.all(np.isnan(d_tot))
    bf, bt, bc = [], [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        bf.append(micro_f1(v["tp"][s], v["fnfp"][s]) - micro_f1(b["tp"][s], b["fnfp"][s]))
        bt.append(d_ted[s].mean())
        if has_tot:
            bc.append(np.nanmean(d_tot[s]))
    ci = lambda x: (100 * np.percentile(x, 2.5), 100 * np.percentile(x, 97.5))
    row = {"n": n,
           "f1_none": micro_f1(b["tp"], b["fnfp"]), "f1_crop": micro_f1(v["tp"], v["fnfp"]),
           "ted_none": b["ted"].mean(), "ted_crop": v["ted"].mean(),
           "d_f1_ci": ci(bf), "d_ted_ci": ci(bt)}
    if has_tot:
        row.update(total_none=np.nanmean(b["total_ok"]), total_crop=np.nanmean(v["total_ok"]), d_total_ci=ci(bc))
    return row


def fmt_ci(ci):
    return f"[{ci[0]:+.1f}, {ci[1]:+.1f}]" + (" ✔" if ci[0] > 0 else (" ✘" if ci[1] < 0 else ""))


def copy(src, dst):
    if os.path.exists(src):
        utils.ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
    else:
        print(f"[!] thiếu {src}")


# ----------------------------------------------------------------------------
# Phần 1 – Tái lập
# ----------------------------------------------------------------------------
def part1(out):
    d = utils.ensure_dir(os.path.join(out, "1_tai_lap_donut"))
    run = "outputs/donut_cord_reproduce"
    copy(f"{run}/config.yaml", f"{d}/cau_hinh_train_da_dung.yaml")
    copy(f"{run}/train_summary.json", f"{d}/tom_tat_train.json")
    copy(f"{run}/history.csv", f"{d}/lich_su_train_theo_epoch.csv")
    copy(f"{run}/train.log", f"{d}/train.log")
    copy("configs/train_cord_reproduce.yaml", f"{d}/cau_hinh_train_co_chu_thich.yaml")
    copy("results/reproduce/reproduce_train_curves.png", f"{d}/duong_cong_train.png")
    copy("results/reproduce/official_imagenet_fp16.json", f"{d}/test_checkpoint_chinh_thuc.json")
    copy("results/reproduce/reproduce_test_fp16.json", f"{d}/test_model_tu_train.json")

    off = utils.load_json("results/reproduce/official_imagenet_fp16.json")["summary"]
    ours = utils.load_json("results/reproduce/reproduce_test_fp16.json")["summary"]
    ts = utils.load_json(f"{run}/train_summary.json")
    hist = list(csv.DictReader(open(f"{run}/history.csv", encoding="utf-8")))
    val_rows = [h for h in hist if h.get("val_ted_acc")]

    md = f"""# 1. Tái lập Donut trên CORD-v2

## Kết quả trên tập test CORD-v2 (100 ảnh)

| Mô hình | Nguồn | F1 | TED Acc | NED ↓ |
|---|---|---|---|---|
| Donut (bài báo / README repo `clovaai/donut`) | số công bố | ~0.841 | 0.913 | – |
| Checkpoint chính thức `naver-clova-ix/donut-base-finetuned-cord-v2`, chạy lại bằng pipeline của nhóm | **tái lập inference** | **{off['f1']:.4f}** | **{off['ted_acc']:.4f}** | {off['ned']:.4f} |
| Model nhóm tự fine-tune từ `donut-base` theo đúng công thức gốc | **tái lập huấn luyện** | **{ours['f1']:.4f}** | **{ours['ted_acc']:.4f}** | {ours['ned']:.4f} |

- Tái lập inference: lệch so với số công bố {100 * (off['ted_acc'] - 0.913):+.2f} điểm TED, {100 * (off['f1'] - 0.841):+.2f} điểm F1 (≈ dao động của 100 ảnh test; 1 ảnh ≈ 1 điểm TED).
- Tái lập huấn luyện: F1 lệch {100 * (ours['f1'] - off['f1']):+.2f} điểm, TED lệch {100 * (ours['ted_acc'] - off['ted_acc']):+.2f} điểm so với checkpoint chính thức – chỉ 1 lần chạy (1 seed), batch 8 mô phỏng bằng tích lũy gradient, khác phiên bản thư viện.
- Số công bố lấy theo README repo; cần đối chiếu lại bảng trong arXiv 2111.15664 trước khi trích dẫn.

## Thiết lập test (giống `test.py` gốc)

| Thông số | Giá trị |
|---|---|
| Tập | CORD-v2 test, {off['n']} ảnh |
| Kích thước ảnh vào (H×W) | {off['input_size']} |
| Chuẩn hóa ảnh | {off['image_norm']} (mean/std ImageNet như `SwinEncoder` gốc; `preprocessor_config.json` trên HF ghi 0.5 – làm TED giảm còn 0.9048) |
| Giải mã | greedy (num_beams = {off['num_beams']}), max_length = {off['max_length']} |
| Độ chính xác số | {off['precision']} (`model.half()` như `test.py`; fp32 cho kết quả trùng) |
| Thước đo | `JSONParseEvaluator` port nguyên từ `donut/util.py` (F1 field-level micro, TED Acc) |
| GPU | {off['gpu']} |

## Thiết lập huấn luyện – so với công thức gốc (`config/train_cord.yaml`, `lightning_module.py`)

| Thông số | Code gốc Donut | Nhóm | Giống? |
|---|---|---|---|
| Khởi tạo | `naver-clova-ix/donut-base` | `{ts['model']}` | ✅ |
| Kích thước ảnh | [1280, 960] | {ts['input_size']} | ✅ |
| Chuẩn hóa ảnh | ImageNet | {ts['image_norm']} | ✅ |
| Số epoch | 30 | {ts['epochs']} | ✅ |
| Batch | 8 | 1 × tích lũy gradient 8 (GPU 8 GB) | ≈ (gradient tương đương) |
| Loss | cross-entropy TB theo token trên batch | TB theo token trên cả 8 ảnh ({ts['loss_reduction']}) | ✅ |
| Optimizer / lr | Adam, 3e-5 | AdamW (weight_decay = 0 ≡ Adam), 3e-5 | ✅ |
| Lịch lr | cosine, warmup 300 bước | cosine, warmup 300 bước | ✅ |
| Gradient clip | 1.0 | 1.0 | ✅ |
| Độ chính xác | 16-bit (AMP) | fp16 AMP | ✅ |
| max_length | 768 | 768 | ✅ |
| Thứ tự key JSON | `sort_json_key: False` | False | ✅ |
| Validation / chọn checkpoint | mỗi 3 epoch, chọn theo normalized edit distance nhỏ nhất | mỗi 3 epoch, theo {ts['select_metric'].upper()} | ✅ |
| Seed | 2022 | 2022 | ✅ |
| Thư viện | torch 1.11, transformers 4.11, timm 0.5.4 | torch 2.11, transformers 4.57 (DonutSwin của HF, cùng trọng số) | ✗ (không thể giống) |

Cấu hình: `cau_hinh_train_co_chu_thich.yaml` (có chú thích giá trị gốc từng dòng) và `cau_hinh_train_da_dung.yaml`
(bản lưu tự động lúc train, run_id khi đó là `B0_official_recipe`).

## Quá trình huấn luyện

- Thời gian: **{ts['train_hours']} giờ** trên {ts['gpu']} ({ts['sec_per_epoch']} s/epoch), VRAM đỉnh {ts['peak_vram_gb']} GB.
- Checkpoint tốt nhất: epoch **{ts['best_epoch']}** (val F1 {ts['val_f1']}, val TED {ts['val_ted_acc']}, val NED {ts['val_ned']}).

| Epoch | {' | '.join(h['epoch'] for h in val_rows)} |
|---|{'---|' * len(val_rows)}
| Train loss | {' | '.join(f"{float(h['train_loss']):.3f}" for h in val_rows)} |
| Val TED | {' | '.join(f"{float(h['val_ted_acc']):.3f}" for h in val_rows)} |
| Val NED ↓ | {' | '.join(f"{float(h['val_ned']):.3f}" for h in val_rows)} |

![Đường cong train](duong_cong_train.png)

## File trong thư mục

| File | Nội dung |
|---|---|
| `cau_hinh_train_co_chu_thich.yaml`, `cau_hinh_train_da_dung.yaml` | cấu hình huấn luyện |
| `tom_tat_train.json`, `lich_su_train_theo_epoch.csv`, `train.log`, `duong_cong_train.png` | quá trình huấn luyện |
| `test_checkpoint_chinh_thuc.json`, `test_model_tu_train.json` | kết quả test: tóm tắt + dự đoán từng ảnh (raw sequence, pred, gt, TED, F1) |
"""
    open(os.path.join(d, "README.md"), "w", encoding="utf-8").write(md)
    return off, ours


# ----------------------------------------------------------------------------
# Phần 2 – Phân tích
# ----------------------------------------------------------------------------
def part_analysis(out):
    d = utils.ensure_dir(os.path.join(out, "2_phan_tich_donut"))
    A = "results/analysis"
    copy(f"{A}/architecture.png", f"{d}/kien_truc.png")
    copy(f"{A}/model_info.json", f"{d}/model_info.json")
    copy(f"{A}/error_analysis.md", f"{d}/phan_tich_loi.md")
    for name in ("sample1_summary.png", "sample94_summary.png", "sample39_summary.png"):
        copy(f"{A}/attention/{name}", f"{d}/attention_{name}")
    mi = utils.load_json(f"{A}/model_info.json")
    pr, ec, dc = mi["params"], mi["encoder_config"], mi["decoder_config"]
    shapes = " → ".join(f"[{s[1]}, {s[2]}, {s[3]}]" for s in mi["encoder_hidden_states"][:4])
    md = f"""# 2. Donut hoạt động thế nào

Phân tích trên checkpoint chính thức `{mi['model']}` (số liệu đo trực tiếp, `model_info.json`).

## Kiến trúc và kích thước tensor (ảnh 1280×960)

| Thành phần | Cấu hình | Shape |
|---|---|---|
| Ảnh vào | – | `{mi['pixel_values']}` |
| Encoder Swin | patch {ec['patch_size']}×{ec['patch_size']}, embed {ec['embed_dim']}, depths {ec['depths']}, heads {ec['num_heads']}, window {ec['window_size']} | {shapes} |
| Đầu ra encoder | lưới 40×30 vector | `{mi['encoder_last_hidden_state']}` |
| Decoder BART | {dc['decoder_layers']} lớp, d_model {dc['d_model']}, {dc['decoder_attention_heads']} head, FFN {dc['decoder_ffn_dim']}, tối đa {dc['max_position_embeddings']} token | – |
| Từ điển | {mi['tokenizer_len']} token (gồm {mi['num_added_tokens']} special token của CORD) | – |

| Tham số | Triệu |
|---|---|
| Encoder (Swin) | {pr['encoder_M']} |
| Decoder (BART) | {pr['decoder_M']} (trong đó embedding {pr['decoder_embeddings_M']}) |
| **Tổng** | **{pr['total_M']}** |

- Swin chỉ tính attention trong cửa sổ 10×10 (+ shifted window) → chi phí tuyến tính theo số patch; với 320×240 = 76 800
  patch ở stage 1, self-attention toàn cục kiểu ViT là không khả thi.
- Decoder chỉ 4 lớp nhưng chiếm 63% tham số, gần một nửa là bảng embedding của từ điển đa ngữ.

![Kiến trúc](kien_truc.png)

## Cross-attention – "định vị chữ ngầm"
Lấy cross-attention lớp cuối của decoder khi sinh từng giá trị, reshape về lưới 40×30, chồng lên ảnh.

- Mẫu thành công (`attention_sample1_summary.png`): mỗi tên món / giá tiền "nhìn" đúng dòng tương ứng trên hóa đơn – model
  tự học căn chỉnh token ↔ vùng ảnh mà không có bounding box nào khi train.
- Mẫu thất bại (`attention_sample94_summary.png`): model đọc cả chữ trang trí in dọc lề thành các món thừa.
- Mẫu thất bại (`attention_sample39_summary.png`): ảnh tối, giấy in nhiệt mờ → chỉ đọc được một phần.

![Thành công](attention_sample1_summary.png)
![Thất bại – chữ in dọc](attention_sample94_summary.png)

## Phân tích lỗi (checkpoint chính thức, test CORD-v2) – chi tiết `phan_tich_loi.md`

| Loại lỗi | Số ảnh mắc (/100) |
|---|---|
| Đọc sai 1–2 ký tự | 46 |
| Thiếu field | 29 |
| Thừa field | 27 |
| Sai giá trị (> 2 ký tự) | 18 |
| Sai cấu trúc JSON | 10 |

Lỗi phổ biến nhất là đọc sai 1–2 ký tự (dấu `.`/`,` trong số tiền, `0`/`O`…); các ca tệ nhất là ảnh tối/mờ, chữ nhỏ.
→ Gợi ý cho phần cải tiến: chữ càng nhỏ/mờ thì càng dễ sai – làm chữ to hơn trước khi vào encoder.
"""
    open(os.path.join(d, "README.md"), "w", encoding="utf-8").write(md)


# ----------------------------------------------------------------------------
# Phần 3 – Cải tiến tiền xử lý
# ----------------------------------------------------------------------------
E = "results/preprocess/eval"
GROUPS = [  # (tên hiển thị, file none, file crop, loại)
    ("CORD chụp xa (hóa đơn nhỏ trên nền bàn)", E + "/{m}__far__none.json", E + "/{m}__far__crop.json", "border"),
    ("SROIE (hóa đơn nhỏ giữa trang scan trắng)", E + "/{m}__sroie__none.json", E + "/{m}__sroie__crop.json", "border"),
    ("CORD-v2 test gốc (ảnh chụp sát)", E + "/{m}__test__none.json", E + "/{m}__test__crop.json", "clean"),
    ("WildReceipt (ảnh chụp, phần lớn đã sát)", E + "/{m}__wildreceipt__none.json", E + "/{m}__wildreceipt__crop.json", "clean"),
]


def part2(out):
    d = utils.ensure_dir(os.path.join(out, "3_cai_tien_preprocess"))
    rows, tables = [], {}
    for m in ("official", "reproduce"):
        lines = []
        for name, fb, fc, kind in GROUPS:
            fb, fc = fb.format(m=m), fc.format(m=m)
            if not (os.path.exists(fb) and os.path.exists(fc)):
                print(f"[!] thiếu {fb} hoặc {fc}")
                continue
            b, c = load(fb), load(fc)
            gains = np.array([p["gain"] for p in c["prep"]])
            subsets = [("tất cả ảnh", np.arange(len(gains)))]
            if kind == "border" and "sroie" in fb:
                subsets.append((f"ảnh nhiều viền (chữ được phóng to ≥ {BORDER_GAIN})", np.where(gains >= BORDER_GAIN)[0]))
            for sname, idx in subsets:
                r = compare(b, c, idx)
                tot = (f"{100 * r['total_none']:.1f}% → **{100 * r['total_crop']:.1f}%** {fmt_ci(r['d_total_ci'])}"
                       if "total_none" in r else "–")
                lines.append(f"| {name} | {sname} | {r['n']} | {r['f1_none']:.3f} → **{r['f1_crop']:.3f}** "
                             f"{fmt_ci(r['d_f1_ci'])} | {r['ted_none']:.3f} → **{r['ted_crop']:.3f}** {fmt_ci(r['d_ted_ci'])} "
                             f"| {tot} |")
                rows.append({"model": m, "dataset": name, "subset": sname, "n": r["n"],
                             "f1_none": round(r["f1_none"], 4), "f1_crop": round(r["f1_crop"], 4),
                             "d_f1_ci95": "%.2f..%.2f" % r["d_f1_ci"],
                             "ted_none": round(r["ted_none"], 4), "ted_crop": round(r["ted_crop"], 4),
                             "d_ted_ci95": "%.2f..%.2f" % r["d_ted_ci"],
                             "total_acc_none": round(r.get("total_none", np.nan), 4),
                             "total_acc_crop": round(r.get("total_crop", np.nan), 4),
                             "d_total_ci95": "%.2f..%.2f" % r["d_total_ci"] if "d_total_ci" in r else "",
                             "loai": "có viền" if kind == "border" else "không/ít viền"})
        tables[m] = lines

    with open(os.path.join(d, "bang_ket_qua.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # thống kê chất lượng cắt
    crop_stats = {}
    for tag, path in (("CORD test", "results/preprocess/crop/test_crop_summary.json"),
                      ("CORD chụp xa", "results/preprocess/crop/far_crop_summary.json")):
        if os.path.exists(path):
            crop_stats[tag] = utils.load_json(path)
    ex = utils.ensure_dir(os.path.join(d, "vi_du"))
    for name in ("sroie", "cord_far"):
        src = os.path.join("results/preprocess/examples", name)
        if os.path.isdir(src):
            for fn in sorted(os.listdir(src)):
                if fn.endswith("_so_sanh.png"):
                    copy(os.path.join(src, fn), os.path.join(ex, f"{name}_{fn}"))
    copy("results/preprocess/far_preview.png", os.path.join(d, "tap_cord_chup_xa_minh_hoa.png"))
    copy("results/preprocess/crop/sroie_crop_grid.png", os.path.join(d, "sroie_vung_cat.png"))
    copy("src/preprocess.py", os.path.join(d, "preprocess.py"))

    head = ("| Bộ dữ liệu | Tập ảnh | n | F1: không cắt → cắt [CI ΔF1] | TED: không cắt → cắt [CI ΔTED] "
            "| Đọc đúng tổng tiền: không cắt → cắt [CI Δ] |\n|---|---|---|---|---|---|")
    cs = crop_stats.get("CORD test", {})
    examples = sorted(os.listdir(ex)) if os.path.isdir(ex) else []
    md = f"""# 3. Cải tiến: cắt vùng chữ trước encoder (tiền xử lý "crop")

## Vấn đề
Donut thu nhỏ mọi ảnh về 1280×960 (giữ tỉ lệ, đệm phần thiếu). Khi hóa đơn chỉ chiếm một phần nhỏ của ảnh
(chụp xa, đặt trên bàn, scan cả trang giấy), chữ sau khi thu nhỏ quá bé để đọc. Ở phần tái lập, ablation độ
phân giải cho thấy chữ nhỏ đi làm mất 13–25 điểm TED.

## Phương pháp (`preprocess.py`, hàm `DocPreprocessor("crop")`)
Cắt ảnh sát vùng có chữ trước khi đưa vào Donut → chữ to hơn mà **không đổi model, không train lại, không tăng
chi phí encoder**. Thuật toán (OpenCV, ~35–50 ms/ảnh):
1. Đưa ảnh về cạnh dài 1000 px và 1600 px (2 tỉ lệ, để bắt cả chữ to lẫn chữ nhỏ).
2. Ảnh xám → CLAHE (chỉ để dò cạnh) → Gaussian 3×3 → **Canny** với ngưỡng tự động (1 ± 0.33) × median.
3. Giãn nở ngang 7×3 để nối nét chữ thành "khối từ"; giữ khối có hình dạng giống chữ: cao 0.4–5% cạnh dài,
   rộng < 60% ảnh, mật độ cạnh ≥ 12%, độ lệch chuẩn độ sáng ≥ 8 → loại vân gỗ, mép bàn, ngón tay, nhiễu.
4. Hộp bao các khối chữ (bỏ 1% điểm ngoại lai) + lề an toàn 8% ngang, 6% dọc → cắt.
5. An toàn: < 6 cụm chữ → giữ ảnh gốc; hộp > 92% ảnh → giữ ảnh gốc; giới hạn phóng to tối đa 2.5 lần.

Chất lượng cắt đo bằng toạ độ từng từ trong ground truth CORD (100 ảnh test): {cs.get('cropped', '?')} ảnh được cắt,
{cs.get('fallback', '?')} giữ nguyên, **{100 * cs.get('word_recall_mean', 0):.1f}% số từ nằm trong vùng cắt**
({cs.get('n_cut_images', '?')} ảnh mất vài từ), chữ được phóng to trung vị ×{cs.get('gain_median', '?')}.



## Kết quả – checkpoint chính thức (đã train trên CORD), không train lại
"không cắt" và "cắt" chạy trên **cùng model, cùng ảnh**; CI 95% bằng paired bootstrap ({N_BOOT} lần);
✔ = cải thiện có ý nghĩa thống kê (CI không chứa 0), ✘ = giảm có ý nghĩa.

{head}
{chr(10).join(tables.get('official', []))}

## Kết quả – model nhóm tự train (tái lập)

{head}
{chr(10).join(tables.get('reproduce', []))}

Ghi chú:
- **CORD chụp xa**: 100 ảnh test CORD được thu nhỏ còn 50–70% khung, xoay 3–8°, đặt lên nền tổng hợp (gỗ, nền tối,
  mặt bàn, vải), nén JPEG – mô phỏng chụp xa (`tap_cord_chup_xa_minh_hoa.png`).
- **SROIE**: 300 ảnh hóa đơn scan (ICDAR 2019), chỉ chấm trường tổng tiền (`total.total_price`); "ảnh nhiều viền"
  = ảnh mà bước cắt phóng to chữ ≥ {BORDER_GAIN} lần (xác định từ bước tiền xử lý, không dựa vào kết quả model).
- SROIE và WildReceipt là đánh giá **zero-shot** (model chưa thấy bộ dữ liệu này, nhãn chỉ ánh xạ một phần sang
  schema CORD) nên điểm tuyệt đối thấp; chỉ so sánh tương đối "không cắt → cắt".

## Kết luận
- **Ảnh có nhiều viền** (hóa đơn chiếm phần nhỏ của ảnh): cắt vùng chữ cải thiện rõ và có ý nghĩa thống kê với
  checkpoint chính thức – CORD chụp xa F1 tăng ~14 điểm; SROIE nhiều viền độ chính xác đọc tổng tiền tăng ~28 điểm.
- **Ảnh đã chụp sát** (CORD gốc, WildReceipt): không cải thiện, cũng không giảm đáng kể.
- Với model nhóm tự train: CORD chụp xa cải thiện tương tự (F1 tăng ~14 điểm, có ý nghĩa thống kê); trên SROIE
  không cải thiện (ảnh nhiều viền tăng nhưng không có ý nghĩa thống kê, toàn bộ tập giảm nhẹ) – model này yếu hơn
  checkpoint chính thức khi gặp dữ liệu khác CORD.
- Bước cắt nên được dùng khi tài liệu thực sự nhỏ trong ảnh; với ảnh đã chụp sát thì không cần.

## Ví dụ (`vi_du/`)
{chr(10).join(f'![{fn}](vi_du/{fn})' for fn in examples)}

## File trong thư mục
| File | Nội dung |
|---|---|
| `preprocess.py` | mã nguồn tiền xử lý (bản sao của `src/preprocess.py`) |
| `bang_ket_qua.csv` | toàn bộ số liệu bảng trên |
| `vi_du/*.png` | ảnh gốc vs ảnh đã cắt + điểm (5 SROIE, 5 CORD chụp xa) |
| `tap_cord_chup_xa_minh_hoa.png`, `sroie_vung_cat.png` | minh họa dữ liệu và vùng cắt |
"""
    open(os.path.join(d, "README.md"), "w", encoding="utf-8").write(md)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Gom kết quả chính vào một thư mục báo cáo")
    ap.add_argument("--out", default="bao_cao_ket_qua")
    args = ap.parse_args()
    out = utils.ensure_dir(args.out)
    off, ours = part1(out)
    part_analysis(out)
    rows = part2(out)
    pick = {(r["model"], r["dataset"][:5], r["subset"][:3]): r for r in rows}
    far = pick.get(("official", "CORD ", "tất"), {})
    sro = next((r for r in rows if r["model"] == "official" and r["dataset"].startswith("SROIE")
                and r["subset"].startswith("ảnh")), {})
    md = f"""# Tổng hợp kết quả chính – Donut trên CORD-v2

## 1. Tái lập Donut → [1_tai_lap_donut/](1_tai_lap_donut/README.md)

| Mô hình | F1 | TED Acc |
|---|---|---|
| Donut công bố (README repo) | ~0.841 | 0.913 |
| Checkpoint chính thức, chạy lại | {off['f1']:.4f} | {off['ted_acc']:.4f} |
| Nhóm tự fine-tune theo công thức gốc (30 epoch, 3.35 giờ) | {ours['f1']:.4f} | {ours['ted_acc']:.4f} |

## 2. Donut hoạt động thế nào → [2_phan_tich_donut/](2_phan_tich_donut/README.md)

Kiến trúc và số tham số đo trực tiếp từ model, cross-attention của decoder (định vị chữ ngầm), phân tích lỗi trên test.

## 3. Cải tiến: cắt vùng chữ trước encoder → [3_cai_tien_preprocess/](3_cai_tien_preprocess/README.md)

Checkpoint chính thức, không train lại, chỉ thêm bước cắt ảnh:

| Dữ liệu có viền | Không cắt | Có cắt |
|---|---|---|
| CORD chụp xa (100 ảnh) – F1 | {far.get('f1_none', float('nan')):.3f} | **{far.get('f1_crop', float('nan')):.3f}** |
| CORD chụp xa – TED | {far.get('ted_none', float('nan')):.3f} | **{far.get('ted_crop', float('nan')):.3f}** |
| SROIE nhiều viền ({sro.get('n', '?')} ảnh) – đọc đúng tổng tiền | {100 * sro.get('total_acc_none', float('nan')):.1f}% | **{100 * sro.get('total_acc_crop', float('nan')):.1f}%** |

Trên ảnh đã chụp sát (CORD test gốc): không thay đổi đáng kể (F1 0.837 → 0.845, TED 0.910 → 0.907, đều không có ý nghĩa thống kê).
"""
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(md)
    print(f"Đã tạo: {out}/")


if __name__ == "__main__":
    main()
