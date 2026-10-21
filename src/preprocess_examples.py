"""Xuất ảnh minh họa: các mẫu mà bước cắt viền (crop) cho kết quả tốt hơn rõ nhất so với ảnh gốc.

Mỗi mẫu: <idx>_1_goc.png, <idx>_2_da_cat.png, <idx>_so_sanh.png (đặt cạnh nhau + điểm), <idx>_ketqua.json (GT, dự đoán).

    python -m src.preprocess_examples      # mặc định: SROIE và CORD chụp xa, model chính thức, 5 ảnh mỗi bộ
Kết quả: results/preprocess/examples/<tên bộ>/
"""
import argparse
import os

import matplotlib.pyplot as plt

from src import utils
from src.data import load_split
from src.preprocess import DocPreprocessor

SETS = {"sroie": ("sroie", "data/ext_sroie"), "cord_far": ("far", "data/cord_far")}  # tên -> (tên tập trong eval, dataset)


def total_of(pred):
    t = pred.get("total") if isinstance(pred, dict) else None
    t = t[0] if isinstance(t, list) and t else t
    return t.get("total_price") if isinstance(t, dict) else None


def main():
    ap = argparse.ArgumentParser(description="Xuất ảnh minh họa (crop tốt hơn ảnh gốc)")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--eval_dir", default="results/preprocess/eval")
    ap.add_argument("--model", default="official")
    ap.add_argument("--out_dir", default="results/preprocess/examples")
    args = ap.parse_args()
    prep = DocPreprocessor("crop")

    for name in args.sets:
        key, data = SETS[name]
        f_base = os.path.join(args.eval_dir, f"{args.model}__{key}__none.json")
        f_crop = os.path.join(args.eval_dir, f"{args.model}__{key}__crop.json")
        base = {r["idx"]: r for r in utils.load_json(f_base)["records"]}
        crop = {r["idx"]: r for r in utils.load_json(f_crop)["records"]}
        # tiêu chí: TED tăng nhiều nhất; với SROIE ưu tiên ảnh từ đọc SAI tổng tiền thành đọc ĐÚNG
        def score(i):
            fixed = (not base[i].get("total_ok", False)) and crop[i].get("total_ok", False)
            return (fixed, crop[i]["ted_acc"] - base[i]["ted_acc"], crop[i]["f1"] - base[i]["f1"])
        best = sorted((i for i in crop if crop[i]["ted_acc"] > base[i]["ted_acc"]), key=score, reverse=True)[:args.n]

        ds = load_split(data, "test")
        out = utils.ensure_dir(os.path.join(args.out_dir, name))
        for i in best:
            img = ds[i]["image"].convert("RGB")
            cropped, info = prep(img)
            img.save(os.path.join(out, f"{i:03d}_1_goc.png"))
            cropped.save(os.path.join(out, f"{i:03d}_2_da_cat.png"))
            b, c = base[i], crop[i]
            gt_total = total_of(c["gt"])

            fig, axes = plt.subplots(1, 2, figsize=(11, 8), gridspec_kw={"width_ratios": [img.width / img.height,
                                                                                         cropped.width / cropped.height]})
            for ax, im, r, title in ((axes[0], img, b, "Ảnh gốc (không tiền xử lý)"),
                                     (axes[1], cropped, c, f"Đã cắt viền (chữ to x{info['gain']:.2f})")):
                ax.imshow(im)
                ax.axis("off")
                ok = r.get("total_ok", total_of(r["pred"]) == gt_total)
                sub = f"TED {r['ted_acc']:.2f} | F1 {r['f1']:.2f}"
                if gt_total is not None:
                    sub += f"\ntổng tiền đọc được: {total_of(r['pred'])} {'ĐÚNG' if ok else 'SAI'}"
                ax.set_title(f"{title}\n{sub}", fontsize=11, color="green" if r is c else "black")
            fig.suptitle(f"{name} #{i}" + (f" (GT tổng tiền: {gt_total})" if gt_total else ""), fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(os.path.join(out, f"{i:03d}_so_sanh.png"), dpi=110)
            plt.close(fig)

            utils.save_json({"idx": i, "gain": info["gain"], "gt": c["gt"],
                             "khong_cat": {"ted": b["ted_acc"], "f1": b["f1"], "pred": b["pred"]},
                             "co_cat": {"ted": c["ted_acc"], "f1": c["f1"], "pred": c["pred"]}},
                            os.path.join(out, f"{i:03d}_ketqua.json"))
            print(f"[{name}] #{i}: TED {b['ted_acc']:.2f} -> {c['ted_acc']:.2f} | F1 {b['f1']:.2f} -> {c['f1']:.2f}"
                  + (f" | total {total_of(b['pred'])} -> {total_of(c['pred'])} (GT {gt_total})" if gt_total else ""))
        print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
