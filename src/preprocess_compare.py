"""So sánh "có cắt vùng chữ" (crop) với "không tiền xử lý" (none) + paired bootstrap.

Đọc các file kết quả của src.evaluate đặt tên theo quy ước  <model>__<dataset>__<variant>.json
(variant = none | crop), nhóm theo (model, dataset), lấy
variant "none" làm baseline, rồi với từng biến thể tính:
  - ΔTED, ΔF1 (F1 micro như evaluator gốc) + khoảng tin cậy 95% bằng paired bootstrap trên cùng các ảnh;
  - p_boot = tỉ lệ lần lấy mẫu có Δ <= 0 (một phía; nhỏ = cải thiện đáng tin);
  - số ảnh tốt hơn / kém hơn / như cũ (theo TED từng ảnh, ngưỡng 0.005);
  - ΔTED theo nhóm: ảnh được cắt vs fallback, và theo mức phóng to chữ (gain);
  - thời gian: sec_per_img (model) và prep_sec_per_img (tiền xử lý).

    python -m src.preprocess_compare
Kết quả: results/preprocess/compare.csv, results/preprocess/report.md, results/preprocess/per_sample/<model>__<dataset>.csv
"""
import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from src import utils
from src.metrics import JSONParseEvaluator

_EV = JSONParseEvaluator()
GAIN_BINS = [(0.0, 1.1, "gain<1.1"), (1.1, 1.3, "1.1≤gain<1.3"), (1.3, 99.0, "gain≥1.3")]


def f1_counts(pred: dict, gt: dict) -> Tuple[int, int]:
    """(tp, fn_or_fp) của một mẫu – cộng dồn qua các mẫu cho đúng F1 micro của JSONParseEvaluator.cal_f1."""
    p, a = _EV.flatten(_EV.normalize_dict(pred)), _EV.flatten(_EV.normalize_dict(gt))
    tp = fnfp = 0
    for field in p:
        if field in a:
            tp += 1
            a.remove(field)
        else:
            fnfp += 1
    return tp, fnfp + len(a)


def micro_f1(tp: np.ndarray, fnfp: np.ndarray) -> float:
    t, f = tp.sum(), fnfp.sum()
    return float(t / (t + f / 2)) if (t + f) > 0 else 0.0


def load(path: str) -> Dict:
    d = utils.load_json(path)
    recs = sorted(d["records"], key=lambda r: r["idx"])
    counts = np.array([f1_counts(r["pred"], r["gt"]) for r in recs])
    return {
        "summary": d["summary"],
        "idx": [r["idx"] for r in recs],
        "ted": np.array([r["ted_acc"] for r in recs]),
        "tp": counts[:, 0], "fnfp": counts[:, 1],
        "prep": [r.get("prep") for r in recs],
        # tập ngoài CORD (--ext_eval): 1/0 = đọc đúng/sai total, nan = GT không có total
        "total_ok": np.array([float(r["total_ok"]) if "total_ok" in r else np.nan for r in recs]),
    }


def bootstrap(base: Dict, var: Dict, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(base["ted"])
    d_ted = var["ted"] - base["ted"]
    d_tot = var["total_ok"] - base["total_ok"]
    has_tot = ~np.isnan(d_tot)
    bt, bf, bc = np.empty(n_boot), np.empty(n_boot), np.full(n_boot, np.nan)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        bt[b] = d_ted[s].mean()
        bf[b] = micro_f1(var["tp"][s], var["fnfp"][s]) - micro_f1(base["tp"][s], base["fnfp"][s])
        if has_tot[s].any():
            bc[b] = np.nanmean(d_tot[s])
    tot = {}
    if has_tot.any():
        tot = {"total_ci_lo": float(np.nanpercentile(bc, 2.5)), "total_ci_hi": float(np.nanpercentile(bc, 97.5)),
               "total_p": float(np.nanmean(bc <= 0))}
    return {**tot,
        "ted_ci_lo": float(np.percentile(bt, 2.5)), "ted_ci_hi": float(np.percentile(bt, 97.5)),
        "ted_p": float((bt <= 0).mean()),
        "f1_ci_lo": float(np.percentile(bf, 2.5)), "f1_ci_hi": float(np.percentile(bf, 97.5)),
        "f1_p": float((bf <= 0).mean()),
    }


def pts(x: float) -> str:
    """Hiệu số -> điểm phần trăm có dấu."""
    return f"{100 * x:+.2f}"


def main():
    ap = argparse.ArgumentParser(description="So sánh crop với none + paired bootstrap")
    ap.add_argument("--eval_dir", default="results/preprocess/eval")
    ap.add_argument("--out_dir", default="results/preprocess")
    ap.add_argument("--baseline", default="none")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    groups: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.eval_dir, "*.json"))):
        parts = os.path.splitext(os.path.basename(path))[0].split("__")
        if len(parts) == 3:
            groups[(parts[0], parts[1])][parts[2]] = path
    if not groups:
        raise SystemExit(f"Không thấy file <model>__<dataset>__<variant>.json trong {args.eval_dir}")

    rows: List[Dict] = []
    md = ["# Cải tiến – cắt vùng chữ (Canny) trước encoder: crop vs none", "",
          f"Sinh bởi `python -m src.preprocess_compare` lúc {utils.now_str()}. Baseline = `{args.baseline}` "
          f"(cùng model, cùng ảnh). Δ tính bằng điểm phần trăm; CI 95% bằng paired bootstrap "
          f"({args.n_boot} lần); p = P(Δ ≤ 0) theo bootstrap (một phía).", ""]
    per_dir = utils.ensure_dir(os.path.join(args.out_dir, "per_sample"))

    for (model, data), variants in sorted(groups.items()):
        if args.baseline not in variants:
            print(f"[!] Bỏ qua {model}/{data}: thiếu baseline '{args.baseline}'")
            continue
        base = load(variants[args.baseline])
        loaded = {v: load(p) for v, p in variants.items()}
        md += [f"## {model} · {data}", "",
               "| Biến thể | F1 | TED | ΔF1 [CI 95%] (p) | ΔTED [CI 95%] (p) | Tốt hơn / kém hơn / như cũ | "
               "Cắt / fallback | gain TB | s/ảnh (model) | s/ảnh (tiền xử lý) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        per_sample = {"idx": base["idx"], f"ted_{args.baseline}": base["ted"]}
        sub_lines, total_lines = [], []
        has_total = base["summary"].get("total_acc") not in (None, "")
        order = [args.baseline] + sorted(v for v in loaded if v != args.baseline)
        for v in order:
            var = loaded[v]
            s = var["summary"]
            if var["idx"] != base["idx"]:
                print(f"[!] {model}/{data}/{v}: tập ảnh khác baseline -> bỏ qua")
                continue
            row = {"model": model, "dataset": data, "variant": v, "n": len(var["ted"]),
                   "f1": s["f1"], "ted_acc": s["ted_acc"], "ned": s.get("ned"),
                   "sec_per_img": s.get("sec_per_img"), "prep_sec_per_img": s.get("prep_sec_per_img", 0.0),
                   "prep_cropped": s.get("prep_cropped", 0), "prep_fallback": s.get("prep_fallback", 0),
                   "prep_gain_mean": s.get("prep_gain_mean", 1.0)}
            if has_total:
                row["total_acc"] = s.get("total_acc")
            if v == args.baseline:
                if has_total:
                    total_lines.append(f"| **{v}** | {s['total_acc']:.4f} | – | – |")
                md.append(f"| **{v}** | {s['f1']:.4f} | {s['ted_acc']:.4f} | – | – | – | – | 1.00 | "
                          f"{s.get('sec_per_img', '')} | – |")
                rows.append(row)
                continue
            bs = bootstrap(base, var, args.n_boot, args.seed)
            d = var["ted"] - base["ted"]
            better, worse = int((d > 0.005).sum()), int((d < -0.005).sum())
            d_f1 = micro_f1(var["tp"], var["fnfp"]) - micro_f1(base["tp"], base["fnfp"])
            row.update(d_f1=round(d_f1, 4), d_ted=round(float(d.mean()), 4), better=better, worse=worse,
                       same=len(d) - better - worse, **{k: round(x, 4) for k, x in bs.items()})
            rows.append(row)
            if has_total and "total_ci_lo" in bs:
                d_tot = float(s["total_acc"]) - float(base["summary"]["total_acc"])
                row["d_total_acc"] = round(d_tot, 4)
                fixed = int(np.nansum((var["total_ok"] == 1) & (base["total_ok"] == 0)))
                broken = int(np.nansum((var["total_ok"] == 0) & (base["total_ok"] == 1)))
                total_lines.append(f"| {v} | {s['total_acc']:.4f} | {pts(d_tot)} [{pts(bs['total_ci_lo'])}, "
                                   f"{pts(bs['total_ci_hi'])}] ({bs['total_p']:.3f}) | {fixed} / {broken} |")
            md.append(
                f"| {v} | {s['f1']:.4f} | {s['ted_acc']:.4f} | "
                f"{pts(d_f1)} [{pts(bs['f1_ci_lo'])}, {pts(bs['f1_ci_hi'])}] ({bs['f1_p']:.3f}) | "
                f"{pts(d.mean())} [{pts(bs['ted_ci_lo'])}, {pts(bs['ted_ci_hi'])}] ({bs['ted_p']:.3f}) | "
                f"{better} / {worse} / {len(d) - better - worse} | "
                f"{row['prep_cropped']} / {row['prep_fallback']} | {row['prep_gain_mean']:.2f} | "
                f"{row['sec_per_img']} | {row['prep_sec_per_img']} |")

            # ---- phân nhóm theo kết quả cắt / mức phóng to
            preps = var["prep"]
            per_sample[f"ted_{v}"] = var["ted"]
            if preps and preps[0] is not None:
                gains = np.array([p["gain"] for p in preps])
                ok = np.array([bool(p["ok"]) for p in preps])
                per_sample[f"gain_{v}"] = gains
                per_sample[f"status_{v}"] = [p["status"] for p in preps]
                cells = [("được cắt", ok), ("fallback (giữ nguyên)", ~ok)]
                cells += [(lab, (gains >= lo) & (gains < hi)) for lo, hi, lab in GAIN_BINS]
                sub_lines.append(f"| {v} | " + " | ".join(
                    f"{pts(d[m].mean())} (n={int(m.sum())})" if m.any() else "– (n=0)" for _, m in cells) + " |")
                sub_header = "| Biến thể | " + " | ".join(lab for lab, _ in cells) + " |"
        if total_lines:
            md += ["", f"Độ chính xác trường total ({model} · {data}, n={int((~np.isnan(base['total_ok'])).sum())} ảnh có total):", "",
                   "| Biến thể | total_acc | Δ [CI 95%] (p) | sửa đúng / làm sai so với baseline |", "|---|---|---|---|"] + total_lines
        if sub_lines:
            md += ["", f"ΔTED theo nhóm ảnh ({model} · {data}):", "", sub_header,
                   "|" + "---|" * (sub_header.count("|") - 1)] + sub_lines
        md.append("")

        keys = list(per_sample.keys())
        with open(os.path.join(per_dir, f"{model}__{data}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(len(base["idx"])):
                w.writerow([(round(float(per_sample[k][i]), 4) if isinstance(per_sample[k][i], (float, np.floating))
                             else per_sample[k][i]) for k in keys])

    out_csv = os.path.join(args.out_dir, "compare.csv")
    header = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    out_md = os.path.join(args.out_dir, "report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nĐã lưu: {out_csv}, {out_md}, {per_dir}/")


if __name__ == "__main__":
    main()
