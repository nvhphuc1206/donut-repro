"""Đánh giá Donut: F1 + TED Acc + NED + latency (hàm đánh giá DUY NHẤT của dự án).

    # tái lập: checkpoint chính thức, đúng như test.py gốc (chuẩn hóa ImageNet + fp16)
    python -m src.evaluate --model naver-clova-ix/donut-base-finetuned-cord-v2 --image_norm imagenet
        --precision fp16 --max_length 768 --run_id official-imagenet-fp16 --out results/reproduce/official_imagenet_fp16.json
    # cải tiến: thêm bước cắt vùng chữ trước encoder; bộ dữ liệu ngoài CORD thêm --ext_eval
    python -m src.evaluate ... --preprocess crop --dataset data/ext_sroie --ext_eval
"""
import argparse
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel

from src import utils
from src.data import get_gt_parses, image_norm_name, load_split, set_image_norm, set_input_size
from src.json_tokens import json2token, token2json
from src.metrics import edit_distance, sample_f1, score_predictions
from src.ext_eval import apply_ext_eval
from src.preprocess import DocPreprocessor


# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
def load_model_and_processor(
    model_path: str,
    input_size: Optional[List[int]] = None,
    device: str = "cuda",
    precision: str = "fp32",
    eager_attention: bool = False,
    image_norm: Optional[str] = None,
):
    """precision: fp32 | fp16 (GPU).
    image_norm: None = giữ theo preprocessor_config của checkpoint | "imagenet" | "hf"."""
    processor = DonutProcessor.from_pretrained(model_path)
    set_image_norm(processor, image_norm)
    kwargs = {"attn_implementation": "eager"} if eager_attention else {}
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_path, **kwargs)
    except (TypeError, ValueError):
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
    if input_size:
        set_input_size(processor, input_size)
        model.config.encoder.image_size = list(input_size)
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] Không có GPU -> chạy trên CPU")
        device = "cpu"
    if precision == "fp16":
        model = model.half()
    model.to(device).eval()
    return model, processor, device


# ----------------------------------------------------------------------------
# Sinh chuỗi + hậu xử lý
# ----------------------------------------------------------------------------
def clean_sequence(seq: str, tokenizer) -> str:
    """Bỏ eos/pad và prompt token đầu (giống code gốc / HF tutorial)."""
    seq = seq.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
    return re.sub(r"<.*?>", "", seq, count=1).strip()


@torch.no_grad()
def generate_sequences(
    model,
    processor,
    pixel_values: torch.Tensor,
    task_prompt: str,
    max_length: int,
    num_beams: int = 1,
    logits_processor=None,
    **extra,
) -> Tuple[List[str], Any]:
    tok = processor.tokenizer
    prompt_ids = tok(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    prompt_ids = prompt_ids.repeat(pixel_values.size(0), 1).to(pixel_values.device)
    gen_kwargs = dict(
        decoder_input_ids=prompt_ids,
        max_length=max_length,
        num_beams=num_beams,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        bad_words_ids=[[tok.unk_token_id]],
        return_dict_in_generate=True,
    )
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor
    gen_kwargs.update(extra)
    outputs = model.generate(pixel_values, **gen_kwargs)
    seqs = tok.batch_decode(outputs.sequences)
    return [clean_sequence(s, tok) for s in seqs], outputs


def sequence_ned(pred_seq: str, gt: dict) -> float:
    """Normalized edit distance giữa chuỗi token dự đoán và chuỗi đích – đúng metric validation
    của code gốc (lightning_module.validation_step), dùng để chọn checkpoint. Càng nhỏ càng tốt."""
    pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred_seq)
    answer = json2token(gt, sort_json_key=False)
    return edit_distance(pred, answer) / max(len(pred), len(answer), 1)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def evaluate_dataset(
    model,
    processor,
    ds,
    task_prompt: str = "<s_cord-v2>",
    max_length: int = 768,
    num_beams: int = 1,
    batch_size: int = 1,
    limit: Optional[int] = None,
    amp: bool = False,
    warmup: int = 0,
    desc: str = "eval",
    preprocess=None,
    ext_eval: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Hàm đánh giá DUY NHẤT của dự án. Trả về (summary, records từng mẫu).
    preprocess: None hoặc callable PIL.Image -> (PIL.Image, info) áp dụng trước processor (src/preprocess.py).
    Thời gian tiền xử lý đo riêng (prep_sec_per_img), không cộng vào sec_per_img.
    ext_eval: chấm trên bộ dữ liệu ngoài CORD (lọc theo schema của tập + chuẩn hóa giá trị, src/ext_eval.py)."""
    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    tok = processor.tokenizer
    added_vocab = tok.get_added_vocab()
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")

    n = len(ds) if limit is None else min(int(limit), len(ds))
    was_training = model.training
    model.eval()

    if warmup:  # vài lượt chạy không tính giờ (khởi tạo CUDA kernel)
        img0 = ds[0]["image"].convert("RGB")
        if preprocess:
            img0 = preprocess(img0)[0]
        pv = processor.image_processor(img0, return_tensors="pt").pixel_values
        for _ in range(warmup):
            with autocast:
                generate_sequences(model, processor, pv.to(device, dtype), task_prompt, max_length, num_beams)

    records = []
    for start in tqdm(range(0, n, batch_size), desc=desc):
        idxs = list(range(start, min(start + batch_size, n)))
        samples = [ds[i] for i in idxs]
        images = [s["image"].convert("RGB") for s in samples]
        infos = [None] * len(images)
        if preprocess:
            t_prep = time.perf_counter()
            images, infos = map(list, zip(*(preprocess(im) for im in images)))
            prep_dt = (time.perf_counter() - t_prep) / len(idxs)
        pv = processor.image_processor(images, return_tensors="pt").pixel_values.to(device, dtype)
        _sync(device)
        t0 = time.perf_counter()
        with autocast:
            seqs, _ = generate_sequences(model, processor, pv, task_prompt, max_length, num_beams)
        _sync(device)
        dt = (time.perf_counter() - t0) / len(idxs)
        for i, s, seq, info in zip(idxs, samples, seqs, infos):
            pred = token2json(seq, added_vocab)
            gt = get_gt_parses(s["ground_truth"])[0]
            records.append({"idx": i, "raw_sequence": seq, "pred": pred, "gt": gt, "seconds": dt})
            if info is not None:
                records[-1].update(prep=info, prep_seconds=prep_dt)

    ext_info = apply_ext_eval(records) if ext_eval else None
    scores = score_predictions([r["pred"] for r in records], [r["gt"] for r in records])
    for r, ted in zip(records, scores["ted_per_sample"]):
        r["ted_acc"] = ted
        r["f1"] = sample_f1(r["pred"], r["gt"])
    summary = {
        "n": n,
        "f1": round(scores["f1"], 4),
        "ted_acc": round(scores["ted_acc"], 4),
        "ned": round(sum(sequence_ned(r["raw_sequence"], r.get("gt_raw", r["gt"])) for r in records) / max(n, 1), 4),
        "sec_per_img": round(sum(r["seconds"] for r in records) / max(n, 1), 4),
    }
    if ext_info:
        summary.update(ext_eval=True, total_acc=ext_info["total_acc"], n_total=ext_info["n_total"],
                       ext_schema=ext_info["ext_schema"])
    if preprocess:
        statuses = [r["prep"]["status"] for r in records]
        summary.update({
            "preprocess": preprocess.mode,
            "prep_sec_per_img": round(sum(r["prep_seconds"] for r in records) / max(n, 1), 4),
            "prep_cropped": statuses.count("cropped"),
            "prep_fallback": sum(s.startswith(("fail", "skip")) for s in statuses),
            "prep_gain_mean": round(sum(r["prep"]["gain"] for r in records) / max(n, 1), 4),
        })
    if was_training:
        model.train()
    return summary, records


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Đánh giá Donut trên CORD (F1, TED Acc, latency)")
    p.add_argument("--model", default="naver-clova-ix/donut-base-finetuned-cord-v2", help="HF id hoặc thư mục checkpoint")
    p.add_argument("--dataset", default="naver-clova-ix/cord-v2", help="HF id hoặc thư mục save_to_disk (tập nhiễu)")
    p.add_argument("--split", default="test")
    p.add_argument("--task_prompt", default="<s_cord-v2>")
    p.add_argument("--input_size", type=int, nargs=2, metavar=("H", "W"), help="Ghi đè độ phân giải processor")
    p.add_argument("--max_length", type=int, default=None, help="Mặc định = decoder max_position_embeddings (768)")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1, help="Giữ 1 khi đo latency")
    p.add_argument("--limit", type=int, default=None, help="Chỉ đánh giá N mẫu đầu")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image_norm", choices=["imagenet", "hf"], default=None,
                   help="Ghi đè chuẩn hóa ảnh (mặc định: theo checkpoint). imagenet = giống code gốc")
    p.add_argument("--preprocess", default="none", choices=["none", "crop"],
                   help="Cải tiến: cắt vùng chữ (Canny) trước encoder (src/preprocess.py)")
    p.add_argument("--ext_eval", action="store_true",
                   help="Bộ dữ liệu ngoài CORD (src/make_external_test.py): lọc dự đoán theo schema của tập, "
                        "chuẩn hóa giá trị, thêm total_acc (src/ext_eval.py)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", default="eval")
    p.add_argument("--notes", default="")
    p.add_argument("--out", default=None, help="File JSON lưu từng dự đoán (mặc định results/<run_id>.json)")
    p.add_argument("--summary_csv", default=None, help="CSV tóm tắt riêng (tùy chọn)")
    p.add_argument("--runs_csv", default="results/runs.csv")
    return p.parse_args()


def main():
    args = parse_args()
    utils.start_log(f"results/logs/{args.run_id}.log")
    utils.set_seed(args.seed)
    model, processor, device = load_model_and_processor(
        args.model, args.input_size, args.device, args.precision, image_norm=args.image_norm)
    max_length = args.max_length or model.decoder.config.max_position_embeddings
    size = processor.image_processor.size
    input_size = [size["height"], size["width"]] if isinstance(size, dict) else list(size)[::-1]
    print(f"Model: {args.model} | input_size(H,W)={input_size} | max_length={max_length} | device={device}")

    ds = load_split(args.dataset, args.split)
    prep = DocPreprocessor(args.preprocess, input_size) if args.preprocess != "none" else None
    utils.reset_peak_vram()
    summary, records = evaluate_dataset(
        model, processor, ds, args.task_prompt, max_length, args.num_beams, args.batch_size,
        args.limit, amp=False, warmup=args.warmup, desc=args.run_id,
        preprocess=prep, ext_eval=args.ext_eval,
    )
    summary = {
        "id": args.run_id,
        "stage": "eval",
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "input_size": utils.size_str(input_size),
        "num_beams": args.num_beams,
        "precision": args.precision,
        "image_norm": image_norm_name(processor),
        "max_length": max_length,
        **summary,
        "peak_vram_gb": utils.peak_vram_gb(),
        "model_size_mb": utils.model_size_mb(model),
        "gpu": utils.gpu_name() if device == "cuda" else "cpu",
        "time": utils.now_str(),
        "notes": args.notes,
    }
    out = args.out or f"results/{args.run_id}.json"
    utils.save_json({"summary": summary, "records": records}, out)
    if args.summary_csv:
        utils.append_csv(args.summary_csv, summary)
    utils.append_csv(args.runs_csv, summary)
    print("\n===== KẾT QUẢ =====")
    keys = ["id", "n", "input_size", "image_norm", "num_beams", "precision", "f1", "ted_acc", "ned",
            "sec_per_img", "peak_vram_gb"]
    if args.ext_eval:
        keys += ["total_acc", "n_total"]
    if prep:
        keys += ["preprocess", "prep_sec_per_img", "prep_cropped", "prep_fallback", "prep_gain_mean"]
    for k in keys:
        print(f"{k:>14}: {summary[k]}")
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
