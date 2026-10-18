"""Đánh giá Donut trên CORD: F1 + TED Acc + latency (Task 1.1, I1, I3, I4, I5).

Ví dụ:
    python -m src.evaluate --model naver-clova-ix/donut-base-finetuned-cord-v2 \
        --run_id P1-official --out results/p1_official_cord.json --summary_csv results/p1_summary.csv
"""
import argparse
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import DonutProcessor, LogitsProcessor, LogitsProcessorList, VisionEncoderDecoderModel

from src import utils
from src.data import get_gt_parses, load_split, set_input_size
from src.json_tokens import token2json
from src.metrics import sample_f1, score_predictions


# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
def load_model_and_processor(
    model_path: str,
    input_size: Optional[List[int]] = None,
    device: str = "cuda",
    precision: str = "fp32",
    eager_attention: bool = False,
):
    """precision: fp32 | fp16 (GPU) | int8 (dynamic quantization, chỉ CPU)."""
    processor = DonutProcessor.from_pretrained(model_path)
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
    if precision == "int8":
        device = "cpu"
        quantize = getattr(torch.ao.quantization, "quantize_dynamic", None) or torch.quantization.quantize_dynamic
        model = quantize(model, {torch.nn.Linear}, dtype=torch.qint8)
    elif precision == "fp16":
        model = model.half()
    model.to(device).eval()
    return model, processor, device


# ----------------------------------------------------------------------------
# Giải mã có ràng buộc (I4 – nâng cao)
# ----------------------------------------------------------------------------
class StructureLogitsProcessor(LogitsProcessor):
    """Ràng buộc cấu trúc khi sinh:
    - chỉ cho phép thẻ đóng </s_k> khớp với thẻ mở gần nhất;
    - cấm thẻ mở không có thẻ đóng tương ứng (token tác vụ như <s_cord-v2>) và key ngoài schema;
    - cấm EOS khi vẫn còn thẻ đang mở.
    """

    def __init__(self, tokenizer, prompt_len: int, allowed_keys: Optional[set] = None):
        opens, closes = {}, {}
        for tok, i in tokenizer.get_added_vocab().items():
            m = re.fullmatch(r"<s_(.+)>", tok)
            if m:
                opens[i] = m.group(1)
                continue
            m = re.fullmatch(r"</s_(.+)>", tok)
            if m:
                closes[i] = m.group(1)
        close_keys = set(closes.values())
        self.opens, self.closes = opens, closes
        self.close_id = {k: i for i, k in closes.items()}
        self.banned_ids = list(closes) + [
            i for i, k in opens.items() if k not in close_keys or (allowed_keys is not None and k not in allowed_keys)
        ]
        self.prompt_len = prompt_len
        self.eos = tokenizer.eos_token_id
        self._base = None

    def _base_mask(self, vocab_size: int, device) -> torch.Tensor:
        if self._base is None or self._base.shape[0] != vocab_size or self._base.device != device:
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            mask[[i for i in self.banned_ids if i < vocab_size]] = True
            self._base = mask.to(device)
        return self._base

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        base = self._base_mask(scores.shape[-1], scores.device)
        for b in range(input_ids.shape[0]):
            stack = []
            for t in input_ids[b, self.prompt_len:].tolist():
                if t in self.opens:
                    stack.append(self.opens[t])
                elif t in self.closes and self.closes[t] in stack:
                    while stack and stack.pop() != self.closes[t]:
                        pass
            mask = base.clone()
            if stack and stack[-1] in self.close_id:
                mask[self.close_id[stack[-1]]] = False
                mask[self.eos] = True
            scores[b, mask] = -float("inf")
        return scores


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
    constrained: bool = False,
    warmup: int = 0,
    desc: str = "eval",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Hàm đánh giá DUY NHẤT của dự án. Trả về (summary, records từng mẫu)."""
    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    tok = processor.tokenizer
    added_vocab = tok.get_added_vocab()
    lp = None
    if constrained:
        prompt_len = len(tok(task_prompt, add_special_tokens=False).input_ids)
        lp = LogitsProcessorList([StructureLogitsProcessor(tok, prompt_len)])
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")

    n = len(ds) if limit is None else min(int(limit), len(ds))
    was_training = model.training
    model.eval()

    if warmup:  # vài lượt chạy không tính giờ (khởi tạo CUDA kernel)
        pv = processor.image_processor(ds[0]["image"].convert("RGB"), return_tensors="pt").pixel_values
        for _ in range(warmup):
            with autocast:
                generate_sequences(model, processor, pv.to(device, dtype), task_prompt, max_length, num_beams, lp)

    records = []
    for start in tqdm(range(0, n, batch_size), desc=desc):
        idxs = list(range(start, min(start + batch_size, n)))
        samples = [ds[i] for i in idxs]
        images = [s["image"].convert("RGB") for s in samples]
        pv = processor.image_processor(images, return_tensors="pt").pixel_values.to(device, dtype)
        _sync(device)
        t0 = time.perf_counter()
        with autocast:
            seqs, _ = generate_sequences(model, processor, pv, task_prompt, max_length, num_beams, lp)
        _sync(device)
        dt = (time.perf_counter() - t0) / len(idxs)
        for i, s, seq in zip(idxs, samples, seqs):
            pred = token2json(seq, added_vocab)
            gt = get_gt_parses(s["ground_truth"])[0]
            records.append({"idx": i, "raw_sequence": seq, "pred": pred, "gt": gt, "seconds": dt})

    scores = score_predictions([r["pred"] for r in records], [r["gt"] for r in records])
    for r, ted in zip(records, scores["ted_per_sample"]):
        r["ted_acc"] = ted
        r["f1"] = sample_f1(r["pred"], r["gt"])
    summary = {
        "n": n,
        "f1": round(scores["f1"], 4),
        "ted_acc": round(scores["ted_acc"], 4),
        "sec_per_img": round(sum(r["seconds"] for r in records) / max(n, 1), 4),
    }
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
    p.add_argument("--input_size", type=int, nargs=2, metavar=("H", "W"), help="Ghi đè độ phân giải processor (I1)")
    p.add_argument("--max_length", type=int, default=None, help="Mặc định = decoder max_position_embeddings (768)")
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1, help="Giữ 1 khi đo latency")
    p.add_argument("--limit", type=int, default=None, help="Chỉ đánh giá N mẫu đầu")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--constrained", action="store_true", help="Bật giải mã có ràng buộc cấu trúc (I4)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", default="eval")
    p.add_argument("--notes", default="")
    p.add_argument("--out", default=None, help="File JSON lưu từng dự đoán (mặc định results/<run_id>.json)")
    p.add_argument("--summary_csv", default=None, help="CSV tóm tắt riêng (vd results/p1_summary.csv)")
    p.add_argument("--runs_csv", default="results/runs.csv")
    return p.parse_args()


def main():
    args = parse_args()
    utils.start_log(f"results/logs/{args.run_id}.log")
    utils.set_seed(args.seed)
    model, processor, device = load_model_and_processor(args.model, args.input_size, args.device, args.precision)
    max_length = args.max_length or model.decoder.config.max_position_embeddings
    size = processor.image_processor.size
    input_size = [size["height"], size["width"]] if isinstance(size, dict) else list(size)[::-1]
    print(f"Model: {args.model} | input_size(H,W)={input_size} | max_length={max_length} | device={device}")

    ds = load_split(args.dataset, args.split)
    utils.reset_peak_vram()
    summary, records = evaluate_dataset(
        model, processor, ds, args.task_prompt, max_length, args.num_beams, args.batch_size,
        args.limit, amp=False, constrained=args.constrained, warmup=args.warmup, desc=args.run_id,
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
        "constrained": args.constrained,
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
    for k in ["id", "n", "input_size", "num_beams", "precision", "f1", "ted_acc", "sec_per_img", "peak_vram_gb"]:
        print(f"{k:>14}: {summary[k]}")
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
