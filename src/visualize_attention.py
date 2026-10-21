"""Phân tích – trực quan hóa cross-attention của decoder lên ảnh (minh chứng "định vị chữ ngầm").

Gom các token thuộc cùng một giá trị field (vd. nội dung trong <s_nm>…</s_nm>), lấy trung bình
cross-attention (trung bình head; lớp cuối hoặc trung bình các lớp) -> lưới (H/32, W/32) -> chồng lên ảnh.

    python -m src.visualize_attention --indices 0 1 2 --keys nm price total_price
Output: results/analysis/attention/sample{idx}_*.png
"""
import argparse
import math
import os
import re

import numpy as np
import torch
from PIL import Image

from src import utils
from src.data import get_gt_parses, load_split
from src.evaluate import generate_sequences, load_model_and_processor
from src.json_tokens import token2json
from src.metrics import JSONParseEvaluator


def encoder_grid(h: int, w: int, n_stages: int = 4):
    """Kích thước lưới feature cuối của Swin: patch 4 rồi patch-merging (÷2) qua n_stages-1 lần."""
    gh, gw = math.ceil(h / 4), math.ceil(w / 4)
    for _ in range(n_stages - 1):
        gh, gw = math.ceil(gh / 2), math.ceil(gw / 2)
    return gh, gw


def step_maps(cross_attentions, layer: str) -> torch.Tensor:
    """cross_attentions: tuple(step) of tuple(layer) of [B, heads, q, n] -> [steps, n]."""
    maps = []
    for layers in cross_attentions:
        stacked = torch.stack([l[0, :, -1, :].float() for l in layers])  # [L, heads, n]
        if layer == "mean":
            m = stacked.mean(dim=(0, 1))
        else:
            m = stacked[int(layer) if layer != "last" else -1].mean(0)
        maps.append(m.cpu())
    return torch.stack(maps)


def group_field_tokens(token_ids, tokenizer, keys, prompt_len):
    """Gom token văn bản theo field đang mở gần nhất. Trả về list {key, steps, ids}."""
    added = tokenizer.get_added_vocab()
    id2tok = {i: t for t, i in added.items()}
    special = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    stack, groups, current = [], [], None
    for t, tid in enumerate(token_ids[prompt_len:]):
        tokstr = id2tok.get(tid)
        if tid in special:
            current = None
            continue
        if tokstr is not None:
            m_open, m_close = re.fullmatch(r"<s_(.+)>", tokstr), re.fullmatch(r"</s_(.+)>", tokstr)
            if m_open:
                stack.append(m_open.group(1))
            elif m_close and m_close.group(1) in stack:
                while stack and stack.pop() != m_close.group(1):
                    pass
            current = None
            continue
        key = stack[-1] if stack else None
        if key in keys:
            if current is None:
                current = {"key": key, "steps": [], "ids": []}
                groups.append(current)
            current["steps"].append(t)
            current["ids"].append(tid)
    return groups


def overlay(base: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> Image.Image:
    import matplotlib.cm as cm

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_img = Image.fromarray(np.uint8(heat * 255)).resize(base.size, Image.BILINEAR)
    colored = Image.fromarray(np.uint8(cm.jet(np.asarray(heat_img) / 255.0)[..., :3] * 255))
    return Image.blend(base.convert("RGB"), colored, alpha)


def denormalize(pv: torch.Tensor, processor) -> Image.Image:
    mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
    img = (pv.float().cpu() * std + mean).clamp(0, 1)
    return Image.fromarray(np.uint8(img.permute(1, 2, 0).numpy() * 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="naver-clova-ix/donut-base-finetuned-cord-v2")
    ap.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--task_prompt", default="<s_cord-v2>")
    ap.add_argument("--keys", nargs="+", default=["nm", "price", "cnt", "total_price"])
    ap.add_argument("--max_groups", type=int, default=6)
    ap.add_argument("--layer", default="last", help="last | mean | chỉ số lớp (0..3)")
    ap.add_argument("--input_size", type=int, nargs=2, metavar=("H", "W"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results/analysis/attention")
    args = ap.parse_args()
    utils.start_log("results/logs/visualize_attention.log")

    import matplotlib.pyplot as plt

    utils.set_seed(42)
    model, processor, device = load_model_and_processor(args.model, args.input_size, args.device, eager_attention=True)
    tok = processor.tokenizer
    max_length = model.decoder.config.max_position_embeddings
    prompt_len = len(tok(args.task_prompt, add_special_tokens=False).input_ids)
    ds = load_split(args.dataset, args.split)
    evaluator = JSONParseEvaluator()
    utils.ensure_dir(args.out_dir)

    for idx in args.indices:
        sample = ds[idx]
        pv = processor.image_processor(sample["image"].convert("RGB"), return_tensors="pt").pixel_values.to(device)
        seqs, out = generate_sequences(model, processor, pv, args.task_prompt, max_length, 1, output_attentions=True)
        pred = token2json(seqs[0], tok.get_added_vocab())
        gt = get_gt_parses(sample["ground_truth"])[0]
        ted = evaluator.cal_acc(pred, gt)

        maps = step_maps(out.cross_attentions, args.layer)  # [steps, n]
        gh, gw = encoder_grid(pv.shape[-2], pv.shape[-1], len(model.config.encoder.depths))
        if gh * gw != maps.shape[1]:
            raise RuntimeError(f"Lưới {gh}x{gw} không khớp n={maps.shape[1]} encoder tokens")

        base = denormalize(pv[0], processor)
        groups = group_field_tokens(out.sequences[0].tolist(), tok, set(args.keys), prompt_len)[: args.max_groups]
        if not groups:
            print(f"[sample {idx}] không có token thuộc keys {args.keys}")
            continue

        cols = min(3, len(groups) + 1)
        rows = math.ceil((len(groups) + 1) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6.5 * rows))
        axes = np.array(axes).reshape(-1)
        axes[0].imshow(base)
        axes[0].set_title(f"sample {idx} (TED Acc={ted:.3f})")
        for g_i, g in enumerate(groups):
            heat = maps[g["steps"]].mean(0).reshape(gh, gw).numpy()
            text = tok.decode(g["ids"]).strip()
            img = overlay(base, heat)
            img.save(os.path.join(args.out_dir, f"sample{idx}_{g_i:02d}_{g['key']}.png"))
            axes[g_i + 1].imshow(img)
            axes[g_i + 1].set_title(f"<s_{g['key']}> \"{text[:30]}\"", fontsize=10)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Cross-attention (layer={args.layer}, trung bình head) – lưới {gh}x{gw}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"sample{idx}_summary.png"), dpi=120)
        plt.close(fig)
        utils.save_json(
            {"idx": idx, "ted_acc": ted, "pred": pred, "gt": gt,
             "groups": [{"key": g["key"], "text": tok.decode(g["ids"]).strip()} for g in groups]},
            os.path.join(args.out_dir, f"sample{idx}.json"),
        )
        print(f"[sample {idx}] TED={ted:.3f} -> {args.out_dir}/sample{idx}_summary.png")


if __name__ == "__main__":
    main()
