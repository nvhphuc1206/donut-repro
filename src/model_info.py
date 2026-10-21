"""Phân tích – in kích thước tensor thực tế + số tham số, vẽ sơ đồ kiến trúc có số liệu.

    python -m src.model_info --model naver-clova-ix/donut-base-finetuned-cord-v2
Output: results/analysis/model_info.json, architecture.mmd (Mermaid), architecture.png
"""
import argparse
import os

import torch
from PIL import Image

from src import utils
from src.data import load_split
from src.evaluate import load_model_and_processor


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def draw_architecture(info, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    enc, dec = info["encoder_config"], info["decoder_config"]
    pv = info["pixel_values"]
    stages = info["encoder_hidden_states"]
    boxes = [
        ("Ảnh tài liệu", f"pixel_values {pv}", "#dbeafe"),
        ("Patch embedding 4×4", f"→ {stages[0]}" if stages else "", "#bfdbfe"),
    ]
    for i, d in enumerate(enc["depths"]):
        shp = stages[i + 1] if len(stages) > i + 1 else ""
        boxes.append((f"Swin stage {i + 1}: {d} block, {enc['num_heads'][i]} head, window {enc['window_size']}",
                      f"→ {shp}", "#93c5fd"))
    boxes += [
        ("Encoder output {z_i}", f"last_hidden_state {info['encoder_last_hidden_state']}", "#60a5fa"),
        (f"BART decoder: {dec['decoder_layers']} layer, d={dec['d_model']}, {dec['decoder_attention_heads']} head",
         "self-attn (causal) + cross-attn vào {z_i}  | prompt: <s_cord-v2>", "#fde68a"),
        ("LM head → token", f"vocab = {info['tokenizer_len']} (gồm {info['num_added_tokens']} special token)", "#fcd34d"),
        ("token2json", "<s_menu><s_nm>Tea</s_nm>…  →  {\"menu\": {\"nm\": \"Tea\"}}", "#bbf7d0"),
    ]
    fig_h = 1.05 * len(boxes) + 0.8
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(boxes) + 0.5)
    ax.axis("off")
    for k, (title, sub, colr) in enumerate(boxes):
        y = len(boxes) - k - 0.5
        ax.add_patch(FancyBboxPatch((1, y - 0.35), 8, 0.75, boxstyle="round,pad=0.05", fc=colr, ec="#334155"))
        ax.text(5, y + 0.12, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(5, y - 0.17, sub, ha="center", va="center", fontsize=8.5, family="monospace")
        if k < len(boxes) - 1:
            ax.annotate("", xy=(5, y - 0.45), xytext=(5, y - 0.37), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title(f"Donut – tổng {info['params']['total_M']}M tham số "
                 f"(encoder {info['params']['encoder_M']}M, decoder {info['params']['decoder_M']}M)", fontsize=11)
    utils.ensure_dir(os.path.dirname(path))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mermaid(info) -> str:
    enc, dec = info["encoder_config"], info["decoder_config"]
    return f"""flowchart TD
    A["Ảnh tài liệu<br/>pixel_values {info['pixel_values']}"] --> B["Swin Transformer encoder<br/>depths {enc['depths']}, window {enc['window_size']}, embed_dim {enc['embed_dim']}"]
    B --> C["Chuỗi embedding z<br/>{info['encoder_last_hidden_state']}"]
    P["Prompt token &lt;s_cord-v2&gt;"] --> D
    C -- cross-attention --> D["BART decoder<br/>{dec['decoder_layers']} layer, d_model {dec['d_model']}"]
    D --> E["Chuỗi token (vocab {info['tokenizer_len']})"]
    E --> F["token2json → JSON"]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="naver-clova-ix/donut-base-finetuned-cord-v2")
    ap.add_argument("--dataset", default="naver-clova-ix/cord-v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--input_size", type=int, nargs=2, metavar=("H", "W"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="results/analysis")
    args = ap.parse_args()
    utils.start_log("results/logs/model_info.log")

    model, processor, device = load_model_and_processor(args.model, args.input_size, args.device)
    try:
        image = load_split(args.dataset, args.split)[0]["image"].convert("RGB")
    except Exception as e:  # không có mạng -> ảnh trắng
        print(f"[!] Không load được ảnh mẫu ({e}), dùng ảnh trắng")
        image = Image.new("RGB", (960, 1280), "white")
    pv = processor.image_processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        enc_out = model.encoder(pv, output_hidden_states=True)
    hidden = getattr(enc_out, "reshaped_hidden_states", None) or enc_out.hidden_states
    ec, dc = model.config.encoder, model.config.decoder
    tok = processor.tokenizer
    info = {
        "model": args.model,
        "processor_size": processor.image_processor.size,
        "pixel_values": list(pv.shape),
        "encoder_last_hidden_state": list(enc_out.last_hidden_state.shape),
        "encoder_hidden_states": [list(h.shape) for h in hidden],
        "encoder_hidden_states_note": "[B, C, h, w]: phần tử 0 = sau patch embedding, i = sau stage i (đã patch-merging nếu có)",
        "encoder_config": {
            "depths": list(ec.depths), "num_heads": list(ec.num_heads), "window_size": ec.window_size,
            "embed_dim": ec.embed_dim, "patch_size": ec.patch_size, "image_size": list(ec.image_size)
            if isinstance(ec.image_size, (list, tuple)) else ec.image_size,
        },
        "decoder_config": {
            "decoder_layers": dc.decoder_layers, "d_model": dc.d_model,
            "decoder_attention_heads": dc.decoder_attention_heads, "decoder_ffn_dim": dc.decoder_ffn_dim,
            "vocab_size": dc.vocab_size, "max_position_embeddings": dc.max_position_embeddings,
        },
        "tokenizer_len": len(tok),
        "num_added_tokens": len(tok.get_added_vocab()),
        "params": {
            "encoder_M": round(n_params(model.encoder) / 1e6, 2),
            "decoder_M": round(n_params(model.decoder) / 1e6, 2),
            "decoder_embeddings_M": round(model.decoder.get_input_embeddings().weight.numel() / 1e6, 2),
            "total_M": round(n_params(model) / 1e6, 2),
        },
    }

    for k, v in info.items():
        print(f"{k}: {v}")
    utils.save_json(info, os.path.join(args.out_dir, "model_info.json"))
    with open(os.path.join(args.out_dir, "architecture.mmd"), "w", encoding="utf-8") as f:
        f.write(mermaid(info))
    draw_architecture(info, os.path.join(args.out_dir, "architecture.png"))
    print(f"\nĐã lưu {args.out_dir}/model_info.json, architecture.mmd, architecture.png")


if __name__ == "__main__":
    main()
