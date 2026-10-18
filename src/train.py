"""Fine-tune Donut trên CORD (Task 1.2, I1-b, I2 LoRA, I3 augmentation).

    python -m src.train --config configs/train_cord_full.yaml
    python -m src.train --config configs/train_cord_full.yaml --set output_dir=/content/drive/MyDrive/donut/outputs/B0 epochs=10

- Checkpoint `last.pt` được ghi mỗi epoch -> chạy lại cùng lệnh để resume (resume theo epoch).
- Checkpoint tốt nhất theo TED Acc trên val được lưu ở `<output_dir>/best` (định dạng HF, dùng thẳng cho evaluate.py).
"""
import argparse
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from src import utils
from src.augment import TrainAugment
from src.data import DonutDataset, add_tokens, collate_train, collect_task_tokens, load_split, set_input_size
from src.evaluate import evaluate_dataset

DEFAULTS = dict(
    run_id="B0",
    model_name_or_path="naver-clova-ix/donut-base",
    dataset="naver-clova-ix/cord-v2",
    train_split="train",
    val_split="validation",
    task_token="<s_cord-v2>",
    input_size=[1280, 960],  # [H, W]
    align_long_axis=False,
    max_length=768,
    sort_json_key=False,
    epochs=12,
    batch_size=1,
    grad_accum=8,
    lr=3.0e-5,
    weight_decay=0.0,
    warmup_steps=300,
    scheduler="cosine",
    max_grad_norm=1.0,
    fp16=True,
    gradient_checkpointing=True,
    num_workers=2,
    seed=42,
    log_every=10,
    val_every=1,
    train_limit=None,
    val_limit=None,
    val_batch_size=4,
    output_dir=None,
    resume=True,
    save_optimizer=True,
    lora=None,
    augment=None,
    runs_csv="results/runs.csv",
    notes="",
)


# ----------------------------------------------------------------------------
# Model / tokenizer
# ----------------------------------------------------------------------------
def build_model_and_processor(cfg):
    path = cfg["model_name_or_path"]
    processor = DonutProcessor.from_pretrained(path)
    config = VisionEncoderDecoderConfig.from_pretrained(path)
    config.encoder.image_size = list(cfg["input_size"])
    config.decoder.max_length = cfg["max_length"]
    model = VisionEncoderDecoderModel.from_pretrained(path, config=config)
    set_input_size(processor, cfg["input_size"], cfg.get("align_long_axis"))
    return model, processor


def setup_special_tokens(model, processor, train_hf, task_token):
    """Thêm <s_key>/</s_key>, <sep/>, <s_cord-v2>; resize embedding; đặt decoder_start = prompt."""
    tok = processor.tokenizer
    first_new_id = len(tok)
    n_added = add_tokens(tok, collect_task_tokens(train_hf, task_token))
    if model.decoder.get_input_embeddings().weight.shape[0] != len(tok):
        model.decoder.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    model.config.decoder_start_token_id = tok.convert_tokens_to_ids(task_token)
    print(f"Đã thêm {n_added} special tokens (vocab = {len(tok)}), decoder_start = {task_token}")
    return n_added, first_new_id


def embedding_params(model):
    """Ma trận embedding của decoder (+ lm_head nếu không bị tie)."""
    emb = model.decoder.get_input_embeddings().weight
    params = [emb]
    out = model.decoder.get_output_embeddings()
    if out is not None and out.weight is not emb:
        params.append(out.weight)
    return params


def apply_lora(model, lcfg, first_new_id):
    """LoRA cho q/v của decoder (self + cross attention) và (tuỳ chọn) query/value của Swin stage 3–4.

    Embedding: mặc định chỉ học các hàng của special token mới (`train_embeddings: new_rows`), tương đương
    modules_to_save nhưng giữ được weight tying embed_tokens <-> lm_head khi merge/lưu.
    """
    from peft import LoraConfig, get_peft_model

    patterns = []
    if lcfg.get("target_decoder", True):
        patterns.append(r"decoder\.model\.decoder\.layers\.\d+\.(?:self_attn|encoder_attn)\.(?:q_proj|v_proj)")
    stages = lcfg.get("target_encoder_stages") or []
    if stages:
        st = "|".join(str(s) for s in stages)
        patterns.append(rf"encoder\.encoder\.layers\.(?:{st})\.blocks\.\d+\.attention\.self\.(?:query|value)")
    target = "|".join(f"(?:{p})" for p in patterns)
    peft_cfg = LoraConfig(
        r=lcfg.get("r", 8),
        lora_alpha=lcfg.get("alpha", 16),
        lora_dropout=lcfg.get("dropout", 0.05),
        target_modules=target,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)

    mode = lcfg.get("train_embeddings", "new_rows")  # new_rows | full | frozen
    extra_effective = 0
    if mode in ("new_rows", "full"):
        for p in embedding_params(model.get_base_model()):
            p.requires_grad_(True)
            if mode == "new_rows":
                mask = torch.zeros(p.shape[0], 1)
                mask[first_new_id:] = 1.0
                p.register_hook(lambda g, m=mask: g * m.to(g.device, g.dtype))
                extra_effective += max(p.shape[0] - first_new_id, 0) * p.shape[1]
    return model, mode, extra_effective


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_save(obj, path, retries: int = 30):
    """Ghi file tạm rồi đổi tên. Windows: file đích có thể đang bị khóa tạm thời
    (Windows Defender/indexer quét file lớn) -> thử lại thay vì làm dừng cả quá trình train."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(10)


def shift_right(labels: torch.Tensor, pad_id: int, start_id: int) -> torch.Tensor:
    """decoder_input_ids = [<s_cord-v2>, y_1, ..., y_{n-1}] (giống shift_tokens_right của HF / code gốc)."""
    dec = labels.new_full(labels.shape, pad_id)
    dec[:, 1:] = labels[:, :-1]
    dec[:, 0] = start_id
    dec[dec == -100] = pad_id
    return dec


def seq2seq_loss(model, pixel_values, labels, pad_id, start_id):
    """Tự tính cross-entropy: logits[t] dự đoán labels[t] (KHÔNG dịch thêm).

    Không dùng `model(..., labels=labels).loss`: từ transformers ~4.46, VisionEncoderDecoderModel gọi
    `self.loss_function` và với config không khai báo loss_type sẽ rơi về ForCausalLMLoss – hàm này
    dịch labels thêm 1 vị trí. Cộng với việc labels đã được shift-right làm decoder input -> model học
    dự đoán token cách 2 vị trí (loss vẫn giảm nhưng sinh chuỗi sai hoàn toàn).
    """
    out = model(pixel_values=pixel_values, decoder_input_ids=shift_right(labels, pad_id, start_id), use_cache=False)
    logits = out.logits
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune Donut trên CORD")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="Ghi đè config: key=value (vd: epochs=10 lora.r=16)")
    args = ap.parse_args()

    cfg = {**DEFAULTS, **utils.load_config(args.config, args.set)}
    cfg["output_dir"] = cfg["output_dir"] or f"outputs/{cfg['run_id']}"
    out_dir = utils.ensure_dir(cfg["output_dir"])
    best_dir = os.path.join(out_dir, "best")
    last_path = os.path.join(out_dir, "last.pt")
    utils.start_log(os.path.join(out_dir, "train.log"))
    utils.save_yaml(cfg, os.path.join(out_dir, "config.yaml"))
    utils.set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(cfg["fp16"]) and device.type == "cuda"
    print(f"Run {cfg['run_id']} | GPU: {utils.gpu_name()} | output: {out_dir}")

    # ---- dữ liệu + model
    train_hf = load_split(cfg["dataset"], cfg["train_split"])
    val_hf = load_split(cfg["dataset"], cfg["val_split"])
    model, processor = build_model_and_processor(cfg)
    _, first_new_id = setup_special_tokens(model, processor, train_hf, cfg["task_token"])
    if cfg["train_limit"]:  # chỉ dùng cho smoke test
        train_hf = train_hf.select(range(min(int(cfg["train_limit"]), len(train_hf))))

    if cfg["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    lora_cfg = cfg.get("lora")
    emb_mode, extra_effective = None, 0
    if lora_cfg:
        model, emb_mode, extra_effective = apply_lora(model, lora_cfg, first_new_id)
        model.print_trainable_parameters()
    pad_id = processor.tokenizer.pad_token_id
    start_id = processor.tokenizer.convert_tokens_to_ids(cfg["task_token"])
    gen_model = model.get_base_model() if lora_cfg else model  # model dùng để generate khi validate
    model.to(device)

    total_params, trainable_params = utils.count_parameters(model)
    if lora_cfg and emb_mode == "new_rows":
        emb_numel = sum(p.numel() for p in embedding_params(gen_model))
        effective_trainable = trainable_params - emb_numel + extra_effective
    else:
        effective_trainable = trainable_params
    print(f"Tham số: tổng {total_params/1e6:.1f}M | train {trainable_params/1e6:.2f}M "
          f"| train hiệu dụng {effective_trainable/1e6:.2f}M")

    augment = TrainAugment(cfg["augment"] if isinstance(cfg["augment"], dict) else None) if cfg.get("augment") else None
    train_ds = DonutDataset(train_hf, processor, cfg["max_length"], train=True, augment=augment,
                            sort_json_key=cfg["sort_json_key"])
    loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_train,
        pin_memory=device.type == "cuda",
    )

    # ---- optimizer / scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    grad_accum = cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    sched_fn = get_cosine_schedule_with_warmup if cfg["scheduler"] == "cosine" else get_linear_schedule_with_warmup
    scheduler = sched_fn(optimizer, cfg["warmup_steps"], total_steps)
    scaler = make_scaler(fp16)
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=fp16)

    state = dict(epoch=0, global_step=0, best_ted=-1.0, best_f1=0.0, best_epoch=-1, train_seconds=0.0, peak_vram=0.0)

    # ---- resume
    if cfg["resume"] and os.path.exists(last_path):
        ckpt = torch_load(last_path)
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
        state.update(ckpt["state"])
        random.setstate(ckpt["rng"]["python"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        print(f"==> Resume từ epoch {state['epoch']} (best TED val = {state['best_ted']:.4f})")

    history_csv = os.path.join(out_dir, "history.csv")
    steps_csv = os.path.join(out_dir, "train_steps.csv")

    # ---- vòng lặp huấn luyện
    for epoch in range(state["epoch"], cfg["epochs"]):
        model.train()
        utils.reset_peak_vram()
        t0 = time.time()
        running, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg['epochs']}")
        for i, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast:
                loss = seq2seq_loss(model, pixel_values, labels, pad_id, start_id)
            scaler.scale(loss / grad_accum).backward()
            running += loss.item()
            n_batches += 1

            if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                state["global_step"] += 1
                if state["global_step"] % cfg["log_every"] == 0:
                    utils.append_csv(steps_csv, {
                        "global_step": state["global_step"], "epoch": epoch + 1,
                        "loss": round(running / n_batches, 5), "lr": scheduler.get_last_lr()[0],
                    })
            pbar.set_postfix(loss=f"{running / n_batches:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        epoch_time = time.time() - t0
        state["train_seconds"] += epoch_time
        epoch_vram = utils.peak_vram_gb()
        state["peak_vram"] = max(state["peak_vram"], epoch_vram)

        # ---- validation (cùng hàm đánh giá với test)
        row = {"epoch": epoch + 1, "train_loss": round(running / max(n_batches, 1), 5),
               "lr": scheduler.get_last_lr()[0], "epoch_seconds": round(epoch_time, 1), "peak_vram_gb": epoch_vram}
        if (epoch + 1) % cfg["val_every"] == 0 or (epoch + 1) == cfg["epochs"]:
            val, _ = evaluate_dataset(
                gen_model, processor, val_hf, cfg["task_token"], cfg["max_length"], num_beams=1,
                batch_size=cfg["val_batch_size"], limit=cfg["val_limit"], amp=fp16, desc="val",
            )
            row.update({"val_f1": val["f1"], "val_ted_acc": val["ted_acc"]})
            print(f"[epoch {epoch + 1}] loss={row['train_loss']:.4f} val F1={val['f1']:.4f} "
                  f"TED={val['ted_acc']:.4f} ({epoch_time / 60:.1f} phút, VRAM {epoch_vram} GB)")
            if val["ted_acc"] > state["best_ted"]:
                state.update(best_ted=val["ted_acc"], best_f1=val["f1"], best_epoch=epoch + 1)
                if lora_cfg:
                    atomic_save(trainable_state(model), os.path.join(out_dir, "best_trainable.pt"))
                else:
                    model.save_pretrained(best_dir)
                    processor.save_pretrained(best_dir)
                print(f"   -> checkpoint tốt nhất mới (epoch {epoch + 1})")
        utils.append_csv(history_csv, row)

        # ---- checkpoint để resume
        state["epoch"] = epoch + 1
        ckpt = {
            "model": trainable_state(model),
            "state": state,
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if cfg["save_optimizer"]:
            ckpt.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
        atomic_save(ckpt, last_path)

    # ---- hoàn tất: LoRA -> merge adapter tốt nhất vào model và lưu dạng HF
    if lora_cfg and not os.path.exists(os.path.join(best_dir, "config.json")):
        best_path = os.path.join(out_dir, "best_trainable.pt")
        if os.path.exists(best_path):
            model.load_state_dict(torch_load(best_path), strict=False)
        merged = model.merge_and_unload()
        merged.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        print(f"Đã merge LoRA và lưu {best_dir}")

    n_epochs = max(state["epoch"], 1)
    summary = {
        "id": cfg["run_id"],
        "stage": "train",
        "config": args.config,
        "model": cfg["model_name_or_path"],
        "input_size": utils.size_str(cfg["input_size"]),
        "epochs": state["epoch"],
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "effective_trainable_params_M": round(effective_trainable / 1e6, 2),
        "peak_vram_gb": state["peak_vram"],
        "train_hours": round(state["train_seconds"] / 3600, 2),
        "sec_per_epoch": round(state["train_seconds"] / n_epochs, 1),
        "best_epoch": state["best_epoch"],
        "val_f1": state["best_f1"],
        "val_ted_acc": state["best_ted"],
        "gpu": utils.gpu_name(),
        "time": utils.now_str(),
        "notes": cfg.get("notes", ""),
    }
    utils.save_json(summary, os.path.join(out_dir, "train_summary.json"))
    utils.append_csv(cfg["runs_csv"], summary)
    print("\n===== HUẤN LUYỆN XONG =====")
    for k, v in summary.items():
        print(f"{k:>30}: {v}")
    print(f"Checkpoint tốt nhất: {best_dir}")


if __name__ == "__main__":
    main()
