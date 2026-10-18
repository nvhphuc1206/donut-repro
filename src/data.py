"""Dữ liệu CORD (HF Hub hoặc thư mục save_to_disk) -> (pixel_values, labels)."""
import json
import os
import random
import re
from typing import Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset

from src.json_tokens import SEP_TOKEN, collect_keys, json2token, key_tokens


def _split_parquet_files(dataset: str, split: str) -> Optional[List[str]]:
    """Các file parquet của đúng một split trên Hub (vd. data/test-00000-of-00001-*.parquet)."""
    try:
        from huggingface_hub import list_repo_files

        files = list_repo_files(dataset, repo_type="dataset")
    except Exception:  # offline / không truy cập được Hub -> để load_dataset tự xử lý
        return None
    pattern = re.compile(rf"(^|/){re.escape(split)}-[^/]*\.parquet$")
    return [f for f in files if pattern.search(f)] or None


def load_split(dataset: str, split: str, config_name: Optional[str] = None, streaming: bool = False):
    """`dataset` là id trên HF Hub (naver-clova-ix/cord-v2) hoặc thư mục local (save_to_disk).

    Với dataset parquet trên Hub, chỉ tải file của split được yêu cầu (load_dataset mặc định tải mọi split).
    """
    if os.path.isdir(dataset):
        ds = load_from_disk(dataset)
        if isinstance(ds, DatasetDict):
            ds = ds[split]
        return ds
    if config_name is None and not streaming:
        files = _split_parquet_files(dataset, split)
        if files:
            return load_dataset(dataset, data_files={split: files}, split=split, verification_mode="no_checks")
    return load_dataset(dataset, config_name, split=split, streaming=streaming)


def get_gt_parses(ground_truth: str) -> List[dict]:
    gt = json.loads(ground_truth)
    if "gt_parses" in gt:
        return gt["gt_parses"]
    return [gt["gt_parse"]]


def collect_task_tokens(hf_ds, task_token: str) -> List[str]:
    """Special tokens cần thêm: prompt + <sep/> + <s_key>/</s_key> cho mọi key trong gt_parse."""
    keys = set()
    for gt_str in hf_ds["ground_truth"]:
        for parse in get_gt_parses(gt_str):
            collect_keys(parse, keys)
    return [task_token, SEP_TOKEN] + key_tokens(keys)


def add_tokens(tokenizer, tokens: List[str]) -> int:
    """Thêm special tokens chưa có trong vocab; trả về số token mới."""
    vocab = tokenizer.get_vocab()
    tokens = [t for t in dict.fromkeys(tokens) if t not in vocab]
    if not tokens:
        return 0
    try:
        return tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens}, replace_additional_special_tokens=False
        )
    except TypeError:  # transformers cũ
        return tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokenizer.additional_special_tokens) + tokens}
        )


def set_input_size(processor, input_size, align_long_axis: Optional[bool] = None):
    """input_size = [H, W] (giống config gốc: [1280, 960] = cao 1280, rộng 960)."""
    h, w = int(input_size[0]), int(input_size[1])
    processor.image_processor.size = {"height": h, "width": w}
    if align_long_axis is not None:
        processor.image_processor.do_align_long_axis = bool(align_long_axis)


class DonutDataset(Dataset):
    """Mỗi mẫu: pixel_values [3,H,W] và labels = ids của json2token(gt) + </s>.

    sort_json_key=False: giữ thứ tự key như trong gt_parse (khớp checkpoint chính thức CORD-v2,
    vốn sinh <s_menu> trước <s_sub_total>, <s_total>).

    Không đưa prompt token vào labels: VisionEncoderDecoderModel tự shift-right labels và chèn
    decoder_start_token_id (= <s_cord-v2>) ở đầu -> prompt không bị tính loss.
    """

    def __init__(self, hf_ds, processor, max_length: int, train: bool, augment=None, sort_json_key: bool = False):
        self.ds = hf_ds
        self.processor = processor
        self.max_length = max_length
        self.train = train
        self.augment = augment
        eos = processor.tokenizer.eos_token
        self.targets = [
            [json2token(p, sort_json_key) + eos for p in get_gt_parses(g)] for g in hf_ds["ground_truth"]
        ]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.ds[idx]
        image = sample["image"].convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        pixel_values = self.processor.image_processor(
            image, random_padding=self.train, return_tensors="pt"
        ).pixel_values[0]
        target = random.choice(self.targets[idx]) if self.train else self.targets[idx][0]
        labels = self.processor.tokenizer(
            target, add_special_tokens=False, max_length=self.max_length, truncation=True
        )["input_ids"]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels, dtype=torch.long)}


def collate_train(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Padding động cho labels bằng -100 (bỏ qua khi tính loss)."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    max_len = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, : len(b["labels"])] = b["labels"]
    return {"pixel_values": pixel_values, "labels": labels}
