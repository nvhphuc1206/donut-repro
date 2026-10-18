"""Chuyển đổi JSON <-> chuỗi token cho Donut.

Port từ clovaai/donut (donut/model.py: json2token / token2json),
MIT License, Copyright (c) 2022-present NAVER Corp.
và notebook fine-tune Donut của HuggingFace (N. Rogge).
"""
import re
from typing import Any, Iterable, List, Optional, Set

SEP_TOKEN = "<sep/>"


def collect_keys(obj: Any, keys: Optional[Set[str]] = None) -> Set[str]:
    """Thu thập mọi key (ở mọi cấp) trong một cây JSON."""
    if keys is None:
        keys = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            collect_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            collect_keys(item, keys)
    return keys


def key_tokens(keys: Iterable[str]) -> List[str]:
    """{"nm", "price"} -> ["<s_nm>", "</s_nm>", "<s_price>", "</s_price>"]."""
    tokens = []
    for k in sorted(keys):
        tokens += [f"<s_{k}>", f"</s_{k}>"]
    return tokens


def json2token(obj: Any, sort_json_key: bool = True, categorical_tokens: Optional[Set[str]] = None) -> str:
    """{"menu": {"nm": "Tea"}} -> "<s_menu><s_nm>Tea</s_nm></s_menu>".

    Giống code gốc: key được sắp xếp giảm dần (reverse=True), phần tử list nối bằng <sep/>.
    """
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        return "".join(
            f"<s_{k}>" + json2token(obj[k], sort_json_key, categorical_tokens) + f"</s_{k}>" for k in keys
        )
    if isinstance(obj, list):
        return SEP_TOKEN.join(json2token(item, sort_json_key, categorical_tokens) for item in obj)
    obj = str(obj)
    if categorical_tokens and f"<{obj}/>" in categorical_tokens:
        obj = f"<{obj}/>"  # token phân loại, ví dụ <advertisement/>
    return obj


def token2json(tokens: str, added_vocab: Optional[Iterable[str]] = None, is_inner_value: bool = False):
    """Chuyển chuỗi token (đã bỏ prompt/eos/pad) thành JSON.

    Nếu chuỗi sai cấu trúc (thiếu thẻ đóng), field tương ứng bị bỏ qua — đúng hành vi code gốc.
    """
    added_vocab = set(added_vocab or [])
    output = {}
    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        end_token = re.search(rf"</s_{re.escape(key)}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
            continue
        end_token = end_token.group()
        content = re.search(f"{re.escape(start_token)}(.*?){re.escape(end_token)}", tokens, re.IGNORECASE)
        if content is not None:
            content = content.group(1).strip()
            if "<s_" in content and "</s_" in content:  # node trong
                value = token2json(content, added_vocab, is_inner_value=True)
                if value:
                    if len(value) == 1:
                        value = value[0]
                    output[key] = value
            else:  # node lá
                output[key] = []
                for leaf in content.split(SEP_TOKEN):
                    leaf = leaf.strip()
                    if leaf in added_vocab and leaf[:1] == "<" and leaf[-2:] == "/>":
                        leaf = leaf[1:-2]  # token phân loại
                    output[key].append(leaf)
                if len(output[key]) == 1:
                    output[key] = output[key][0]

        tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
        if tokens[:6] == SEP_TOKEN:  # phần tử kế tiếp của list
            return [output] + token2json(tokens[6:], added_vocab, is_inner_value=True)

    if len(output):
        return [output] if is_inner_value else output
    return [] if is_inner_value else {"text_sequence": tokens}
