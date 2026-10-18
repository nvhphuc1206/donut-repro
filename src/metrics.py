"""Thước đo cho Donut: field-level F1, TED-based accuracy (nTED), ANLS.

Class JSONParseEvaluator được port nguyên vẹn từ clovaai/donut (donut/util.py),
MIT License, Copyright (c) 2022-present NAVER Corp.
Mọi thí nghiệm trong dự án đều dùng DUY NHẤT evaluator này.
"""
from typing import Any, Dict, List, Union

import zss
from nltk import edit_distance
from zss import Node

# nltk >= 3.9.x giới hạn edit_distance ở 2000 ký tự (chống DoS). Model đầu quá trình train hay sinh chuỗi
# lặp rất dài -> ValueError làm dừng train. Code gốc của Donut không có giới hạn này -> bỏ giới hạn
# để giữ nguyên cách tính metric.
if "MAX_DISTANCE_INPUT_LEN" in edit_distance.__globals__:
    edit_distance.__globals__["MAX_DISTANCE_INPUT_LEN"] = 10 ** 9


class JSONParseEvaluator:
    """Calculate n-TED (Normalized Tree Edit Distance) based accuracy and F1 accuracy score."""

    @staticmethod
    def flatten(data: dict):
        """{"menu": [{"nm": ["cake"]}, ...]} -> [("menu.nm", "cake"), ...]"""
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(node1: Node, node2: Node):
        label1 = node1.label
        label2 = node2.label
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf and label2_leaf:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif not label1_leaf and label2_leaf:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf and not label2_leaf:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """Sort by key, while iterate over element if data is list."""
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key in sorted(data.keys(), key=lambda k: (len(k), k)):
                value = self.normalize_dict(data[key])
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
            else:
                new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
        else:
            new_data = [str(data).strip()]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]):
        """Global field-level F1 (micro-averaged)."""
        total_tp, total_fn_or_fp = 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fn_or_fp += 1
            total_fn_or_fp += len(answer)
        return total_tp / (total_tp + total_fn_or_fp / 2)

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(item, "<subtree>")
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict):
        """nTED-based accuracy = max(0, 1 - TED(pred, gt) / TED(empty, gt))."""
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )


# ----------------------------------------------------------------------------
# Hàm tiện ích (không thuộc code gốc)
# ----------------------------------------------------------------------------
_EVALUATOR = JSONParseEvaluator()


def sample_f1(pred: dict, gt: dict) -> float:
    """F1 của một mẫu (an toàn khi cả hai đều rỗng)."""
    try:
        return _EVALUATOR.cal_f1([pred], [gt])
    except ZeroDivisionError:
        return 1.0


def score_predictions(preds: List[dict], gts: List[dict]) -> Dict[str, Any]:
    """Trả về F1 tổng (micro), TED Acc trung bình và TED Acc từng mẫu."""
    teds = [_EVALUATOR.cal_acc(p, g) for p, g in zip(preds, gts)]
    try:
        f1 = _EVALUATOR.cal_f1(preds, gts)
    except ZeroDivisionError:
        f1 = 0.0
    return {"f1": f1, "ted_acc": sum(teds) / max(len(teds), 1), "ted_per_sample": teds}


def anls(pred: str, answers: List[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity cho một câu hỏi (DocVQA)."""
    pred = " ".join(str(pred).strip().lower().split())
    best = 0.0
    for ans in answers:
        ans = " ".join(str(ans).strip().lower().split())
        if not ans and not pred:
            return 1.0
        dist = edit_distance(pred, ans)
        nl = dist / max(len(pred), len(ans), 1)
        best = max(best, 1 - nl if nl < threshold else 0.0)
    return best
