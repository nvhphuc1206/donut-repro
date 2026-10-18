"""Kiểm tra nhanh (CPU, không tải model) các hàm thuần logic: json<->token, metric, hậu xử lý.

    python -m src.selftest
"""
from src.json_tokens import json2token, token2json
from src.metrics import JSONParseEvaluator, anls, score_predictions
from src.postprocess import filter_schema, fix_structure, normalize_value, schema_from_gts


def main():
    gt = {
        "menu": [{"nm": "Lemon Tea", "cnt": "1", "price": "25.000"}, {"nm": "Bread", "cnt": "2", "price": "10.000"}],
        "total": {"total_price": "45.000"},
    }
    seq = json2token(gt)
    print("json2token:", seq)
    back = token2json(seq)
    assert back == gt, back
    print("token2json round-trip: OK")

    ev = JSONParseEvaluator()
    assert abs(ev.cal_acc(gt, gt) - 1.0) < 1e-9
    assert abs(ev.cal_f1([gt], [gt]) - 1.0) < 1e-9
    wrong = {"menu": [{"nm": "Lemon Tea", "cnt": "1", "price": "25.000"}], "total": {"total_price": "45.00"}}
    sc = score_predictions([wrong], [gt])
    print(f"Dự đoán thiếu 1 món + sai 1 ký tự: F1={sc['f1']:.3f} TED Acc={sc['ted_acc']:.3f}")
    assert 0 < sc["ted_acc"] < 1 and 0 < sc["f1"] < 1

    broken = "<s_total><s_total_price>45.000</s_total><s_menu><s_nm>Lemon Tea<s_price>25.000</s_price></s_menu>"
    paths, leaf_keys = schema_from_gts([gt])
    fixed = fix_structure(broken, leaf_keys)
    print("fix_structure:", fixed)
    parsed = token2json(fixed)
    assert parsed["total"]["total_price"] == "45.000" and parsed["menu"]["nm"] == "Lemon Tea", parsed
    assert filter_schema({"menu": {"nm": "a", "xyz": "b"}}, paths) == {"menu": {"nm": "a"}}
    assert normalize_value("25,000") == normalize_value("25.000") == "25000"
    assert normalize_value("x 1") == "x1"
    assert anls("hello", ["hello"]) == 1.0 and anls("abc", ["xyz"]) == 0.0
    print("Tất cả kiểm tra: OK")


if __name__ == "__main__":
    main()
