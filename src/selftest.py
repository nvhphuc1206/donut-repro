"""Kiểm tra nhanh (CPU, không tải model) các hàm thuần logic: json<->token, metric, chấm điểm bộ ngoài, bước cắt.

    python -m src.selftest
"""
import numpy as np
from PIL import Image, ImageDraw

from src.ext_eval import apply_ext_eval
from src.json_tokens import json2token, token2json
from src.metrics import JSONParseEvaluator, score_predictions
from src.preprocess import DocPreprocessor


def main():
    gt = {
        "menu": [{"nm": "Lemon Tea", "cnt": "1", "price": "25.000"}, {"nm": "Bread", "cnt": "2", "price": "10.000"}],
        "total": {"total_price": "45.000"},
    }
    seq = json2token(gt)
    print("json2token:", seq)
    assert token2json(seq) == gt
    print("token2json round-trip: OK")

    ev = JSONParseEvaluator()
    assert abs(ev.cal_acc(gt, gt) - 1.0) < 1e-9
    assert abs(ev.cal_f1([gt], [gt]) - 1.0) < 1e-9
    wrong = {"menu": [{"nm": "Lemon Tea", "cnt": "1", "price": "25.000"}], "total": {"total_price": "45.00"}}
    sc = score_predictions([wrong], [gt])
    print(f"Dự đoán thiếu 1 món + sai 1 ký tự: F1={sc['f1']:.3f} TED Acc={sc['ted_acc']:.3f}")
    assert 0 < sc["ted_acc"] < 1 and 0 < sc["f1"] < 1

    # chấm bộ ngoài CORD: lọc key ngoài schema, bỏ khoảng trắng + ký hiệu tiền tệ
    recs = [{"pred": {"total": {"total_price": "$ 7.60", "cashprice": "10.00"}, "menu": {"nm": "A B"}},
             "gt": {"total": {"total_price": "7.60"}}}]
    info = apply_ext_eval(recs)
    assert recs[0]["pred"] == {"total": {"total_price": "7.60"}} and info["total_acc"] == 1.0, recs[0]["pred"]
    print("ext_eval: OK")

    # bước cắt: ảnh trắng lớn có một khối "chữ" nhỏ ở giữa -> phải cắt sát khối chữ, không mất chữ
    img = Image.new("RGB", (1600, 2400), "white")
    d = ImageDraw.Draw(img)
    for row in range(15):
        for col in range(6):
            x, y = 600 + col * 70, 900 + row * 50
            d.rectangle([x, y, x + 50, y + 16], outline="black", width=3)
    out, info = DocPreprocessor("crop")(img)
    x0, y0, x1, y1 = info["box"]
    assert info["status"] == "cropped" and x0 <= 600 and y0 <= 900 and x1 >= 1000 and y1 >= 1616, info
    assert info["gain"] > 1.5, info
    assert np.asarray(DocPreprocessor("none")(img)[0]).shape == np.asarray(img).shape
    print(f"preprocess crop: OK (box={info['box']}, gain={info['gain']})")
    print("Tất cả kiểm tra: OK")


if __name__ == "__main__":
    main()
