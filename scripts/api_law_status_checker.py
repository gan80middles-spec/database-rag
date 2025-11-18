# app.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS  # 可选，如果前端跨域访问就用

# 从你的 detect.py 里导入
from detect import analyze_law_text


app = Flask(__name__)
CORS(app)  

# ====== 从 bytes 中尝试多种编码解码 ======

def read_text_from_bytes(data: bytes, encodings: List[str] | None = None) -> str:
    if encodings is None:
        encodings = ["utf-8", "gb18030", "gbk"]

    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue

    return data.decode("latin-1", errors="ignore")


# ====== 健康检查 ======

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


# ====== 直接传文本的接口 ======
#
# POST /analyze/text
# JSON body: { "text": "……全文……" }
# Response: { "modified_laws": [...], "repealed_laws": [...] }

@app.route("/analyze/text", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": '请求 JSON 里需要字段 "text"'}), 400

    text = data.get("text") or ""
    if not text.strip():
        return jsonify({"error": "text 为空"}), 400

    result = analyze_law_text(text)
    # result 已经是 {"modified_laws": [...], "repealed_laws": [...]}
    return jsonify(result)


if __name__ == "__main__":
    # 开发环境可以打开 debug，线上建议关掉
    app.run(host="0.0.0.0", port=8001, debug=True)
