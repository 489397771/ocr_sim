# -*- coding: utf-8 -*-
import json

from flask import Flask, request, jsonify

from idcard import ocr_id_card
from lawyerLicense import ocr_bar_license

app = Flask(__name__)


@app.route('/idCard', methods=["POST"])
def id_card():
    # 接收图片
    upload_file = request.files['img']
    # 获取图片路径
    img_path = upload_file.img_path
    # 模型识别返回结果
    result = ocr_id_card(img_path)
    json_info = json.dumps(result, ensure_ascii=False)
    return json_info


@app.route('/barLicence', methods=["POST"])
def bar_licence():
    # 接收图片
    upload_file = request.files['img']
    # 获取图片路径
    img_path = upload_file.img_path
    # 模型识别返回结果
    result = ocr_bar_license(img_path)
    json_info = json.dumps(result, ensure_ascii=False)
    return json_info


if __name__ == "__main__":
    app.run("0.0.0.0", 8080, debug=False)
