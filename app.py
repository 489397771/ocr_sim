# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify

from idcard import ocr_id_card
from lawyerLicense import ocr_bar_license

app = Flask(__name__)


@app.route('/idCard', methods=["POST"])
def id_card():
    data = request.get_json()
    img = data.get('path')
    return jsonify(ocr_id_card(img))


@app.route('/barLicence', methods=["POST"])
def bar_licence():
    data = request.get_json()
    img = data.get("path")
    return jsonify(ocr_bar_license(img))


if __name__ == "__main__":
    app.run("0.0.0.0", 8080, debug=False)
