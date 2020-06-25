from flask import Flask
from ocr import *
from crime_qa import *
from crime_classify import *
from question_classify import *
from flask import request, jsonify, render_template

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/question/<question>')
def gg(question):
    final_answer = handler.search_main(question)
    # huida = ''.join(final_answer).replace('#def::','')
    print('answers:', ''.join(str(final_answer[1])))
    answer = ''.join(str(final_answer))
    return answer


@app.route('/crimec', methods=['GET', 'POST'])
def class_input():
    try:
        data = request.get_json()
        input = data.get("input")
    except:
        input = request.args.get("input", "")
    label = handler1.predict(input)
    result = {
        "class": label

    }
    return jsonify(result)


@app.route('/matchQa', methods=['GET', 'POST'])
def match_question():
    try:
        data = request.get_json()
        question = data.get("question")
    except:
        question = request.args.get("question", "")

    final_answer, sim1, sim2, sim3 = handler.search_main(question)
    huida = ''.join(final_answer).replace('#def::', '')
    print('answers:', huida)
    print('sim', sim1)
    answer = str(huida)
    result = {
        "answers": answer,
        "相似问题推荐1": sim1.get("sim_question").replace('"', ''),
        "相似问题推荐2": sim2.get("sim_question").replace('"', ''),
        "相似问题推荐3": sim3.get("sim_question").replace('"', ''),

    }
    return jsonify(result)


@app.route('/QA', methods=['GET', 'POST'])
def qa_input():
    try:
        data = request.get_json()
        qa = data.get("qa")
    except:
        qa = request.args.get("qa", "")
    label, prob = handler2.predict(qa)
    result = {
        "label": label,
        "prob": prob
    }
    return jsonify(result)


@app.route('/hello', methods=['GET', 'POST'])
def hello(answer=None, sim1=None):
    try:
        data = request.get_json()
        question = data.get("question")
    except:
        question = request.args.get("question", "")

    final_answer, sim1, sim2, sim3 = handler.search_main(question)
    huida = ''.join(final_answer).replace('#def::', '')
    print('answers:', huida)
    print('sim', sim1)
    answer = str(huida)
    result = {
        "answers": answer,
        "相似问题推荐1": sim1.get("sim_question").replace('"', ''),
        "相似问题推荐2": sim2.get("sim_question").replace('"', ''),
        "相似问题推荐3": sim3.get("sim_question").replace('"', ''),

    }
    # return jsonify(result)
    return render_template('index.html', name=answer, sim1=sim1)


#
handler1 = CrimeClassify()
handler = CrimeQA()
handler2 = QuestionClassify()

app.run(host='0.0.0.0')
