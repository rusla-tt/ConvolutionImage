# -*- encoding: utf-8 -*-
from flask import Flask, request, redirect, url_for, jsonify
import numpy as np
import lib.KerasNeuralNetwork as KNN
import lib.ImagePkl as IP
import lib.Marcov as marcov

app = Flask(__name__)

kemono = KNN.DeepLearning()
imagepkl = IP.ImagePkl()
sentence = marcov.Marcov()

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        img = request.files['img']
        img.save('./tmp/tmp.jpg')
        kemono_name = kemono.prediction()
        marcov_text = sentence.marcov_main(kemono_name[0], False, 90)
        rnn_text = kemono.prediction_rnn(kemono_name[0], 300)
        result = {
                "name": kemono_name[0],
                "ratio": str(kemono_name[1]),
                "marcov_text": marcov_text,
                "rnn_text": rnn_text
                }
        return jsonify(ResultSet=result)
    else:
        return jsonify(ResultSet={
            "status_code":"405",
            "message":"method not allowed"
            })

@app.route('/model', methods=['CREATE'])
def create():
    if request.method == 'CREATE':
        kemono.create_model()
        return jsonify(ResultSet={
            "status_code":"200",
            "message":"create model"
            })
    else:
        return jsonify(ResultSet={
            "status_code":"405",
            "message":"method not allowed"})

@app.route('/rnn/model', methods=['CREATE'])
def create_rnn():
    if request.method == 'CREATE':
        kemono.create_model_rnn(50)
        return jsonify(ResultSet={
            "status_code": "200",
            "message": "create model"
            })
    else:
        return jsonify(ResultSet={
            "status_code": "405",
            "message": "method not allowed"
            })


@app.route('/pkl', methods=['CREATE'])
def pkl():
    if request.method == 'CREATE':
        imagepkl.create_data_target()
        return jsonify(ResultSet={
            "status_code":"200",
            "message":"create pickle file"
            })
    else:
        return jsonify(ResultSet={
            "status_code":"405",
            "message":"method not allowed"
            })

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0')
