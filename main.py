# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:07:40 2021

@author: User
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import model as model

app = Flask(__name__)
port = int(os.getenv("PORT", 8085))
model = pickle.load(open('.\model\modelrandomf.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    
    final_features = [np.array(int_features)]
    print('Caracteristicas finales ' , final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    output = 'El valor del m2 es de '+ str(output)
    return render_template('result.html', prediction_text= output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    
    
    app.run(host='localhost', port=port)