# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:56:27 2023

@author: God
"""

import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='Forecasted Quantity is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)