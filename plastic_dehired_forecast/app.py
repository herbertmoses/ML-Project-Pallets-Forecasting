from flask import Flask, request, render_template
import numpy as np
import sqlite3
import pickle

app = Flask(__name__)

# Load model from SQLite
def load_model_from_db():
    conn = sqlite3.connect('dehired_forecast.db')
    cursor = conn.cursor()
    cursor.execute('SELECT model FROM model_store LIMIT 1')
    model_blob = cursor.fetchone()[0]
    conn.close()
    return pickle.loads(model_blob)

model = load_model_from_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text='Forecasted Quantity is {:.2f}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
