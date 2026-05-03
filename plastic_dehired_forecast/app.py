import os
import numpy as np
import sqlite3
import pickle
from flask import Flask, request, render_template

# -------------------------------
# App Init
# -------------------------------
app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "dehired_forecast.db")

# -------------------------------
# Load Model from SQLite
# -------------------------------
def load_model_from_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT model FROM model_store LIMIT 1")
        row = cursor.fetchone()

        conn.close()

        if row is None:
            raise ValueError("No model found in DB")

        return pickle.loads(row[0])

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


model = load_model_from_db()

# -------------------------------
# Utility Functions
# -------------------------------
def validate_input(form_data):
    try:
        values = [float(x) for x in form_data.values()]
        if len(values) != 4:
            raise ValueError("Expected 4 input features")
        return values, None
    except Exception as e:
        return None, str(e)


def prepare_features(values):
    return np.array(values).reshape(1, -1)


def predict_quantity(features):
    prediction = model.predict(features)
    return round(float(prediction[0]), 2)


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values, error = validate_input(request.form)

    if error:
        return render_template(
            "index.html",
            prediction_text=f"❌ Error: {error}"
        )

    features = prepare_features(values)
    result = predict_quantity(features)

    return render_template(
        "index.html",
        prediction_text=f"✅ Forecasted Dehired Quantity: {result}"
    )


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, use_reloader=False)