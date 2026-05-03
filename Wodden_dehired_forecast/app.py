import os
import numpy as np
import pickle
from flask import Flask, request, render_template

# -------------------------------
# App Initialization
# -------------------------------
app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model1.pkl")

# Load model safely
def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")

model = load_model(MODEL_PATH)

# -------------------------------
# Utility Functions
# -------------------------------
def validate_input(form_data):
    """
    Validate incoming form data
    """
    try:
        values = [float(x) for x in form_data.values()]
        if len(values) != 4:
            raise ValueError("Expected 4 input features")
        return values, None
    except Exception as e:
        return None, str(e)


def prepare_features(values):
    """
    Convert input into model-ready format
    """
    return np.array(values).reshape(1, -1)


def predict_quantity(features):
    """
    Model prediction wrapper
    """
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
        prediction_text=f"✅ Forecasted Quantity: {result}"
    )


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, use_reloader=False)