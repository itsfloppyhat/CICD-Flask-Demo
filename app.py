
"""Flask API that serves predictions from a trained model."""
import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model once at startup (not on every request)
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint. Cloud Run uses this to know the app is alive."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept flower measurements, return a prediction.

    Expected JSON body:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    The four values are: sepal_length, sepal_width, petal_length, petal_width
    """
    data = request.get_json()

    # Validate input
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    features = data["features"]
    if len(features) != 4:
        return jsonify({"error": "Expected 4 features, got " + str(len(features))}), 400

    # Predict
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return jsonify({
        "prediction": IRIS_CLASSES[prediction],
        "confidence": round(float(max(probabilities)), 4),
        "probabilities": {
            name: round(float(prob), 4)
            for name, prob in zip(IRIS_CLASSES, probabilities)
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)