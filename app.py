
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("fraud_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    result = model.predict(features)
    return jsonify({"prediction": int(result[0])})

if __name__ == "__main__":
    app.run(debug=True)
