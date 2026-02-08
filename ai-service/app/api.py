import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "nlp_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    model = None
    vectorizer = None
    print("❌ Model loading failed:", e)


# ---------- Health check ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AI Grievance API running"})


# ---------- Prediction ----------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Text field is required"}), 400

    text = data["text"].strip()

    if text == "":
        return jsonify({"error": "Empty text provided"}), 400

    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]

    category = model.classes_[probs.argmax()]
    confidence = float(probs.max())

    return jsonify({
        "category": category,
        "confidence": round(confidence, 3)
    })


# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
