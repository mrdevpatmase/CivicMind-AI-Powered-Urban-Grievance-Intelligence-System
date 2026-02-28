import os
import re
import joblib
from scipy.sparse import hstack
from flask import Flask, request, jsonify
from severity import calculate_severity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "nlp_model.pkl")
WORD_VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")
CHAR_VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "char_vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
    char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
except Exception as e:
    model = None
    word_vectorizer = None
    char_vectorizer = None
    print("❌ Model loading failed:", e)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AI Grievance API running"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Text field is required"}), 400

    text = data["text"].strip()

    if text == "":
        return jsonify({"error": "Empty text provided"}), 400

    text = clean_text(text)

    # -------- Correct Feature Pipeline --------
    X_word = word_vectorizer.transform([text])
    X_char = char_vectorizer.transform([text])
    X_final = hstack([X_word, X_char])

    probs = model.predict_proba(X_final)[0]

    category = model.classes_[probs.argmax()]
    confidence = float(probs.max())

    severity = calculate_severity(category)

    return jsonify({
        "category": category,
        "confidence": round(confidence, 3),
        "severity": severity
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)