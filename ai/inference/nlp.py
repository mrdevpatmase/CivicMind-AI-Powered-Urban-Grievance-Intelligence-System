import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "nlp_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

def predict_category(text: str):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = float(max(model.predict_proba(vec)[0]))
    return pred, round(prob, 2)
