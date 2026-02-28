import os
import pandas as pd
import joblib

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "nlp_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")
DATA_PATH = os.path.join(BASE_DIR, "unseen_test.csv")

# -------- Load model --------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -------- Load data --------

df = pd.read_csv(DATA_PATH)

texts = df["text"]
true_labels = df["true_category"]

# -------- Predict --------
X = vectorizer.transform(texts)
preds = model.predict(X)

# -------- Show results --------
correct = 0
total = len(preds)

print("\n=== Prediction Results ===\n")

for text, true, pred in zip(texts, true_labels, preds):
    status = "✅" if true == pred else "❌"
    print(f"{status}  {text} → predicted: {pred} | actual: {true}")
    if true == pred:
        correct += 1

accuracy = correct / total * 100

print("\n---------------------------")
print(f"Total samples: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print("---------------------------")
