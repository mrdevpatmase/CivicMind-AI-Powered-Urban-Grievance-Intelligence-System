import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ---------------- TEXT CLEANING ----------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- TRAINING FUNCTION ----------------
def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "complaints.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # -------- Load data --------
    df = pd.read_csv(DATA_PATH)

    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

    X = df["text"]
    y = df["category"]

    # -------- Train/Test split --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------- TF-IDF Vectorizer --------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=6000,      # slightly increased
        min_df=2                # ignore rare noise words
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # -------- Logistic Regression with class balancing --------
    model = LogisticRegression(
        max_iter=1500,
        n_jobs=-1,
        class_weight="balanced"
    )

    # -------- Cross-validation (REAL accuracy) --------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=skf, scoring="accuracy")

    print("\n🔎 Cross-Validation Accuracy Scores:", cv_scores)
    print("📊 Mean CV Accuracy:", round(cv_scores.mean(), 4))

    # -------- Train final model --------
    model.fit(X_train_vec, y_train)

    # -------- Test set evaluation --------
    y_pred = model.predict(X_test_vec)

    print("\n🎯 Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

    # -------- Save model --------
    joblib.dump(model, os.path.join(MODEL_DIR, "nlp_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    print(f"\n✅ Models saved to: {MODEL_DIR}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    train()
