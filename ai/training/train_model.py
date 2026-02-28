import os
import re
import pandas as pd
import joblib
from scipy.sparse import hstack

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
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "complaints_large.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Clean text
    df["text"] = df["text"].astype(str).apply(clean_text)

    X = df["text"]
    y = df["category"]

    print("\n📊 Class Distribution:\n", y.value_counts())

    # -------- Train/Test split --------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------- Word-level TF-IDF --------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=12000,
        min_df=2,
        sublinear_tf=True
    )

    # -------- Character-level TF-IDF --------
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=8000
    )

    print("\n🔄 Vectorizing text...")

    X_train_word = vectorizer.fit_transform(X_train)
    X_test_word = vectorizer.transform(X_test)

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)

    # Combine word + char features
    X_train_vec = hstack([X_train_word, X_train_char])
    X_test_vec = hstack([X_test_word, X_test_char])

    print("✅ Feature shape:", X_train_vec.shape)

    # -------- Logistic Regression --------
    model = LogisticRegression(
        max_iter=2000,
        C=2,
        solver="lbfgs",
        class_weight="balanced"
    )

    # -------- Cross-validation --------
    print("\n🔎 Running Cross-Validation...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=skf, scoring="accuracy")

    print("📊 CV Scores:", cv_scores)
    print("📊 Mean CV Accuracy:", round(cv_scores.mean(), 4))

    # -------- Train Final Model --------
    print("\n🚀 Training final model...")
    model.fit(X_train_vec, y_train)

    # -------- Test Evaluation --------
    y_pred = model.predict(X_test_vec)

    print("\n🎯 Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\n📄 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # -------- Save Models --------
    joblib.dump(model, os.path.join(MODEL_DIR, "nlp_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(char_vectorizer, os.path.join(MODEL_DIR, "char_vectorizer.pkl"))

    print("\n✅ Model and vectorizers saved to:", MODEL_DIR)


# ---------------- RUN ----------------
if __name__ == "__main__":
    train()