import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "complaints.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, os.path.join(MODEL_DIR, "nlp_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    print(f"\nModels saved to {MODEL_DIR}")


if __name__ == "__main__":
    train()
