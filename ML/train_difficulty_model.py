"""
train_difficulty_model.py
Place this in: ML/ (or scripts/)
Run: python ML/train_difficulty_model.py

Trains a difficulty classifier on skill_test_dataset.csv and saves:
  - models/difficulty_model.pkl
  - data/questions_metadata.csv
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "skill_test_dataset.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
META_PATH = os.path.join(ROOT_DIR, "data", "questions_metadata.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Helper: Feature Engineering ───────────────────────────────────────────────
TECH_KEYWORDS = [
    "algorithm", "complexity", "time complexity", "space complexity",
    "recursion", "iteration", "pointer", "memory", "heap", "stack",
    "binary", "neural", "gradient", "backpropagation", "kernel",
    "eigenvalue", "normalization", "regularization", "loss", "epoch",
    "thread", "mutex", "concurrency", "asynchronous", "prototype",
    "inheritance", "polymorphism", "encapsulation", "abstraction",
    "sql", "join", "index", "transaction", "acid", "schema", "query",
    "layer", "activation", "dropout", "convolution", "embedding",
    "token", "bigO", "hash", "graph", "tree", "dynamic programming",
]

def count_tech_keywords(text: str) -> int:
    text_lower = text.lower()
    return sum(1 for kw in TECH_KEYWORDS if kw in text_lower)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric features derived from question text + options."""
    df = df.copy()
    # Combine question + all options into one blob for TF-IDF
    df["full_text"] = (
        df["question_text"].fillna("") + " " +
        df["option_a"].fillna("") + " " +
        df["option_b"].fillna("") + " " +
        df["option_c"].fillna("") + " " +
        df["option_d"].fillna("")
    )
    df["q_word_count"]     = df["question_text"].apply(lambda x: len(str(x).split()))
    df["q_char_count"]     = df["question_text"].apply(len)
    df["tech_keyword_cnt"] = df["question_text"].apply(count_tech_keywords)
    df["has_code_hint"]    = df["question_text"].apply(
        lambda x: int(bool(re.search(r"[(){}\[\]<>]|->|=>|::|def |class |SELECT|WHERE|JOIN", str(x))))
    )
    return df


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} questions from {DATA_PATH}")
    print(f"Class distribution:\n{df['difficulty'].value_counts()}\n")
    df["question_id"] = range(1, len(df) + 1)
    return df


def train_and_evaluate(df):
    df = build_features(df)

    X_text = df["full_text"]
    X_meta = df[["q_word_count", "q_char_count", "tech_keyword_cnt", "has_code_hint"]]
    y      = df["difficulty"]

    X_train_txt, X_test_txt, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
        X_text, X_meta, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Model 1: Logistic Regression (baseline) ────────────────────────────
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
    ])
    lr_pipe.fit(X_train_txt, y_train)
    lr_preds = lr_pipe.predict(X_test_txt)
    lr_acc   = accuracy_score(y_test, lr_preds)
    print("── Logistic Regression ──")
    print(f"Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr_preds))

    # ── Model 2: RandomForestClassifier (primary) ──────────────────────────
    # Why RF over LR?
    # • Handles non-linear relationships between features naturally
    # • Less sensitive to feature scaling — no need to normalize word counts
    # • Provides feature importance for interpretability
    # • More robust to class imbalance with class_weight="balanced"
    rf_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf",   RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train_txt, y_train)
    rf_preds = rf_pipe.predict(X_test_txt)
    rf_acc   = accuracy_score(y_test, rf_preds)
    print("── Random Forest ──")
    print(f"Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf_preds))

    # Choose the better model
    best_model = rf_pipe if rf_acc >= lr_acc else lr_pipe
    best_name  = "RandomForest" if rf_acc >= lr_acc else "LogisticRegression"
    print(f"\n✅ Best model: {best_name} (accuracy={max(rf_acc, lr_acc):.4f})")

    return best_model, df


def save_artifacts(model, df):
    # Save model
    model_path = os.path.join(MODEL_DIR, "difficulty_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved → {model_path}")

    # Save questions metadata
    meta = df[["question_id", "difficulty", "skill_tested",
               "question_text", "option_a", "option_b", "option_c", "option_d", "correct_answer"]].copy()
    meta.to_csv(META_PATH, index=False)
    print(f"✅ Metadata saved → {META_PATH}")


if __name__ == "__main__":
    df           = load_data()
    model, df    = train_and_evaluate(df)
    save_artifacts(model, df)
    print("\nDone! Run scripts/predict_difficulty.py to test prediction on new questions.")