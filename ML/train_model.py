import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "careers.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "career_model.pkl")
MLB_PATH   = os.path.join(MODEL_DIR, "mlb.pkl")


def train():
    random.seed(42)

    df = pd.read_csv(DATA_PATH)
    df["required_skills"] = df["required_skills"].fillna("").str.lower()

    career_skills = {}
    for _, row in df.iterrows():
        skills = [s.strip() for s in row["required_skills"].split(",") if s.strip()]
        career_skills[row["career_title"]] = skills

    careers = df["career_title"].tolist()

    # Generate synthetic training samples
    X_samples, y_labels = [], []
    for career, req_skills in career_skills.items():
        if not req_skills:
            continue
        # Core samples
        for _ in range(30):
            X_samples.append(req_skills)
            y_labels.append(career)
        # Partial samples
        for _ in range(50):
            k = max(1, int(len(req_skills) * random.uniform(0.5, 0.85)))
            X_samples.append(random.sample(req_skills, k))
            y_labels.append(career)
        # Noise samples
        other_skills = [s for c, skills in career_skills.items()
                        if c != career for s in skills]
        for _ in range(20):
            n_extra = random.randint(1, 4)
            extra = random.sample(other_skills, min(n_extra, len(other_skills)))
            X_samples.append(req_skills + extra)
            y_labels.append(career)

    # Encode features
    mlb = MultiLabelBinarizer()
    X   = mlb.fit_transform(X_samples)
    y   = np.array(y_labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(mlb,   MLB_PATH)

    return {
        "status":        "success",
        "accuracy":      round(acc * 100, 2),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "n_careers":     len(career_skills),
    }


# ─── Only runs when you execute this file directly ────────────────────────────
if __name__ == "__main__":
    print("Training model...")
    result = train()
    print(f"Done! Accuracy: {result['accuracy']}%")
    print(f"Saved to models/career_model.pkl")