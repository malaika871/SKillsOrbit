import os
import joblib
import json
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

# ─── Paths ────────────────────────────────────────────────────────────────────
from ML.config import ONET_DIR, MODEL_DIR

MODEL_PATH   = os.path.join(MODEL_DIR, "career_model.pkl")
MLB_PATH     = os.path.join(MODEL_DIR, "mlb.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
SELECTOR_PATH = os.path.join(MODEL_DIR, "selector.pkl")


def load_onet_data():
    print("📂 Loading O*NET 29.0 .txt files...")

    occ = pd.read_csv(
        os.path.join(ONET_DIR, "Occupation Data.txt"),
        sep="\t", usecols=["O*NET-SOC Code", "Title"], dtype=str
    ).rename(columns={"O*NET-SOC Code": "soc_code", "Title": "career_title"})

    skills = pd.read_csv(
        os.path.join(ONET_DIR, "Skills.txt"),
        sep="\t",
        usecols=["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"],
        dtype={"O*NET-SOC Code": str, "Element Name": str,
               "Scale ID": str, "Data Value": float}
    ).rename(columns={
        "O*NET-SOC Code": "soc_code", "Element Name": "skill",
        "Scale ID": "scale_id", "Data Value": "score"
    })
    skills = skills[skills["scale_id"] == "IM"][["soc_code", "skill", "score"]]

    know_path = os.path.join(ONET_DIR, "Knowledge.txt")
    if os.path.exists(know_path):
        know = pd.read_csv(
            know_path, sep="\t",
            usecols=["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"],
            dtype={"O*NET-SOC Code": str, "Element Name": str,
                   "Scale ID": str, "Data Value": float}
        ).rename(columns={
            "O*NET-SOC Code": "soc_code", "Element Name": "skill",
            "Scale ID": "scale_id", "Data Value": "score"
        })
        know = know[know["scale_id"] == "IM"][["soc_code", "skill", "score"]]
        # Prefix knowledge to avoid name collision with skills
        know["skill"] = "know_" + know["skill"].str.strip().str.lower()
        skills = pd.concat([skills, know], ignore_index=True)
        print("   ✅ Knowledge.txt loaded")

    # ── SKIP Technology Skills ──────────────────────────────────────────────
    # Technology Skills.txt is the main source of 8000+ noisy features.
    # Specific tool names ("Adobe Photoshop 2024") are too sparse to help.
    # We only add them if Hot Technology = Yes (actually in-demand tools).
    tech_path = os.path.join(ONET_DIR, "Technology Skills.txt")
    if os.path.exists(tech_path):
        tech = pd.read_csv(tech_path, sep="\t", dtype=str)
        tech.columns = tech.columns.str.strip()
        # Only keep Hot Technologies — these are universally signal-rich
        if "Hot Technology" in tech.columns:
            tech = tech[tech["Hot Technology"].str.strip().str.upper() == "Y"]
        tech = tech.rename(columns={"O*NET-SOC Code": "soc_code", "Example": "skill"})
        tech = tech.dropna(subset=["skill"])
        tech["skill"] = "tech_" + tech["skill"].str.strip().str.lower()
        tech["score"] = 4.0
        skills = pd.concat([skills, tech[["soc_code", "skill", "score"]]], ignore_index=True)
        print(f"   ✅ Technology Skills.txt loaded (hot techs only: {len(tech)} rows)")

    # ── Add Abilities.txt for richer signal ─────────────────────────────────
    abilities_path = os.path.join(ONET_DIR, "Abilities.txt")
    if os.path.exists(abilities_path):
        ab = pd.read_csv(
            abilities_path, sep="\t",
            usecols=["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"],
            dtype={"O*NET-SOC Code": str, "Element Name": str,
                   "Scale ID": str, "Data Value": float}
        ).rename(columns={
            "O*NET-SOC Code": "soc_code", "Element Name": "skill",
            "Scale ID": "scale_id", "Data Value": "score"
        })
        ab = ab[ab["scale_id"] == "IM"][["soc_code", "skill", "score"]]
        ab["skill"] = "ability_" + ab["skill"].str.strip().str.lower()
        skills = pd.concat([skills, ab], ignore_index=True)
        print("   ✅ Abilities.txt loaded")

    # ── Add Work Activities.txt ──────────────────────────────────────────────
    wa_path = os.path.join(ONET_DIR, "Work Activities.txt")
    if os.path.exists(wa_path):
        wa = pd.read_csv(
            wa_path, sep="\t",
            usecols=["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"],
            dtype={"O*NET-SOC Code": str, "Element Name": str,
                   "Scale ID": str, "Data Value": float}
        ).rename(columns={
            "O*NET-SOC Code": "soc_code", "Element Name": "skill",
            "Scale ID": "scale_id", "Data Value": "score"
        })
        wa = wa[wa["scale_id"] == "IM"][["soc_code", "skill", "score"]]
        wa["skill"] = "activity_" + wa["skill"].str.strip().str.lower()
        skills = pd.concat([skills, wa], ignore_index=True)
        print("   ✅ Work Activities.txt loaded")

    skills["skill"] = skills["skill"].str.strip().str.lower()
    print(f"   ✅ {len(occ)} occupations | {len(skills)} total skill records")
    return occ, skills


def build_skill_dict(occ, skills_df, importance_threshold=2.5):
    """
    Build per-career skill dict. Uses SCORE-WEIGHTED features, not just binary.
    Higher threshold (2.5 → 3.0) cuts noise from rarely-important skills.
    """
    print("⚙️  Building skill dictionary (threshold=2.5)...")

    skills_df = skills_df[skills_df["score"] >= importance_threshold].copy()
    merged = skills_df.merge(occ, on="soc_code", how="inner")

    career_skills = {}
    for title, group in merged.groupby("career_title"):
        deduped = group.groupby("skill")["score"].max().reset_index()
        skill_list = list(zip(deduped["skill"], deduped["score"]))
        if len(skill_list) >= 5:
            career_skills[title] = skill_list

    print(f"   ✅ {len(career_skills)} careers with sufficient data")
    return career_skills


def augment_samples(career_skills, samples_per_career=40, seed=42):
    """
    Augment with 40 samples per career (up from 25).
    More samples = better generalization.
    Uses score-weighted sampling so high-importance skills dominate.
    """
    random.seed(seed)
    np.random.seed(seed)

    X_samples, y_labels = [], []

    for career, skill_score_pairs in career_skills.items():
        skills = [s for s, _ in skill_score_pairs]
        scores = np.array([sc for _, sc in skill_score_pairs], dtype=float)
        probs  = scores / scores.sum()

        # 1. Full set — all skills (perfect resume)
        X_samples.append(skills)
        y_labels.append(career)

        # 2. Top-70% by score (strong candidate resume)
        n_top = max(4, int(len(skills) * 0.7))
        top_idx = np.argsort(scores)[-n_top:]
        X_samples.append([skills[i] for i in top_idx])
        y_labels.append(career)

        # 3. Top-50% — partial but focused
        n_mid = max(3, int(len(skills) * 0.5))
        mid_idx = np.argsort(scores)[-n_mid:]
        X_samples.append([skills[i] for i in mid_idx])
        y_labels.append(career)

        # 4. Weighted random samples (simulate real varied resumes)
        for _ in range(samples_per_career - 3):
            k = random.randint(
                max(3, int(len(skills) * 0.45)),
                max(5, int(len(skills) * 0.90))
            )
            k = min(k, len(skills))
            chosen = np.random.choice(len(skills), size=k, replace=False, p=probs)
            X_samples.append([skills[i] for i in chosen])
            y_labels.append(career)

    print(f"   ✅ {len(X_samples)} samples "
          f"({samples_per_career}/career × {len(career_skills)} careers)")
    return X_samples, y_labels


def train():
    occ, skills_df  = load_onet_data()
    career_skills    = build_skill_dict(occ, skills_df, importance_threshold=2.5)

    print("🔀 Augmenting training data...")
    X_samples, y_labels = augment_samples(career_skills, samples_per_career=40)

    # Binary encode
    mlb = MultiLabelBinarizer()
    X   = mlb.fit_transform(X_samples)
    y   = np.array(y_labels)

    print(f"📊 Raw matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # ── Feature selection: remove near-zero-variance features ───────────────
    # This is the key fix for the 8815-feature problem.
    # Features that are 0 for >98% of samples give no signal → remove them.
    print("✂️  Removing low-variance features...")
    selector = VarianceThreshold(threshold=0.02)   # removes features present in <2% of samples
    X = selector.fit_transform(X)
    print(f"   ✅ Reduced to {X.shape[1]} informative features")

    print(f"🎯 Predicting across {len(np.unique(y))} careers")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🌲 Training Random Forest... (2-4 mins)")
    model = RandomForestClassifier(
    n_estimators=100,       # down from 300 — saves ~3x memory
    max_depth=20,           # cap depth — main fix for MemoryError
    min_samples_leaf=2,     # stops splitting tiny leaves
    min_samples_split=4,
    max_features="sqrt",
    random_state=42,
    n_jobs=1,               # single core — avoids parallel memory spikes
    class_weight="balanced"
)
    model.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc     = accuracy_score(y_test, y_pred)

    classes = model.classes_
    top5_correct = sum(
        1 for i, true in enumerate(y_test)
        if true in classes[np.argsort(y_proba[i])[-5:]]
    )
    top5_acc = top5_correct / len(y_test)

    importances  = model.feature_importances_
    # Map back from selected features to original skill names
    selected_mask   = selector.get_support()
    selected_skills = mlb.classes_[selected_mask]
    top20_idx    = np.argsort(importances)[-20:][::-1]
    top20_skills = [
        {"skill": selected_skills[i], "importance": round(float(importances[i]), 4)}
        for i in top20_idx
    ]

    metrics = {
        "status":           "success",
        "accuracy":         round(acc * 100, 2),
        "top5_accuracy":    round(top5_acc * 100, 2),
        "train_samples":    int(len(X_train)),
        "test_samples":     int(len(X_test)),
        "n_careers":        int(len(np.unique(y))),
        "n_skill_features": int(X.shape[1]),
        "dataset_source":   "O*NET 29.0 (US Dept of Labor)",
        "top20_skills":     top20_skills,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,    MODEL_PATH)
    joblib.dump(mlb,      MLB_PATH)
    joblib.dump(selector, SELECTOR_PATH)   # ← save selector for use in recommender.py
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Done!")
    print(f"   Top-1 accuracy : {metrics['accuracy']}%")
    print(f"   Top-5 accuracy : {metrics['top5_accuracy']}%")
    print(f"   Careers        : {metrics['n_careers']}")
    print(f"   Features used  : {metrics['n_skill_features']}")

    return metrics


if __name__ == "__main__":
    result = train()