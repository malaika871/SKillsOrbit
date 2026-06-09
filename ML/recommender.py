import os
import joblib
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "careers.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "career_model.pkl")
MLB_PATH   = os.path.join(BASE_DIR, "models", "mlb.pkl")

_model = None
_mlb   = None
_df    = None


def _load_model():
    global _model, _mlb, _df
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run: python ML/train_model.py"
            )
        _model = joblib.load(MODEL_PATH)
        _mlb   = joblib.load(MLB_PATH)
        _df    = pd.read_csv(DATA_PATH)


def get_matches(user_skills_string, top_n=5):
    _load_model()

    user_skills = [s.strip().lower() for s in user_skills_string.split(",") if s.strip()]
    X_user      = _mlb.transform([user_skills])

    probabilities = _model.predict_proba(X_user)[0]
    classes       = _model.classes_

    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    top    = ranked[:top_n]

    results = []
    for career_name, prob in top:
        row = _df[_df["career_title"] == career_name]
        if row.empty:
            continue
        row = row.iloc[0]

        market_trend = row["market_trend"] if pd.notna(row.get("market_trend")) else "Moderate"
        growth_rate  = float(row["salary_growth_rate"]) if pd.notna(row.get("salary_growth_rate")) else 5.0

        results.append({
            "career":             str(career_name),
            "score":              round(float(prob) * 100, 1),
            "required_skills":    [s.strip() for s in row["required_skills"].split(",")],
            "salary_min":         int(row["salary_min"]),
            "salary_max":         int(row["salary_max"]),
            "demand_level":       int(row["demand_level"]),
            "job_type":           str(row["job_type"]),
            "market_trend":       str(market_trend),
            "salary_growth_rate": float(growth_rate),
            "automation_risk":    int(row["automation_risk"]) if pd.notna(row.get("automation_risk")) else 0,
            "competition_level":  int(row["competition_level"]) if pd.notna(row.get("competition_level")) else 0,
    })
    return results