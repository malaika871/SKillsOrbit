import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "careers.csv")

df = pd.read_csv(DATA_PATH)
df["required_skills"] = df["required_skills"].fillna("").str.lower()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["required_skills"])

def get_matches(user_skills_string, top_n=5):
    user_input = user_skills_string.lower()
    user_vec = vectorizer.transform([user_input])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(top_n)
    results = []
    for _, row in top.iterrows():
        market_trend = row["market_trend"] if "market_trend" in row and pd.notna(row["market_trend"]) else "Moderate"
        growth_rate = row["salary_growth_rate"] if "salary_growth_rate" in row and pd.notna(row["salary_growth_rate"]) else 5.0
        results.append({
            "career": row["career_title"],
            "score": round(float(row["score"]) * 100, 1),
            "required_skills": [s.strip() for s in row["required_skills"].split(",")],
            "salary_min": row["salary_min"],
            "salary_max": row["salary_max"],
            "demand_level": row["demand_level"],
            "job_type": row["job_type"],
            "market_trend": market_trend,
            "salary_growth_rate": float(growth_rate),
            "automation_risk": row.get("automation_risk"),
            "competition_level": row.get("competition_level"),
        })
    return results
