import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "careers.csv")

# Initialize as None for lazy loading
_df = None
_vectorizer = None
_tfidf_matrix = None

def _initialize_data():
    """Lazy load and initialize the data and vectorizer with error handling."""
    global _df, _vectorizer, _tfidf_matrix
    if _df is None or _vectorizer is None or _tfidf_matrix is None:
        try:
            if not os.path.exists(DATA_PATH):
                raise FileNotFoundError(f"Career data file not found at: {DATA_PATH}")
            _df = pd.read_csv(DATA_PATH)
            _df["required_skills"] = _df["required_skills"].fillna("").str.lower()
            
            _vectorizer = TfidfVectorizer()
            _tfidf_matrix = _vectorizer.fit_transform(_df["required_skills"])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize recommender: {str(e)}")


def get_matches(user_skills_string, top_n=5):
    """Get career matches based on user's skills."""
    _initialize_data()
    
    user_input = user_skills_string.lower()
    user_vec = _vectorizer.transform([user_input])
    scores = cosine_similarity(user_vec, _tfidf_matrix).flatten()
    
    # Use a local copy to avoid mutating the global dataframe
    top = _df.assign(score=scores).sort_values("score", ascending=False).head(top_n)
    
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
