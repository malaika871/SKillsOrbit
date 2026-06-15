import os
import re
import joblib
import numpy as np
import pandas as pd

from ML.config import BASE_DIR, ONET_DIR, MODEL_DIR
from ML.shared import (
    PAKISTAN_SALARY_MAP,
    AUTOMATION_RISK_MAP,
    get_pakistan_salary   as _get_pakistan_salary,
    get_automation_risk   as _get_automation_risk,
    infer_job_type        as _infer_job_type,
)
from ML.skill_gap import compute_match_percentage

MODEL_PATH    = os.path.join(MODEL_DIR, "career_model.pkl")
MLB_PATH      = os.path.join(MODEL_DIR, "mlb.pkl")
SELECTOR_PATH = os.path.join(MODEL_DIR, "selector.pkl")
CAREERS_CSV   = os.path.join(BASE_DIR, "data", "careers.csv")

_model    = None
_mlb      = None
_selector = None
_onet_df  = None
_careers_df = None
_career_skills_cache = None
_skill_lookup = None

# Common user-facing skill aliases → O*NET / MLB feature hints
_SKILL_ALIASES = {
    "sql": "tech_structured query language sql",
    "ml": "programming",
    "machine learning": "programming",
    "deep learning": "programming",
    "data science": "science",
    "data analysis": "mathematics",
    "excel": "tech_microsoft excel",
    "power bi": "tech_microsoft power bi",
    "tableau": "tech_tableau",
    "aws": "tech_amazon web services aws",
    "gcp": "tech_google cloud",
    "azure": "tech_microsoft azure",
    "docker": "tech_docker",
    "kubernetes": "tech_kubernetes",
    "k8s": "tech_kubernetes",
    "node.js": "tech_node.js",
    "nodejs": "tech_node.js",
    "react.js": "tech_react",
    "vue": "tech_vue.js",
    "angular": "tech_angular",
    "c++": "tech_c++",
    "c#": "tech_c#",
    ".net": "tech_microsoft .net",
    "git": "tech_git",
    "linux": "tech_linux",
    "html": "tech_hypertext markup language html",
    "css": "tech_cascading style sheets css",
    "rest api": "tech_representational state transfer rest",
    "rest": "tech_representational state transfer rest",
    "nlp": "reading comprehension",
    "tensorflow": "tech_python",
    "pytorch": "tech_python",
    "scikit-learn": "tech_python",
    "pandas": "tech_python",
    "numpy": "tech_python",
}




def _build_skill_lookup(mlb):
    """Build a map from common user skill tokens to MLB feature names."""
    lookup = {}
    for feature in mlb.classes_:
        lookup[feature] = feature

        base = feature
        for prefix in ("tech_", "know_", "ability_", "activity_"):
            if feature.startswith(prefix):
                base = feature[len(prefix):]
                break

        lookup[base] = feature
        for token in re.split(r"[,/()]+", base):
            token = token.strip().lower()
            if len(token) >= 2:
                lookup.setdefault(token, feature)

    for alias, target in _SKILL_ALIASES.items():
        if target in mlb.classes_:
            lookup[alias] = target
        elif f"tech_{alias}" in mlb.classes_:
            lookup[alias] = f"tech_{alias}"

    return lookup


def _normalize_user_skills(user_skills, mlb, lookup):
    """Map resume/advisor skill names to trained MLB feature names."""
    normalized = []
    classes = set(mlb.classes_)

    for skill in user_skills:
        sl = skill.strip().lower()
        if not sl:
            continue

        if sl in classes:
            normalized.append(sl)
            continue

        if sl in lookup:
            normalized.append(lookup[sl])
            continue

        tech_name = f"tech_{sl}"
        if tech_name in classes:
            normalized.append(tech_name)
            continue

        partial = [feat for feat in mlb.classes_ if sl in feat]
        if partial:
            normalized.append(sorted(partial, key=len)[0])
            continue

        for feat in mlb.classes_:
            base = feat.split("_", 1)[-1]
            if base == sl or base.endswith(f" {sl}"):
                normalized.append(feat)
                break

    return list(dict.fromkeys(normalized))


def _load_careers_df():
    """Load curated careers dataset used for skill-based recommendations."""
    global _careers_df
    if _careers_df is not None:
        return _careers_df

    if not os.path.exists(CAREERS_CSV):
        _careers_df = pd.DataFrame()
        return _careers_df

    _careers_df = pd.read_csv(CAREERS_CSV)
    return _careers_df


def _parse_required_skills(skills_value):
    return [s.strip() for s in str(skills_value).split(",") if s.strip()]


def _skills_overlap_score(user_skills, required_skills):
    """Score how well user skills match a career's required skills (0-99)."""
    user_lower = {s.strip().lower() for s in user_skills if s.strip()}
    req_lower = {s.strip().lower() for s in required_skills if s.strip()}
    if not req_lower or not user_lower:
        return 0.0

    score, _, _ = compute_match_percentage(user_lower, req_lower)
    return score


def _load_career_skills():
    """Load top skills per career from O*NET and careers.csv."""
    global _career_skills_cache
    if _career_skills_cache is not None:
        return _career_skills_cache

    _career_skills_cache = {}

    careers_df = _load_careers_df()
    for _, row in careers_df.iterrows():
        title = str(row["career_title"])
        _career_skills_cache[title] = _parse_required_skills(row["required_skills"])

    skills_path = os.path.join(ONET_DIR, "Skills.txt")
    occ_path = os.path.join(ONET_DIR, "Occupation Data.txt")

    if os.path.exists(skills_path) and os.path.exists(occ_path):
        occ = pd.read_csv(occ_path, sep="\t", usecols=["O*NET-SOC Code", "Title"], dtype=str)
        occ = occ.rename(columns={"O*NET-SOC Code": "soc_code", "Title": "career_title"})

        skills = pd.read_csv(
            skills_path, sep="\t",
            usecols=["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"],
            dtype={"O*NET-SOC Code": str, "Element Name": str, "Scale ID": str, "Data Value": float},
        ).rename(columns={
            "O*NET-SOC Code": "soc_code",
            "Element Name": "skill",
            "Scale ID": "scale_id",
            "Data Value": "score",
        })
        skills = skills[skills["scale_id"] == "IM"]
        merged = skills.merge(occ, on="soc_code", how="inner")

        for title, group in merged.groupby("career_title"):
            top = group.sort_values("score", ascending=False)["skill"].head(12).tolist()
            if title not in _career_skills_cache or not _career_skills_cache[title]:
                _career_skills_cache[title] = top

    return _career_skills_cache


def _get_required_skills(career_name):
    cache = _load_career_skills()
    return cache.get(career_name, [])


def _competition_from_demand(demand_level):
    return max(3, min(10, 12 - demand_level * 2))


def _load_model():
    global _model, _mlb, _selector, _onet_df, _skill_lookup
    if _model is not None:
        return

    for path, name in [(MODEL_PATH, "career_model.pkl"), (MLB_PATH, "mlb.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found. Run: python ML/train_model.py")

    _model = joblib.load(MODEL_PATH)
    _mlb   = joblib.load(MLB_PATH)
    _skill_lookup = _build_skill_lookup(_mlb)

    if os.path.exists(SELECTOR_PATH):
        _selector = joblib.load(SELECTOR_PATH)

    # Load O*NET Occupation Data for career descriptions
    occ_path = os.path.join(ONET_DIR, "Occupation Data.txt")
    if os.path.exists(occ_path):
        _onet_df = pd.read_csv(occ_path, sep="\t", dtype=str)
        _onet_df.columns = _onet_df.columns.str.strip()
        _onet_df = _onet_df.rename(columns={
            "O*NET-SOC Code": "soc_code",
            "Title":          "career_title",
            "Description":    "description"
        })
    else:
        _onet_df = pd.DataFrame(columns=["soc_code", "career_title", "description"])

    _load_career_skills()


def get_matches(user_skills_string, top_n=5):
    user_skills = [s.strip() for s in user_skills_string.split(",") if s.strip()]
    if not user_skills:
        return []

    careers_df = _load_careers_df()
    if not careers_df.empty:
        return _get_matches_from_csv(user_skills, careers_df, top_n)

    return _get_matches_ml(user_skills_string, top_n)


def _get_matches_from_csv(user_skills, careers_df, top_n=5):
    """Recommend careers by skill overlap against careers.csv."""
    _load_career_skills()

    scored = []
    for _, row in careers_df.iterrows():
        required = _parse_required_skills(row["required_skills"])
        score = _skills_overlap_score(user_skills, required)
        scored.append((score, row, required))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_n]

    results = []
    for score, row, required in top:
        career_name = str(row["career_title"])
        results.append({
            "career":             career_name,
            "soc_code":           "",
            "score":              score,
            "description":        f"Career path for {career_name} based on your skill profile.",
            "required_skills":    required,
            "salary_min":         int(row["salary_min"]),
            "salary_max":         int(row["salary_max"]),
            "currency":           "PKR",
            "salary_period":      "monthly",
            "demand_level":       int(row["demand_level"]),
            "market_trend":       str(row["market_trend"]),
            "salary_growth_rate": float(row["salary_growth_rate"]),
            "automation_risk":    int(row["automation_risk"]),
            "job_type":           str(row["job_type"]),
            "competition_level":  int(row["competition_level"]),
        })

    return results


def _get_matches_ml(user_skills_string, top_n=5):
    """Fallback ML recommendations when careers.csv is unavailable."""
    _load_model()

    user_skills = [s.strip() for s in user_skills_string.split(",") if s.strip()]
    encoded_skills = _normalize_user_skills(user_skills, _mlb, _skill_lookup)
    if not encoded_skills:
        encoded_skills = [s.strip().lower() for s in user_skills]

    X_user = _mlb.transform([encoded_skills])
    if _selector is not None:
        X_user = _selector.transform(X_user)

    probabilities = _model.predict_proba(X_user)[0]
    classes       = _model.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    top_prob = ranked[0][1] if ranked else 1.0
    top = ranked[:top_n]

    results = []
    for career_name, prob in top:
        row = _onet_df[_onet_df["career_title"] == career_name]
        description = str(row.iloc[0]["description"]) if not row.empty else ""
        soc_code    = str(row.iloc[0]["soc_code"])    if not row.empty else ""

        sal_min, sal_max, demand = _get_pakistan_salary(career_name)
        automation_risk          = _get_automation_risk(career_name)
        display_demand           = min(10, max(1, demand * 2))
        growth_rate = {5: 12.0, 4: 9.0, 3: 6.0, 2: 4.0, 1: 2.0}.get(demand, 6.0)
        market_trend = {5: "High Growth", 4: "Growing", 3: "Moderate",
                        2: "Stable",      1: "Declining"}.get(demand, "Moderate")
        relative_score = round((prob / top_prob) * 100, 1) if top_prob else 0.0

        results.append({
            "career":             str(career_name),
            "soc_code":           soc_code,
            "score":              relative_score,
            "description":        description,
            "required_skills":    _get_required_skills(career_name),
            "salary_min":         sal_min,
            "salary_max":         sal_max,
            "currency":           "PKR",
            "salary_period":      "monthly",
            "demand_level":       display_demand,
            "market_trend":       market_trend,
            "salary_growth_rate": growth_rate,
            "automation_risk":    automation_risk,
            "job_type":           _infer_job_type(career_name),
            "competition_level":  _competition_from_demand(demand),
        })

    return results
