import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONET_DIR = os.path.join(BASE_DIR, "data", "onet")

TREND_DEMAND_DELTA = {
    "Rising": 1.8, "Moderate": 0.8, "Stable": 0.2, "Declining": -1.2,
}
DEFAULT_GROWTH_BY_TREND = {
    "Rising": 6.5, "Moderate": 5.0, "Stable": 3.0, "Declining": 1.5,
}

# Pakistan salary map (PKR/month) — same as recommender.py
PAKISTAN_SALARY_MAP = {
    "software developer":     (80000,  350000, 5),
    "software engineer":      (80000,  350000, 5),
    "data scientist":         (100000, 420000, 5),
    "machine learning":       (120000, 450000, 5),
    "artificial intelligence":(120000, 450000, 5),
    "web developer":          (60000,  250000, 4),
    "frontend":               (60000,  220000, 4),
    "backend":                (70000,  280000, 4),
    "full stack":             (80000,  320000, 5),
    "mobile developer":       (70000,  300000, 4),
    "devops":                 (90000,  380000, 5),
    "cloud":                  (90000,  400000, 5),
    "cybersecurity":          (80000,  350000, 4),
    "network":                (50000,  200000, 3),
    "database":               (60000,  220000, 3),
    "data analyst":           (70000,  250000, 4),
    "business analyst":       (65000,  230000, 4),
    "project manager":        (90000,  350000, 4),
    "product manager":        (100000, 400000, 4),
    "graphic designer":       (40000,  150000, 3),
    "ui":                     (60000,  220000, 4),
    "ux":                     (60000,  220000, 4),
    "accountant":             (40000,  150000, 3),
    "financial":              (60000,  250000, 3),
    "doctor":                 (80000,  500000, 5),
    "physician":              (80000,  500000, 5),
    "surgeon":                (150000, 800000, 5),
    "nurse":                  (35000,  100000, 4),
    "pharmacist":             (50000,  180000, 4),
    "dentist":                (70000,  350000, 4),
    "teacher":                (30000,  80000,  3),
    "professor":              (60000,  200000, 3),
    "lawyer":                 (60000,  350000, 3),
    "civil engineer":         (60000,  250000, 4),
    "mechanical engineer":    (60000,  230000, 3),
    "electrical engineer":    (65000,  250000, 4),
    "architect":              (60000,  250000, 3),
    "marketing":              (45000,  200000, 3),
    "human resource":         (45000,  160000, 3),
    "supply chain":           (50000,  200000, 3),
    "manager":                (70000,  300000, 4),
    "analyst":                (60000,  220000, 4),
    "engineer":               (60000,  250000, 4),
    "consultant":             (70000,  300000, 4),
    "researcher":             (50000,  200000, 3),
    "technician":             (35000,  120000, 3),
    "default":                (40000,  150000, 3),
}

AUTOMATION_RISK_MAP = {
    "data entry": 85, "cashier": 80, "telemarketer": 90,
    "accountant": 60, "driver": 65, "software": 10,
    "data scientist": 15, "machine learning": 10, "doctor": 15,
    "surgeon": 10, "nurse": 20, "teacher": 25, "lawyer": 30,
    "psychologist": 20, "designer": 35, "researcher": 20,
    "manager": 25, "engineer": 30, "default": 40,
}

_onet_df = None


def _load_onet():
    global _onet_df
    if _onet_df is not None:
        return
    occ_path = os.path.join(ONET_DIR, "Occupation Data.txt")
    if os.path.exists(occ_path):
        _onet_df = pd.read_csv(occ_path, sep="\t", dtype=str)
        _onet_df.columns = _onet_df.columns.str.strip()
        _onet_df = _onet_df.rename(columns={
            "O*NET-SOC Code": "soc_code",
            "Title": "career_title",
            "Description": "description"
        })
    else:
        _onet_df = pd.DataFrame(columns=["soc_code", "career_title", "description"])


def _get_pakistan_salary(career_title):
    title_lower = career_title.lower()
    for keyword, (sal_min, sal_max, demand) in PAKISTAN_SALARY_MAP.items():
        if keyword in title_lower:
            return sal_min, sal_max, demand
    return PAKISTAN_SALARY_MAP["default"]


def _get_automation_risk(career_title):
    title_lower = career_title.lower()
    for keyword, risk in AUTOMATION_RISK_MAP.items():
        if keyword in title_lower:
            return risk
    return AUTOMATION_RISK_MAP["default"]


def _infer_market_trend(demand_level):
    if demand_level >= 5:   return "Rising"
    if demand_level >= 4:   return "Moderate"
    if demand_level >= 3:   return "Stable"
    return "Declining"


def _infer_job_type(career_title):
    title_lower = career_title.lower()
    if any(k in title_lower for k in ("remote", "online", "virtual")):
        return "Remote"
    if any(k in title_lower for k in ("software", "developer", "data", "web", "computer", "analyst")):
        return "Hybrid"
    if any(k in title_lower for k in ("nurse", "doctor", "teacher", "construction", "manufacturing")):
        return "On-site"
    return "Hybrid"


def _project_salary(salary_min, salary_max, growth_rate, years=5):
    projections = []
    current_year = pd.Timestamp.now().year
    min_s, max_s = float(salary_min), float(salary_max)
    mid_s = (min_s + max_s) / 2
    rate = growth_rate / 100.0
    for i in range(years + 1):
        f = (1 + rate) ** i
        projections.append({
            "year": str(current_year + i),
            "min": round(min_s * f),
            "mid": round(mid_s * f),
            "max": round(max_s * f),
        })
    return projections


def _project_demand(demand_level, market_trend, years=5):
    base_index = min(100, max(10, demand_level * 20))
    delta = TREND_DEMAND_DELTA.get(market_trend, 0.2)
    current_year = pd.Timestamp.now().year
    projections = []
    for i in range(years + 1):
        index = base_index + (delta * i)
        index = max(15, min(100, index)) if market_trend != "Declining" else max(15, index)
        projections.append({"year": str(current_year + i), "demand_index": round(index, 1)})
    return projections


def _market_factors(demand_level, automation_risk, salary_growth_rate, market_trend):
    demand          = min(100, demand_level * 20)
    salary_potential = min(100, (salary_growth_rate / 8.0) * 100)
    job_security    = min(100, max(0, 100 - automation_risk))
    growth_outlook  = {"Rising": 90, "Moderate": 70, "Stable": 50, "Declining": 25}.get(market_trend, 50)
    return {
        "demand":           round(demand, 1),
        "salary_potential": round(salary_potential, 1),
        "job_security":     round(job_security, 1),
        "growth_outlook":   growth_outlook,
        "competition_ease": 60.0,   # neutral default without CSV data
    }


def _market_outlook_score(factors, market_trend):
    weights = {"demand": 0.25, "salary_potential": 0.2, "job_security": 0.2,
               "growth_outlook": 0.2, "competition_ease": 0.15}
    score = sum(factors[k] * w for k, w in weights.items())
    trend_bonus = {"Rising": 5, "Moderate": 2, "Stable": 0, "Declining": -8}.get(market_trend, 0)
    return round(min(100, max(0, score + trend_bonus)), 1)


def _build_insights(career_name, market_trend, salary_growth_rate,
                    demand_level, automation_risk, match_score, factors):
    insights = [
        f"{career_name} shows a {market_trend.lower()} market trend with ~{salary_growth_rate}% annual salary growth.",
        f"Demand level is {demand_level}/5 — {'strong hiring activity expected' if demand_level >= 4 else 'steady opportunities available' if demand_level >= 3 else 'competition may be tighter'}.",
        f"Automation risk is {automation_risk}% — {'high human oversight still needed' if automation_risk < 25 else 'some tasks may be automated over time'}.",
        "Salary figures reflect Pakistani market rates (PKR/month).",
    ]
    if match_score is not None:
        if match_score >= 75:
            insights.append(f"Your {match_score}% skill match puts you in a strong position.")
        elif match_score >= 50:
            insights.append(f"At {match_score}% match, upskilling in key gaps could unlock this career within 6–12 months.")
        else:
            insights.append(f"Current match is {match_score}% — consider the learning roadmap before targeting this role.")
    if market_trend == "Rising":
        insights.append("Market momentum is positive — early movers may benefit from premium salaries.")
    elif market_trend == "Declining":
        insights.append("Market is contracting — focus on niche specializations to stay competitive.")
    return insights


def simulate_career(career_name: str, match_score=None, years: int = 5):
    _load_onet()

    salary_min, salary_max, demand_level = _get_pakistan_salary(career_name)
    automation_risk = _get_automation_risk(career_name)
    market_trend    = _infer_market_trend(demand_level)
    salary_growth   = DEFAULT_GROWTH_BY_TREND[market_trend]
    factors         = _market_factors(demand_level, automation_risk, salary_growth, market_trend)
    outlook         = _market_outlook_score(factors, market_trend)

    return {
        "career":             career_name,
        "currency":           "PKR",
        "salary_period":      "monthly",
        "job_type":           _infer_job_type(career_name),
        "match_score":        match_score,
        "market_trend":       market_trend,
        "salary_growth_rate": salary_growth,
        "demand_level":       demand_level,
        "automation_risk":    automation_risk,
        "current_salary":     {
            "min": salary_min,
            "max": salary_max,
            "mid": round((salary_min + salary_max) / 2),
        },
        "projected_salary":   _project_salary(salary_min, salary_max, salary_growth, years),
        "demand_projection":  _project_demand(demand_level, market_trend, years),
        "market_factors":     factors,
        "metrics":            {"market_outlook_score": outlook},
        "insights":           _build_insights(career_name, market_trend, salary_growth,
                                              demand_level, automation_risk, match_score, factors),
    }