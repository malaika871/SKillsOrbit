import logging
import numpy as np
import pandas as pd

from ML.config import ONET_DIR
from ML.shared import (
    get_pakistan_salary,
    get_automation_risk,
    infer_job_type,
    get_market_trend,
    get_salary_growth_rate,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

TREND_DEMAND_DELTA = {
    "Rising": 1.8, "Moderate": 0.8, "Stable": 0.2, "Declining": -1.2,
}

_onet_df = None


def _load_onet():
    global _onet_df
    if _onet_df is not None:
        return
    import os
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


def _infer_market_trend(demand_level):
    """Infer trend from demand level (1-5 scale) — used only as final fallback."""
    if demand_level >= 5:   return "Rising"
    if demand_level >= 4:   return "Moderate"
    if demand_level >= 3:   return "Stable"
    return "Declining"


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
    demand           = min(100, demand_level * 20)
    salary_potential = min(100, (salary_growth_rate / 8.0) * 100)
    job_security     = min(100, max(0, 100 - automation_risk))
    growth_outlook   = {"Rising": 90, "Moderate": 70, "Stable": 50, "Declining": 25}.get(market_trend, 50)
    return {
        "demand":           round(demand, 1),
        "salary_potential": round(salary_potential, 1),
        "job_security":     round(job_security, 1),
        "growth_outlook":   growth_outlook,
        "competition_ease": 60.0,
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
    """
    Simulate a 5-year career projection.

    Salary data is read from careers.csv via shared.get_pakistan_salary(),
    which is the SAME source used by recommender.py for career cards.
    This ensures career card salary == simulation modal salary.
    """
    _load_onet()

    # ── Single source of truth via shared.py ─────────────────────────────────
    salary_min, salary_max, demand_level = get_pakistan_salary(career_name)
    automation_risk = get_automation_risk(career_name)
    market_trend    = get_market_trend(career_name)
    salary_growth   = get_salary_growth_rate(career_name)
    job_type        = infer_job_type(career_name)

    factors  = _market_factors(demand_level, automation_risk, salary_growth, market_trend)
    outlook  = _market_outlook_score(factors, market_trend)

    return {
        "career":             career_name,
        "currency":           "PKR",
        "salary_period":      "monthly",
        "job_type":           job_type,
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