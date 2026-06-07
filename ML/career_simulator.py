import os
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "careers.csv")

TREND_DEMAND_DELTA = {
    "Rising": 1.8,
    "Moderate": 0.8,
    "Stable": 0.2,
    "Declining": -1.2,
}

DEFAULT_GROWTH_BY_TREND = {
    "Rising": 6.5,
    "Moderate": 5.0,
    "Stable": 3.0,
    "Declining": 1.5,
}

# Lazy loaded cached data
_df = None

def _initialize_data():
    """Lazy load and cache the careers data with error handling."""
    global _df
    if _df is None:
        try:
            if not os.path.exists(DATA_PATH):
                raise FileNotFoundError(f"Career data file not found at: {DATA_PATH}")
            _df = pd.read_csv(DATA_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize career simulator: {str(e)}")


def _infer_market_trend(demand_level: float) -> str:
    if demand_level >= 9:
        return "Rising"
    if demand_level >= 7:
        return "Moderate"
    if demand_level >= 5:
        return "Stable"
    return "Declining"


def _infer_salary_growth(market_trend: str) -> float:
    return DEFAULT_GROWTH_BY_TREND.get(market_trend, 3.0)


def _safe_float(value, default=0.0) -> float:
    """Safely convert to float, with logging for invalid values."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid numeric value '{value}', using default {default} instead. Error: {e}")
        return default


def _get_career_row(career_name: str):
    """Get career row from cached dataframe."""
    _initialize_data()
    matches = _df[_df["career_title"].str.lower() == career_name.strip().lower()]
    if matches.empty:
        return None
    return matches.iloc[0]


def _project_salary(salary_min, salary_max, growth_rate, years=5):
    projections = []
    current_year = pd.Timestamp.now().year
    min_s, max_s = float(salary_min), float(salary_max)
    mid_s = (min_s + max_s) / 2
    rate = growth_rate / 100.0

    for i in range(years + 1):
        factor = (1 + rate) ** i
        projections.append({
            "year": str(current_year + i),
            "min": round(min_s * factor),
            "mid": round(mid_s * factor),
            "max": round(max_s * factor),
        })
    return projections


def _project_demand(demand_level, market_trend, years=5):
    base_index = min(100, max(10, demand_level * 10))
    delta = TREND_DEMAND_DELTA.get(market_trend, 0.2)
    current_year = pd.Timestamp.now().year
    projections = []

    for i in range(years + 1):
        index = base_index + (delta * i)
        if market_trend == "Declining":
            index = max(15, index)
        else:
            index = min(100, max(10, index))
        projections.append({
            "year": str(current_year + i),
            "demand_index": round(index, 1),
        })
    return projections


def _market_factors(row, market_trend, salary_growth_rate):
    demand = min(100, _safe_float(row.get("demand_level"), 5) * 10)
    salary_potential = min(100, (salary_growth_rate / 8.0) * 100)
    job_security = min(100, max(0, 100 - _safe_float(row.get("automation_risk"), 30)))
    growth_outlook = {
        "Rising": 90,
        "Moderate": 70,
        "Stable": 50,
        "Declining": 25,
    }.get(market_trend, 50)
    competition_ease = min(100, max(0, 100 - _safe_float(row.get("competition_level"), 5) * 10))

    return {
        "demand": round(demand, 1),
        "salary_potential": round(salary_potential, 1),
        "job_security": round(job_security, 1),
        "growth_outlook": growth_outlook,
        "competition_ease": round(competition_ease, 1),
    }


def _market_outlook_score(factors, market_trend):
    weights = {
        "demand": 0.25,
        "salary_potential": 0.2,
        "job_security": 0.2,
        "growth_outlook": 0.2,
        "competition_ease": 0.15,
    }
    score = sum(factors[k] * w for k, w in weights.items())
    trend_bonus = {"Rising": 5, "Moderate": 2, "Stable": 0, "Declining": -8}.get(market_trend, 0)
    return round(min(100, max(0, score + trend_bonus)), 1)


def _build_insights(row, market_trend, salary_growth_rate, match_score, factors):
    career = row["career_title"]
    job_type = row.get("job_type", "Hybrid")
    demand = _safe_float(row.get("demand_level"), 5)
    automation = _safe_float(row.get("automation_risk"), 30)
    competition = _safe_float(row.get("competition_level"), 5)

    insights = [
        f"{career} shows a {market_trend.lower()} market trend with ~{salary_growth_rate}% annual salary growth.",
        f"Demand level is {demand}/10 — {'strong hiring activity expected' if demand >= 8 else 'steady opportunities available' if demand >= 6 else 'competition may be tighter'}.",
        f"Automation risk is {automation}% — {'high human oversight still needed' if automation < 25 else 'some tasks may be automated over time'}.",
        f"Typical work arrangement: {job_type}.",
    ]

    if match_score is not None:
        if match_score >= 75:
            insights.append(f"Your {match_score}% skill match puts you in a strong position to enter this field soon.")
        elif match_score >= 50:
            insights.append(f"At {match_score}% match, upskilling in key gaps could unlock this career within 6–12 months.")
        else:
            insights.append(f"Current match is {match_score}% — consider the learning roadmap before targeting this role.")

    if market_trend == "Rising":
        insights.append("Market momentum is positive — early movers may benefit from premium salaries.")
    elif market_trend == "Declining":
        insights.append("Market is contracting — focus on niche specializations to stay competitive.")

    if factors["job_security"] >= 75:
        insights.append("High job security score — lower automation exposure relative to peers.")
    if competition >= 8:
        insights.append("Competition is high — certifications and portfolio projects will help you stand out.")

    return insights


def simulate_career(career_name: str, match_score=None, years: int = 5):
    """
    Generate market trend and salary simulation data for a career.

    Uses careers.csv fields: salary_min, salary_max, demand_level,
    automation_risk, competition_level, job_type, market_trend,
    salary_growth_rate. Missing trend/growth fields are inferred.
    """
    row = _get_career_row(career_name)
    if row is None:
        return None

    demand_level = _safe_float(row.get("demand_level"), 5)
    market_trend = row.get("market_trend")
    if pd.isna(market_trend) or not str(market_trend).strip():
        market_trend = _infer_market_trend(demand_level)
    else:
        market_trend = str(market_trend).strip()

    salary_growth_rate = _safe_float(row.get("salary_growth_rate"), None)
    if salary_growth_rate is None:  # Only use default if None, not 0.0
        salary_growth_rate = _infer_salary_growth(market_trend)

    salary_min = int(row["salary_min"])
    salary_max = int(row["salary_max"])
    factors = _market_factors(row, market_trend, salary_growth_rate)
    outlook = _market_outlook_score(factors, market_trend)

    return {
        "career": row["career_title"],
        "job_type": row.get("job_type", "Hybrid"),
        "match_score": match_score,
        "market_trend": market_trend,
        "salary_growth_rate": salary_growth_rate,
        "demand_level": demand_level,
        "automation_risk": _safe_float(row.get("automation_risk"), 30),
        "competition_level": _safe_float(row.get("competition_level"), 5),
        "current_salary": {
            "min": salary_min,
            "max": salary_max,
            "mid": round((salary_min + salary_max) / 2),
        },
        "projected_salary": _project_salary(salary_min, salary_max, salary_growth_rate, years),
        "demand_projection": _project_demand(demand_level, market_trend, years),
        "market_factors": factors,
        "metrics": {
            "market_outlook_score": outlook,
        },
        "insights": _build_insights(row, market_trend, salary_growth_rate, match_score, factors),
    }
