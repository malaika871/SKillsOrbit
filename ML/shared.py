"""
ML/shared.py
============
Shared constants and utility functions used by both recommender.py and
career_simulator.py.  Import from here — never duplicate.
"""

import os
import pandas as pd

from ML.config import BASE_DIR

CAREERS_CSV = os.path.join(BASE_DIR, "data", "careers.csv")

# ─── Pakistan Salary Map (PKR/month) ─────────────────────────────────────────
# Source: Rozee.pk / Glassdoor Pakistan 2024
# Format: keyword → (salary_min, salary_max, demand_level 1-5)
# NOTE: This map is the FALLBACK used only when careers.csv does NOT contain
#       the requested career title.  All primary salary data comes from
#       careers.csv so that both career cards and the simulation modal always
#       show identical numbers.
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
    "android":                (70000,  280000, 4),
    "ios":                    (70000,  280000, 4),
    "devops":                 (90000,  380000, 5),
    "cloud":                  (90000,  400000, 5),
    "cybersecurity":          (80000,  350000, 4),
    "network":                (50000,  200000, 3),
    "database":               (60000,  220000, 3),
    "data analyst":           (70000,  250000, 4),
    "business analyst":       (65000,  230000, 4),
    "systems analyst":        (65000,  230000, 3),
    "project manager":        (90000,  350000, 4),
    "product manager":        (100000, 400000, 4),
    "it manager":             (100000, 380000, 4),
    "graphic designer":       (40000,  150000, 3),
    "ui":                     (60000,  220000, 4),
    "ux":                     (60000,  220000, 4),
    "accountant":             (40000,  150000, 3),
    "auditor":                (50000,  180000, 3),
    "financial":              (60000,  250000, 3),
    "banker":                 (50000,  200000, 3),
    "economist":              (60000,  220000, 3),
    "doctor":                 (80000,  500000, 5),
    "physician":              (80000,  500000, 5),
    "surgeon":                (150000, 800000, 5),
    "nurse":                  (35000,  100000, 4),
    "pharmacist":             (50000,  180000, 4),
    "dentist":                (70000,  350000, 4),
    "teacher":                (30000,  80000,  3),
    "professor":              (60000,  200000, 3),
    "lecturer":               (50000,  150000, 3),
    "lawyer":                 (60000,  350000, 3),
    "legal":                  (50000,  250000, 3),
    "civil engineer":         (60000,  250000, 4),
    "mechanical engineer":    (60000,  230000, 3),
    "electrical engineer":    (65000,  250000, 4),
    "chemical engineer":      (60000,  230000, 3),
    "architect":              (60000,  250000, 3),
    "construction":           (50000,  200000, 3),
    "marketing":              (45000,  200000, 3),
    "sales":                  (40000,  180000, 3),
    "content":                (35000,  120000, 3),
    "social media":           (35000,  130000, 3),
    "human resource":         (45000,  160000, 3),
    "recruitment":            (40000,  140000, 3),
    "supply chain":           (50000,  200000, 3),
    "logistics":              (45000,  180000, 3),
    "journalist":             (35000,  120000, 2),
    "writer":                 (30000,  120000, 2),
    "psychologist":           (50000,  200000, 3),
    "counselor":              (40000,  150000, 3),
    "manager":                (70000,  300000, 4),
    "analyst":                (60000,  220000, 4),
    "engineer":               (60000,  250000, 4),
    "consultant":             (70000,  300000, 4),
    "researcher":             (50000,  200000, 3),
    "scientist":              (60000,  250000, 3),
    "technician":             (35000,  120000, 3),
    "administrator":          (40000,  150000, 3),
    "default":                (40000,  150000, 3),
}

# ─── Automation Risk by category (0-100) ─────────────────────────────────────
AUTOMATION_RISK_MAP = {
    "data entry":        85, "cashier":         80, "telemarketer":    90,
    "accountant":        60, "bookkeeper":       70, "driver":          65,
    "software":          10, "data scientist":   15, "machine learning": 10,
    "doctor":            15, "surgeon":          10, "nurse":           20,
    "teacher":           25, "lawyer":           30, "psychologist":    20,
    "designer":          35, "architect":        35, "researcher":      20,
    "manager":           25, "consultant":       30, "engineer":        30,
    "default":           40,
}

# ─── Lazy-loaded careers.csv cache ───────────────────────────────────────────
_careers_cache: dict | None = None   # { career_title_lower: row_dict }


def _load_careers_cache() -> dict:
    """Load careers.csv into a lowercase-keyed dict (cached after first call)."""
    global _careers_cache
    if _careers_cache is not None:
        return _careers_cache

    _careers_cache = {}
    if not os.path.exists(CAREERS_CSV):
        return _careers_cache

    df = pd.read_csv(CAREERS_CSV)
    for _, row in df.iterrows():
        key = str(row["career_title"]).strip().lower()
        _careers_cache[key] = row.to_dict()

    return _careers_cache


# ─── Public helpers ───────────────────────────────────────────────────────────

def get_pakistan_salary(career_title: str) -> tuple[int, int, int]:
    """
    Return (salary_min, salary_max, demand_level) for a career title.

    Priority:
      1. Exact match in careers.csv  (title case-insensitive)
      2. Keyword match in PAKISTAN_SALARY_MAP
      3. 'default' entry in PAKISTAN_SALARY_MAP

    This is the SINGLE SOURCE OF TRUTH used by both recommender.py and
    career_simulator.py so that career cards and simulation modals always
    show the same salary figures.
    """
    title_lower = career_title.strip().lower()

    # 1. Exact match from careers.csv
    cache = _load_careers_cache()
    if title_lower in cache:
        row = cache[title_lower]
        demand_raw = row.get("demand_level", 5)
        # careers.csv stores demand 1-10; normalise to 1-5 for internal use
        demand_norm = max(1, min(5, round(int(demand_raw) / 2)))
        return int(row["salary_min"]), int(row["salary_max"]), demand_norm

    # 2. Keyword fallback from hardcoded map
    for keyword, (sal_min, sal_max, demand) in PAKISTAN_SALARY_MAP.items():
        if keyword in title_lower:
            return sal_min, sal_max, demand

    return PAKISTAN_SALARY_MAP["default"]


def get_automation_risk(career_title: str) -> int:
    """Return estimated automation risk % (0-100) for a career title."""
    # First try careers.csv (has per-row automation_risk column)
    title_lower = career_title.strip().lower()
    cache = _load_careers_cache()
    if title_lower in cache:
        row = cache[title_lower]
        if "automation_risk" in row and pd.notna(row["automation_risk"]):
            return int(row["automation_risk"])

    # Keyword fallback
    for keyword, risk in AUTOMATION_RISK_MAP.items():
        if keyword in title_lower:
            return risk

    return AUTOMATION_RISK_MAP["default"]


def infer_job_type(career_title: str) -> str:
    """Infer Remote / Hybrid / On-site from career title keywords."""
    # First try careers.csv
    title_lower = career_title.strip().lower()
    cache = _load_careers_cache()
    if title_lower in cache:
        row = cache[title_lower]
        if "job_type" in row and pd.notna(row["job_type"]):
            return str(row["job_type"])

    # Keyword fallback
    if any(k in title_lower for k in ("remote", "online", "virtual")):
        return "Remote"
    if any(k in title_lower for k in ("software", "developer", "data", "web", "computer", "analyst")):
        return "Hybrid"
    if any(k in title_lower for k in ("nurse", "doctor", "teacher", "construction", "manufacturing")):
        return "On-site"
    return "Hybrid"


def get_market_trend(career_title: str) -> str:
    """Return market trend string from careers.csv, or infer from demand."""
    title_lower = career_title.strip().lower()
    cache = _load_careers_cache()
    if title_lower in cache:
        row = cache[title_lower]
        if "market_trend" in row and pd.notna(row["market_trend"]):
            return str(row["market_trend"])

    # Infer from demand
    _, _, demand = get_pakistan_salary(career_title)
    if demand >= 5:   return "Rising"
    if demand >= 4:   return "Moderate"
    if demand >= 3:   return "Stable"
    return "Declining"


def get_salary_growth_rate(career_title: str) -> float:
    """Return annual salary growth rate % from careers.csv, or estimate."""
    title_lower = career_title.strip().lower()
    cache = _load_careers_cache()
    if title_lower in cache:
        row = cache[title_lower]
        if "salary_growth_rate" in row and pd.notna(row["salary_growth_rate"]):
            return float(row["salary_growth_rate"])

    trend = get_market_trend(career_title)
    return {"Rising": 6.5, "Moderate": 5.0, "Stable": 3.0, "Declining": 1.5}.get(trend, 5.0)
