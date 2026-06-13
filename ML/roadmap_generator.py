import os
import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONET_DIR = os.path.join(BASE_DIR, "data", "onet")

# ─── Cached O*NET data ────────────────────────────────────────────────────────
_onet_df     = None   # Occupation Data.txt
_skills_df   = None   # Skills.txt (for generating phase skills)
_careers_list = None


def _initialize_data():
    global _onet_df, _skills_df, _careers_list
    if _onet_df is not None:
        return

    occ_path = os.path.join(ONET_DIR, "Occupation Data.txt")
    if not os.path.exists(occ_path):
        raise FileNotFoundError(f"Occupation Data.txt not found at: {occ_path}")

    _onet_df = pd.read_csv(occ_path, sep="\t", dtype=str)
    _onet_df.columns = _onet_df.columns.str.strip()
    _onet_df = _onet_df.rename(columns={
        "O*NET-SOC Code": "soc_code",
        "Title":          "career_title",
        "Description":    "description"
    })
    _careers_list = _onet_df["career_title"].tolist()

    # Load Skills.txt for roadmap phase generation
    skills_path = os.path.join(ONET_DIR, "Skills.txt")
    if os.path.exists(skills_path):
        _skills_df = pd.read_csv(
            skills_path, sep="\t",
            usecols=["O*NET-SOC Code", "Element Name", "Scale ID", "Data Value"],
            dtype={"O*NET-SOC Code": str, "Element Name": str,
                   "Scale ID": str, "Data Value": float}
        ).rename(columns={
            "O*NET-SOC Code": "soc_code",
            "Element Name":   "skill",
            "Scale ID":       "scale_id",
            "Data Value":     "score"
        })
        _skills_df = _skills_df[_skills_df["scale_id"] == "IM"]


# ─── Predefined roadmap templates (kept for popular careers) ──────────────────
ROADMAP_TEMPLATES = {
    "Software Developers": {
        "description": "Build software applications across a wide range of domains and platforms",
        "phases": [
            {
                "phase": "Phase 1: Programming Fundamentals",
                "duration": "2-3 months", "level": "Beginner",
                "description": "Learn core programming concepts and a primary language",
                "skills": ["Python or Java", "Data Structures", "Algorithms", "Git", "Problem Solving"],
                "resources": ["CS50 by Harvard (free)", "Automate the Boring Stuff with Python", "LeetCode Easy Problems"],
                "status": "pending"
            },
            {
                "phase": "Phase 2: Software Design",
                "duration": "2-3 months", "level": "Intermediate",
                "description": "Understand how to design and structure software systems",
                "skills": ["OOP", "Design Patterns", "Clean Code", "Unit Testing", "Debugging"],
                "resources": ["Clean Code (Book)", "Refactoring Guru", "JUnit / PyTest Docs"],
                "status": "pending"
            },
            {
                "phase": "Phase 3: Web / Backend Development",
                "duration": "3-4 months", "level": "Intermediate",
                "description": "Build real applications with APIs and databases",
                "skills": ["REST APIs", "SQL", "Node.js or Django", "Authentication", "Docker"],
                "resources": ["Node.js Official Docs", "Django for Beginners (Book)", "PostgreSQL Tutorial"],
                "status": "pending"
            },
            {
                "phase": "Phase 4: Systems & DevOps",
                "duration": "2-3 months", "level": "Advanced",
                "description": "Deploy, monitor and scale applications",
                "skills": ["CI/CD", "Linux", "Cloud (AWS/GCP)", "Kubernetes", "Monitoring"],
                "resources": ["AWS Free Tier", "The Linux Command Line (Book)", "GitHub Actions Docs"],
                "status": "pending"
            }
        ]
    },
    "Data Scientists": {
        "description": "Extract insights from data using statistics, ML, and visualization",
        "phases": [
            {
                "phase": "Phase 1: Foundations",
                "duration": "3-4 months", "level": "Beginner",
                "description": "Build a strong foundation in programming and mathematics",
                "skills": ["Python", "SQL", "Statistics", "Pandas", "NumPy"],
                "resources": ["Python for Data Analysis (Book)", "Khan Academy Statistics", "DataCamp SQL"],
                "status": "pending"
            },
            {
                "phase": "Phase 2: Data Visualization",
                "duration": "2-3 months", "level": "Intermediate",
                "description": "Learn to explore and present data visually",
                "skills": ["Matplotlib", "Seaborn", "Tableau", "EDA", "Data Storytelling"],
                "resources": ["Storytelling with Data (Book)", "Tableau Public Tutorials", "Matplotlib Docs"],
                "status": "pending"
            },
            {
                "phase": "Phase 3: Machine Learning",
                "duration": "3-4 months", "level": "Intermediate",
                "description": "Apply ML algorithms to real datasets",
                "skills": ["Scikit-learn", "Feature Engineering", "Model Evaluation", "Cross Validation"],
                "resources": ["Andrew Ng ML Course", "Hands-On ML (Book)", "Kaggle Competitions"],
                "status": "pending"
            },
            {
                "phase": "Phase 4: Deep Learning & Deployment",
                "duration": "3-4 months", "level": "Advanced",
                "description": "Build and deploy neural network models",
                "skills": ["TensorFlow", "PyTorch", "NLP", "MLOps", "Cloud Platforms"],
                "resources": ["Fast.ai Course", "Deep Learning Specialization (Coursera)", "MLflow Docs"],
                "status": "pending"
            }
        ]
    },
    "Registered Nurses": {
        "description": "Provide patient care and support in clinical and community settings",
        "phases": [
            {
                "phase": "Phase 1: Basic Sciences",
                "duration": "6 months", "level": "Beginner",
                "description": "Build foundational knowledge in biology and anatomy",
                "skills": ["Anatomy", "Physiology", "Microbiology", "Medical Terminology", "Patient Safety"],
                "resources": ["Anatomy & Physiology (OpenStax)", "Khan Academy Biology", "NCLEX Study Guides"],
                "status": "pending"
            },
            {
                "phase": "Phase 2: Clinical Skills",
                "duration": "6 months", "level": "Intermediate",
                "description": "Develop hands-on nursing and patient care skills",
                "skills": ["Patient Assessment", "Medication Administration", "IV Therapy", "Wound Care"],
                "resources": ["Clinical Nursing Skills (Book)", "ATI Nursing Education", "Hospital Practicum"],
                "status": "pending"
            },
            {
                "phase": "Phase 3: Specialization",
                "duration": "4-6 months", "level": "Advanced",
                "description": "Choose a nursing specialty and build advanced skills",
                "skills": ["ICU Care", "Pediatrics or OB", "Emergency Nursing", "Leadership"],
                "resources": ["AACN Certification Resources", "Specialty Nursing Journals", "Mentorship Programs"],
                "status": "pending"
            }
        ]
    },
    "Web Developers": {
        "description": "Build modern, responsive websites and web applications",
        "phases": [
            {
                "phase": "Phase 1: HTML, CSS & Web Basics",
                "duration": "2-3 months", "level": "Beginner",
                "description": "Learn the fundamentals of web development",
                "skills": ["HTML", "CSS", "Responsive Design", "Flexbox", "Grid"],
                "resources": ["MDN Web Docs", "FreeCodeCamp", "CSS Tricks"],
                "status": "pending"
            },
            {
                "phase": "Phase 2: JavaScript",
                "duration": "3-4 months", "level": "Beginner",
                "description": "Master JavaScript and DOM manipulation",
                "skills": ["JavaScript", "DOM", "ES6+", "Async/Await", "Fetch API"],
                "resources": ["JavaScript.info", "Eloquent JavaScript (Book)", "JavaScript30"],
                "status": "pending"
            },
            {
                "phase": "Phase 3: Frontend Framework",
                "duration": "3-4 months", "level": "Intermediate",
                "description": "Learn a modern frontend framework",
                "skills": ["React", "State Management", "Component Design", "Hooks", "Routing"],
                "resources": ["React Official Docs", "Scrimba React Course", "Vue Mastery"],
                "status": "pending"
            },
            {
                "phase": "Phase 4: Backend & Deployment",
                "duration": "3-4 months", "level": "Advanced",
                "description": "Build APIs and deploy full-stack applications",
                "skills": ["Node.js", "Express", "Databases", "REST APIs", "Docker"],
                "resources": ["Node.js Docs", "MongoDB University", "Vercel/Railway Docs"],
                "status": "pending"
            }
        ]
    },
}


def get_all_careers():
    """Return list of all O*NET career titles."""
    try:
        _initialize_data()
        return _careers_list
    except Exception as e:
        logger.error(f"Error loading careers: {e}")
        return []


def _get_onet_skills_for_career(career_title, top_n=20):
    """
    Fetch top N most important skills for a career from O*NET Skills.txt.
    Returns a list of skill name strings sorted by importance score.
    """
    if _skills_df is None or _onet_df is None:
        return []

    # Get SOC code for this career
    row = _onet_df[_onet_df["career_title"] == career_title]
    if row.empty:
        return []
    soc_code = row.iloc[0]["soc_code"]

    # Get skills for this SOC code, sorted by importance
    career_skills = _skills_df[_skills_df["soc_code"] == soc_code].copy()
    career_skills  = career_skills.sort_values("score", ascending=False)
    return career_skills["skill"].head(top_n).tolist()


def _find_roadmap_template(career_title):
    """Match careers.csv titles to predefined roadmap templates."""
    if career_title in ROADMAP_TEMPLATES:
        return ROADMAP_TEMPLATES[career_title]

    title_lower = career_title.lower()
    for key, template in ROADMAP_TEMPLATES.items():
        if key.lower() == title_lower:
            return template
        if key.lower().rstrip("s") == title_lower.rstrip("s"):
            return template
        if title_lower in key.lower() or key.lower() in title_lower:
            return template
    return None


def generate_roadmap(career_title):
    """
    Generate a learning roadmap for a career.
    Uses predefined templates for popular careers,
    dynamically generates from O*NET data for all others.
    """
    _initialize_data()

    template = _find_roadmap_template(career_title)
    if template is not None:
        roadmap = template.copy()
        roadmap["career"] = career_title
        return roadmap

    # Dynamically generate from O*NET skill data
    skills = _get_onet_skills_for_career(career_title, top_n=20)

    # Fallback if no skills found
    if not skills:
        skills = ["Core Knowledge", "Professional Skills", "Communication",
                  "Critical Thinking", "Domain Expertise"]

    # Get career description from O*NET
    row = _onet_df[_onet_df["career_title"] == career_title]
    description = str(row.iloc[0]["description"]) if not row.empty else \
                  f"Structured learning path to become a {career_title}"

    # Split skills into 3-4 phases based on importance order
    # (O*NET skills are already sorted by importance score)
    total     = len(skills)
    chunk     = max(3, total // 4)
    phases    = []
    levels    = ["Beginner", "Intermediate", "Intermediate", "Advanced"]
    durations = ["2-3 months", "3-4 months", "3-4 months", "2-3 months"]

    skill_chunks = [skills[i:i+chunk] for i in range(0, total, chunk)]
    # Merge any tiny last chunk into previous phase
    if len(skill_chunks) > 1 and len(skill_chunks[-1]) < 2:
        skill_chunks[-2].extend(skill_chunks.pop())

    for idx, chunk_skills in enumerate(skill_chunks[:4]):
        phase_num = idx + 1
        level     = levels[min(idx, 3)]
        duration  = durations[min(idx, 3)]
        phases.append({
            "phase":       f"Phase {phase_num}: {'Foundations' if phase_num == 1 else 'Core Skills' if phase_num == 2 else 'Advanced Skills' if phase_num == 3 else 'Professional Practice'}",
            "duration":    duration,
            "level":       level,
            "description": f"{'Build foundational knowledge' if phase_num == 1 else 'Develop core competencies' if phase_num == 2 else 'Master advanced techniques' if phase_num == 3 else 'Apply skills professionally'} in {career_title}",
            "skills":      chunk_skills,
            "resources": [
                "Online courses (Coursera, edX, Udemy)",
                "Official documentation and textbooks",
                "Practice projects and portfolio building"
            ],
            "status": "pending"
        })

    return {
        "career":      career_title,
        "description": description,
        "phases":      phases
    }


def get_roadmap_for_skill_level(career_title, user_skills):
    """
    Personalize a roadmap by marking phases completed/current/pending
    based on which skills the user already has.
    """
    roadmap = generate_roadmap(career_title)
    if not roadmap:
        return None

    user_skills_lower = {s.strip().lower() for s in user_skills}

    for phase in roadmap["phases"]:
        phase_skills_lower = {s.lower() for s in phase["skills"]}
        if phase_skills_lower.issubset(user_skills_lower):
            phase["status"] = "completed"
        elif phase_skills_lower & user_skills_lower:
            phase["status"] = "current"
        else:
            phase["status"] = "pending"

    return roadmap