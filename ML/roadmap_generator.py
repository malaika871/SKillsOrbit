import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "careers.csv")

# Predefined roadmap templates for different careers
ROADMAP_TEMPLATES = {
    "Data Scientist": {
        "description": "Master data analysis, statistics, and machine learning to extract insights from data",
        "phases": [
            {
                "phase": "Phase 1: Foundations",
                "duration": "3-4 months",
                "level": "Beginner",
                "description": "Build a strong foundation in programming and mathematics",
                "skills": ["Python", "SQL", "Statistics", "Pandas", "NumPy"],
                "resources": [
                    "Python for Data Analysis (Book)",
                    "Khan Academy Statistics Course",
                    "DataCamp SQL Fundamentals"
                ],
                "status": "pending"
            },
            {
                "phase": "Phase 2: Data Manipulation & Visualization",
                "duration": "2-3 months",
                "level": "Intermediate",
                "description": "Learn to manipulate, clean, and visualize data effectively",
                "skills": ["Data Cleaning", "Matplotlib", "Seaborn", "Tableau", "Data Storytelling"],
                "resources": [
                    "Tableau Desktop Specialist Certification",
                    "Matplotlib Documentation",
                    "Storytelling with Data (Book)"
                ],
                "status": "pending"
            },
            {
                "phase": "Phase 3: Machine Learning Basics",
                "duration": "3-4 months",
                "level": "Intermediate",
                "description": "Understand core machine learning algorithms and their applications",
                "skills": ["Scikit-learn", "Machine Learning", "Feature Engineering", "Model Evaluation"],
                "resources": [
                    "Andrew Ng's Machine Learning Course",
                    "Scikit-learn Documentation",
                    "Hands-On Machine Learning (Book)"
                ],
                "status": "pending"
            },
            {
                "phase": "Phase 4: Deep Learning & Advanced Topics",
                "duration": "4-5 months",
                "level": "Advanced",
                "description": "Dive into neural networks and advanced ML techniques",
                "skills": ["Deep Learning", "TensorFlow", "PyTorch", "NLP", "Computer Vision"],
                "resources": [
                    "Fast.ai Deep Learning Course",
                    "Deep Learning Specialization (Coursera)",
                    "PyTorch Tutorials"
                ],
                "status": "pending"
            },
            {
                "phase": "Phase 5: Production & Deployment",
                "duration": "2-3 months",
                "level": "Advanced",
                "description": "Learn to deploy and maintain ML models in production",
                "skills": ["Model Deployment", "Docker", "Git", "MLOps", "Cloud Platforms"],
                "resources": [
                    "AWS Machine Learning Specialty",
                    "Docker for Data Science",
                    "MLOps Fundamentals"
                ],
                "status": "pending"
            }
        ]
    },
    "Web Developer": {
        "description": "Build modern, responsive websites and web applications",
        "phases": [
            {
                "phase": "Phase 1: HTML, CSS & Web Basics",
                "duration": "2-3 months",
                "level": "Beginner",
                "description": "Learn the fundamentals of web development",
                "skills": ["HTML", "CSS", "Responsive Design", "Flexbox", "Grid"],
                "resources": [
                    "MDN Web Docs",
                    "FreeCodeCamp Responsive Web Design",
                    "CSS Tricks"
                ],
                "status": "pending"
            },
            {
                "phase": "Phase 2: JavaScript Fundamentals",
                "duration": "3-4 months",
                "level": "Beginner",
                "description": "Master JavaScript and DOM manipulation",
                "skills": ["JavaScript", "DOM", "ES6+", "Async/Await", "Fetch API"],
                "resources": [
                    "JavaScript.info",
                    "Eloquent JavaScript (Book)",
                    "JavaScript30 Challenge"
                ],
                "status": "pending"
            },
            {
                "phase": "Phase 3: Frontend Framework",
                "duration": "3-4 months",
                "level": "Intermediate",
                "description": "Learn a modern frontend framework",
                "skills": ["React", "Vue.js", "State Management", "Component Design", "Hooks"],
                "resources": [
                    "React Official Documentation",
                    "Scrimba React Course",
                    "Vue Mastery"
                ],
                "status": "pending"
            },
            {
                "phase": "Phase 4: Backend Development",
                "duration": "3-4 months",
                "level": "Intermediate",
                "description": "Build server-side applications and APIs",
                "skills": ["Node.js", "Express.js", "REST APIs", "Databases", "Authentication"],
                "resources": [
                    "Node.js Documentation",
                    "Express.js Guide",
                    "MongoDB University"
                ],
                "status": "pending"
            },
            {
                "phase": "Phase 5: DevOps & Deployment",
                "duration": "2-3 months",
                "level": "Advanced",
                "description": "Deploy and maintain web applications",
                "skills": ["Git", "CI/CD", "Docker", "Cloud Hosting", "Performance Optimization"],
                "resources": [
                    "Git & GitHub Training",
                    "Netlify/Vercel Documentation",
                    "Web Performance Best Practices"
                ],
                "status": "pending"
            }
        ]
    }
}

# Lazy loaded cached data
_df = None
_careers_list = None

def _initialize_data():
    """Lazy load and cache the careers data with error handling."""
    global _df, _careers_list
    if _df is None or _careers_list is None:
        try:
            if not os.path.exists(DATA_PATH):
                raise FileNotFoundError(f"Career data file not found at: {DATA_PATH}")
            _df = pd.read_csv(DATA_PATH)
            _careers_list = _df["career_title"].tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize roadmap generator: {str(e)}")


def get_all_careers():
    """Get list of all available careers from CSV (cached)"""
    try:
        _initialize_data()
        return _careers_list
    except Exception as e:
        logger.error(f"Error loading careers: {e}")
        return []


def generate_roadmap(career_name):
    """Generate a roadmap for a specific career"""
    # Check if we have a predefined template
    if career_name in ROADMAP_TEMPLATES:
        roadmap = ROADMAP_TEMPLATES[career_name].copy()
        roadmap['career'] = career_name
        return roadmap

    # Otherwise, generate a generic roadmap based on skills from CSV
    try:
        _initialize_data()
        career_data = _df[_df['career_title'] == career_name].iloc[0]

        skills = [s.strip() for s in career_data['required_skills'].split(',')]

        # Split skills into phases (fixed logic: min 3, max 5 skills per phase)
        skills_per_phase = min(5, max(3, len(skills) // 3))
        phases = []

        for i in range(0, len(skills), skills_per_phase):
            phase_skills = skills[i:i+skills_per_phase]
            phase_num = (i // skills_per_phase) + 1

            level = "Beginner" if phase_num == 1 else "Intermediate" if phase_num <= 3 else "Advanced"

            phases.append({
                "phase": f"Phase {phase_num}: Learning Stage {phase_num}",
                "duration": "3-4 months",
                "level": level,
                "description": f"Master essential skills for {career_name}",
                "skills": phase_skills,
                "resources": [
                    "Online courses and tutorials",
                    "Official documentation",
                    "Practice projects"
                ],
                "status": "pending"
            })

        return {
            "career": career_name,
            "description": f"Structured learning path to become a {career_name}",
            "phases": phases
        }

    except Exception as e:
        logger.error(f"Error generating roadmap for {career_name}: {e}")
        return None


def get_roadmap_for_skill_level(career_name, user_skills):
    """
    Get a personalized roadmap based on user's current skills.
    Mark phases as completed if user already has those skills.
    """
    roadmap = generate_roadmap(career_name)
    if not roadmap:
        return None

    user_skills_lower = set([s.strip().lower() for s in user_skills])

    for phase in roadmap['phases']:
        phase_skills_lower = set([s.lower() for s in phase['skills']])

        # Check if user has all skills in this phase
        if phase_skills_lower.issubset(user_skills_lower):
            phase['status'] = 'completed'
        # Check if user has some skills in this phase
        elif phase_skills_lower & user_skills_lower:
            phase['status'] = 'current'
        else:
            phase['status'] = 'pending'

    return roadmap
