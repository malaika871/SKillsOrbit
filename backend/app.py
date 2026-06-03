from flask import Flask, render_template, request, jsonify
import os
import sys

# Add project root to path to import modules properly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure Flask to use the frontend folder for templates
template_dir = os.path.join(PROJECT_ROOT, 'frontend')
app = Flask(__name__, template_folder=template_dir, static_folder=template_dir, static_url_path='/static')
app.secret_key = "skillorbit_secret"

# Import ML modules
from ML.recommender import get_matches
from ML.skill_gap import get_detailed_skill_gap, prioritize_skills
from ML.roadmap_generator import get_all_careers, generate_roadmap, get_roadmap_for_skill_level
from ML.career_simulator import simulate_career
from backend.resume_analyzer import SkillExtractor, PetriNet

# Lazy-loaded extractor
_extractor = None

def get_extractor():
    """Get or create the SkillExtractor instance (lazy-loaded with error handling)"""
    global _extractor
    if _extractor is None:
        try:
            _extractor = SkillExtractor()
        except OSError as e:
            raise RuntimeError(f"Failed to initialize resume analyzer: {e}") from e
    return _extractor

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/advisor")
def advisor():
    return render_template("advisor.html")

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """API endpoint to get career recommendations based on user skills"""
    try:
        data = request.get_json()
        skills = data.get("skills", [])

        if not skills:
            return jsonify({"error": "No skills provided"}), 400

        # Convert skills list to comma-separated string
        skills_string = ", ".join(skills)

        # Get top 5 career matches
        matches = get_matches(skills_string, top_n=5)

        # Add user skills to response for comparison
        response = {
            "user_skills": skills,
            "matches": matches
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/skill-gap", methods=["POST"])
def skill_gap_analysis():
    """API endpoint to get detailed skill gap analysis for a specific career"""
    try:
        data = request.get_json()
        user_skills = data.get("user_skills", [])
        career_match = data.get("career_match", {})

        if not user_skills or not career_match:
            return jsonify({"error": "Missing required data"}), 400

        # Get detailed skill gap analysis
        gap_analysis = get_detailed_skill_gap(user_skills, career_match)

        return jsonify(gap_analysis), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/roadmap")
def roadmap():
    return render_template("roadmap.html")

@app.route("/api/careers", methods=["GET"])
def get_careers():
    """API endpoint to get list of all available careers"""
    try:
        careers = get_all_careers()
        return jsonify(careers), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/roadmap", methods=["POST"])
def get_roadmap():
    """API endpoint to get roadmap for a specific career"""
    try:
        data = request.get_json()
        career = data.get("career", "")
        user_skills = data.get("user_skills", [])

        if not career:
            return jsonify({"error": "Career name is required"}), 400

        # Generate roadmap (personalized if user_skills provided)
        if user_skills:
            roadmap = get_roadmap_for_skill_level(career, user_skills)
        else:
            roadmap = generate_roadmap(career)

        if not roadmap:
            return jsonify({"error": "Roadmap not found for this career"}), 404

        return jsonify(roadmap), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/prioritize-skills", methods=["POST"])
def prioritize_skills_endpoint():
    """API endpoint to prioritize skills to learn based on career matches"""
    try:
        data = request.get_json()
        missing_skills = data.get("missing_skills", [])
        career_matches = data.get("career_matches", [])

        if not missing_skills:
            return jsonify({"error": "No missing skills provided"}), 400

        # Prioritize skills
        prioritized = prioritize_skills(missing_skills, career_matches)

        return jsonify({"prioritized_skills": prioritized}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/resume-analyzer")
def resume_analyzer_page():
    return render_template("resume_analyzer.html")


@app.route("/api/analyze-resume", methods=["POST"])
def analyze_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        extractor = get_extractor()
        file_bytes = file.read()
        info = extractor.analyze(file_bytes, file.filename)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse resume: {str(e)}"}), 500

    skills_found = info["skills_found"]
    resume_info = {k: v for k, v in info.items() if k != "_text"}

    # Career matching via existing TF-IDF logic
    if skills_found:
        skills_string = ", ".join(skills_found)
        raw_matches = get_matches(skills_string, top_n=5)
    else:
        raw_matches = []

    career_matches = [
        {"career": m["career"], "match_score": m["score"], "rank": i + 1}
        for i, m in enumerate(raw_matches)
    ]

    # Skill gap for top career
    skill_gaps = {"top_career": "", "missing_skills": [], "you_have": []}
    if raw_matches:
        top = raw_matches[0]
        gap = get_detailed_skill_gap(skills_found, top)
        skill_gaps = {
            "top_career": top["career"],
            "missing_skills": gap["missing_skills"],
            "you_have": gap["exact_matches"],
        }

    return jsonify({
        "resume_info": resume_info,
        "career_matches": career_matches,
        "skill_gaps": skill_gaps,
    }), 200


@app.route("/api/petri-simulate", methods=["POST"])
def petri_simulate():
    try:
        net = PetriNet()
        result = net.simulate_all()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/career-simulation", methods=["POST"])
def career_simulation():
    """API endpoint for market trend and salary simulation for a career."""
    try:
        data = request.get_json() or {}
        career = data.get("career", "")
        match_score = data.get("match_score")

        if not career:
            return jsonify({"error": "Career name is required"}), 400

        result = simulate_career(career, match_score=match_score)
        if not result:
            return jsonify({"error": f"Career '{career}' not found"}), 404

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Startup checks
    print("Starting SkillOrbit app...")
    try:
        # Test import basic dependencies
        import pandas
        import spacy
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        sys.exit(1)
    
    app.run(debug=True)
