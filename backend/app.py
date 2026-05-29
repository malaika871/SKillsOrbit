from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import sys

# Configure Flask to use the frontend folder for templates
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = "skillorbit_secret"

# Add parent directory to path to import ML module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ML.recommender import get_matches
from ML.skill_gap import get_skill_gap, get_detailed_skill_gap, prioritize_skills
from ML.roadmap_generator import get_all_careers, generate_roadmap, get_roadmap_for_skill_level

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


if __name__ == "__main__":
    app.run(debug=True)
