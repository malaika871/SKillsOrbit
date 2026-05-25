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

if __name__ == "__main__":
    app.run(debug=True)
