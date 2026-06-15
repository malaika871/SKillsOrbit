from flask import Flask, render_template, request, jsonify, session
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
from ML.live_simulations import (
    simulate_roadmap_petri,
    simulate_career_journey,
    simulate_skill_gap_pipeline,
    simulate_markov_career,
    simulate_job_queue,
    simulate_skill_extraction,
)

# Import Skill Test module
from backend.skill_test import (
    load_questions, check_answers, save_test_result,
    get_test_history, get_leaderboard, get_available_skills,
    get_next_adaptive_question, update_ability, predict_difficulty,
)

# Lazy-loaded extractor
_extractor = None

def get_extractor():
    global _extractor
    if _extractor is None:
        try:
            _extractor = SkillExtractor()
        except OSError as e:
            raise RuntimeError(f"Failed to initialize resume analyzer: {e}") from e
    return _extractor


# ── Page routes ────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/advisor")
def advisor():
    return render_template("advisor.html")

@app.route("/roadmap")
def roadmap():
    return render_template("roadmap.html")

@app.route("/resume-analyzer")
def resume_analyzer_page():
    return render_template("resume_analyzer.html")

@app.route("/skill-test")
def skill_test_page():
    return render_template("skill_test.html")


# ── Career & Skills API ────────────────────────────────────────────────────────

@app.route("/api/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        skills = data.get("skills", [])
        if not skills:
            return jsonify({"error": "No skills provided"}), 400
        skills_string = ", ".join(skills)
        matches = get_matches(skills_string, top_n=5)
        return jsonify({"user_skills": skills, "matches": matches}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/skill-gap", methods=["POST"])
def skill_gap_analysis():
    try:
        data = request.get_json()
        user_skills = data.get("user_skills", [])
        career_match = data.get("career_match", {})
        if not user_skills or not career_match:
            return jsonify({"error": "Missing required data"}), 400
        gap_analysis = get_detailed_skill_gap(user_skills, career_match)
        return jsonify(gap_analysis), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/careers", methods=["GET"])
def get_careers():
    try:
        careers = get_all_careers()
        return jsonify(careers), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/roadmap", methods=["POST"])
def get_roadmap():
    try:
        data = request.get_json()
        career = data.get("career", "")
        user_skills = data.get("user_skills", [])
        if not career:
            return jsonify({"error": "Career name is required"}), 400
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
    try:
        data = request.get_json()
        missing_skills = data.get("missing_skills", [])
        career_matches = data.get("career_matches", [])
        if not missing_skills:
            return jsonify({"error": "No missing skills provided"}), 400
        prioritized = prioritize_skills(missing_skills, career_matches)
        return jsonify({"prioritized_skills": prioritized}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Resume Analyzer API ────────────────────────────────────────────────────────

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

    if skills_found:
        skills_string = ", ".join(skills_found)
        raw_matches = get_matches(skills_string, top_n=5)
    else:
        raw_matches = []

    career_matches = [
        {
            "career": m["career"],
            "match_score": m["score"],
            "rank": i + 1,
            "required_skills": m.get("required_skills", []),
            "salary_min": m["salary_min"],
            "salary_max": m["salary_max"],
            "demand_level": m["demand_level"],
            "job_type": m.get("job_type", "Hybrid"),
            "market_trend": m["market_trend"],
            "salary_growth_rate": m["salary_growth_rate"],
            "automation_risk": m["automation_risk"],
            "competition_level": m.get("competition_level", 5),
        }
        for i, m in enumerate(raw_matches)
    ]

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


# ── Simulation API ─────────────────────────────────────────────────────────────

@app.route("/api/petri-simulate", methods=["POST"])
def petri_simulate():
    try:
        net = PetriNet()
        result = net.simulate_all()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/roadmap-simulate", methods=["POST"])
def roadmap_simulate():
    try:
        data = request.get_json() or {}
        career = data.get("career", "")
        user_skills = data.get("user_skills", [])
        if not career:
            return jsonify({"error": "Career name is required"}), 400
        result = simulate_roadmap_petri(career, user_skills)
        if result.get("error"):
            return jsonify(result), 404
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/career-journey-simulate", methods=["POST"])
def career_journey_simulate():
    try:
        data = request.get_json() or {}
        career = data.get("career", "")
        match_score = data.get("match_score")
        if not career:
            return jsonify({"error": "Career name is required"}), 400
        result = simulate_career_journey(career, match_score=match_score)
        if result.get("error"):
            return jsonify(result), 404
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/skill-gap-simulate", methods=["POST"])
def skill_gap_simulate():
    try:
        data = request.get_json() or {}
        user_skills = data.get("user_skills", [])
        career_match = data.get("career_match", {})
        if not user_skills or not career_match:
            return jsonify({"error": "Missing required data"}), 400
        return jsonify(simulate_skill_gap_pipeline(user_skills, career_match)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/markov-simulate", methods=["POST"])
def markov_simulate():
    try:
        data = request.get_json() or {}
        career = data.get("career", "")
        if not career:
            return jsonify({"error": "Career name is required"}), 400
        return jsonify(simulate_markov_career(
            career,
            match_score=data.get("match_score"),
            user_skills=data.get("user_skills", []),
        )), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/queue-simulate", methods=["POST"])
def queue_simulate():
    try:
        data = request.get_json() or {}
        career = data.get("career", "")
        if not career:
            return jsonify({"error": "Career name is required"}), 400
        return jsonify(simulate_job_queue(
            career,
            competition_level=int(data.get("competition_level", 5)),
            demand_level=int(data.get("demand_level", 5)),
        )), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/skill-extraction-simulate", methods=["POST"])
def skill_extraction_simulate():
    try:
        data = request.get_json() or {}
        skills = data.get("skills_found", [])
        return jsonify(simulate_skill_extraction(skills)), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/career-simulation", methods=["POST"])
def career_simulation():
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

@app.route("/api/train-model", methods=["POST"])
def train_model_endpoint():
    try:
        from ML.train_model import train
        result = train()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Skill Test API ─────────────────────────────────────────────────────────────

@app.route("/api/test/skills", methods=["GET"])
def api_test_skills():
    return jsonify({"skills": get_available_skills()})

@app.route("/api/test/start", methods=["POST"])
def api_test_start():
    data = request.get_json(force=True) or {}
    difficulty = data.get("difficulty", "all")
    skill = data.get("skill", "all")
    num_questions = int(data.get("num_questions", 10))
    adaptive = bool(data.get("adaptive", False))

    if adaptive and skill and skill.lower() != "all":
        session["adaptive_ability"] = 0.5
        session["adaptive_answered"] = []
        session["adaptive_skill"] = skill
        session["adaptive_score"] = {"correct": 0, "total": 0}
        first_q = get_next_adaptive_question(skill, [], 0.5)
        if not first_q:
            return jsonify({"error": "No questions found"}), 404
        session["adaptive_current_qid"] = first_q["question_id"]
        return jsonify({
            "mode": "adaptive",
            "question": first_q,
            "time_limit_per_question": 30,
        })

    test_id, questions = load_questions(difficulty, skill, num_questions)
    if not questions:
        return jsonify({"error": "No questions found for given filters"}), 404

    session["current_test_id"] = test_id
    session["current_test_difficulty"] = difficulty
    session["current_test_skill"] = skill

    return jsonify({
        "test_id": test_id,
        "questions": questions,
        "time_limit_per_question": 30,
        "mode": "standard",
    })

@app.route("/api/test/submit", methods=["POST"])
def api_test_submit():
    data = request.get_json(force=True) or {}
    answers = data.get("answers", [])
    test_id = data.get("test_id") or session.get("current_test_id", "unknown")
    difficulty = session.get("current_test_difficulty", "all")
    skill = session.get("current_test_skill", "all")

    result = check_answers(test_id, answers)

    user_id = session.get("user_id", "anonymous")
    save_test_result(
        user_id=user_id,
        score=result["score"],
        difficulty=difficulty,
        skill=skill,
        questions_attempted=result["total_questions"],
    )

    skills_needing_review = []
    if result["score"] < 70:
        skills_needing_review.append(skill)
    result["skills_needing_review"] = skills_needing_review

    return jsonify(result)

@app.route("/api/test/history", methods=["GET"])
def api_test_history():
    user_id = session.get("user_id", request.args.get("user_id", "anonymous"))
    return jsonify({"history": get_test_history(user_id)})

@app.route("/api/test/leaderboard", methods=["GET"])
def api_test_leaderboard():
    top_n = int(request.args.get("top", 10))
    return jsonify({"leaderboard": get_leaderboard(top_n)})

@app.route("/api/test/adaptive/answer", methods=["POST"])
def api_adaptive_answer():
    data = request.get_json(force=True) or {}
    question_id = int(data.get("question_id"))
    selected = str(data.get("selected", "")).upper()

    result = check_answers("adaptive", [{"question_id": question_id, "selected": selected}])
    feedback_item = result["feedback"][0] if result["feedback"] else {}
    is_correct = feedback_item.get("correct", False)
    difficulty = data.get("difficulty", "medium")

    ability = session.get("adaptive_ability", 0.5)
    answered = session.get("adaptive_answered", [])
    score_state = session.get("adaptive_score", {"correct": 0, "total": 0})

    ability = update_ability(ability, is_correct, difficulty)
    answered.append(question_id)
    score_state["total"] += 1
    if is_correct:
        score_state["correct"] += 1

    session["adaptive_ability"] = ability
    session["adaptive_answered"] = answered
    session["adaptive_score"] = score_state

    skill = session.get("adaptive_skill", "Python")
    next_q = get_next_adaptive_question(skill, answered, ability)

    return jsonify({
        "feedback": feedback_item,
        "current_ability": ability,
        "next_question": next_q,
        "score_so_far": round(score_state["correct"] / score_state["total"] * 100, 1)
                        if score_state["total"] > 0 else 0,
    })

@app.route("/api/update-user-skills", methods=["POST"])
def api_update_user_skills():
    data = request.get_json(force=True) or {}
    skill = data.get("skill")
    action = data.get("action", "needs_review")

    user_skills = session.get("user_skills", [])
    user_needs_review = session.get("user_needs_review", [])

    if action == "remove" and skill in user_skills:
        user_skills.remove(skill)
    elif action == "needs_review" and skill not in user_needs_review:
        user_needs_review.append(skill)

    session["user_skills"] = user_skills
    session["user_needs_review"] = user_needs_review

    return jsonify({
        "success": True,
        "user_skills": user_skills,
        "needs_review": user_needs_review,
        "redirect_roadmap": "/roadmap",
    })

@app.route("/api/admin/predict-difficulty", methods=["POST"])
def api_predict_difficulty():
    data = request.get_json(force=True) or {}
    text = data.get("question_text", "")
    if not text:
        return jsonify({"error": "question_text is required"}), 400
    label = predict_difficulty(text)
    return jsonify({"predicted_difficulty": label, "question_text": text})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting SkillOrbit...")
    try:
        import pandas
        import spacy
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    app.run(debug=True)