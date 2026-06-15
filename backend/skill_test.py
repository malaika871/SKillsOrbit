"""
backend/skill_test.py
─────────────────────
Skill Test module for SkillOrbit.
"""

import os
import csv
import uuid
import pickle
import pandas as pd
from datetime import datetime

# ── Paths (everything relative to project root) ────────────────────────────────
# backend/skill_test.py  →  go up one level to reach project root
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BACKEND_DIR)
META_PATH    = os.path.join(ROOT_DIR, "data", "questions_metadata.csv")
RESULTS_PATH = os.path.join(ROOT_DIR, "data", "test_results.csv")
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "difficulty_model.pkl")

# ── In-memory test store (keyed by test_id) ────────────────────────────────────
_active_tests: dict = {}

# ── Lazy-load metadata ─────────────────────────────────────────────────────────
_questions_df = None

def _get_df():
    global _questions_df
    if _questions_df is None:
        if not os.path.isfile(META_PATH):
            raise FileNotFoundError(
                f"questions_metadata.csv not found at {META_PATH}. "
                "Run: python ML/train_difficulty_model.py"
            )
        _questions_df = pd.read_csv(META_PATH)
    return _questions_df


# ── Public API ─────────────────────────────────────────────────────────────────

def get_available_skills():
    return sorted(_get_df()["skill_tested"].unique().tolist())


def load_questions(difficulty=None, skill=None, num_questions=10):
    df = _get_df().copy()

    if difficulty and difficulty.lower() != "all":
        df = df[df["difficulty"].str.lower() == difficulty.lower()]

    if skill and skill.lower() != "all":
        df = df[df["skill_tested"].str.lower() == skill.lower()]

    if df.empty:
        return None, []

    sample = df.sample(n=min(num_questions, len(df)), random_state=None)

    questions = []
    for _, row in sample.iterrows():
        questions.append({
            "question_id": int(row["question_id"]),
            "text":        row["question_text"],
            "options":     [row["option_a"], row["option_b"], row["option_c"], row["option_d"]],
            "skill":       row["skill_tested"],
            "difficulty":  row["difficulty"],
        })

    test_id = str(uuid.uuid4())[:8]
    _active_tests[test_id] = {
        "questions": {q["question_id"]: q for q in questions},
        "difficulty": difficulty or "all",
        "skill": skill or "all",
    }
    return test_id, questions


def check_answers(test_id, answers):
    df = _get_df()
    feedback = []
    correct = 0

    for ans in answers:
        qid      = int(ans["question_id"])
        selected = str(ans.get("selected", "")).strip().upper()

        row = df[df["question_id"] == qid]
        if row.empty:
            feedback.append({"question_id": qid, "correct": False,
                             "correct_answer": "?", "selected": selected})
            continue

        correct_answer = str(row.iloc[0]["correct_answer"]).strip().upper()
        is_correct = (selected == correct_answer)
        if is_correct:
            correct += 1

        feedback.append({
            "question_id":    qid,
            "correct":        is_correct,
            "correct_answer": correct_answer,
            "selected":       selected,
            "text":           row.iloc[0]["question_text"],
            "options": {
                "A": row.iloc[0]["option_a"],
                "B": row.iloc[0]["option_b"],
                "C": row.iloc[0]["option_c"],
                "D": row.iloc[0]["option_d"],
            },
        })

    total = len(answers)
    score = round((correct / total * 100) if total > 0 else 0, 1)
    return {
        "score":           score,
        "correct_count":   correct,
        "total_questions": total,
        "feedback":        feedback,
    }


def save_test_result(user_id, score, difficulty, skill, questions_attempted):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "user_id", "score", "difficulty", "skill", "questions_attempted"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":           datetime.utcnow().isoformat(),
            "user_id":             user_id,
            "score":               score,
            "difficulty":          difficulty,
            "skill":               skill,
            "questions_attempted": questions_attempted,
        })


def get_test_history(user_id):
    if not os.path.isfile(RESULTS_PATH):
        return []
    df = pd.read_csv(RESULTS_PATH)
    user_df = df[df["user_id"].astype(str) == str(user_id)]
    return user_df.sort_values("timestamp", ascending=False).to_dict(orient="records")


def get_leaderboard(top_n=10):
    if not os.path.isfile(RESULTS_PATH):
        return []
    df = pd.read_csv(RESULTS_PATH)
    leaderboard = (
        df.groupby("user_id")["score"]
        .max()
        .reset_index()
        .sort_values("score", ascending=False)
        .head(top_n)
    )
    return leaderboard.to_dict(orient="records")


def get_next_adaptive_question(skill, answered_ids, current_ability):
    if current_ability < 0.4:
        target_diff = "easy"
    elif current_ability < 0.7:
        target_diff = "medium"
    else:
        target_diff = "hard"

    df = _get_df()
    pool = df[
        (df["skill_tested"].str.lower() == skill.lower()) &
        (df["difficulty"] == target_diff) &
        (~df["question_id"].isin(answered_ids))
    ]
    if pool.empty:
        pool = df[
            (df["skill_tested"].str.lower() == skill.lower()) &
            (~df["question_id"].isin(answered_ids))
        ]
    if pool.empty:
        return None

    row = pool.sample(1).iloc[0]
    return {
        "question_id": int(row["question_id"]),
        "text":        row["question_text"],
        "options":     [row["option_a"], row["option_b"], row["option_c"], row["option_d"]],
        "skill":       row["skill_tested"],
        "difficulty":  row["difficulty"],
    }


def update_ability(current_ability, is_correct, difficulty):
    diff_weight = {"easy": 0.05, "medium": 0.10, "hard": 0.15}.get(difficulty, 0.10)
    if is_correct:
        return min(1.0, current_ability + diff_weight)
    return max(0.0, current_ability - diff_weight)


# ── Difficulty Predictor ───────────────────────────────────────────────────────
_model = None

def predict_difficulty(question_text):
    global _model
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            return "unknown"
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model.predict([question_text])[0]