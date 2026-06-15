"""
scripts/predict_difficulty.py
──────────────────────────────
Demonstrates the trained difficulty classifier on new question text.

Usage:
    python scripts/predict_difficulty.py
    python scripts/predict_difficulty.py --text "What is a Python generator?"
"""

import os
import sys
import pickle
import argparse

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "difficulty_model.pkl")

EXAMPLE_QUESTIONS = [
    ("What does len() return in Python?", "easy"),
    ("What is the difference between a list and a tuple?", "medium"),
    ("Explain the GIL and its impact on multi-threaded Python programs.", "hard"),
    ("Which SQL clause filters rows?", "easy"),
    ("What is a window function in SQL?", "hard"),
    ("What is dropout regularization in deep learning?", "hard"),
    ("How do you declare a variable in JavaScript?", "easy"),
    ("What is event bubbling?", "medium"),
]


def load_model():
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Run: python ML/train_difficulty_model.py")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(model, text: str) -> str:
    return model.predict([text])[0]


def main():
    parser = argparse.ArgumentParser(description="Predict difficulty of a question.")
    parser.add_argument("--text", type=str, default=None, help="Question text to classify")
    args = parser.parse_args()

    model = load_model()
    print("✅ Difficulty classifier loaded.\n")

    if args.text:
        label = predict(model, args.text)
        print(f"Question : {args.text}")
        print(f"Predicted: {label.upper()}")
        return

    print("─── Example Predictions ───────────────────────────────")
    print(f"{'Question':<55} {'True':>8}  {'Predicted':>10}")
    print("─" * 78)
    for q_text, true_label in EXAMPLE_QUESTIONS:
        pred = predict(model, q_text)
        match = "✅" if pred == true_label else "❌"
        print(f"{q_text[:54]:<55} {true_label:>8}  {pred:>10}  {match}")
    print()
    print("Use --text 'your question here' to predict a single question.")


if __name__ == "__main__":
    main()