# SkillOrbit — AI-Powered Career Guidance Platform

SkillOrbit is a Flask-based web application that helps students, fresh graduates, and professionals make informed career decisions. It uses a trained Machine Learning model to analyze a user's skills and recommend the most suitable career paths, identify skill gaps, and generate personalized learning roadmaps.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Machine Learning Model](#machine-learning-model)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)

---

## Project Overview

Career decision-making is one of the most challenging tasks for students and professionals. SkillOrbit solves this by combining Machine Learning, Natural Language Processing, and data analytics into a single platform. A user simply enters their skills or uploads their resume — the system handles everything else.

The platform covers four core modules:

1. **Career Recommendation** — ML model predicts best-fit careers based on user skills
2. **Skill Gap Analysis** — identifies which skills the user has and which are missing
3. **Learning Roadmap** — generates a phase-by-phase learning plan for any career
4. **Career Simulator** — shows salary projections and market trends for a chosen career

---

## Features

- **Resume Analyzer** — upload a PDF or DOCX resume; the system extracts skills automatically using NLP (spaCy + fuzzy matching)
- **ML-based Career Recommendations** — trained Random Forest classifier predicts career match with confidence percentage
- **Skill Gap Analysis** — exact and partial skill matching against required skills for any career
- **Personalized Roadmap Generator** — phase-by-phase learning path based on current skill level
- **Career Market Simulator** — 5-year salary projection and demand index based on market trends
- **Petri Net Simulation** — animated visualization of the resume analysis workflow
- **Retrain Endpoint** — `/api/train-model` allows retraining the ML model on demand

---

## Machine Learning Model

### Algorithm
**Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)

### Why Random Forest?
- Handles high-dimensional binary feature vectors well
- Robust against overfitting with many trees
- Returns `predict_proba()` — confidence scores for every career class
- Works well even with synthetically generated training data

### Training Pipeline

```
careers.csv (60 careers, ~300 unique skills)
        ↓
Synthetic Data Generation (6000 labeled samples)
  - 30 core samples per career (full skill set)
  - 50 partial samples per career (50–85% of skills)
  - 20 noise samples per career (skills + random extras)
        ↓
Feature Encoding — MultiLabelBinarizer
  skills list → binary vector  e.g. ["python","sql"] → [0,1,0,...,1,0]
        ↓
Train / Test Split — 80% train, 20% test (stratified)
        ↓
model.fit(X_train, y_train)  ← actual ML training
        ↓
Evaluation — accuracy_score + classification_report
        ↓
joblib.dump() → models/career_model.pkl
               models/mlb.pkl
```

### Evaluation Metrics
| Metric | Value |
|--------|-------|
| Accuracy | ~98% on test set |
| Evaluation method | Train/test split (stratified) |
| Report | Per-career precision, recall, F1-score |

### At Prediction Time (Flask)
```
User skills input
      ↓
MultiLabelBinarizer.transform()   ← same encoder used in training
      ↓
model.predict_proba()             ← confidence % for each career
      ↓
Top 5 careers ranked by confidence
```

### Model Files
The trained model is **not stored in the repository**. It is generated locally by running the training script. This is standard practice — code is versioned, models are generated fresh.

```
models/                  ← auto-created by train_model.py
├── career_model.pkl     ← trained Random Forest
└── mlb.pkl              ← fitted MultiLabelBinarizer
```

---

## Project Structure

```
SkillOrbit/
├── backend/
│   ├── app.py               # Flask application — all API routes
│   └── resume_analyzer.py   # NLP-based resume parsing + Petri Net
├── data/
│   └── careers.csv          # 60 careers with required skills, salary, demand
├── frontend/
│   ├── index.html           # Landing page
│   ├── advisor.html         # Career advisor (skill input + recommendations)
│   ├── resume_analyzer.html # Resume upload and analysis
│   ├── roadmap.html         # Learning roadmap viewer
│   └── nav.css              # Shared navigation styles
├── ML/
│   ├── train_model.py       # Training script — run this first
│   ├── recommender.py       # Loads trained model, returns career matches
│   ├── skill_gap.py         # Skill gap and partial match analysis
│   ├── roadmap_generator.py # Phase-by-phase roadmap generation
│   └── career_simulator.py  # Salary projection and market trend simulation
├── models/                  # Auto-generated (not in repo)
│   ├── career_model.pkl
│   └── mlb.pkl
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/SkillOrbit.git
cd SkillOrbit
```

### Step 2 — Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Train the ML model

This step generates the `models/` folder with the trained classifier.

```bash
python ML/train_model.py
```

Expected output:
```
Training model...
Done! Accuracy: 98.5%
Saved to models/career_model.pkl
```

---

## How to Run

```bash
python backend/app.py
```

Open your browser and go to: `http://127.0.0.1:5000`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/recommend` | Get top 5 career matches from skills |
| POST | `/api/skill-gap` | Get skill gap for a specific career |
| POST | `/api/roadmap` | Get learning roadmap for a career |
| GET  | `/api/careers` | List all available careers |
| POST | `/api/career-simulation` | Get salary projection and market trend |
| POST | `/api/analyze-resume` | Upload resume and extract skills |
| POST | `/api/petri-simulate` | Run Petri Net workflow simulation |
| POST | `/api/train-model` | Retrain the ML model on demand |
| POST | `/api/prioritize-skills` | Prioritize missing skills by frequency |

### Example Request — Career Recommendation

```bash
POST /api/recommend
Content-Type: application/json

{
  "skills": ["Python", "Machine Learning", "TensorFlow", "NLP"]
}
```

```json
{
  "user_skills": ["Python", "Machine Learning", "TensorFlow", "NLP"],
  "matches": [
    {
      "career": "NLP Engineer",
      "score": 94.5,
      "salary_min": 85000,
      "salary_max": 150000,
      "demand_level": 9,
      "market_trend": "Rising"
    }
  ]
}
```

---

## Technologies Used

| Category | Technology |
|----------|------------|
| Backend | Flask 3.x |
| ML / Data | scikit-learn, pandas, numpy, joblib |
| NLP | spaCy, rapidfuzz, PyMuPDF |
| Frontend | HTML, CSS, JavaScript |
| Resume Parsing | PyMuPDF (PDF), python-docx (DOCX) |
| Model | Random Forest Classifier |
| Feature Encoding | MultiLabelBinarizer |
| Formal Modeling | Petri Net (custom implementation) |

---


## Notes

- The `models/` directory is excluded from the repository via `.gitignore`
- Always run `python ML/train_model.py` after cloning before starting the server
- The training script generates fresh model files each time it runs