"""Live simulation engines for SkillOrbit (Petri Net, journey, gap pipeline, Markov, queue)."""

from __future__ import annotations

import random
from ML.career_simulator import simulate_career
from ML.roadmap_generator import get_roadmap_for_skill_level, generate_roadmap
from ML.skill_gap import get_detailed_skill_gap, prioritize_skills


def _petri_log(step, event, description, active_place, fired_transition=None, extra=None):
    entry = {
        "step": step,
        "event": event,
        "description": description,
        "active_place": active_place,
        "fired_transition": fired_transition,
    }
    if extra:
        entry.update(extra)
    return entry


def simulate_roadmap_petri(career: str, user_skills: list | None = None) -> dict:
    """Petri net over roadmap phases — token moves as phases complete."""
    user_skills = user_skills or []
    if user_skills:
        roadmap = get_roadmap_for_skill_level(career, user_skills)
    else:
        roadmap = generate_roadmap(career)

    if not roadmap or not roadmap.get("phases"):
        return {"error": f"No roadmap for '{career}'", "log": [], "places": [], "transitions": []}

    phases = roadmap["phases"]
    places = [f"P{i}" for i in range(len(phases))]
    place_labels = {
        f"P{i}": p.get("phase", f"Phase {i + 1}") for i, p in enumerate(phases)
    }
    transitions = []
    for i in range(len(phases) - 1):
        transitions.append({
            "id": f"T{i}",
            "label": "complete_phase",
            "input": f"P{i}",
            "output": f"P{i + 1}",
            "phase_name": phases[i].get("phase", ""),
        })

    tokens = {p: 0 for p in places}
    log = [_petri_log(0, "init", f"Learning path started for {career}", places[0])]

    for i, phase in enumerate(phases):
        pid = places[i]
        tokens[pid] = 1
        status = phase.get("status", "pending")
        status_msg = {
            "completed": "Phase completed",
            "current": "Phase in progress",
            "pending": "Phase pending",
        }.get(status, "Phase")

        log.append(_petri_log(
            len(log), "place",
            f"{status_msg}: {phase.get('phase', pid)} ({phase.get('duration', '')})",
            pid,
            extra={"phase_status": status, "skills": phase.get("skills", [])[:6]},
        ))

        if i < len(transitions):
            t = transitions[i]
            if status in ("completed", "current"):
                tokens[t["input"]] = 0
                tokens[t["output"]] = 1
                log.append(_petri_log(
                    len(log), "fire",
                    f"Transition {t['id']}: advancing to next phase",
                    t["output"],
                    t["id"],
                ))

    return {
        "career": career,
        "places": places,
        "place_labels": place_labels,
        "transitions": transitions,
        "final_tokens": tokens,
        "phases": phases,
        "log": log,
        "model": "Petri Net — Learning Path",
    }


def simulate_career_journey(career: str, match_score: float | None = None, years: int = 5) -> dict:
    """Year-by-year career journey timeline for live step animation."""
    market = simulate_career(career, match_score=match_score, years=years)
    if not market:
        return {"error": f"Career '{career}' not found", "steps": []}

    salary_proj = market.get("projected_salary", [])
    demand_proj = market.get("demand_projection", [])
    match = match_score if match_score is not None else 50

    stages = [
        (0, "Current State", f"Starting point with {match}% skill match for {career}."),
        (1, "Upskill Phase", "Close skill gaps through focused learning and projects."),
        (2, "Entry / Transition", f"Apply for {career} roles aligned with your profile."),
    ]

    steps = []
    step_idx = 0
    for year_idx, sal in enumerate(salary_proj[: years + 1]):
        year = sal.get("year", str(year_idx))
        if year_idx < 3:
            title, desc = stages[year_idx][1], stages[year_idx][2]
        elif year_idx == years:
            title, desc = "Career Milestone", f"Projected market position for {career}."
        else:
            title, desc = f"Growth Year {year_idx}", "Salary and demand continue trending."

        demand_pt = demand_proj[year_idx] if year_idx < len(demand_proj) else {}
        steps.append({
            "step": step_idx,
            "year": year,
            "title": title,
            "description": desc,
            "salary_mid": sal.get("mid", 0),
            "salary_min": sal.get("min", 0),
            "salary_max": sal.get("max", 0),
            "demand_index": demand_pt.get("demand_index", 50),
            "match_score": min(99, match + year_idx * 5) if year_idx > 0 else match,
        })
        step_idx += 1

    return {
        "career": career,
        "currency": market.get("currency", "PKR"),
        "salary_period": market.get("salary_period", "monthly"),
        "market_trend": market.get("market_trend"),
        "steps": steps,
        "model": "Discrete-Event Timeline — Career Journey",
    }


def simulate_skill_gap_pipeline(user_skills: list, career_match: dict, max_skills: int = 8) -> dict:
    """Pipeline simulation: each missing skill learned raises match %."""
    gap = get_detailed_skill_gap(user_skills, career_match)
    prioritized = prioritize_skills(gap["missing_skills"], [career_match])
    to_learn = [p["skill"] for p in prioritized[:max_skills]]

    if not to_learn:
        to_learn = gap["missing_skills"][:max_skills]

    base_match = gap.get("match_percentage", 0)
    total = max(gap.get("total_required", 1), 1)
    steps = [{
        "step": 0,
        "event": "start",
        "skill": None,
        "description": f"Current match: {base_match}% for {gap.get('career', 'career')}",
        "match_percentage": base_match,
    }]

    current_match = base_match
    for i, skill in enumerate(to_learn, start=1):
        boost = min(12, 100 / total)
        current_match = min(99.0, round(current_match + boost, 1))
        steps.append({
            "step": i,
            "event": "learn",
            "skill": skill,
            "description": f"Learn {skill} — match improves",
            "match_percentage": current_match,
        })

    steps.append({
        "step": len(steps),
        "event": "complete",
        "skill": None,
        "description": f"Target match after upskilling: {current_match}%",
        "match_percentage": current_match,
    })

    return {
        "career": gap.get("career", ""),
        "initial_match": base_match,
        "final_match": current_match,
        "skills_to_learn": to_learn,
        "steps": steps,
        "model": "Pipeline — Skill Gap Closure",
    }


def simulate_markov_career(career: str, match_score: float | None = None, user_skills: list | None = None) -> dict:
    """Markov chain over career stages with transition probabilities."""
    random.seed(hash(career) % 2**32)
    match = (match_score or 50) / 100.0
    skill_bonus = min(0.25, len(user_skills or []) * 0.02)

    states = ["Student / Fresher", "Junior", "Mid-Level", "Senior", "Lead / Expert"]
    base_probs = [
        [0.0, 0.55, 0.30, 0.10, 0.05],
        [0.0, 0.20, 0.45, 0.25, 0.10],
        [0.0, 0.05, 0.25, 0.45, 0.25],
        [0.0, 0.0, 0.10, 0.35, 0.55],
        [0.0, 0.0, 0.0, 0.20, 0.80],
    ]

    transitions = []
    for i, src in enumerate(states):
        for j, dst in enumerate(states):
            if i == j or base_probs[i][j] <= 0:
                continue
            prob = min(0.95, base_probs[i][j] + skill_bonus + match * 0.1)
            transitions.append({
                "from": src,
                "to": dst,
                "probability": round(prob * 100, 1),
            })

    path = [states[0]]
    current = 0
    path_log = [{
        "step": 0,
        "state_index": 0,
        "state": states[0],
        "description": f"Starting as {states[0]} targeting {career}",
    }]

    for step in range(1, 5):
        probs = base_probs[current][:]
        for j in range(len(probs)):
            if j > current:
                probs[j] = min(0.95, probs[j] + skill_bonus + match * 0.08)
        total = sum(probs)
        if total <= 0:
            break
        probs = [p / total for p in probs]
        nxt = random.choices(range(len(states)), weights=probs, k=1)[0]
        if nxt <= current:
            nxt = min(current + 1, len(states) - 1)
        current = nxt
        path.append(states[current])
        path_log.append({
            "step": step,
            "state_index": current,
            "state": states[current],
            "description": f"Year {step}: transition to {states[current]}",
        })
        if current >= len(states) - 1:
            break

    return {
        "career": career,
        "states": states,
        "transitions": transitions,
        "path": path,
        "path_log": path_log,
        "match_score": match_score,
        "model": "Markov Chain — Career Progression",
    }


def simulate_job_queue(career: str, competition_level: int = 5, demand_level: int = 5) -> dict:
    """Queueing simulation: applicants → screening → interview → offer."""
    random.seed(hash(career + str(competition_level)) % 2**32)

    service_rate = max(1, demand_level)
    arrival_rate = max(2, competition_level + 3)
    slots = max(1, demand_level // 2 + 1)

    events = []
    queue_len = 0
    processed = 0
    offers = 0
    t = 0

    events.append({
        "time": t,
        "event": "init",
        "queue_length": 0,
        "description": f"Job market opens for {career} — {slots} interview slots",
    })

    for step in range(1, 16):
        t += 1
        arrivals = random.randint(1, max(1, arrival_rate // 2))
        queue_len += arrivals
        served = min(queue_len, service_rate)
        queue_len -= served
        processed += served
        new_offers = 1 if random.random() < (demand_level / 12) and served > 0 else 0
        offers += new_offers

        label = "arrival"
        desc = f"{arrivals} applicant(s) arrive — queue: {queue_len}"
        if served > 0:
            label = "serve"
            desc = f"{served} screened — queue: {queue_len}"
        if new_offers:
            label = "offer"
            desc = f"Offer extended! Total offers: {offers}"

        events.append({
            "time": t,
            "event": label,
            "queue_length": queue_len,
            "processed": processed,
            "offers": offers,
            "slots": slots,
            "description": desc,
        })

    success_rate = round((offers / max(processed, 1)) * 100, 1)
    return {
        "career": career,
        "competition_level": competition_level,
        "demand_level": demand_level,
        "events": events,
        "summary": {
            "total_processed": processed,
            "total_offers": offers,
            "success_rate": success_rate,
            "avg_wait": round(queue_len / max(slots, 1), 1),
        },
        "model": "Queueing Network — Job Market",
    }


def simulate_skill_extraction(skills_found: list, sample_size: int = 12) -> dict:
    """NLP skill extraction pipeline for resume analyzer animation."""
    skills = skills_found[:sample_size] if skills_found else [
        "Python", "SQL", "Communication", "Git", "JavaScript",
    ]
    stages = ["tokenize", "ner_scan", "taxonomy_match", "fuzzy_match", "dedupe"]

    steps = [{
        "step": 0,
        "stage": "upload",
        "description": "Resume text loaded into NLP pipeline",
        "skills_found": [],
    }]

    found_so_far = []
    for i, stage in enumerate(stages, start=1):
        chunk = skills[: max(1, (i * len(skills)) // len(stages))]
        found_so_far = list(dict.fromkeys(found_so_far + chunk))
        steps.append({
            "step": i,
            "stage": stage,
            "description": {
                "tokenize": "Tokenizing document text",
                "ner_scan": "Running spaCy NER scan",
                "taxonomy_match": "Matching skill taxonomy terms",
                "fuzzy_match": "Fuzzy matching near-miss skills",
                "dedupe": "Deduplicating and ranking skills",
            }.get(stage, stage),
            "skills_found": found_so_far.copy(),
        })

    return {
        "total_skills": len(skills_found) if skills_found else len(skills),
        "steps": steps,
        "model": "NLP Pipeline — Skill Extraction",
    }
