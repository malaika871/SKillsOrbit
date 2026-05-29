def get_skill_gap(user_skills_string, required_skills_list):
    """
    Analyzes the skill gap between user's skills and required skills for a career.

    Args:
        user_skills_string (str): Comma-separated string of user's skills
        required_skills_list (list): List of required skills for a career

    Returns:
        dict: Dictionary containing 'matched' and 'missing' skill lists
    """
    user_skills = set([s.strip().lower() for s in user_skills_string.split(",")])
    required = set([s.strip().lower() for s in required_skills_list])
    missing = required - user_skills
    matched = required & user_skills
    return {
        "matched": list(matched),
        "missing": list(missing)
    }


def get_detailed_skill_gap(user_skills_list, career_match):
    """
    Get detailed skill gap analysis for a specific career match.

    Args:
        user_skills_list (list): List of user's skills
        career_match (dict): Career match dictionary from recommender

    Returns:
        dict: Detailed skill gap analysis
    """
    user_skills_lower = set([s.strip().lower() for s in user_skills_list])
    required_skills = career_match.get('required_skills', [])
    required_skills_lower = set([s.strip().lower() for s in required_skills])

    # Find exact matches
    exact_matches = user_skills_lower & required_skills_lower

    # Find missing skills
    missing_skills = required_skills_lower - user_skills_lower

    # Find partial matches (fuzzy matching)
    partial_matches = []
    for user_skill in user_skills_lower:
        for req_skill in required_skills_lower:
            if user_skill in req_skill or req_skill in user_skill:
                if req_skill not in exact_matches:
                    partial_matches.append(req_skill)

    # Calculate match percentage
    total_required = len(required_skills_lower)
    matched_count = len(exact_matches) + len(partial_matches)
    match_percentage = (matched_count / total_required * 100) if total_required > 0 else 0

    return {
        "career": career_match.get('career', ''),
        "exact_matches": list(exact_matches),
        "partial_matches": list(set(partial_matches)),
        "missing_skills": list(missing_skills - set(partial_matches)),
        "match_percentage": round(match_percentage, 1),
        "total_required": total_required,
        "matched_count": matched_count
    }


def prioritize_skills(missing_skills, career_matches):
    """
    Prioritize missing skills based on their frequency across top career matches.

    Args:
        missing_skills (list): List of missing skills
        career_matches (list): List of career match dictionaries

    Returns:
        list: Prioritized skills with frequency count
    """
    skill_frequency = {}

    for skill in missing_skills:
        skill_lower = skill.lower()
        count = 0
        for career in career_matches:
            required_skills = [s.lower() for s in career.get('required_skills', [])]
            if skill_lower in required_skills:
                count += 1
        skill_frequency[skill] = count

    # Sort by frequency (descending)
    prioritized = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)

    return [{"skill": skill, "frequency": freq} for skill, freq in prioritized]
