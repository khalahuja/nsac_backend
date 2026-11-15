"""Reject plans that omit common explicit numeric constraints before execution."""
import re


def audit_constraints(action, text):
    if action is None:
        return ""
    text = text.lower()
    for word, value in {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}.items():
        text = re.sub(r"\b" + word + r"\b", str(value), text)
    ranking = action.site_ranking
    criteria = ranking.criteria if ranking else action.criteria
    mission = action.mission_windows
    issues = []
    number = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    def duration(body):
        patterns = [number + r"\s*(days?|hours?|hrs?)\s*(?:of\s+)?" + body,
                    body + r"\s+(?:for\s+)?(?:at\s*least\s+)?" + number + r"\s*(days?|hours?|hrs?)"]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match[1]) * (24 if match[2].startswith("day") else 1)
        return None
    sun = duration(r"(?:continuous\s+)?(?:sunlight|sunshine)")
    earth = duration(r"(?:continuous\s+)?earth(?:\s+(?:in view|visibility|visible|contact))?")
    if sun is not None:
        applied = max(ranking.min_sunlight_days * 24 if ranking else 0,
                      mission.min_sunlight_days * 24 if mission else 0,
                      criteria.minHours if criteria and criteria.requireSun else 0)
        if applied < sun:
            issues.append(f"sunlight duration {sun:g} hours")
    if earth is not None:
        applied = max(mission.min_earth_hours if mission else 0,
                      max(criteria.minEarthHours, criteria.minHours if criteria.requireEarth else 0) if criteria else 0)
        if applied < earth:
            issues.append(f"Earth visibility duration {earth:g} hours")
    match = re.search(r"temperature\s+(?:between|from)\s*" + number + r"\s*(?:c|celsius)?\s*(?:and|to)\s*" + number, text)
    if match and (not criteria or criteria.minTemperature < float(match[1]) or criteria.maxTemperature > float(match[2])):
        issues.append("temperature range")
    for pattern, field, lower in [
        (r"earth\s*(?:elevation\s*)?(?:above|over|>=|>)\s*" + number, "minEarthElevation", True),
        (r"(?:irradiance|solar flux)\s*(?:above|over|at least|>=|>)\s*" + number, "minIrradiance", True),
        (r"temperature\s*(?:above|over|at least|>=|>)\s*" + number, "minTemperature", True),
        (r"temperature\s*(?:below|under|at most|<=|<)\s*" + number, "maxTemperature", False),
    ]:
        match = re.search(pattern, text)
        if match:
            applied = getattr(criteria, field) if criteria else None
            if applied is None or (applied < float(match[1]) if lower else applied > float(match[1])):
                issues.append(field)
            if field == "minEarthElevation" and criteria and not (criteria.requireEarth or criteria.minEarthHours > 0):
                issues.append("Earth visibility")
    match = re.search(r"slope\s*(?:below|under|at most|<=|<)\s*" + number, text)
    if match and (not ranking or ranking.max_slope_deg is None or ranking.max_slope_deg > float(match[1])):
        issues.append("maximum slope")
    return "The action omitted or weakened requested conditions: " + ", ".join(issues) if issues else ""
