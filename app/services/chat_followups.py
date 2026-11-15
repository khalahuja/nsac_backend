"""Short, contextual next steps for chat replies without another model call."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..api.chat import ChatRequest, Reply

_GENERIC = {"tell me more", "learn more", "more information", "continue", "yes", "no",
            "anything else", "what else", "help me", "show more"}


def _key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def clean_options(value: object) -> list[str]:
    """Reject incomplete labels rather than cutting off a useful question."""
    if not isinstance(value, list):
        return []
    kept, seen = [], set()
    for item in value:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        key = _key(text)
        if not key or key in seen or key in _GENERIC or len(text) > 60:
            continue
        kept.append(text)
        seen.add(key)
        if len(kept) == 4:
            break
    return kept


_PAGE_OPTIONS = {
    "selene": ["Explain sunlight at the selected site", "Open the Earth-visibility plot", "Compare this site with Apollo 11"],
    "horizon": ["Explain Earth visibility at this site", "Find windows with Earth in view", "Compare this site with Apollo 11"],
    "windows": ["Explain the current search constraints", "Open the Earth-visibility plot", "Explain the sampling limits"],
    "forecast": ["Explain the temperature model", "Find science windows for this period", "Explain the forecast's uncertainty"],
    "sites": ["Explain the site-ranking criteria", "Compare the terrain and ice evidence", "Explain which site data are missing"],
    "compare": ["Explain the tradeoffs between these sites", "Compare their Earth visibility", "Explain which terrain data are missing"],
    "swarm": ["Explain how these dates are screened", "Check terrain and landing hazards", "Explain the mission assumptions"],
    "mission-plan": ["Explain how these dates are screened", "Check terrain and landing hazards", "Explain the mission assumptions"],
    "surface": ["Explain the terrain data limits", "Check the landing footprint", "Open the Earth-visibility plot"],
    "trajectories": ["Explain the transfer tradeoffs", "Explain the trajectory assumptions", "Open the lunar orbit calculator"],
    "landing": ["Which inputs most affect landing risk?", "Explain the landing model's limits", "Check terrain and landing hazards"],
    "orbit": ["Explain the orbit insertion assumptions", "Which inputs most affect the timeline?", "Explain the transfer tradeoffs"],
    "water": ["Explain the evidence for water ice", "Which ice measurements are missing?", "Compare terrain and ice evidence"],
    "solar": ["Which inputs most affect solar output?", "Explain the solar model's assumptions", "Open the battery calculator"],
    "battery": ["Which inputs most affect battery mass?", "Explain the battery model's assumptions", "Open the solar power calculator"],
    "link": ["Explain the link budget assumptions", "Open the Earth-visibility plot", "Which inputs most affect link margin?"],
    "temperature": ["Explain the temperature model", "Open the temperature forecast", "Explain the thermal assumptions"],
    "mission-lab": ["Explain the mission assumptions", "Which inputs most affect mission feasibility?", "Check terrain and landing hazards"],
    "models": ["Explain the temperature model", "Which results require measured terrain?", "Explain the astronomy model's limits"],
}
_PAGE_ALIASES = {
    "selene": {"moon atlas", "moon map"},
    "windows": {"windows", "science windows"},
    "sites": {"candidate finder", "candidate sites"},
    "horizon": {"earth visibility plot", "earth visibility", "dte sky plot", "horizon"},
    "forecast": {"forecast", "temperature forecast"},
    "swarm": {"agent swarm", "mission planner"},
    "mission-plan": {"agent swarm", "mission planner"},
}


def improve_followups(reply: Reply, payload: ChatRequest, facts: dict | None = None) -> Reply:
    """Prefer the next decision, preserve clarification, and avoid repeat work."""
    asked = {_key(m.content) for m in payload.messages if m.role == "user"}
    action = reply.action or payload.active_search
    page = action.page if action else payload.page
    criteria = action.criteria if action else None
    ranking = action.site_ranking if action else None
    if ranking and ranking.criteria:
        criteria = ranking.criteria
    mission = action.mission_windows if action else None
    earth_required = bool(criteria and criteria.requireEarth)
    has_contact = earth_required or bool(criteria and criteria.minEarthHours > 0) or bool(mission and mission.min_earth_hours > 0)
    sun_required = bool(criteria and criteria.requireSun) or bool(mission and mission.min_sunlight_days > 0) or bool(ranking and ranking.min_sunlight_days > 0)
    far_side = str((facts or {}).get("side_of_moon", "")).lower().startswith("far side")

    def useful(text: str) -> bool:
        key = _key(text)
        if key in asked:
            return False
        navigation = re.fullmatch(r"(?:open|show|go to) (?:the )?(.+)", key)
        if navigation and navigation[1] in _PAGE_ALIASES.get(page, set()):
            return False
        adding_contact = bool(re.match(r"(?:add|require|include)\b.*\b(?:earth|contact)\b", key))
        if adding_contact and (far_side or earth_required or has_contact and key == "add earth contact"):
            return False
        if far_side and re.match(r"find\b.*\b(?:earth|contact)\b", key):
            return False
        if sun_required and key in {"require sunlight", "add sunlight", "require sun", "require continuous sunlight"}:
            return False
        return True

    options = [o for o in clean_options(reply.options) if useful(o)]
    clarifying = reply.action is None and (
        bool(reply.unresolved_conditions) or "?" in reply.reply
        or bool(re.search(r"\bplease (?:give|provide|specify|clarify)\b", reply.reply, re.I))
    )
    if not clarifying and len(options) < 2:
        candidates = list(_PAGE_OPTIONS.get(page, [
            "Explain the assumptions behind this answer",
            "Which inputs are needed for the next step?",
        ]))
        if action and (criteria or mission or ranking) and not has_contact and not far_side:
            candidates.insert(0, "Require Earth contact throughout the window")
        for option in candidates:
            if useful(option) and _key(option) not in {_key(o) for o in options}:
                options.append(option)
            if len(options) >= 3:
                break
    return reply.model_copy(update={"options": options[:3]})
