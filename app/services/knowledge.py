"""Answers to explanatory questions about the website's own models, without an AI call.

These come from the Models & sources page and the documented limitations of each
calculation, so they are free, instant and work while the AI provider's quota
is exhausted. Only explanation-style questions are matched; anything that needs
a calculation still goes to the assistant.
"""
from __future__ import annotations

import re

_EXPLAIN = re.compile(
    r"\b(explain|what (is|are|does)|how (does|do|is|accurate|reliable)|why|limit|limitation|assumption|uncertain|accuracy|"
    r"source|data set|dataset|model|method|based on|trust|caveat)\b", re.I)

# (topic, trigger, answer). Order matters: the first matching topic wins.
TOPICS: list[tuple[str, re.Pattern[str], str]] = [
    ("landing", re.compile(r"landing (model|probability|confidence|success|safety)|dispersion|containment|landing ellipse", re.I),
     "The landing model screens touchdown risk; it is not a certified success probability. "
     "Containment uses P(r ≤ R) = 1 − exp(−R²/2σ²), which assumes unbiased, independent Gaussian north/east errors (real guidance errors can be biased or correlated). "
     "Terrain safety is the share of the 3σ landing ellipse with slope ≤ 12° on the NASA LOLA 80 m DEM; rocks and craters smaller than one 80 m pixel are not seen, and the DEM only covers 80°S to the pole. "
     "Multiplying containment by terrain safety assumes the two are independent. A flight probability needs Monte Carlo guidance simulation, high-resolution hazard maps and vehicle reliability data."),
    ("temperature", re.compile(r"temperature|thermal|cold|heat|kelvin|°c", re.I),
     "Surface temperature is a reduced-order radiative model, not a measurement. It uses albedo 0.12, emissivity 0.95 and an assumed 100 K night floor, "
     "relaxing towards equilibrium with T_next = T_eq + (T − T_eq)·exp(−Δt/τ) on a 30-minute step; the heating (10 h) and cooling (55 h) time constants are uncalibrated. "
     "The night floor makes it unsuitable for permanently shadowed cold traps or ice stability. Measured temperatures would need the LRO Diviner data set, which is not installed."),
    ("sunlight", re.compile(r"sunlight|illumination|irradiance|shadow|skyline|horizon|terrain|dem|lola", re.I),
     "Sun positions come from the JPL DE440 ephemeris via SPICE. Irradiance uses S₀ = 1,361 W/m² scaled by the Sun's distance. "
     "Where the NASA LOLA LDEM_80S_80M terrain model covers the site (80°S to the pole, 80 m/px), the Sun counts as up only above the measured skyline, traced out to 150 km at 2 m above the ground; elsewhere a smooth horizon is used. "
     "Results are sampled (typically hourly), so window edges are uncertain by about one step, and partial Sun disks or sub-pixel terrain are not resolved."),
    ("earth", re.compile(r"earth (contact|visib|in view)|direct.to.earth|\bdte\b|libration|communicat|link", re.I),
     "Earth positions come from JPL DE440 with the MOON_ME body frame, so libration is included. Earth counts as visible when it is above the local horizon (the measured skyline where the LOLA DEM covers the site). "
     "Visibility is sampled in time, so rise and set times are uncertain by about one step. The radio link budget assumes X-band, Eb/N0 of 2.5 dB plus a 3 dB margin, no pointing or atmospheric losses, and is capped at 50 Mbit/s by the X-band allocation; ground-station scheduling is not modelled."),
    ("mission", re.compile(r"mission|swarm|planner|launch|transfer|delta.?v|Δv|descent|orbit insertion|\bloi\b|\btli\b|trajector", re.I),
     "The mission planner chains deterministic equations: site selection, landing dates, two-stage launch, a coplanar Hohmann transfer aimed at the Moon's real distance on arrival (JPL DE440), "
     "lunar orbit insertion at perilune, powered descent and surface operations. Assumptions: 1.8 km/s ascent losses, 0.35 km/s descent losses plus the hover reserve, a constant average descent throttle and a required Δv reserve (default 10 %). "
     "It does not phase-target the Moon's direction, plane changes, finite burns or navigation errors, so launch dates are screening estimates, not launch windows."),
    ("windows", re.compile(r"science window|window search|\bwindows?\b", re.I),
     "A science window is a continuous run of samples where every chosen condition holds at once (temperature, irradiance, Sun up, Earth elevation, minimum duration). "
     "Edges depend on the sampling step: a measured duration d means the true duration lies between d and d + 2 steps. Temperatures in the filter are model estimates, and Earth conditions need the analysis server (the offline browser model fixes Earth at 45°)."),
    ("ranking", re.compile(r"rank|score|candidate site|site finder|water|\bice\b", re.I),
     "The site ranking scores published south-pole points. Landing = share of a 500 m footprint with DEM slope ≤ 12°. Water = exp(−d/14.4 km), where d is the distance to the rim of the nearest crater with Moon Mineralogy Mapper surface-ice detections (Li et al. 2018); "
     "it ranks access to known ice, not the chance that ice is present. Sunlight and Earth visibility are measured against the real skyline. "
     "Overall = 0.4·landing + 0.3·water + 0.2·sunlight + 0.1·Earth; the weights are a transparent choice, and none of these scores is a probability."),
    ("data", re.compile(r"data|source|ephemeris|spice|nasa|jpl|where.*(numbers|values) come", re.I),
     "Data sources: Sun and Earth positions from JPL DE440 via SPICE (MOON_ME frame); terrain from NASA LRO LOLA LDEM_80S_80M (80°S to the pole, 80 m/px); "
     "crater centres and diameters from the USGS/IAU Gazetteer of Planetary Nomenclature; published illumination statistics from Gläser et al. 2018 and others (cited on Compare Sites); "
     "ice detections from Li et al. 2018. Surface temperatures are modelled, not measured. The Models & sources page lists every assumption."),
]


def explain(question: str) -> tuple[str, str] | None:
    """(topic, answer) for an explanation question about the site's models, else None."""
    if not _EXPLAIN.search(question):
        return None
    # Requests to compute something are not explanations.
    if re.search(r"\b(find|search|plan|rank|compare|calculate|compute|run|show me|open)\b", question, re.I) and not re.search(r"\b(explain|limit|assumption|how accurate|why)\b", question, re.I):
        return None
    for topic, trigger, answer in TOPICS:
        if trigger.search(question):
            return topic, answer
    return None
