"""Deterministic, auditable launch-to-landing screening graph.

These are patched-conic and reduced-order engineering estimates, not a flown
trajectory, launch-window solution, or certified landing analysis.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from math import ceil, cos, exp, log, pi, sqrt
from typing import Literal, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .forecast import compute_forecast
from .positions import body_position

MU_E = 398600.4418  # km^3/s^2
MU_M = 4902.8001
R_E = 6378.137  # km
R_M = 1737.4
MOON_DISTANCE = 384400.0
G0 = 9.80665  # m/s^2


class MissionInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    #: Launch time (UTC). Replaced by the landing-window agent when a landing range is given.
    departure_utc: datetime
    launch_latitude_deg: float = Field(default=28.5, ge=-90, le=90)
    parking_altitude_km: float = Field(default=200, ge=150, le=2000)
    earth_parking_orbits: int = Field(default=1, ge=0, le=100)
    earth_staging_altitude_km: float = Field(default=0, ge=0, le=60000, description="0 disables the higher Earth staging orbit")
    earth_staging_orbits: int = Field(default=1, ge=0, le=100)
    # Two-stage launcher (defaults: a medium kerosene first stage and a hydrogen upper stage).
    launch_wet_mass_kg: float = Field(default=420_000, gt=0, description="First stage, fuelled")
    launch_dry_mass_kg: float = Field(default=25_000, gt=0, description="First stage, empty")
    launch_isp_s: float = Field(default=300, ge=100, le=550, description="First stage, average over ascent")
    launch_thrust_n: float = Field(default=7_000_000, gt=0)
    upper_stage_wet_mass_kg: float = Field(default=110_000, gt=0)
    upper_stage_dry_mass_kg: float = Field(default=5_000, gt=0)
    upper_stage_isp_s: float = Field(default=450, ge=100, le=550)
    #: The upper stage performs trans-lunar injection (as real missions do); otherwise the spacecraft does.
    tli_by_upper_stage: bool = True
    spacecraft_wet_mass_kg: float = Field(default=6500, gt=0)
    spacecraft_dry_mass_kg: float = Field(default=2400, gt=0, description="Empty lander including payload")
    lunar_wait_orbits: int = Field(default=1, ge=0, le=100)
    spacecraft_isp_s: float = Field(default=450, ge=100, le=5000)
    landing_thrust_n: float = Field(default=22_000, gt=0)
    #: Average throttle during powered descent (real landers throttle deeply).
    descent_throttle: float = Field(default=0.6, ge=0.1, le=1)
    #: Hover and divert time held in reserve before touchdown.
    hover_reserve_s: float = Field(default=60, ge=0, le=600)
    #: Δv reserve the spacecraft must keep beyond the nominal LOI and descent.
    dv_margin_pct: float = Field(default=10, ge=0, le=100)
    lunar_orbit_altitude_km: float = Field(default=100, ge=15, le=1000)
    target_latitude_deg: float = Field(default=-89.4398, ge=-90, le=90)
    target_longitude_deg: float = Field(default=-137.17, ge=-180, le=180)
    landing_radius_m: float = Field(default=100, gt=0, le=10000)
    landing_sigma_m: float = Field(default=30, gt=0, le=1000)
    payload_power_w: float = Field(default=300, ge=0, le=100000)
    #: Heater/keep-alive power drawn through the lunar night.
    survival_power_w: float = Field(default=80, ge=0, le=100000)
    battery_capacity_wh: float = Field(default=20000, ge=0)
    array_area_m2: float = Field(default=8, ge=0)
    array_efficiency: float = Field(default=0.28, ge=0, le=1)
    #: Surface stay checked for power, night survival and Earth contact.
    stay_days: float = Field(default=14, ge=1, le=60)
    # Direct-to-Earth X-band link.
    tx_power_w: float = Field(default=20, gt=0, le=1000)
    antenna_gain_dbi: float = Field(default=24, ge=0, le=60)
    link_frequency_ghz: float = Field(default=8.4, gt=0.1, le=40)
    ground_gt_db_k: float = Field(default=50, ge=0, le=70, description="DSN 34 m dish at X-band is about 50-54 dB/K")
    #: Site-selection agent: "auto" ranks candidate sites (the target becomes one candidate); "fixed" keeps the target.
    site_mode: Literal["auto", "fixed"] = "fixed"
    #: Landing-window agent: when both are set, the departure date is replaced by the best computed launch date.
    landing_start_utc: datetime | None = None
    landing_end_utc: datetime | None = None
    min_sunlight_days: float = Field(default=8, ge=1 / 24, le=30)
    min_earth_contact_hours: float = Field(default=0, ge=0, le=720)

    @model_validator(mode="after")
    def masses(self):
        if self.departure_utc.tzinfo is None or self.departure_utc.utcoffset() is None:
            raise ValueError("Departure date must include a UTC offset")
        if self.launch_dry_mass_kg >= self.launch_wet_mass_kg:
            raise ValueError("First-stage empty mass must be below its fuelled mass")
        if self.upper_stage_dry_mass_kg >= self.upper_stage_wet_mass_kg:
            raise ValueError("Upper-stage empty mass must be below its fuelled mass")
        if (self.landing_start_utc is None) != (self.landing_end_utc is None):
            raise ValueError("Give both the first and last landing date, or neither")
        if self.landing_start_utc and self.landing_end_utc:
            for value in (self.landing_start_utc, self.landing_end_utc):
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError("Landing dates must include a UTC offset")
            if self.landing_end_utc <= self.landing_start_utc or (self.landing_end_utc - self.landing_start_utc).days > 366:
                raise ValueError("The landing date range must be ordered and at most 366 days")
        if self.earth_staging_altitude_km and self.earth_staging_altitude_km <= self.parking_altitude_km:
            raise ValueError("Earth staging altitude must exceed the parking altitude, or be 0 for direct TLI")
        if self.spacecraft_dry_mass_kg >= self.spacecraft_wet_mass_kg:
            raise ValueError("Spacecraft dry mass must be below spacecraft wet mass")
        return self


class Stage(BaseModel):
    agent: str
    equation: str
    inputs: dict[str, float | str]
    outputs: dict[str, float | str | bool | None]
    checks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    event: str
    utc: datetime
    note: str = ""


class MissionPlan(BaseModel):
    inputs: MissionInputs
    stages: list[Stage]
    timeline: list[TimelineEvent] = Field(default_factory=list)
    overall_status: str
    total_spacecraft_delta_v_km_s: float
    spacecraft_delta_v_capacity_km_s: float
    assumptions: list[str]
    missing_verifications: list[str]
    execution_log: list[str]


class MissionState(TypedDict, total=False):
    inputs: MissionInputs
    providers: Any
    site: Stage
    window: Stage
    launch: Stage
    transfer: Stage
    capture: Stage
    descent: Stage
    surface: Stage
    surface_events: list
    review: MissionPlan



#: Candidate south-pole points with published illumination studies (see the
#: frontend's Compare Sites sources). The user's own target is always added.
CANDIDATE_SITES: list[tuple[str, float, float]] = [
    ("Connecting Ridge (C1)", -89.4398, -137.17),
    ("Connecting Ridge (C2)", -89.3130, -114.02),
    ("Shackleton rim (S0)", -89.6866, -162.81),
    ("Shackleton rim (S1)", -89.7815, -155.30),
    ("de Gerlache Rim (GR1)", -88.6952, -68.2846),
    ("Mons Malapert summit", -85.98, -1.67),
    ("Nobile Rim (VIPER area)", -85.42088, 31.6218),
]
MAX_SITE_SLOPE_DEG = 10.0  # Artemis lander requirement (NASA LaRC, Dwyer Cianciolo)


def _site_metrics(lat: float, lon: float, start: datetime, end: datetime, providers: Any) -> dict:
    """Sunlight and Earth visibility against the real skyline (when a DEM is loaded), 6-hourly."""
    from .surface_screen import horizon_or_none  # noqa: PLC0415 - avoid import cycle

    terrain = providers.terrain
    horizon = horizon_or_none(terrain, lat, lon, list(range(360)))
    use_horizon = horizon is not None and all(h is not None for h in horizon)
    sun = earth = n = 0
    moment = start
    while moment <= end:
        g = providers.astronomy.get_geometry(moment)
        s = body_position(lat, lon, g.sun_direction_body, g.sun_distance_km)
        e = body_position(lat, lon, g.earth_direction_body, g.earth_distance_km)
        sky = (lambda az: horizon[int(round(az)) % 360]) if use_horizon else (lambda az: 0.0)
        sun += s.elevation_deg > sky(s.azimuth_deg)
        earth += e.elevation_deg > sky(e.azimuth_deg)
        n += 1
        moment += timedelta(hours=6)
    slope = terrain.get_slope(lat, lon) if terrain.is_measured else None
    return {
        "sunlit_pct": 100 * sun / n, "earth_visible_pct": 100 * earth / n, "slope_deg": slope,
        "elevation_m": terrain.get_elevation(lat, lon) if terrain.is_measured else None, "terrain_horizon": use_horizon,
    }


def site_agent(state: MissionState) -> dict:
    p = state["inputs"]
    providers = state.get("providers")
    named = next((c for c in CANDIDATE_SITES if abs(c[1] - p.target_latitude_deg) < 1e-3 and abs(c[2] - p.target_longitude_deg) < 1e-3), None)
    target = (named[0] if named else "Requested target", p.target_latitude_deg, p.target_longitude_deg)
    if p.site_mode == "fixed" or providers is None:
        return {"site": Stage(agent="Site selection agent", equation="fixed target (no ranking requested)",
            inputs={"mode": p.site_mode}, outputs={"selected_site": target[0], "latitude_deg": target[1], "longitude_deg": target[2]},
            limitations=["Site ranking skipped: the requested coordinate is used as given."])}
    start = (p.landing_start_utc or p.departure_utc).astimezone(timezone.utc)
    end = (p.landing_end_utc or start + timedelta(days=30)).astimezone(timezone.utc)
    candidates = [target] + [c for c in CANDIDATE_SITES if abs(c[1] - target[1]) > 1e-4 or abs(c[2] - target[2]) > 1e-4]
    scored = []
    for name, lat, lon in candidates:
        m = _site_metrics(lat, lon, start, end, providers)
        slope_ok = m["slope_deg"] is None or m["slope_deg"] <= MAX_SITE_SLOPE_DEG
        # Power first, communications second; unsafe slopes are excluded.
        score = (m["sunlit_pct"] * 0.6 + m["earth_visible_pct"] * 0.4) if slope_ok else -1
        scored.append((score, name, lat, lon, m, slope_ok))
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0]
    outputs: dict[str, float | str | bool | None] = {
        "selected_site": best[1], "latitude_deg": best[2], "longitude_deg": best[3],
        "selected_sunlit_pct": best[4]["sunlit_pct"], "selected_earth_visible_pct": best[4]["earth_visible_pct"],
        "selected_slope_deg": best[4]["slope_deg"], "selected_elevation_m": best[4]["elevation_m"],
        "candidates_evaluated": len(scored),
    }
    for rank, (score, name, _lat, _lon, m, ok) in enumerate(scored, start=1):
        slope = "n/a" if m["slope_deg"] is None else f"{m['slope_deg']:.1f}°"
        outputs[f"rank_{rank}"] = f"{name}: sun {m['sunlit_pct']:.0f}%, Earth {m['earth_visible_pct']:.0f}%, slope {slope}{'' if ok else ' (too steep)'}"
    checks = [] if best[0] >= 0 else ["No candidate meets the 10° slope limit."]
    new_inputs = p.model_copy(update={"target_latitude_deg": best[2], "target_longitude_deg": best[3]})
    return {"inputs": new_inputs, "site": Stage(
        agent="Site selection agent",
        equation="score = 0.6·sunlit% + 0.4·Earth-visible%  (excluded if DEM slope > 10°)",
        inputs={"window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(), "candidates": len(scored)},
        outputs=outputs, checks=checks,
        limitations=[("Sunlight and Earth visibility use the measured LOLA skyline (2 m height, out to 150 km), 6-hourly."
                      if best[4]["terrain_horizon"] else "No polar DEM installed: a smooth horizon is used and slopes are unknown."),
                     "Candidates are published study points plus the requested target; hazards smaller than the DEM pixel are not seen."])}


def window_agent(state: MissionState) -> dict:
    p = state["inputs"]
    providers = state.get("providers")
    if p.landing_start_utc is None or providers is None:
        return {"window": Stage(agent="Landing window agent", equation="fixed departure (no landing range given)",
            inputs={"departure_utc": p.departure_utc.isoformat()}, outputs={"departure_utc": p.departure_utc.isoformat()},
            limitations=["No landing date range given, so the requested departure date is used without a sunlight search."])}
    from .mission_windows import WindowSearchInputs, search_mission_windows  # noqa: PLC0415 - avoid import cycle

    search = search_mission_windows(WindowSearchInputs(
        inputs=p.model_copy(update={"landing_start_utc": None, "landing_end_utc": None}),
        landing_start_utc=p.landing_start_utc, landing_end_utc=p.landing_end_utc,
        min_continuous_sunlight_days=p.min_sunlight_days,
        min_continuous_earth_hours=p.min_earth_contact_hours,
    ), providers, run_best_plan=False)
    if not search.candidates:
        return {"window": Stage(agent="Landing window agent", equation="Sun above skyline for every hour of the stay",
            inputs={"min_sunlight_days": p.min_sunlight_days, "min_earth_contact_hours": p.min_earth_contact_hours}, outputs={"candidates_found": 0},
            checks=[f"No landing date in the range keeps the Sun up for {p.min_sunlight_days * 24:g} hours"
                    + (f" and Earth continuously visible for {p.min_earth_contact_hours:g} hours." if p.min_earth_contact_hours else ".")], limitations=search.limitations)}
    # Artemis requires direct-to-Earth contact during landing: within each sunlit
    # interval, move the landing to the earliest hour with Earth above the real
    # skyline that still leaves the required sunlight.
    from .surface_screen import horizon_or_none  # noqa: PLC0415 - avoid import cycle
    lat, lon = p.target_latitude_deg, p.target_longitude_deg
    horizon = horizon_or_none(providers.terrain, lat, lon, list(range(360)))
    use_horizon = horizon is not None and all(h is not None for h in horizon)

    def earth_up(moment: datetime) -> bool:
        direction, distance = providers.astronomy.get_earth_vector_from_moon(moment)
        e = body_position(lat, lon, direction, distance)
        return e.elevation_deg > (horizon[int(round(e.azimuth_deg)) % 360] if use_horizon else 0.0)

    def earth_requirement_met(moment: datetime) -> bool:
        if not p.min_earth_contact_hours:
            return earth_up(moment)
        return all(earth_up(moment + timedelta(hours=hour))
                   for hour in range(ceil(p.min_earth_contact_hours) + 1))

    required = timedelta(days=p.min_sunlight_days)
    adjusted = []
    for c in search.candidates:
        flight = c.estimated_landing_utc - c.estimated_launch_utc
        moment, latest = c.sunlight_interval_start_utc, c.sunlight_interval_end_utc - required
        found = None
        while moment <= latest:
            if moment - flight >= search.landing_start_utc.astimezone(timezone.utc) - timedelta(days=30) and earth_requirement_met(moment):
                found = moment
                break
            moment += timedelta(hours=1)
        if found is not None:
            c = c.model_copy(update={"estimated_landing_utc": found, "estimated_launch_utc": found - flight,
                                     "continuous_sunlight_days_screened": (c.sunlight_interval_end_utc - found).total_seconds() / 86400})
        adjusted.append((found is not None, c))
    # Prefer Earth in view at touchdown, then any Earth contact, then the Sun margin.
    with_earth, chosen = max(adjusted, key=lambda item: (item[0], item[1].earth_contact_hours_first_8_days > 0, item[1].minimum_sun_elevation_deg))
    # The flight time depends on the Moon's distance on the actual dates; iterate so touchdown lands on the chosen epoch.
    launch = chosen.estimated_launch_utc
    for _ in range(4):
        launch = chosen.estimated_landing_utc - timedelta(days=flight_duration_days(p.model_copy(update={"departure_utc": launch}), providers))
    chosen = chosen.model_copy(update={"estimated_launch_utc": launch})
    outputs: dict[str, float | str | bool | None] = {
        "chosen_launch_utc": chosen.estimated_launch_utc.isoformat(), "chosen_landing_utc": chosen.estimated_landing_utc.isoformat(),
        "chosen_sunlight_days": chosen.continuous_sunlight_days_screened, "chosen_sun_margin_deg": chosen.minimum_sun_elevation_deg,
        "chosen_earth_contact_hours_first_8_days": chosen.earth_contact_hours_first_8_days,
        "chosen_continuous_earth_hours": chosen.continuous_earth_hours_screened,
        "earth_in_view_at_chosen_landing": with_earth,
        "terrain_verified": chosen.terrain_verified, "candidates_found": len(search.candidates),
    }
    for i, c in enumerate(search.candidates, start=1):
        outputs[f"option_{i}"] = (f"launch {c.estimated_launch_utc:%Y-%m-%d %H:%M} → land {c.estimated_landing_utc:%Y-%m-%d %H:%M} UTC, "
                                  f"sun {c.continuous_sunlight_days_screened:.1f} d, Earth {c.earth_contact_hours_first_8_days:.0f} h")
    return {"inputs": p.model_copy(update={"departure_utc": chosen.estimated_launch_utc}), "window": Stage(
        agent="Landing window agent",
        equation="landing t: Sun above skyline ∀ t..t+N days; launch = landing − (transfer + configured lunar orbits + descent burn)",
        inputs={"landing_start_utc": p.landing_start_utc.isoformat(), "landing_end_utc": p.landing_end_utc.isoformat(), "min_sunlight_days": p.min_sunlight_days, "min_earth_contact_hours": p.min_earth_contact_hours},
        outputs=outputs, limitations=search.limitations)}


X_BAND_MAX_BPS = 50e6
ASCENT_MINUTES = 10.0      # liftoff to parking-orbit insertion (assumed)
LOSS_ALLOWANCE_KM_S = 1.8  # gravity, drag and steering losses on ascent (assumed)
DESCENT_LOSS_KM_S = 0.35   # gravity and steering losses in powered descent (assumed)
G_MOON = MU_M * 1000 / R_M**2  # m/s², 1.624


def _rocket_mass(m0: float, dv_km_s: float, isp: float) -> float:
    """Mass left after a burn (Tsiolkovsky)."""
    return m0 / exp(dv_km_s * 1000 / (isp * G0))


def _rocket_dv(m0: float, m1: float, isp: float) -> float:
    return isp * G0 * log(m0 / m1) / 1000


def earth_orbit_raise(p: MissionInputs) -> dict[str, float]:
    """Two impulsive burns between circular Earth parking and staging orbits."""
    r1 = R_E + p.parking_altitude_km
    if not p.earth_staging_altitude_km:
        return {"departure_burn_km_s": 0.0, "circularization_burn_km_s": 0.0, "total_delta_v_km_s": 0.0, "coast_hours": 0.0}
    r2 = R_E + p.earth_staging_altitude_km
    a = (r1 + r2) / 2
    departure = sqrt(MU_E * (2 / r1 - 1 / a)) - sqrt(MU_E / r1)
    circularization = sqrt(MU_E / r2) - sqrt(MU_E * (2 / r2 - 1 / a))
    return {"departure_burn_km_s": departure, "circularization_burn_km_s": circularization,
            "total_delta_v_km_s": departure + circularization, "coast_hours": pi * sqrt(a**3 / MU_E) / 3600}


def tli_delta_v(p: MissionInputs, moon_distance_km: float = MOON_DISTANCE) -> float:
    rp = R_E + (p.earth_staging_altitude_km or p.parking_altitude_km)
    a = (rp + moon_distance_km) / 2
    return sqrt(MU_E * (2 / rp - 1 / a)) - sqrt(MU_E / rp)


def launch_agent(state: MissionState) -> dict:
    p = state["inputs"]
    r = R_E + p.parking_altitude_km
    circular = sqrt(MU_E / r)
    rotation = 0.4651 * cos(p.launch_latitude_deg * pi / 180)
    required = circular - rotation + LOSS_ALLOWANCE_KM_S
    payload = p.spacecraft_wet_mass_kg
    liftoff = p.launch_wet_mass_kg + p.upper_stage_wet_mass_kg + payload
    dv1 = _rocket_dv(liftoff, p.launch_dry_mass_kg + p.upper_stage_wet_mass_kg + payload, p.launch_isp_s)
    stage2_needed = max(0.0, required - dv1)
    stage2_start = p.upper_stage_wet_mass_kg + payload
    in_orbit = _rocket_mass(stage2_start, stage2_needed, p.upper_stage_isp_s)
    stage2_capacity = _rocket_dv(stage2_start, p.upper_stage_dry_mass_kg + payload, p.upper_stage_isp_s)
    left_for_tli = _rocket_dv(in_orbit, p.upper_stage_dry_mass_kg + payload, p.upper_stage_isp_s) if in_orbit > p.upper_stage_dry_mass_kg + payload else 0.0
    twr = p.launch_thrust_n / (liftoff * G0)
    reaches_orbit = dv1 + stage2_capacity >= required
    tli_needed = tli_delta_v(p) + earth_orbit_raise(p)["total_delta_v_km_s"]
    checks = []
    if not reaches_orbit:
        checks.append("The two stages cannot reach the parking orbit with this payload.")
    if twr <= 1.15:
        checks.append("Lift-off thrust-to-weight is below 1.15; the rocket would barely leave the pad.")
    if p.tli_by_upper_stage and reaches_orbit and left_for_tli < tli_needed:
        checks.append(f"The upper stage has {left_for_tli:.2f} km/s left in orbit but Earth orbit raising and lunar injection need {tli_needed:.2f} km/s.")
    return {"launch": Stage(
        agent="Launch and parking-orbit agent",
        equation="v_c=√(μ_E/r); Δv_to_orbit≈v_c−v_rotation+1.8 km/s; per stage Δv=Isp·g0·ln(m0/m1)",
        inputs={"parking_altitude_km": p.parking_altitude_km, "stage1_wet_kg": p.launch_wet_mass_kg, "stage1_dry_kg": p.launch_dry_mass_kg,
                "stage1_isp_s": p.launch_isp_s, "upper_stage_wet_kg": p.upper_stage_wet_mass_kg, "upper_stage_dry_kg": p.upper_stage_dry_mass_kg,
                "upper_stage_isp_s": p.upper_stage_isp_s, "payload_kg": payload, "launch_thrust_n": p.launch_thrust_n},
        outputs={"circular_speed_km_s": circular, "orbit_period_hours": 2 * pi * sqrt(r**3 / MU_E) / 3600,
                 "required_delta_v_km_s": required, "stage1_delta_v_km_s": dv1, "upper_stage_delta_v_capacity_km_s": stage2_capacity,
                 "upper_stage_delta_v_left_in_orbit_km_s": left_for_tli, "liftoff_mass_kg": liftoff, "initial_twr": twr,
                 "screen_pass": not checks},
        checks=checks,
        limitations=["1.8 km/s ascent losses are an assumed allowance; no atmosphere, guidance or trajectory simulation.",
                     "Stage burns are sequential with no inter-stage losses, fairing or engine-out modelled."])}


def _moon_distance_at(moment: datetime, providers: Any) -> float | None:
    if providers is None:
        return None
    try:
        return float(providers.astronomy.get_earth_vector_from_moon(moment)[1])
    except Exception:  # noqa: BLE001 - fall back to the mean distance
        return None


def transfer_agent(state: MissionState) -> dict:
    p = state["inputs"]
    providers = state.get("providers")
    parking_radius = R_E + p.parking_altitude_km
    rp = R_E + (p.earth_staging_altitude_km or p.parking_altitude_km)
    parking_period_h = 2 * pi * sqrt(parking_radius**3 / MU_E) / 3600
    staging_period_h = 2 * pi * sqrt(rp**3 / MU_E) / 3600
    raising = earth_orbit_raise(p)
    parking_time = p.departure_utc.astimezone(timezone.utc) + timedelta(minutes=ASCENT_MINUTES)
    raise_start = parking_time + timedelta(hours=p.earth_parking_orbits * parking_period_h)
    staging_time = raise_start + timedelta(hours=raising["coast_hours"])
    staging_wait_h = p.earth_staging_orbits * staging_period_h if p.earth_staging_altitude_km else 0.0
    tli_time = staging_time + timedelta(hours=staging_wait_h)
    # The Moon's distance varies 356,000–406,000 km; aim the transfer apogee at its distance on arrival.
    distance, source = MOON_DISTANCE, "mean distance (no ephemeris)"
    for _ in range(4):
        a = (rp + distance) / 2
        coast = pi * sqrt(a**3 / MU_E) / 86400
        real = _moon_distance_at(tli_time + timedelta(days=coast), providers)
        if real is None:
            break
        distance, source = real, getattr(providers.astronomy, "source", "ephemeris")
    a = (rp + distance) / 2
    v_c = sqrt(MU_E / rp)
    v_tli = sqrt(MU_E * (2 / rp - 1 / a))
    time_days = pi * sqrt(a**3 / MU_E) / 86400
    v_ap = sqrt(MU_E * (2 / distance - 1 / a))
    v_moon = sqrt(MU_E / distance)
    vinf = abs(v_moon - v_ap)
    arrival = tli_time + timedelta(days=time_days)
    checks = []
    if p.tli_by_upper_stage:
        launch = state.get("launch") or launch_agent(state)["launch"]
        available = float(launch.outputs["upper_stage_delta_v_left_in_orbit_km_s"])
        required = raising["total_delta_v_km_s"] + v_tli - v_c
        if available < required:
            checks.append(f"Upper-stage Earth orbit raising and TLI require {required:.2f} km/s; only {available:.2f} km/s is available after ascent.")
    return {"transfer": Stage(
        agent="Translunar transfer agent",
        equation="a=(r_LEO+r_Moon)/2; v=√(μ_E(2/r−1/a)); t=π√(a³/μ_E); v∞≈|v_Moon−v_apogee|",
        inputs={"parking_radius_km": parking_radius, "tli_radius_km": rp, "moon_distance_at_arrival_km": distance, "moon_distance_source": source,
                "performed_by": "upper stage" if p.tli_by_upper_stage else "spacecraft"},
        outputs={"parking_orbit_utc": parking_time.isoformat(), "orbit_raise_start_utc": raise_start.isoformat(),
                 "staging_orbit_utc": staging_time.isoformat() if p.earth_staging_altitude_km else None,
                 "earth_parking_orbits": p.earth_parking_orbits, "earth_staging_orbits": p.earth_staging_orbits if p.earth_staging_altitude_km else 0,
                 "earth_staging_altitude_km": p.earth_staging_altitude_km, "earth_staging_speed_km_s": sqrt(MU_E / rp),
                 "earth_staging_period_hours": staging_period_h, "earth_orbit_raise_delta_v_km_s": raising["total_delta_v_km_s"],
                 "earth_raise_departure_burn_km_s": raising["departure_burn_km_s"], "earth_raise_circularization_burn_km_s": raising["circularization_burn_km_s"],
                 "earth_raise_coast_hours": raising["coast_hours"], "tli_utc": tli_time.isoformat(), "tli_delta_v_km_s": v_tli - v_c, "tli_speed_km_s": v_tli,
                 "transfer_semimajor_axis_km": a, "coast_days": time_days, "arrival_utc": arrival.isoformat(),
                 "arrival_v_infinity_km_s": vinf, "moon_distance_at_arrival_km": distance},
        checks=checks,
        limitations=["Coplanar two-body Hohmann screening: the Moon's distance at arrival comes from the ephemeris, but its direction is not phase-targeted.",
                     "Earth orbit counts are configured waits; optional orbit raising uses a two-impulse Hohmann transfer, with no plane changes or optimal phasing."])}


def capture_agent(state: MissionState) -> dict:
    p = state["inputs"]
    vinf = float(state["transfer"].outputs["arrival_v_infinity_km_s"])
    r = R_M + p.lunar_orbit_altitude_km
    hyperbolic = sqrt(vinf**2 + 2 * MU_M / r)
    circular = sqrt(MU_M / r)
    return {"capture": Stage(
        agent="Lunar orbit insertion agent",
        equation="Δv_LOI=√(v∞²+2μ_M/r_p)−√(μ_M/r_p)",
        inputs={"v_infinity_km_s": vinf, "orbit_altitude_km": p.lunar_orbit_altitude_km},
        outputs={"loi_delta_v_km_s": hyperbolic - circular, "arrival_periapsis_speed_km_s": hyperbolic, "circular_speed_km_s": circular,
                 "wait_orbits": p.lunar_wait_orbits, "orbit_period_hours": 2 * pi * sqrt(r**3 / MU_M) / 3600},
        limitations=["Tangential burn at lunar periapsis into a circular orbit; no finite-burn or plane-change cost. A polar site lies under any polar orbit, so no phasing wait is added."])}


def descent_agent(state: MissionState) -> dict:
    p = state["inputs"]
    v = float(state["capture"].outputs["circular_speed_km_s"])
    hover = G_MOON * p.hover_reserve_s / 1000
    dv = v + DESCENT_LOSS_KM_S + hover
    tli = 0.0 if p.tli_by_upper_stage else float(state["transfer"].outputs["tli_delta_v_km_s"])
    loi = float(state["capture"].outputs["loi_delta_v_km_s"])
    isp = p.spacecraft_isp_s
    earth_raise = 0.0 if p.tli_by_upper_stage else float(state["transfer"].outputs["earth_orbit_raise_delta_v_km_s"])
    after_earth_raise = _rocket_mass(p.spacecraft_wet_mass_kg, earth_raise, isp)
    after_tli = _rocket_mass(after_earth_raise, tli, isp)
    after_loi = _rocket_mass(after_tli, loi, isp)
    after_descent = _rocket_mass(after_loi, dv, isp)
    nominal = earth_raise + tli + loi + dv
    reserve = nominal * p.dv_margin_pct / 100
    capacity = _rocket_dv(p.spacecraft_wet_mass_kg, p.spacecraft_dry_mass_kg, isp)
    twr_start = p.landing_thrust_n / (after_loi * G_MOON)
    twr_end = p.landing_thrust_n / (after_descent * G_MOON)
    mdot = p.landing_thrust_n / (isp * G0)
    burn_s = (after_loi - after_descent) / (mdot * p.descent_throttle)
    arrival = datetime.fromisoformat(str(state["transfer"].outputs["arrival_utc"]))
    descent_start = arrival + timedelta(hours=p.lunar_wait_orbits * float(state["capture"].outputs["orbit_period_hours"]))
    landing = descent_start + timedelta(seconds=burn_s)
    checks = []
    if after_descent < p.spacecraft_dry_mass_kg:
        checks.append("Propellant is insufficient for the burns the spacecraft performs.")
    elif capacity < nominal + reserve:
        checks.append(f"Δv reserve is below {p.dv_margin_pct:.0f} %: capacity {capacity:.2f} km/s vs {nominal + reserve:.2f} km/s needed with reserve.")
    if twr_start * p.descent_throttle <= 1:
        checks.append("At the planned average throttle the engine cannot hold the lander up at the start of descent.")
    if twr_end > 6:
        checks.append(f"Engine is oversized: thrust-to-weight {twr_end:.1f} at touchdown would need throttling below 17 % to hover.")
    return {"descent": Stage(
        agent="Powered descent and propellant agent",
        equation="Δv_descent=v_orbit+0.35 km/s+g_Moon·t_hover; m1=m0·e^(−Δv/(Isp·g0)); t_burn=Δm/(ṁ_max·throttle)",
        inputs={"spacecraft_wet_mass_kg": p.spacecraft_wet_mass_kg, "spacecraft_dry_mass_kg": p.spacecraft_dry_mass_kg, "isp_s": isp,
                "landing_thrust_n": p.landing_thrust_n, "descent_throttle": p.descent_throttle, "hover_reserve_s": p.hover_reserve_s,
                "dv_margin_pct": p.dv_margin_pct},
        outputs={"descent_delta_v_km_s": dv, "hover_reserve_delta_v_km_s": hover, "spacecraft_tli_delta_v_km_s": tli, "spacecraft_earth_raise_delta_v_km_s": earth_raise,
                 "nominal_spacecraft_delta_v_km_s": nominal, "required_reserve_km_s": reserve, "spacecraft_delta_v_capacity_km_s": capacity,
                 "delta_v_margin_pct": 100 * (capacity - nominal) / nominal if nominal else None,
                 "loaded_propellant_kg": p.spacecraft_wet_mass_kg - p.spacecraft_dry_mass_kg,
                 "propellant_used_kg": p.spacecraft_wet_mass_kg - after_descent,
                 "propellant_remaining_kg": max(0.0, after_descent - p.spacecraft_dry_mass_kg),
                 "mass_after_loi_kg": after_loi, "landing_mass_kg": after_descent,
                 "twr_at_descent_start": twr_start, "twr_at_touchdown": twr_end,
                 "powered_descent_minutes": burn_s / 60, "descent_start_utc": descent_start.isoformat(),
                 "estimated_landing_utc": landing.isoformat(),
                 "mission_duration_hours": (landing - p.departure_utc).total_seconds() / 3600, "screen_pass": not checks},
        checks=checks,
        limitations=["0.35 km/s descent losses are an assumed allowance; constant average throttle, no optimal-control trajectory or terrain-relative navigation.",
                     "Reference: Apollo 11 powered descent took 12.6 min (mission report MSC-00171)."])}


def flight_duration_days(inputs: MissionInputs, providers: Any = None) -> float:
    """Launch to touchdown, using the same agents as the mission graph."""
    state: MissionState = {"inputs": inputs, "providers": providers}
    state.update(transfer_agent(state))
    state.update(capture_agent(state))
    state.update(descent_agent(state))
    landing = datetime.fromisoformat(str(state["descent"].outputs["estimated_landing_utc"]))
    return (landing - inputs.departure_utc).total_seconds() / 86400


def _link_rate_bps(p: MissionInputs, distance_km: float) -> float:
    """Direct-to-Earth data rate with 2.5 dB Eb/N0 required and 3 dB margin."""
    from math import log10  # noqa: PLC0415
    wavelength = 299_792_458 / (p.link_frequency_ghz * 1e9)
    eirp = 10 * log10(p.tx_power_w) + p.antenna_gain_dbi
    fspl = 20 * log10(4 * pi * distance_km * 1000 / wavelength)
    c_n0 = eirp - fspl + p.ground_gt_db_k + 228.6
    return 10 ** ((c_n0 - 2.5 - 3.0) / 10)


def surface_agent(state: MissionState) -> dict:
    p = state["inputs"]
    providers = state.get("providers")
    landing = datetime.fromisoformat(str(state["descent"].outputs["estimated_landing_utc"]))
    r, sigma = p.landing_radius_m, p.landing_sigma_m
    containment = 1 - exp(-r * r / (2 * sigma * sigma))
    outputs: dict[str, float | str | bool | None] = {"landing_containment_probability": containment}
    limitations = ["Landing dispersion is isotropic Gaussian.",
                   "Power assumes a Sun-tracking array (no cosine loss, no dust or temperature derating)."]
    checks: list[str] = []
    events: list[tuple[str, datetime, str]] = []
    terrain = getattr(providers, "terrain", None)
    if terrain is not None and terrain.is_measured:
        from .surface_screen import compute_surface_screen  # noqa: PLC0415 - avoid import cycle
        ellipse = max(r, 3 * sigma)
        screen = compute_surface_screen(p.target_latitude_deg, p.target_longitude_deg, landing, ellipse, providers)
        safe = screen.get("safe_fraction")
        outputs.update({"terrain_safe_fraction": safe, "landing_ellipse_radius_m": ellipse,
                        "landing_success_screen": containment * safe if safe is not None else None,
                        "terrain_source": terrain.source})
        if safe is not None and safe < 0.9:
            checks.append(f"Only {safe * 100:.0f} % of the {ellipse:.0f} m landing ellipse is under the 12° slope limit.")
        limitations.append(f"Slope safety from {terrain.source}; rocks smaller than the DEM pixel are not seen.")
    else:
        outputs.update({"terrain_safe_fraction": None, "landing_success_screen": None})
        limitations.append("No DEM covers this site, so landing-ellipse terrain safety is not computed.")
    if providers is not None:
        step = 6
        end = landing + timedelta(days=p.stay_days)
        forecast = compute_forecast(latitude_deg=p.target_latitude_deg, longitude_deg=p.target_longitude_deg, start=landing, end=end,
                                    step_hours=step, providers=providers, max_samples=int(p.stay_days * 24 / step) + 2)
        samples = forecast.samples
        from .surface_screen import horizon_or_none  # noqa: PLC0415 - avoid import cycle
        horizon = horizon_or_none(terrain, p.target_latitude_deg, p.target_longitude_deg, list(range(360))) if terrain is not None else None
        skyline_ok = horizon is not None and all(h is not None for h in horizon)

        def above(el, az):
            return el is not None and az is not None and el > (horizon[int(round(az)) % 360] if skyline_ok else 0.0)

        lit = [above(x.solar_elevation_deg, x.solar_azimuth_deg) for x in samples[:-1]]
        seen = [above(x.earth_elevation_deg, x.earth_azimuth_deg) for x in samples[:-1]]
        longest_dark = dark = longest_blackout = blackout = 0
        for i, (is_lit, is_seen) in enumerate(zip(lit, seen)):
            dark = dark + step if not is_lit else 0
            blackout = blackout + step if not is_seen else 0
            longest_dark = max(longest_dark, dark)
            longest_blackout = max(longest_blackout, blackout)
            if i:
                when = datetime.fromisoformat(str(samples[i].datetime_utc).replace("Z", "+00:00"))
                if lit[i - 1] != is_lit:
                    events.append(("Sunrise" if is_lit else "Sunset", when, "Sun crosses the local skyline (±6 h sampling)"))
                if seen[i - 1] != is_seen:
                    events.append(("Earth rises" if is_seen else "Earth sets", when, "direct-to-Earth contact " + ("begins" if is_seen else "ends") + " (±6 h sampling)"))

        def beam(x):
            d = x.sun_distance_km or 149_597_870.7
            return 1361.0 * (149_597_870.7 / d) ** 2

        generated = sum(beam(x) * p.array_area_m2 * p.array_efficiency * step for x, is_lit in zip(samples[:-1], lit) if is_lit)
        lit_hours = step * sum(lit)
        dark_hours = step * (len(lit) - sum(lit))
        demand = p.payload_power_w * lit_hours + p.survival_power_w * dark_hours
        night_need = p.survival_power_w * longest_dark
        distances = [x.earth_distance_km for x in samples if x.earth_distance_km]
        link_rate = _link_rate_bps(p, max(distances)) if distances else None
        # X-band space-research allocation near Earth is 50 MHz; about 1 bit/s per Hz with standard coding.
        rate = min(link_rate, X_BAND_MAX_BPS) if link_rate else None
        outputs.update({"stay_days": p.stay_days, "sunlight_hours": lit_hours, "earth_contact_hours": step * sum(seen),
                        "longest_night_hours": longest_dark, "longest_blackout_hours": longest_blackout,
                        "solar_energy_wh": generated, "load_energy_wh": demand,
                        "night_survival_energy_wh": night_need, "battery_capacity_wh": p.battery_capacity_wh,
                        "dte_data_rate_kbps_worst_case": rate / 1000 if rate else None,
                        "dte_link_limited_kbps": link_rate / 1000 if link_rate else None,
                        "sun_up_at_touchdown": lit[0] if lit else None, "earth_in_view_at_touchdown": seen[0] if seen else None,
                        "astronomy_source": forecast.sources.astronomy})
        if generated < demand:
            checks.append("Solar generation over the stay is below the electrical load.")
        if night_need > p.battery_capacity_wh:
            checks.append(f"The longest night ({longest_dark} h) needs {night_need / 1000:.1f} kWh of heater power; the battery holds {p.battery_capacity_wh / 1000:.1f} kWh.")
        if seen and not seen[0]:
            checks.append("Earth is below the local horizon at touchdown: no direct-to-Earth link during landing (a relay is needed).")
        if rate is not None and rate < 1000:
            checks.append(f"Direct-to-Earth link supports only {rate:.0f} bit/s at the farthest Earth distance.")
        limitations.append(f"Six-hour samples over a {p.stay_days:g}-day stay; "
                           + (f"Sun and Earth tested against the measured skyline ({terrain.source})." if skyline_ok else "smooth horizon (no polar DEM for this site)."))
        limitations.append("Link budget: X-band, required Eb/N0 2.5 dB plus 3 dB margin, no pointing or atmospheric losses; capped at 50 Mbit/s by the 50 MHz X-band allocation.")
    return {"surface": Stage(
        agent="Landing safety, power and contact agent",
        equation="P(r≤R)=1−e^(−R²/2σ²); safe = DEM slope ≤12° share of the 3σ ellipse; E_solar=Σ1361(AU/d)²·A·η·Δt when Sun above skyline; "
                 "E_night=P_survival·t_dark; C/N0=EIRP−FSPL+G/T+228.6",
        inputs={"latitude_deg": p.target_latitude_deg, "longitude_deg": p.target_longitude_deg, "landing_radius_m": r, "dispersion_sigma_m": sigma,
                "array_area_m2": p.array_area_m2, "array_efficiency": p.array_efficiency, "payload_power_w": p.payload_power_w,
                "survival_power_w": p.survival_power_w, "tx_power_w": p.tx_power_w, "antenna_gain_dbi": p.antenna_gain_dbi},
        outputs=outputs, checks=checks, limitations=limitations), "surface_events": events}


def review_agent(state: MissionState) -> dict:
    p = state["inputs"]
    stages = [state[name] for name in ("site", "window", "launch", "transfer", "capture", "descent", "surface")]
    descent = state["descent"].outputs
    total = float(descent["nominal_spacecraft_delta_v_km_s"])
    capacity = float(descent["spacecraft_delta_v_capacity_km_s"])
    failed = any(stage.checks for stage in stages)
    status = "screening_failed" if failed else "provisional_screening_pass"
    t = state["transfer"].outputs
    timeline = [
        TimelineEvent(event="Launch", utc=p.departure_utc, note=f"from {p.launch_latitude_deg:.1f}° latitude"),
        TimelineEvent(event="Parking orbit", utc=p.departure_utc + timedelta(minutes=ASCENT_MINUTES), note=f"{p.parking_altitude_km:.0f} km"),
        TimelineEvent(event="Trans-lunar injection", utc=datetime.fromisoformat(str(t["tli_utc"])), note=f"{float(t['tli_delta_v_km_s']):.2f} km/s by the {'upper stage' if p.tli_by_upper_stage else 'spacecraft'}"),
        TimelineEvent(event="Lunar orbit insertion", utc=datetime.fromisoformat(str(t["arrival_utc"])), note=f"{float(state['capture'].outputs['loi_delta_v_km_s']):.2f} km/s into {p.lunar_orbit_altitude_km:.0f} km orbit"),
        TimelineEvent(event="Powered descent starts", utc=datetime.fromisoformat(str(descent["descent_start_utc"])), note=f"after {p.lunar_wait_orbits} orbit(s)"),
        TimelineEvent(event="Touchdown", utc=datetime.fromisoformat(str(descent["estimated_landing_utc"])), note=f"{float(descent['powered_descent_minutes']):.1f} min burn"),
    ]
    if p.earth_staging_altitude_km:
        timeline += [
            TimelineEvent(event="Earth orbit-raising burn", utc=datetime.fromisoformat(str(t["orbit_raise_start_utc"])), note=f"{float(t['earth_raise_departure_burn_km_s']):.2f} km/s"),
            TimelineEvent(event="Earth staging orbit insertion", utc=datetime.fromisoformat(str(t["staging_orbit_utc"])), note=f"{p.earth_staging_altitude_km:g} km; circularization {float(t['earth_raise_circularization_burn_km_s']):.2f} km/s; {p.earth_staging_orbits} orbit(s)"),
        ]
    timeline += [TimelineEvent(event=name, utc=when, note=note) for name, when, note in state.get("surface_events", [])]
    landing = datetime.fromisoformat(str(descent["estimated_landing_utc"]))
    timeline.append(TimelineEvent(event="End of planned stay", utc=landing + timedelta(days=p.stay_days), note=f"{p.stay_days:g} days"))
    timeline.sort(key=lambda e: e.utc)
    plan = MissionPlan(
        inputs=p, stages=stages, timeline=timeline, overall_status=status,
        total_spacecraft_delta_v_km_s=total, spacecraft_delta_v_capacity_km_s=capacity,
        assumptions=[
            "Coplanar patched-conic transfer to the Moon's ephemeris distance at arrival",
            f"Ascent losses {LOSS_ALLOWANCE_KM_S} km/s; descent losses {DESCENT_LOSS_KM_S} km/s plus {p.hover_reserve_s:.0f} s hover",
            f"{p.dv_margin_pct:.0f} % Δv reserve required on the spacecraft",
            "Six-hour surface samples over the planned stay",
        ],
        missing_verifications=[
            "Launch-date phasing of the Moon's direction (only its distance is used)",
            "High-fidelity propagation, plane changes and maneuver targeting",
            "Rocks and hazards smaller than the DEM pixel; navigation covariance",
            "Thermal design beyond a constant survival-heater load",
        ],
        execution_log=[f"{i + 1}. {stage.agent}: deterministic equations evaluated" for i, stage in enumerate(stages)]
        + [f"{len(stages) + 1}. Review agent: constraints aggregated; no LLM numerical output used"])
    return {"review": plan}


@lru_cache(maxsize=1)
def mission_graph():
    graph=StateGraph(MissionState)
    for name, node in (("site",site_agent),("window",window_agent),("launch",launch_agent),("transfer",transfer_agent),("capture",capture_agent),("descent",descent_agent),("surface",surface_agent),("review",review_agent)):
        graph.add_node(name,node)
    graph.add_edge(START,"site")
    graph.add_edge("site","window")
    graph.add_edge("window","launch")
    for a,b in (("launch","transfer"),("transfer","capture"),("capture","descent"),("descent","surface"),("surface","review")):
        graph.add_edge(a,b)
    graph.add_edge("review",END)
    return graph.compile()


def plan_mission(inputs: MissionInputs, providers: Any = None) -> MissionPlan:
    return mission_graph().invoke({"inputs":inputs,"providers":providers})["review"]
