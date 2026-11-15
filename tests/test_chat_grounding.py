"""Grounding: the assistant may only state numbers it can source."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.api import chat as module
from app.api.chat import ChatRequest, ForecastRequest, Reply

CONTEXT = dict(latitude=-54.3838, longitude=162.9562, start_datetime_utc="2027-01-01T00:00:00Z",
               end_datetime_utc="2027-02-01T00:00:00Z", step_hours=1, display_timezone="UTC")
FACTS = {"side_of_moon": "far side", "earth_elevation_min_deg": -39.3, "earth_elevation_max_deg": -28.5, "sun_up_pct": 49.6}


def request(text):
    return ChatRequest(messages=[dict(role="user", content=text)], context=CONTEXT)


def test_invented_quantity_is_removed_and_grounded_ones_kept():
    reply = Reply(reply="It is on the far side; Earth stays between -39.3° and -28.5°. The Sun is up 49.6% of the time. Earth contact lasts 14 hours a day.")
    out = module._guard_ungrounded_numbers(reply, request("is Earth visible here?"), FACTS).reply
    assert "-39.3°" in out and "49.6%" in out
    assert "14 hours" not in out and "could not verify" in out


def test_user_numbers_are_allowed():
    reply = Reply(reply="Searching for 5 h of sunlight and 8 h of Earth contact.")
    assert module._guard_ungrounded_numbers(reply, request("at least 5 hours sunlight and 8 hours earth"), FACTS).reply == reply.reply


def test_require_earth_throughout_folds_into_date_search():
    """The transcript's 'Require Earth throughout window' follow-up crashed the action."""
    reply, problem = module._parse_reply(json.dumps({
        "reply": "Requiring Earth throughout the stay.",
        "action": {"page": "swarm", "criteria": {"requireEarth": True},
                   "mission_windows": {"latitude": -54.3838, "longitude": 162.9562, "landing_start_utc": "2027-01-01T00:00:00Z",
                                       "landing_end_utc": "2027-02-01T00:00:00Z", "min_sunlight_days": 5 / 24, "min_earth_hours": 8}},
    }), ForecastRequest(**CONTEXT))
    assert reply is not None, problem
    assert reply.action.criteria is None and reply.action.page == "swarm"
    assert reply.action.mission_windows.min_earth_hours == 8      # max(8, 5 h stay)


DATA = Path(__file__).resolve().parents[2] / "data"


@pytest.mark.skipif(not (DATA / "spice" / "de440s.bsp").exists(), reason="SPICE not installed")
def test_far_side_site_is_identified():
    from app.config import Settings
    from app.services.providers import build_providers
    from app.services.site_facts import site_facts
    providers = build_providers(Settings(use_spice=True))
    facts = site_facts(-54.3838, 162.9562, datetime(2027, 1, 1, tzinfo=timezone.utc), providers)
    assert facts["side_of_moon"].startswith("far side") and facts["earth_up_pct"] == 0
    near = site_facts(0.6741, 23.4729, datetime(2027, 1, 1, tzinfo=timezone.utc), providers)
    assert near["side_of_moon"].startswith("near side")
