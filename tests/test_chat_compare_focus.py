"""A comparison follow-up must not spend model quota or replace the site list."""
import asyncio

import pytest
from pydantic import ValidationError

from app.api import chat as module
from app.api.chat import Action, ChatRequest

CONTEXT = dict(latitude=-89.54, longitude=0, start_datetime_utc="2027-01-01T00:00:00Z",
               end_datetime_utc="2027-02-01T00:00:00Z", step_hours=1, display_timezone="UTC")


@pytest.mark.parametrize("text", ["Compare their Earth visibility", "Compare their Earth contact?"])
def test_earth_visibility_comparison_never_calls_the_model(monkeypatch, text):
    def unexpected(*args, **kwargs):
        raise AssertionError("This comparison must not require Groq or a key")
    monkeypatch.setattr(module, "get_settings", unexpected)
    payload = ChatRequest(context=CONTEXT, page="compare", messages=[dict(role="user", content=text)],
                          active_search=Action(page="compare", compare_sites={"mode": "replace", "sites": [
                              {"name": "Custom site", "latitude": -88.2, "longitude": 35},
                              {"name": "Malapert", "latitude": -85.95, "longitude": 0},
                          ]}))
    before = payload.model_dump()
    result = asyncio.run(module.chat(payload))
    assert result.action.page == "compare"
    assert result.action.compare_focus == "earth_visibility"
    assert result.action.compare_sites is None
    assert result.action.forecast is None
    assert payload.model_dump() == before


@pytest.mark.parametrize("text,page", [
    ("Compare their Earth visibility", "selene"),
    ("Compare their Earth visibility in March 2028", "compare"),
    ("Compare their Earth visibility and require sunlight", "compare"),
])
def test_additional_conditions_and_unknown_comparison_context_are_not_discarded(text, page):
    payload = ChatRequest(context=CONTEXT, page=page, messages=[dict(role="user", content=text)])
    assert module._explicit_compare_focus(payload) is None


def test_focus_action_is_only_valid_for_the_comparison_page():
    with pytest.raises(ValidationError):
        Action(page="horizon", compare_focus="earth_visibility")
