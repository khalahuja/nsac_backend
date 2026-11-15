"""Follow-ups should advance this conversation without inventing settings."""
import asyncio

import pytest
from pydantic import ValidationError

from app.api import chat as module
from app.api.chat import Action, ChatRequest, Reply, SearchOutcome
from app.config import Settings
from app.services.chat_followups import clean_options, improve_followups

CONTEXT = dict(latitude=-89.54, longitude=0,
               start_datetime_utc="2027-01-01T00:00:00Z",
               end_datetime_utc="2027-02-01T00:00:00Z",
               step_hours=1, display_timezone="UTC")


def request(text="Find sunlight windows", **kwargs):
    return ChatRequest(messages=[dict(role="user", content=text)], context=CONTEXT, **kwargs)


def test_labels_are_complete_distinct_and_specific():
    assert clean_options(["  Explain   the model  ", "Explain the model?",
                          "Tell me more", 12, "", "x" * 61,
                          "Open the sky plot"]) == ["Explain the model", "Open the sky plot"]


def test_clarification_preserves_choices_without_filler():
    reply = Reply(reply="Should temperature be above or below your limit?",
                  options=["Use it as a lower temperature limit", "Use it as an upper temperature limit"])
    assert improve_followups(reply, request()).options == reply.options
    assert improve_followups(Reply(reply="Which year should I use?"), request()).options == []
    unresolved = Reply(reply="Please specify the irradiance limit.", unresolved_conditions=["Irradiance missing"])
    assert improve_followups(unresolved, request()).options == []


def test_search_followups_skip_redundant_constraints_and_navigation():
    action = Action(page="windows", forecast=CONTEXT,
                    criteria={"requireSun": True, "requireEarth": True, "minHours": 24})
    reply = Reply(reply="Searching with both conditions.", action=action,
                  options=["Add Earth contact", "Require sunlight", "Open Science Windows", "Explain the sampling limits"])
    result = improve_followups(reply, request())
    assert "Add Earth contact" not in result.options
    assert "Require sunlight" not in result.options
    assert "Open Science Windows" not in result.options
    assert "Explain the sampling limits" in result.options
    assert result.action == action


def test_fallback_uses_candidate_context_and_preserves_every_constraint():
    action = Action(page="sites", site_ranking={"priority": "water", "min_sunlight_days": 8,
                                              "max_slope_deg": 10, "criteria": {"requireEarth": True}})
    payload = request("Explain the evidence", page="selene", active_search=action)
    result = improve_followups(Reply(reply="The ranking is a screening calculation."), payload)
    assert result.options[0] == "Explain the site-ranking criteria"
    assert all("Earth contact" not in text for text in result.options)
    assert payload.active_search == action


def test_does_not_repeat_a_previous_user_question():
    payload = request("Explain the sampling limits", page="windows")
    result = improve_followups(Reply(reply="The boundaries depend on the sample spacing.",
                                    options=["Explain the sampling limits?"]), payload)
    assert all("sampling limits" not in option.lower() for option in result.options)
    assert 1 <= len(result.options) <= 3


def test_far_side_does_not_offer_impossible_earth_contact():
    reply = Reply(reply="The selected site is on the far side.",
                  action=Action(page="windows", forecast=CONTEXT, criteria={"requireSun": True}),
                  options=["Require Earth contact throughout the window", "Find windows with Earth in view"])
    result = improve_followups(reply, request(), {"side_of_moon": "far side"})
    assert not any("Earth contact" in option or "Earth in view" in option for option in result.options)


def test_deterministic_search_gets_followups_without_calling_model(monkeypatch):
    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key=""))
    result = asyncio.run(module.chat(request("Find windows with 2 days sunlight and 1 days Earth in view in January 2027")))
    assert result.action is not None
    assert result.options
    assert all(len(option) <= 60 for option in result.options)


def test_model_receives_previous_computation_and_followup_guidance(monkeypatch):
    prompts = []
    async def ask(settings, messages, context):
        prompts.append(messages[0]["content"])
        return Reply(reply="We can examine the active constraints.",
                     options=["Tell me more", "Explain the active constraints"]), True

    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key="test-key"))
    monkeypatch.setattr(module, "_ask_groq", ask)
    monkeypatch.setattr(module, "site_facts", lambda *args: None)
    payload = request("Why did this search return no matches?", page="windows",
                      search_outcome={"kind": "windows", "match_count": 0})
    reply = asyncio.run(module.chat(payload))
    assert "match_count" in prompts[0]
    assert "Do not invent dates" in prompts[0]
    assert "Tell me more" not in reply.options


@pytest.mark.parametrize("count", [-1, 20001])
def test_result_summary_is_bounded(count):
    with pytest.raises(ValidationError):
        SearchOutcome(kind="windows", match_count=count)
