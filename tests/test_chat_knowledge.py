"""Explanation questions are answered locally, without an AI call."""
import asyncio

from app.api import chat as module
from app.api.chat import ChatRequest
from app.config import Settings
from app.services.knowledge import explain

CONTEXT = dict(latitude=-89.44, longitude=-137.17, start_datetime_utc="2027-01-01T00:00:00Z",
               end_datetime_utc="2027-02-01T00:00:00Z", step_hours=1, display_timezone="UTC")


def test_explanations_route_locally_and_searches_do_not():
    assert explain("xplain the landing model's limits")[0] == "landing"
    assert explain("how accurate is the temperature?")[0] == "temperature"
    assert explain("find windows in jan 2027 with earth visible") is None
    assert explain("rank south pole sites for water") is None


def test_explanation_needs_no_api_key(monkeypatch):
    monkeypatch.setattr(module, "get_settings", lambda: Settings(groq_api_key=""))
    reply = asyncio.run(module.chat(ChatRequest(messages=[dict(role="user", content="explain the landing model's limits")], context=CONTEXT)))
    assert "Gaussian" in reply.reply and reply.action is None and reply.options
