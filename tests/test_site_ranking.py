from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from app.services import site_ranking as module

START = datetime(2027, 1, 2, tzinfo=timezone.utc)


def setup_ranking(monkeypatch, durations):
    monkeypatch.setattr(module, "CANDIDATE_SITES", [("Near ice", -89, 1), ("Far ice", -88, 2), ("Too short", -87, 3)])
    monkeypatch.setattr(module, "_site_metrics", lambda *args: {"slope_deg": 1, "sunlit_pct": 99, "earth_visible_pct": 50})
    monkeypatch.setattr(module, "compute_surface_screen", lambda *args: {"safe_fraction": 0.9})
    monkeypatch.setattr(module, "nearest_ice_crater", lambda lat, lon: ("Ice crater", lon * 10))
    def search(request, providers, run_best_plan):
        assert not run_best_plan
        assert request.min_continuous_sunlight_days == 3
        assert request.landing_start_utc == START
        duration = durations[int(request.inputs.target_longitude_deg) - 1]
        candidates = [SimpleNamespace(sunlight_interval_start_utc=START,
            sunlight_interval_end_utc=START + timedelta(days=duration))] if duration >= 3 else []
        return SimpleNamespace(candidates=candidates, limitations=["Hourly sampling", "Eclipse caveat", "Short shadows may be missed"])
    monkeypatch.setattr(module, "search_mission_windows", search)
    return SimpleNamespace(terrain=SimpleNamespace(source="test terrain"), astronomy=SimpleNamespace(source="test astronomy"))


def test_filters_duration_before_ranking_ice_access(monkeypatch):
    providers = setup_ranking(monkeypatch, [3, 4, 2.9])
    result = module.rank_sites(module.SiteRankingRequest(start_utc=START, end_utc=START + timedelta(days=34),
        min_sunlight_days=3, priority="water"), providers)
    assert [s.name for s in result.sites] == ["Near ice", "Far ice"]
    assert [s.continuous_sunlight_days for s in result.sites] == [3, 4]
    assert result.best_water == "Near ice"
    assert all(s.sunlight_start_utc == START for s in result.sites)
    assert "Hourly sampling" in result.method


def test_no_qualifying_sites_returns_empty_result(monkeypatch):
    providers = setup_ranking(monkeypatch, [0, 2, 2.99])
    result = module.rank_sites(module.SiteRankingRequest(start_utc=START, end_utc=START + timedelta(days=34),
        min_sunlight_days=3), providers)
    assert result.sites == []
    assert result.best_water is None and result.best_overall is None and result.best_landing is None


def test_duration_range_validated_before_search():
    for days, period in [(0.5, 34), (3, 367)]:
        with pytest.raises(ValidationError):
            module.SiteRankingRequest(start_utc=START, end_utc=START + timedelta(days=period), min_sunlight_days=days)
