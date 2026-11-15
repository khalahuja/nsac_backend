from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
import pytest
from pydantic import ValidationError
from app.services.operations import MissionRequirements, OperationsRequest, summarize_intervals, plan_operations
from app.services.operations_chat import requirement_command

START = datetime(2027, 1, 1, tzinfo=timezone.utc)
def request(**changes):
    return OperationsRequest(latitude=-89.4, longitude=-137.17, landing_start_utc=START,
                             landing_end_utc=START + timedelta(hours=3), **changes)

def test_endpoint_screening_outages_and_simultaneous_checks():
    q = MissionRequirements(mission_hours=4, min_sunlight_pct=50, min_earth_pct=25,
                            max_dark_hours=2, max_outage_hours=3, min_both_hours=1)
    times = [START + timedelta(hours=i) for i in range(5)]
    result = summarize_intervals(times, [True, True, True, False, False], [True, True, False, True, False], q)
    assert result['metrics'] == dict(sunlight_pct=50, earth_pct=25, sunlight_hours=2,
        earth_hours=1, both_hours=1, longest_dark_hours=2, longest_outage_hours=3)
    assert result['feasible']
    q.max_outage_hours = 2
    assert not summarize_intervals(times, [True] * 5, [True, True, False, True, False], q)['feasible']
    assert sum(x['hours'] for x in result['tracks'][0]['segments']) == 4

def test_impossible_and_unbounded_requests_are_rejected():
    for changes in [dict(mission_hours=0), dict(mission_hours=1.5), dict(mission_hours=4, min_both_hours=5), dict(min_sunlight_pct=math.nan)]:
        with pytest.raises(ValidationError): MissionRequirements(**changes)
    data = request().model_dump()
    for patch in [dict(landing_end_utc=START + timedelta(days=91)), dict(landing_start_utc=START.replace(tzinfo=None)), dict(nearby_radius_km=51)]:
        with pytest.raises(ValidationError): OperationsRequest.model_validate({**data, **patch})

def test_search_includes_full_stay_and_nearby_polar_points(providers):
    q = MissionRequirements(mission_hours=4, min_both_hours=0, min_sunlight_pct=0,
        min_earth_pct=0, max_dark_hours=4, max_outage_hours=4)
    result = plan_operations(request(requirements=q), providers)
    assert result['candidate_count'] == result['feasible_count'] == 4
    assert result['terrain_verified'] is False
    assert len(result['nearby']) == 8
    assert all(c['feasible'] for c in result['candidates'])
    assert any(c['end_utc'] == (START + timedelta(hours=7)).isoformat() for c in result['candidates'])
    for point in result['nearby']:
        assert -90 <= point['latitude'] <= 90 and -180 <= point['longitude'] <= 180
        assert point['landing_utc'] == result['best']['landing_utc']
    assert result['candidates'] == sorted(result['candidates'], key=lambda c: (-c['score'], c['landing_utc']))

def test_chat_preserves_other_requirements_and_validates_updates():
    current = request()
    proposed = requirement_command('Set mission duration to 8 days', current)
    assert proposed.requirements.mission_hours == 192
    assert proposed.requirements.max_outage_hours == current.requirements.max_outage_hours
    assert proposed.landing_start_utc == current.landing_start_utc
    assert requirement_command('Set minimum sunlight to 80% and maximum outage to 3 hours', current) is None
    with pytest.raises(ValidationError): requirement_command('Set minimum sunlight to 101%', current)

def test_operations_route_and_chat_without_llm(client):
    payload = request(requirements=MissionRequirements(mission_hours=4, min_both_hours=0)).model_dump(mode='json')
    response = client.post('/api/mission/operations', json=payload)
    assert response.status_code == 200, response.text
    assert response.json()['best']['end_utc']
    chat = dict(page='operations', operations_context=payload, context=dict(latitude=-89.4,
        longitude=-137.17, start_datetime_utc=START.isoformat(), end_datetime_utc=(START + timedelta(days=7)).isoformat(), step_hours=1, display_timezone='UTC'),
        messages=[dict(role='user', content='Set minimum sunlight to 80%')])
    response = client.post('/api/chat', json=chat, headers={'Origin': 'http://localhost:5173'})
    assert response.status_code == 200, response.text
    assert response.json()['action']['operations']['requirements']['min_sunlight_pct'] == 80
    chat['messages'][0]['content'] = 'Explain my mission operations result'
    response = client.post('/api/chat', json=chat, headers={'Origin': 'http://localhost:5173'})
    assert response.status_code == 200, response.text
    assert 'spherical horizon' in response.json()['reply']


def test_no_matching_dates_keeps_failed_limits(providers):
    payload = request(requirements=MissionRequirements(mission_hours=4, min_sunlight_pct=100,
        min_earth_pct=100, max_dark_hours=0, max_outage_hours=0, min_both_hours=4, min_earth_elevation_deg=30))
    result = plan_operations(payload, providers)
    assert result['feasible_count'] == 0
    assert not result['best']['feasible']
    assert any(not c['passed'] for c in result['best']['checks'])
    assert result['request']['requirements']['min_earth_elevation_deg'] == 30


def test_terrain_skyline_blocks_visibility_and_partial_dem_is_unverified(providers):
    class BlockedTerrain:
        is_measured = True
        source = 'test-dem'
        def horizon_profile(self, lat, lon, azimuths): return [90.0] * len(azimuths)
        def get_elevation(self, lat, lon): return 0
    providers = replace(providers, terrain=BlockedTerrain())
    payload = request(requirements=MissionRequirements(mission_hours=4, min_both_hours=0))
    result = plan_operations(payload, providers)
    assert result['terrain_verified']
    assert result['best']['metrics']['sunlight_hours'] == 0
    assert result['best']['metrics']['earth_hours'] == 0
    providers.terrain.horizon_profile = lambda lat, lon, azimuths: [None] * len(azimuths)
    assert not plan_operations(payload, providers)['terrain_verified']


def test_nearby_points_are_distinct_and_at_requested_distance_at_both_poles():
    from math import radians, sin, cos, asin, sqrt
    from app.services.operations import nearby_point
    for latitude in (-90, -89.9999, 90):
        points = [nearby_point(latitude, 37, 10, bearing) for bearing in range(0, 360, 45)]
        assert len({(round(lat, 8), round(lon, 8)) for lat, lon in points}) == 8
        for lat, lon in points:
            dlat, dlon = radians(lat - latitude), radians(lon - 37)
            haversine = sin(dlat / 2)**2 + cos(radians(latitude)) * cos(radians(lat)) * sin(dlon / 2)**2
            distance = 2 * 1737.4 * asin(sqrt(haversine))
            assert distance == pytest.approx(10, abs=1e-8)
