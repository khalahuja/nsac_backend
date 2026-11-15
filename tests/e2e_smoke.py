"""End-to-end smoke test against a RUNNING backend, the way the website calls it.

Usage (backend running on :8000):
    python tests/e2e_smoke.py [http://127.0.0.1:8000]

Not collected by pytest (no test_ prefix): it needs a live server and data.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
HEADERS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (Selene e2e)", "Origin": "http://localhost:5173"}
SITE = {"latitude": -89.4398, "longitude": -137.17}
PERIOD = {"start_datetime_utc": "2027-01-01T00:00:00Z", "end_datetime_utc": "2027-01-31T00:00:00Z", "step_hours": 1, "display_timezone": "UTC"}


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=HEADERS)
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read()
            ctype = response.headers.get("Content-Type", "")
            return response.status, (json.loads(raw) if "json" in ctype else raw), time.time() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read()[:300], time.time() - started


CHECKS = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


@check("status reports real data providers")
def _():
    code, body, _ = call("GET", "/api/status")
    assert code == 200, body
    text = json.dumps(body).lower()
    assert "spice" in text, "SPICE not active"
    assert "ldem" in text or "lola" in text, "LOLA DEM not active"


@check("environment at Connecting Ridge (terrain-measured)")
def _():
    code, body, _ = call("POST", "/api/environment", {**SITE, "datetime_utc": "2027-01-10T00:00:00Z"})
    assert code == 200, body
    assert body["terrain"]["is_measured"] is True


@check("timeline for the surface view sky paths")
def _():
    code, body, _ = call("POST", "/api/timeline", {**SITE, "center_datetime_utc": "2027-01-10T12:00:00Z", "hours_before": 354, "hours_after": 354, "step_hours": 4})
    assert code == 200 and len(body["samples"]) >= 170, str(body)[:200]
    assert all(s["earth_elevation_deg"] is not None for s in body["samples"]), "timeline has no Earth positions"


@check("forecast (one chunk, 721 samples)")
def _():
    code, body, _ = call("POST", "/api/forecast", {**SITE, **PERIOD})
    assert code == 200 and body["sample_count"] == 721, body


@check("surface skyline from LOLA (150 km horizon)")
def _():
    code, body, _ = call("POST", "/api/surface/skyline", {**SITE, "step_deg": 10})
    assert code == 200 and body["available"] and len(body["skyline"]) == 36, body
    assert all(p["elevation_deg"] < 2 for p in body["skyline"]), "Connecting Ridge should see past every horizon"


@check("surface safety screen with real slopes and shadows")
def _():
    code, body, _ = call("POST", "/api/surface/screen", {**SITE, "datetime_utc": "2027-01-10T00:00:00Z", "radius_m": 500})
    assert code == 200 and body["available"] and body["coverage_fraction"] == 1, body


@check("Excel export of a 427-day period (10,273 samples)")
def _():
    code, body, _ = call("POST", "/api/export/xlsx", {**SITE, "start_datetime_utc": "2027-01-01T00:00:00Z", "end_datetime_utc": "2028-03-03T23:59:00Z", "step_hours": 1, "display_timezone": "UTC"})
    assert code == 200 and isinstance(body, bytes) and body[:2] == b"PK", body


@check("mission agent swarm simulation")
def _():
    code, body, _ = call("POST", "/api/mission/simulate", {"inputs": {"departure_utc": "2027-01-05T12:00:00Z", "target_latitude_deg": SITE["latitude"], "target_longitude_deg": SITE["longitude"]}})
    assert code == 200, body


@check("named features list")
def _():
    code, body, _ = call("GET", "/api/moon/features")
    assert code == 200, body


@check("crawlers are refused")
def _():
    req = urllib.request.Request(BASE + "/api/status", headers={"User-Agent": "GPTBot/1.1"})
    try:
        urllib.request.urlopen(req, timeout=20)
        raise AssertionError("bot was served")
    except urllib.error.HTTPError as error:
        assert error.code == 403


failures = 0
for name, fn in CHECKS:
    started = time.time()
    try:
        fn()
        print(f"PASS  {name}  ({time.time() - started:.1f} s)")
    except Exception as error:  # noqa: BLE001
        failures += 1
        print(f"FAIL  {name}: {str(error)[:300]}")
print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed")
sys.exit(1 if failures else 0)
