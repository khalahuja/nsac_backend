"""API contract tests: request validation, response shape, and provenance."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

REFERENCE_ISO = "2026-09-26T12:00:00Z"


def environment_payload(**overrides: object) -> dict:
    payload = {
        "latitude": -89.4,
        "longitude": 137.3,
        "datetime_utc": REFERENCE_ISO,
    }
    payload.update(overrides)
    return payload


class TestHealthAndStatus:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_status_reports_active_providers(self, client: TestClient) -> None:
        response = client.get("/api/status")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["astronomy_source"] == "simple-analytic-ephemeris"
        assert body["terrain_source"] == "flat-sphere"
        assert body["terrain_is_measured"] is False
        assert body["diviner_available"] is False

    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "/api/environment" in response.json()["paths"]


class TestEnvironmentValidation:
    @pytest.mark.parametrize("latitude", [-90.0, -89.4, 0.0, 45.0, 90.0])
    def test_accepts_valid_latitudes(self, client: TestClient, latitude: float) -> None:
        response = client.post(
            "/api/environment", json=environment_payload(latitude=latitude)
        )
        assert response.status_code == 200

    @pytest.mark.parametrize("longitude", [-180.0, -137.3, 0.0, 137.3, 180.0])
    def test_accepts_valid_longitudes(self, client: TestClient, longitude: float) -> None:
        response = client.post(
            "/api/environment", json=environment_payload(longitude=longitude)
        )
        assert response.status_code == 200

    @pytest.mark.parametrize("latitude", [-90.1, 90.1, 1000.0, -1000.0])
    def test_rejects_out_of_range_latitude(
        self, client: TestClient, latitude: float
    ) -> None:
        response = client.post(
            "/api/environment", json=environment_payload(latitude=latitude)
        )
        assert response.status_code == 422
        assert "latitude" in response.json()["detail"]

    @pytest.mark.parametrize("longitude", [-180.1, 180.1, 999.0])
    def test_rejects_out_of_range_longitude(
        self, client: TestClient, longitude: float
    ) -> None:
        response = client.post(
            "/api/environment", json=environment_payload(longitude=longitude)
        )
        assert response.status_code == 422
        assert "longitude" in response.json()["detail"]

    def test_rejects_invalid_datetime(self, client: TestClient) -> None:
        response = client.post(
            "/api/environment", json=environment_payload(datetime_utc="not-a-date")
        )
        assert response.status_code == 422
        assert response.json()["error_type"] == "validation_error"

    def test_rejects_missing_fields(self, client: TestClient) -> None:
        response = client.post("/api/environment", json={"latitude": 0.0})
        assert response.status_code == 422

    def test_rejects_non_numeric_latitude(self, client: TestClient) -> None:
        response = client.post(
            "/api/environment", json=environment_payload(latitude="north")
        )
        assert response.status_code == 422


class TestEnvironmentResponse:
    @pytest.fixture
    def body(self, client: TestClient) -> dict:
        response = client.post("/api/environment", json=environment_payload())
        assert response.status_code == 200
        return response.json()

    def test_echoes_the_requested_coordinate(self, body: dict) -> None:
        assert body["coordinate"]["latitude"] == pytest.approx(-89.4)
        assert body["coordinate"]["longitude"] == pytest.approx(137.3)

    def test_contains_every_documented_section(self, body: dict) -> None:
        for section in ("coordinate", "terrain", "solar", "thermal", "sources", "fidelity"):
            assert section in body

    def test_solar_fields_are_in_range(self, body: dict) -> None:
        solar = body["solar"]
        assert -90.0 <= solar["elevation_deg"] <= 90.0
        assert 0.0 <= solar["azimuth_deg"] < 360.0
        assert 0.0 <= solar["incidence_angle_deg"] <= 180.0
        assert solar["solar_irradiance_w_m2"] >= 0.0
        assert len(solar["sun_vector_body"]) == 3
        assert len(solar["sun_vector_j2000"]) == 3

    def test_thermal_is_labelled_as_a_model_not_a_measurement(self, body: dict) -> None:
        thermal = body["thermal"]
        assert thermal["is_measured"] is False
        assert thermal["model"] == "simplified-radiative-equilibrium"
        assert thermal["confidence"] == "experimental"
        assert thermal["temperature_c"] == pytest.approx(
            thermal["temperature_k"] - 273.15
        )

    def test_terrain_is_labelled_as_a_placeholder(self, body: dict) -> None:
        assert body["terrain"]["source"] == "flat-sphere"
        assert body["terrain"]["is_measured"] is False

    def test_terrain_shadow_is_unknown_without_a_dem(self, body: dict) -> None:
        assert body["solar"]["terrain_shadowed"] is None

    def test_diviner_observation_is_null_when_no_dataset(self, body: dict) -> None:
        """The spec forbids fabricating an observed temperature."""
        assert body["diviner_observation"] is None

    def test_provenance_is_present(self, body: dict) -> None:
        assert body["sources"]["astronomy"] == "simple-analytic-ephemeris"
        assert body["sources"]["terrain"] == "flat-sphere"
        assert body["sources"]["temperature"] == "simplified-radiative-model"

    def test_fidelity_labels_are_human_readable(self, body: dict) -> None:
        fidelity = body["fidelity"]
        assert "Approximate" in fidelity["astronomy"]
        assert "Placeholder" in fidelity["terrain"]
        assert "Experimental" in fidelity["temperature"]
        assert fidelity["overall_note"]

    def test_accuracy_notes_disclose_limitations(self, body: dict) -> None:
        notes = " ".join(body["accuracy_notes"])
        assert "not a JPL-precision" in notes
        assert "Diviner" in notes

    def test_sun_events_are_returned(self, body: dict) -> None:
        assert "sun_events" in body
        events = body["sun_events"]
        assert events is not None
        assert "search_window_hours" in events

    def test_sun_events_can_be_skipped(self, client: TestClient) -> None:
        response = client.post(
            "/api/environment", json=environment_payload(include_sun_events=False)
        )
        assert response.json()["sun_events"] is None


class TestDayNightConsistency:
    def test_subsolar_point_is_illuminated_and_hot(self, client: TestClient) -> None:
        """Query the sub-solar point itself; it must be lit and near peak temperature."""
        first = client.post("/api/environment", json=environment_payload()).json()
        subsolar_latitude = first["solar"]["subsolar_latitude_deg"]
        subsolar_longitude = first["solar"]["subsolar_longitude_deg"]

        response = client.post(
            "/api/environment",
            json=environment_payload(
                latitude=subsolar_latitude,
                longitude=subsolar_longitude,
                include_sun_events=False,
            ),
        )
        body = response.json()
        assert body["solar"]["illuminated"] is True
        assert body["solar"]["incidence_angle_deg"] == pytest.approx(0.0, abs=0.01)
        assert body["solar"]["elevation_deg"] == pytest.approx(90.0, abs=0.01)
        assert body["thermal"]["temperature_k"] > 350.0

    def test_antisolar_point_is_dark_and_cold(self, client: TestClient) -> None:
        """The antipode of the sub-solar point is in full lunar night."""
        first = client.post("/api/environment", json=environment_payload()).json()
        latitude = -first["solar"]["subsolar_latitude_deg"]
        longitude = first["solar"]["subsolar_longitude_deg"] - 180.0
        if longitude < -180.0:
            longitude += 360.0

        body = client.post(
            "/api/environment",
            json=environment_payload(
                latitude=latitude, longitude=longitude, include_sun_events=False
            ),
        ).json()
        assert body["solar"]["illuminated"] is False
        assert body["solar"]["solar_irradiance_w_m2"] == 0.0
        assert body["thermal"]["temperature_k"] == pytest.approx(100.0, abs=0.1)


class TestTimeline:
    @pytest.fixture
    def timeline_payload(self) -> dict:
        return {
            "latitude": -89.4,
            "longitude": 137.3,
            "center_datetime_utc": REFERENCE_ISO,
            "hours_before": 72,
            "hours_after": 72,
            "step_hours": 3,
        }

    def test_returns_the_expected_sample_count(
        self, client: TestClient, timeline_payload: dict
    ) -> None:
        response = client.post("/api/timeline", json=timeline_payload)
        assert response.status_code == 200
        body = response.json()
        # 144 h span / 3 h = 48 intervals -> 49 samples.
        assert body["sample_count"] == 49
        assert len(body["samples"]) == 49

    def test_samples_are_chronological(
        self, client: TestClient, timeline_payload: dict
    ) -> None:
        samples = client.post("/api/timeline", json=timeline_payload).json()["samples"]
        times = [sample["datetime_utc"] for sample in samples]
        assert times == sorted(times)

    def test_sample_fields_are_consistent(
        self, client: TestClient, timeline_payload: dict
    ) -> None:
        samples = client.post("/api/timeline", json=timeline_payload).json()["samples"]
        for sample in samples:
            assert -90.0 <= sample["solar_elevation_deg"] <= 90.0
            assert sample["solar_irradiance_w_m2"] >= 0.0
            assert sample["temperature_k"] >= 100.0
            # Darkness must mean exactly zero flux and the temperature floor.
            if not sample["illuminated"]:
                assert sample["solar_irradiance_w_m2"] == 0.0
                assert sample["temperature_k"] == pytest.approx(100.0, abs=1e-6)

    def test_carries_provenance(self, client: TestClient, timeline_payload: dict) -> None:
        body = client.post("/api/timeline", json=timeline_payload).json()
        assert body["sources"]["astronomy"] == "simple-analytic-ephemeris"
        assert body["synodic_month_hours"] == pytest.approx(708.7, abs=0.5)

    def test_supports_a_full_lunar_day_window(self, client: TestClient) -> None:
        """The UI offers a ~one-lunar-day window; the API must accept it."""
        response = client.post(
            "/api/timeline",
            json={
                "latitude": 0.0,
                "longitude": 0.0,
                "center_datetime_utc": REFERENCE_ISO,
                "hours_before": 355,
                "hours_after": 355,
                "step_hours": 6,
            },
        )
        assert response.status_code == 200
        assert response.json()["sample_count"] > 100

    def test_equatorial_timeline_covers_day_and_night(self, client: TestClient) -> None:
        """Over a full lunar day the equator must see both light and darkness."""
        samples = client.post(
            "/api/timeline",
            json={
                "latitude": 0.0,
                "longitude": 0.0,
                "center_datetime_utc": REFERENCE_ISO,
                "hours_before": 355,
                "hours_after": 355,
                "step_hours": 6,
            },
        ).json()["samples"]
        illuminated = [sample["illuminated"] for sample in samples]
        assert any(illuminated)
        assert not all(illuminated)

    def test_rejects_too_many_samples(self, client: TestClient) -> None:
        """Excessive requests fail loudly instead of being silently coarsened."""
        response = client.post(
            "/api/timeline",
            json={
                "latitude": 0.0,
                "longitude": 0.0,
                "center_datetime_utc": REFERENCE_ISO,
                "hours_before": 1440,
                "hours_after": 1440,
                "step_hours": 0.1,
            },
        )
        assert response.status_code == 422
        assert "exceeds the limit" in response.json()["detail"]

    def test_rejects_invalid_step(self, client: TestClient) -> None:
        response = client.post(
            "/api/timeline",
            json={
                "latitude": 0.0,
                "longitude": 0.0,
                "center_datetime_utc": REFERENCE_ISO,
                "step_hours": 0,
            },
        )
        assert response.status_code == 422

    def test_rejects_out_of_range_coordinates(self, client: TestClient) -> None:
        response = client.post(
            "/api/timeline",
            json={
                "latitude": 120.0,
                "longitude": 0.0,
                "center_datetime_utc": REFERENCE_ISO,
            },
        )
        assert response.status_code == 422


class TestMoonFeatures:
    def test_returns_a_feature_catalogue(self, client: TestClient) -> None:
        response = client.get("/api/moon/features")
        assert response.status_code == 200
        features = response.json()
        assert isinstance(features, list)
        for feature in features:
            assert -90.0 <= feature["lat"] <= 90.0
            assert -180.0 <= feature["lon"] <= 180.0


def test_timeline_includes_earth_positions(client):
    """The surface view draws Earth's sky path from the timeline."""
    response = client.post("/api/timeline", json={"latitude": -89.5, "longitude": 0, "center_datetime_utc": "2027-01-10T12:00:00Z", "hours_before": 48, "hours_after": 48, "step_hours": 6})
    assert response.status_code == 200
    assert all(sample["earth_elevation_deg"] is not None for sample in response.json()["samples"])
