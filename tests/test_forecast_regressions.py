"""Regression checks for sampling, model provenance and workbook timestamps."""
from datetime import datetime, timedelta, timezone
from openpyxl import load_workbook
import pytest

from app.services.forecast import compute_forecast, ForecastError
from app.services.thermal import temperature_confidence_label
from app.services.workbook import forecast_workbook


def test_changing_output_interval_preserves_common_temperatures(providers):
    start = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    args = dict(latitude_deg=0.6741, longitude_deg=23.4729, start=start,
                end=start + timedelta(days=2), providers=providers)
    hourly = compute_forecast(**args, step_hours=1)
    coarse = compute_forecast(**args, step_hours=6)
    quarter = compute_forecast(**args, step_hours=0.25)
    indexed = {row.datetime_utc: row for row in hourly.samples}
    for row in [*coarse.samples, *(r for r in quarter.samples if r.datetime_utc in indexed)]:
        assert row.temperature_k == pytest.approx(indexed[row.datetime_utc].temperature_k, abs=1e-9)
        assert row.solar_irradiance_w_m2 == indexed[row.datetime_utc].solar_irradiance_w_m2


def test_workbook_preserves_forecast_values_and_selected_timezone(providers):
    start = datetime(2026, 9, 26, 12, tzinfo=timezone.utc)
    forecast = compute_forecast(latitude_deg=0, longitude_deg=0, start=start,
                                end=start + timedelta(hours=1), step_hours=1, providers=providers)
    book = load_workbook(forecast_workbook(forecast, providers, 'Asia/Kolkata'))
    sheet = book['Hourly Predictions']
    assert sheet['A2'].value == start.replace(tzinfo=None)
    assert sheet['B2'].value == (start + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
    assert sheet['C2'].value == pytest.approx(forecast.samples[0].temperature_c)
    assert dict(book['Metadata'].values)['Selected display time zone'] == 'Asia/Kolkata'


def test_extreme_forecast_range_rejected_before_grid_allocation(providers):
    with pytest.raises(ForecastError, match='exceeds'):
        compute_forecast(latitude_deg=0, longitude_deg=0,
                         start=datetime(1900, 1, 1, tzinfo=timezone.utc),
                         end=datetime(2050, 1, 1, tzinfo=timezone.utc),
                         step_hours=0.001, providers=providers)


def test_diviner_availability_does_not_claim_calibration():
    assert temperature_confidence_label(False, True) == 'experimental'
