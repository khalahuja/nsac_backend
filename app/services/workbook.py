"""Excel workbook generation for lunar forecasts."""

from __future__ import annotations

from io import BytesIO
from zoneinfo import ZoneInfo

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from ..models.schemas import ForecastResponse
from .providers import ScienceProviders


def forecast_workbook(
    forecast: ForecastResponse, providers: ScienceProviders, display_timezone: str = "UTC"
) -> BytesIO:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Hourly Predictions"

    headers = [
        "Date / time (UTC)",
        f"Date / time ({display_timezone})",
        "Temperature (°C)",
        "Temperature (K)",
        "Horizontal irradiance (W/m²)",
        "Terrain-surface irradiance (W/m²)",
        "Sun elevation (°)",
        "Sun azimuth (°)",
        "Incidence angle (°)",
        "Status",
        "Earth elevation (degrees)",
        "Earth azimuth (degrees)",
        "Earth distance from site (km)",
        "DTE visibility (active horizon)",
        "Terrain shadow (unknown if blank)",
        "Sun visibility margin (degrees)",
        "Earth visibility margin (degrees)",
    ]
    sheet.append(headers)
    header_fill = PatternFill("solid", fgColor="182536")
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True)

    local_zone = ZoneInfo(display_timezone)
    for sample in forecast.samples:
        moment = sample.datetime_utc.replace(tzinfo=None)
        local_moment = sample.datetime_utc.astimezone(local_zone).replace(tzinfo=None)
        sheet.append(
            [
                moment,
                local_moment,
                sample.temperature_c,
                sample.temperature_k,
                sample.horizontal_irradiance_w_m2,
                sample.solar_irradiance_w_m2,
                sample.solar_elevation_deg,
                sample.solar_azimuth_deg,
                sample.incidence_angle_deg,
                "Sunlit" if sample.illuminated else "Shadowed",
                sample.earth_elevation_deg,
                sample.earth_azimuth_deg,
                sample.earth_distance_km,
                "Visible" if sample.earth_visible else "Below threshold",
                sample.terrain_shadowed,
                sample.solar_horizon_margin_deg,
                sample.earth_horizon_margin_deg,
            ]
        )

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.column_dimensions["A"].width = 22
    sheet.column_dimensions["B"].width = 28
    for column in "CDEFGHI":
        sheet.column_dimensions[column].width = 22
    sheet.column_dimensions["J"].width = 14
    for column in 'KLMNOPQ':
        sheet.column_dimensions[column].width = 26
    for column in ("A", "B"):
        for cell in sheet[column][1:]:
            cell.number_format = "yyyy-mm-dd hh:mm"
    for row in sheet.iter_rows(min_row=2, min_col=3, max_col=9):
        for cell in row:
            cell.number_format = "0.00"

    metadata = workbook.create_sheet("Metadata")
    rows = [
        ("Field", "Value"),
        ("Latitude (°; north positive)", forecast.coordinate.latitude),
        ("Longitude (°; east positive)", forecast.coordinate.longitude),
        ("Start UTC", forecast.start_datetime_utc.isoformat()),
        ("End UTC", forecast.end_datetime_utc.isoformat()),
        ("Selected display time zone", display_timezone),
        ("Start in selected time zone", forecast.start_datetime_utc.astimezone(local_zone).isoformat()),
        ("End in selected time zone", forecast.end_datetime_utc.astimezone(local_zone).isoformat()),
        ("Sampling interval (hours)", forecast.step_hours),
        ("Sample count", forecast.sample_count),
        ("Earth elevation threshold (degrees)", forecast.min_earth_elevation_deg),
        ("Geometric Earth visibility (hours)", forecast.earth_visible_hours),
        ("Sun above horizon (hours)", forecast.sun_above_horizon_hours),
        ("Horizontal solar energy (Wh/m2)", forecast.horizontal_energy_wh_m2),
        ("Terrain skyline applied", forecast.terrain_visibility_available),
        ("Longest sampled darkness (hours)", forecast.longest_darkness_hours),
        ("Darkness extends beyond forecast boundary", forecast.darkness_boundary_limited),
        ("Position and communication limitations", forecast.position_notes),
        ("Astronomy source", forecast.sources.astronomy),
        ("Terrain source", forecast.sources.terrain),
        ("Thermal model", forecast.model_name),
        ("Diviner source", forecast.sources.diviner),
        ("NASA texture dataset", "LRO WAC global color mosaic (visualization only)"),
        ("Terrain dataset role", "LOLA / SLDEM when configured; spherical fallback otherwise"),
        ("Ephemeris role", "JPL SPICE when configured; analytic fallback otherwise"),
        ("Albedo", providers.albedo),
        ("Emissivity", providers.emissivity),
        ("Temperature assumptions", forecast.model_notes),
        (
            "Irradiance assumptions",
            "Direct solar beam only; no atmosphere, diffuse light, earthshine, or self-heating.",
        ),
        (
            "Terrain shadow limitation",
            "A 1-degree measured terrain skyline is interpolated when available; otherwise a spherical horizon is used. " + forecast.position_notes,
        ),
        (
            "Scientific limitation",
            "Predictions are model outputs, not retrieved future NASA measurements. Validate against Diviner before operational use.",
        ),
    ]
    for row in rows:
        metadata.append(row)
    for index, window in enumerate(forecast.earth_visibility_windows, 1):
        metadata.append((f"DTE window {index} (UTC; interpolated)", f"{window.start_utc.isoformat()} to {window.end_utc.isoformat()} ({window.duration_hours:.3f} h)"))
    for cell in metadata[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
    metadata.column_dimensions["A"].width = 34
    metadata.column_dimensions["B"].width = 110
    for row in metadata.iter_rows():
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    stream = BytesIO()
    workbook.save(stream)
    stream.seek(0)
    return stream


__all__ = ["forecast_workbook"]
