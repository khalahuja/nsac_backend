"""On-demand JPL Horizons observer angles at a lunar surface coordinate."""
from __future__ import annotations

import json
from functools import lru_cache
import re
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


class HorizonsUnavailable(Exception):
    pass


def _angle_row(text: str) -> tuple[float, float]:
    if "$$SOE" not in text or "$$EOE" not in text:
        raise HorizonsUnavailable("Horizons did not return an observer ephemeris for this request.")
    row = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip().splitlines()[0]
    # Quantity 4 is the apparent azimuth/elevation pair. CSV rows may also
    # include a solar-presence marker, so parse only the last two numeric cells.
    values = [float(cell.strip()) for cell in row.split(",")[1:] if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", cell.strip())]
    if len(values) < 2:
        raise HorizonsUnavailable("Horizons returned a row without usable azimuth and elevation.")
    azimuth, elevation = values[-2:]
    if not 0 <= azimuth <= 360 or not -90 <= elevation <= 90:
        raise HorizonsUnavailable("Horizons returned angles outside the expected range.")
    return azimuth, elevation


@lru_cache(maxsize=512)
def observer_angles(lat: float, lon: float, moment: datetime, target: str) -> dict:
    if target not in {"10", "399"}:
        raise ValueError("Only Sun and Earth are supported")
    utc = moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
    params = {
        "format": "json", "COMMAND": f"'{target}'", "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'OBSERVER'", "CENTER": "'coord@301'",
        "COORD_TYPE": "'GEODETIC'", "SITE_COORD": f"'{lon:.6f},{lat:.6f},0'",
        "TIME_DIGITS": "\'FRACSEC\'", "TLIST": f"'{utc:%Y-%b-%d %H:%M:%S.%f}'", "QUANTITIES": "'4'", "CSV_FORMAT": "'YES'",
    }
    try:
        with urlopen(HORIZONS_URL + "?" + urlencode(params), timeout=8) as response:
            payload = json.load(response)
        azimuth, elevation = _angle_row(payload.get("result", ""))
    except HorizonsUnavailable:
        raise
    except Exception as error:
        raise HorizonsUnavailable("JPL Horizons is unreachable or returned an invalid response.") from error
    return {"azimuth_deg": azimuth, "elevation_deg": elevation, "source": "JPL Horizons observer quantity 4", "utc": utc.isoformat()}
