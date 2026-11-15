"""Native equal-area Lunar Prospector hydrogen bins, in ppm by weight."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import math
import numpy as np

SOURCE = "https://pds-geosciences.wustl.edu/missions/lunarp/reduced/hydrogenlow.txt"
DOCUMENTATION = "https://pds-geosciences.wustl.edu/missions/lunarp/reduced_special.html"

@lru_cache(maxsize=8)
def _grid(path: str, modified: int):
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] != 5 or not np.isfinite(data).all():
        raise ValueError("Invalid Lunar Prospector hydrogen table.")
    return data


def hydrogen_summary(directory: Path, latitude: float, longitude: float) -> dict:
    path = directory / "hydrogenlow.txt"
    base = {
        "available": False, "source_url": SOURCE, "documentation_url": DOCUMENTATION,
        "title": "Lunar Prospector Hydrogen_low, June 2001", "units": "ppm by weight",
        "selected_value_ppm": None, "selected_bounds": None, "cells": [],
        "hemisphere": "south" if latitude < 0 else "north",
        "note": "Native approximately 60 km equal-area bins from a NASA-hosted team special product, not a PDS peer-reviewed archive. Hydrogen is a resource proxy, not exposed ice or accessible water reserves. Estimates may be unreliable in high-thorium/rare-earth regions.",
    }
    if not path.is_file():
        return base
    data = _grid(str(path), path.stat().st_mtime_ns)
    lon = (longitude + 180) % 360 - 180
    hits = data[(data[:,0] <= latitude) & ((latitude < data[:,1]) | ((latitude == 90) & (data[:,1] == 90))) & (data[:,2] <= lon) & (lon < data[:,3])]
    polar = data[data[:,1] <= -75] if latitude < 0 else data[data[:,0] >= 75]
    cells = [{"lat_min": float(a), "lat_max": float(b), "lon_min": float(c), "lon_max": float(d),
              "hydrogen_ppm": float(value) if value >= 0 else None} for a,b,c,d,value in polar]
    hit = hits[0] if len(hits) else None
    return {**base, "available": True, "cells": cells,
            "selected_value_ppm": float(hit[4]) if hit is not None and hit[4] >= 0 else None,
            "selected_bounds": hit[:4].tolist() if hit is not None else None}
