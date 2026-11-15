"""Validate sourced evidence footprints without turning proxies into ice detections."""
from __future__ import annotations
import json
import math
from pathlib import Path

from .nasa_catalog import ProductError

KINDS = {"ice_detection", "cold_conditions", "resource_potential"}
RADIUS_KM = 1737.4


def validate_evidence(data: dict) -> dict:
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ProductError("Water evidence must be a GeoJSON FeatureCollection.")
    if len(data["features"]) > 1000:
        raise ProductError("At most 1000 evidence features are allowed.")
    total = 0
    normalized = []
    for feature in data["features"]:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ProductError("Each evidence feature must be a GeoJSON Feature object.")
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            raise ProductError("Evidence properties must be an object.")
        if properties.get("evidence_type") not in KINDS:
            raise ProductError("Each feature must declare ice_detection, cold_conditions or resource_potential.")
        source = properties.get("source_url", "")
        if not isinstance(source, str) or not source.startswith("https://"):
            raise ProductError("Each evidence feature requires an HTTPS source_url.")
        geometry = feature.get("geometry") or {}
        if not isinstance(geometry, dict):
            raise ProductError("Evidence geometry must be an object.")
        kind, coordinates = geometry.get("type"), geometry.get("coordinates")
        if kind not in {"Point", "Polygon", "MultiPolygon"}:
            raise ProductError("Evidence geometry must be Point, Polygon or MultiPolygon.")
        def validate_coordinate(pair):
            nonlocal total
            if not isinstance(pair, list) or len(pair) < 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in pair[:2]):
                raise ProductError("Evidence coordinates must be numeric longitude/latitude pairs.")
            lon, lat = pair[:2]
            if not math.isfinite(lon) or not math.isfinite(lat) or abs(lon) > 180 or abs(lat) > 90:
                raise ProductError("Evidence coordinates must use east-positive longitude [-180,180] and latitude [-90,90].")
            total += 1
            if total > 20_000:
                raise ProductError("Evidence geometry exceeds 20,000 vertices.")
            return [lon, lat]
        if kind == "Point":
            normalized_coordinates = validate_coordinate(coordinates)
        else:
            polygons = [coordinates] if kind == "Polygon" else coordinates
            if not isinstance(polygons, list) or not polygons:
                raise ProductError("Evidence polygons must contain closed rings.")
            normalized_polygons = []
            for polygon in polygons:
                if not isinstance(polygon, list) or not polygon:
                    raise ProductError("Evidence polygon is empty.")
                rings = []
                for ring in polygon:
                    if not isinstance(ring, list) or len(ring) < 4:
                        raise ProductError("Evidence rings require at least four coordinates.")
                    checked = [validate_coordinate(pair) for pair in ring]
                    if checked[0] != checked[-1]:
                        raise ProductError("Evidence polygon rings must be closed.")
                    rings.append(checked)
                normalized_polygons.append(rings)
            normalized_coordinates = normalized_polygons[0] if kind == "Polygon" else normalized_polygons
        normalized.append({
            "type": "Feature", "geometry": {"type": kind, "coordinates": normalized_coordinates},
            "properties": {
                "name": str(properties.get("name", "Evidence footprint"))[:120],
                "evidence_type": properties["evidence_type"], "source_url": source[:2048],
                "dataset": str(properties.get("dataset", ""))[:200],
                "note": str(properties.get("note", ""))[:500],
            },
        })
    return {"type": "FeatureCollection", "features": normalized}


def load_evidence(directory: Path) -> dict:
    path = directory / "water.geojson"
    if not path.is_file():
        return {"type": "FeatureCollection", "features": []}
    if path.stat().st_size > 1_000_000:
        raise ProductError("Configured water evidence exceeds 1 MB.")
    return validate_evidence(json.loads(path.read_text(encoding="utf-8")))


def summarize_evidence(data: dict, latitude: float, longitude: float) -> dict:
    """Distances to sourced geometry; no inference from a crater's circular outline."""
    lat0, lon0 = math.radians(latitude), math.radians(longitude)
    def local(pair):
        lon, lat = map(math.radians, pair)
        dl = lon - lon0
        cosine = min(1.0, max(-1.0, math.sin(lat0)*math.sin(lat)+math.cos(lat0)*math.cos(lat)*math.cos(dl)))
        angle = math.acos(cosine)
        scale = 1.0 if angle < 1e-12 else angle / max(1e-12, math.sin(angle))
        return (RADIUS_KM*scale*math.cos(lat)*math.sin(dl),
                RADIUS_KM*scale*(math.cos(lat0)*math.sin(lat)-math.sin(lat0)*math.cos(lat)*math.cos(dl)))
    def inside(ring):
        hit = False
        for a, b in zip(ring, ring[1:]):
            if (a[1] > 0) != (b[1] > 0) and 0 < (b[0]-a[0])*(-a[1])/(b[1]-a[1])+a[0]:
                hit = not hit
        return hit
    def segment(a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        t = max(0, min(1, -(a[0]*dx+a[1]*dy)/(dx*dx+dy*dy))) if dx*dx+dy*dy else 0
        return math.hypot(a[0]+t*dx, a[1]+t*dy)
    rows = []
    for feature in data["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "Point":
            p = local(geometry["coordinates"])
            distance = math.hypot(*p)
            contains = distance < 1e-6
        else:
            polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
            projected = [[list(map(local, ring)) for ring in polygon] for polygon in polygons]
            contains = any(inside(polygon[0]) and not any(inside(hole) for hole in polygon[1:]) for polygon in projected)
            distance = 0.0 if contains else min(segment(a,b) for polygon in projected for ring in polygon for a,b in zip(ring,ring[1:]))
        rows.append({**feature["properties"], "distance_km": distance, "contains_selected_site": contains})
    return {
        "feature_collection": data, "features": sorted(rows, key=lambda r: r["distance_km"]),
        "counts": {kind: sum(r["evidence_type"] == kind for r in rows) for kind in sorted(KINDS)},
        "notes": [
            "Imported source attribution is supplied by the dataset author; Selene does not independently verify each detection.",
            "Cold conditions and hydrogen/resource proxies remain separate from spectroscopic ice detections.",
            "Polygon distances use a site-centred azimuthal-equidistant projection with straight edges; interpret as regional screening distances.",
            "No footprint does not mean no ice. Remote sensing does not establish accessible reserves or extraction yield.",
        ],
    }
