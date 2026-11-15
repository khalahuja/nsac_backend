"""Deterministic requirement commands. Unsupported compound edits go to the assistant."""
import re
from .operations import OperationsRequest

def requirement_command(message: str, current: OperationsRequest):
    text = message.lower().strip().rstrip('.!?')
    patterns = [
        (r'(?:set |change )?(?:my )?mission duration (?:to )?(\d+(?:\.\d+)?) (hours?|days?)', 'mission_hours'),
        (r'(?:set |change )?minimum sunlight(?: coverage)? (?:to )?(\d+(?:\.\d+)?)\s*(%|percent)', 'min_sunlight_pct'),
        (r'(?:set |change )?minimum earth visibility (?:to )?(\d+(?:\.\d+)?)\s*(%|percent)', 'min_earth_pct'),
        (r'(?:set |change )?maximum (?:contact|communication) outage (?:to )?(\d+(?:\.\d+)?) (hours?|days?)', 'max_outage_hours'),
        (r'(?:set |change )?maximum (?:continuous )?darkness (?:to )?(\d+(?:\.\d+)?) (hours?|days?)', 'max_dark_hours'),
        (r'(?:set |change )?minimum (?:simultaneous |sunlight and contact )?overlap (?:to )?(\d+(?:\.\d+)?) (hours?|days?)', 'min_both_hours'),
        (r'(?:set |change )?minimum earth elevation (?:to )?(\d+(?:\.\d+)?) (degrees?)', 'min_earth_elevation_deg'),
        (r'(?:set |change )?nearby (?:search )?radius (?:to )?(\d+(?:\.\d+)?) (km)', 'nearby_radius_km'),
    ]
    for pattern, key in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            value = float(match[1]) * (24 if match[2].startswith('day') else 1)
            data = current.model_dump()
            if key == 'nearby_radius_km': data[key] = value
            else: data['requirements'][key] = value
            return OperationsRequest.model_validate(data)
    return None
