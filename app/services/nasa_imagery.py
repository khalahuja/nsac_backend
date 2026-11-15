"""Discover WMTS matrices from a fixed allow-list of NASA services."""
from __future__ import annotations
from functools import lru_cache
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from .nasa_catalog import LAYERS, ProductError


@lru_cache(maxsize=8)
def layer_capabilities(layer_id: str) -> dict:
    if layer_id not in LAYERS:
        raise ProductError("Unknown NASA map layer.")
    layer = LAYERS[layer_id]
    with urlopen(layer["endpoint"] + "/1.0.0/WMTSCapabilities.xml", timeout=12) as response:
        data = response.read(1_000_001)
    if len(data) > 1_000_000:
        raise ProductError("WMTS capabilities response exceeds the limit.")
    root = ET.fromstring(data)
    ns = {"w": "http://www.opengis.net/wmts/1.0", "o": "http://www.opengis.net/ows/1.1"}
    product = root.find("w:Contents/w:Layer", ns)
    if product is None:
        raise ProductError("NASA service did not return a WMTS layer.")
    matrix_id = product.findtext("w:TileMatrixSetLink/w:TileMatrixSet", namespaces=ns)
    matrix_set = next((m for m in root.findall("w:Contents/w:TileMatrixSet", ns) if m.findtext("o:Identifier", namespaces=ns) == matrix_id), None)
    if matrix_set is None:
        raise ProductError("NASA service returned no matching tile matrix.")
    format_name = product.findtext("w:Format", namespaces=ns)
    extension = "png" if format_name == "image/png" else "jpg" if format_name == "image/jpeg" else None
    if extension is None:
        raise ProductError("NASA service returned an unsupported tile image format.")
    matrices = []
    for matrix in matrix_set.findall("w:TileMatrix", ns):
        def get(name):
            return matrix.findtext("w:" + name, namespaces=ns)
        matrices.append({
            "id": matrix.findtext("o:Identifier", namespaces=ns),
            "width": int(float(get("MatrixWidth"))), "height": int(float(get("MatrixHeight"))),
            "tile_width": int(get("TileWidth")), "tile_height": int(get("TileHeight")),
            "top_left": [float(v) for v in get("TopLeftCorner").split()],
            "pixel_size": float(get("ScaleDenominator")) * 0.00028,
        })
    return {
        "id": layer_id, **layer, "matrix_set": matrix_id, "matrices": matrices,
        "style": product.findtext("w:Style/o:Identifier", namespaces=ns) or "default",
        "extension": extension,
        "attribution": "NASA / GSFC / ASU (LROC); NASA PDS and instrument teams",
    }
