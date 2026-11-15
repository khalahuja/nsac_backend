"""Fetch a bounded starter set from verified NASA/PDS product links.

No packages are installed. Run from the repository:
    env/Scripts/python.exe backend/scripts/fetch_nasa_data.py
Downloads four 41 MB LOLA files and one 213 MB Diviner source table.
The Diviner table is streamed, retaining only 88-90 S; it is not kept in full.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGION = ROOT / "data" / "nasa"
DIVINER = ROOT / "data" / "diviner"
LOLA = "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site01/"
PDS = "https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_pcp/diurnal/ltim/pols/"
PRODUCT = "pcp_avg_tbol_pols_sum_ltim01_240"


def terrain():
    REGION.mkdir(parents=True, exist_ok=True)
    fields = {"elevation": "surf", "slope": "slp", "height_uncertainty": "toterr", "slope_uncertainty": "slperr"}
    region = {
        "id": "connecting-ridge-5m", "title": "LOLA Connecting Ridge (Site 01), 5 m",
        "source_url": "https://pgda.gsfc.nasa.gov/products/78",
        "resolution_m": 5, "height_units": "m", "slope_units": "deg",
        "frame": "MOON_ME", "ephemeris": "DE421", "files": {},
    }
    for key, suffix in fields.items():
        name = f"Site01_final_adj_5mpp_{suffix}.tif"
        target = REGION / name
        if not target.is_file():
            temporary = target.with_suffix(".download")
            digest = hashlib.sha256()
            with urlopen(LOLA + name, timeout=60) as response, temporary.open("wb") as out:
                if "tiff" not in response.headers.get("Content-Type", "").lower():
                    raise ValueError(f"NASA returned a non-TIFF response for {name}")
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 100_000_000:
                        raise ValueError("LOLA starter product exceeds 100 MB.")
                    digest.update(block)
                    out.write(block)
            # Validate it can be opened before marking it installed.
            import rasterio
            with rasterio.open(temporary) as ds:
                if ds.crs is None or ds.width < 2:
                    raise ValueError("Invalid LOLA raster.")
            temporary.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        region[key] = name
        region["files"][key] = {"source_url": LOLA + name, "sha256": digest}
        print(f"Verified {name}", flush=True)
    catalog_path = REGION / "catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {"version": 1, "regions": []}
    catalog["regions"] = [r for r in catalog["regions"] if r["id"] != region["id"]] + [region]
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")


def diviner():
    DIVINER.mkdir(parents=True, exist_ok=True)
    target = DIVINER / (PRODUCT + "_88s.npz")
    if target.is_file() and (DIVINER / "manifest.json").is_file():
        print("Diviner starter product already prepared.", flush=True)
        return
    with urlopen(PDS + PRODUCT + ".lbl", timeout=30) as response:
        label = response.read().decode()
    if "CLOCTIME_MIN" not in label or "0.25 <hour>" not in label:
        raise ValueError("PDS local-time metadata changed; review the product before importing.")
    (DIVINER / (PRODUCT + ".lbl")).write_text(label, encoding="utf-8")
    columns = []
    digest = hashlib.sha256()
    # PCP fixed-length ASCII: normalized pole x/y, longitude, latitude, tbol K.
    with urlopen(PDS + PRODUCT + ".tab", timeout=60) as response:
        total = 0
        for line in response:
            total += len(line)
            if total > 250_000_000:
                raise ValueError("Diviner starter table exceeds 250 MB.")
            digest.update(line)
            try:
                values = [float(v) for v in line.decode("ascii").replace(",", " ").split()]
            except ValueError:
                continue
            if len(values) == 5 and values[3] <= -88 and 0 < values[4] < 500:
                columns.append(values)
    if not columns:
        raise ValueError("No valid Diviner polar measurements were read.")
    data = np.asarray(columns)
    np.savez_compressed(target, x_m=data[:, 0] * 1_737_400, y_m=data[:, 1] * 1_737_400, temperature_k=data[:, 4])
    product = {
        "file": target.name, "dataset": PRODUCT.upper(), "units": "K",
        "kind": "local_time_climatology", "season": "summer",
        "local_time_min_hours": 0, "local_time_max_hours": 0.25,
        "reference_period": "2009-07-05 to 2019-02-17", "resolution_m": 240,
        "source_url": PDS + PRODUCT + ".xml", "source_sha256": digest.hexdigest(),
        "preparation": "Original x/y normalized by mean lunar radius, scaled to m; retained native cells at 88-90 S. No temperature interpolation.",
    }
    path = DIVINER / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"version": 1, "products": []}
    manifest["products"] = [p for p in manifest["products"] if p["file"] != product["file"]] + [product]
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(columns)} real Diviner pixels, summer local time 00:00-00:15.", flush=True)


def hydrogen():
    REGION.mkdir(parents=True, exist_ok=True)
    url = "https://pds-geosciences.wustl.edu/missions/lunarp/reduced/hydrogenlow.txt"
    target = REGION / "hydrogenlow.txt"
    if not target.is_file():
        with urlopen(url, timeout=30) as response:
            data = response.read(1_000_001)
        if len(data) > 1_000_000 or b"H ppm by weight" not in data[:2000]:
            raise ValueError("Lunar Prospector metadata changed; review the product.")
        import io
        values = np.loadtxt(io.BytesIO(data), comments="#", delimiter=",")
        if values.shape != (11306, 5):
            raise ValueError("Unexpected Lunar Prospector equal-area bin table.")
        target.write_bytes(data)
    (REGION / "hydrogen-source.json").write_text(json.dumps({
        "source_url": url, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "units": "H ppm by weight", "product": "Hydrogen_low, June 2001",
        "documentation": "https://pds-geosciences.wustl.edu/missions/lunarp/reduced_special.html",
    }, indent=2), encoding="utf-8")
    print("Verified local Lunar Prospector hydrogen bins.", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as executor:
        for future in [executor.submit(terrain), executor.submit(diviner), executor.submit(hydrogen)]:
            future.result()
