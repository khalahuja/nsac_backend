# NASA dataset integration guide

Current NASA workspace, prepared Diviner reader, 5 m products and import schemas: [data/nasa/README.md](../../../data/nasa/README.md).

This project ships with **no NASA data**. Every dataset below is optional, and
the application runs without any of them by falling back to documented
approximate models. Nothing is downloaded automatically — these archives are
large (tens of MB to >100 GB) and their use terms should be read by a human
first.

The rule the codebase enforces: **when a dataset is absent, the API labels its
output as approximate or returns `null`. It never invents a measurement.**

---

## 1. JPL/NAIF SPICE — Sun/Earth/Moon geometry

| | |
|---|---|
| **Consumed by** | `app/services/astronomy.py` → `SpiceAstronomyProvider` |
| **Enable with** | `USE_SPICE=true` |
| **Install to** | `data/spice/` |
| **Source** | <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/> |
| **Replaces** | `SimpleAstronomyProvider` (analytic, ~0.1–0.3° error) |

### Required kernels

| Kernel type | Example file | Purpose | Size |
|---|---|---|---|
| Leapseconds (LSK) | `naif0012.tls` | UTC → ephemeris time | ~5 KB |
| Planetary ephemeris (SPK) | `de440s.bsp` | Sun/Earth/Moon positions | ~32 MB |
| Planetary constants (PCK) | `pck00011.tpc` | Body radii, rotation elements | ~130 KB |
| Moon orientation (binary PCK) | `moon_pa_de440_200625.bpc` | High-precision lunar libration | ~50 MB |
| Moon frame definitions (FK) | `moon_de440_220930.tf` | Defines `MOON_ME` / `MOON_PA` frames | ~5 KB |

Download paths under the NAIF root above:

```text
lsk/naif0012.tls
spk/planets/de440s.bsp
pck/pck00011.tpc
pck/moon_pa_de440_200625.bpc
fk/satellites/moon_de440_220930.tf
```

### Install

```bash
cd data/spice
curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls
curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
curl -O https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
```

Then set `USE_SPICE=true` and restart. Every file with a recognised kernel
suffix (`.tls .bsp .tpc .bpc .tf .tm .ti .tsc`) in `data/spice/` is furnished
automatically, sorted by name for reproducibility.

The provider runs a **self-test** at construction: it computes Sun, Earth and
orientation vectors for a probe date. A kernel set that loads but cannot answer
those queries (e.g. an LSK with no SPK) is rejected, and the app falls back to
the analytic provider with a logged warning rather than failing at request time.

Confirm which provider is live at `GET /api/status`.

---

## 2. LRO LOLA / SLDEM2015 — terrain

| | |
|---|---|
| **Consumed by** | `app/services/terrain.py` → `RasterTerrainProvider` |
| **Enable with** | `USE_TERRAIN_DEM=true` |
| **Install to** | `data/terrain/` |
| **Source** | <https://pgda.gsfc.nasa.gov/products/54> (SLDEM2015), <https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/> (LOLA GDR) |
| **Replaces** | `FlatTerrainProvider` (perfect sphere, elevation/slope = 0) |

### Products

| Product | Coverage | Resolution | Size |
|---|---|---|---|
| SLDEM2015 | ±60° latitude | 64 px/deg (~474 m) | ~90 GB global |
| LOLA LDEM 118 | Global incl. poles | 4–128 px/deg | 0.2–40 GB |
| LOLA LDEM 875S | South pole | 5 m/px | ~10 GB |

For a first run, a coarse global LOLA GDR (`LDEM_16.IMG`, 16 px/deg, ~200 MB)
is enough to exercise the code path.

### Preparation

Convert the PDS `.IMG`/`.LBL` pair to a GeoTIFF in simple cylindrical
projection with degrees as the affine units:

```bash
gdal_translate -of GTiff LDEM_16.IMG lunar_dem.tif
gdalinfo lunar_dem.tif   # confirm the transform is in degrees
```

Place the result at `data/terrain/lunar_dem.tif` (or set
`TERRAIN_RASTER_FILENAME`) and set `USE_TERRAIN_DEM=true`.

### What becomes real

With a DEM loaded, `TerrainInfo.is_measured` flips to `true` and these stop
being placeholders:

- `elevation_m` — windowed read at the query pixel
- `slope_deg` — from a 3×3 central-difference gradient, with east–west spacing
  scaled by `cos(latitude)`
- `aspect_deg` — downhill azimuth, clockwise from North
- **terrain-corrected surface normal** — tilts the normal by the local
  gradient, so the solar incidence angle (and therefore irradiance and
  temperature) responds to real topography

Reads are always windowed (`rasterio` `Window`, `boundless=True`). The full
raster is never loaded into memory.

---

## 3. Terrain shadowing (horizon ray-marching)

| | |
|---|---|
| **Consumed by** | `RasterTerrainProvider.is_terrain_shadowed` |
| **Enable with** | `USE_TERRAIN_SHADOWS=true` (requires `USE_TERRAIN_DEM=true`) |

Marches outward along the solar azimuth, computing
`terrain_angle = atan((h_i - h_0) / d)` at each sample and comparing the
maximum against the solar elevation. Lunar curvature is removed by subtracting
`d² / (2R)` from each neighbour's height.

Without a DEM this returns `None`, which the API surfaces as
`"terrain_shadowed": null` — **unknown**, not "not shadowed". This distinction
matters most exactly where the tool is most useful: permanently shadowed
regions near the poles.

Cost note: the ray-march is per-query, so it runs on `/api/environment` but is
deliberately skipped in `/api/timeline`, where it would dominate runtime.

---

## 4. LRO Diviner - historical temperature comparisons

The reader supports prepared south-polar PDS PCP local-time/season climatologies.
Enable USE_DIVINER=true with products and manifest.json in data/diviner.
The importer streams a summer 00:00-00:15 bin and retains native cells at 88-90 S.
Unmatched pixels, local-time bins or seasons remain missing. These are historical
averages, not observations of the forecast date.

See [preparation and metadata](../../../data/nasa/README.md).

---

## 5. LROC imagery — surface texture

| | |
|---|---|
| **Consumed by** | Frontend (`data/textures/`) |
| **Source** | <https://wms.lroc.asu.edu/lroc/view_rdr/WAC_GLOBAL> |

The frontend currently draws a **procedural** lunar texture. It places the
major maria and the named craters from the feature catalogue at their real
coordinates, but the fine crater field is synthetic and is a visualisation aid
only — no pixel of it is ever read back as data.

To use real imagery, drop an equirectangular mosaic (e.g.
`WAC_GLOBAL_E000N0000_016P.tif`, ~1 GB at 100 m/px) into `data/textures/` and
load it instead of calling the procedural builder. Downsample to ≤8192×4096
first; browsers handle larger textures poorly.

---

## Verifying what is actually active

```bash
curl http://localhost:8000/api/status
```

```json
{
  "astronomy_source": "simple-analytic-ephemeris",
  "terrain_source": "flat-sphere",
  "terrain_is_measured": false,
  "diviner_available": false,
  "spice_requested": false
}
```

`*_requested` versus `*_source` is the key pair: if `spice_requested` is `true`
while `astronomy_source` is still `simple-analytic-ephemeris`, the kernels
failed to load and the server log explains why.
