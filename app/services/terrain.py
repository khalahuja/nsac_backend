"""Lunar terrain: elevation, slope, aspect, surface normal, and horizon shadows.

Provider hierarchy
------------------

``FlatTerrainProvider``
    The MVP default. Models the Moon as a perfect sphere of radius
    :data:`~app.constants.MOON_RADIUS_KM`. It returns **zero** elevation and
    **zero** slope and labels the source ``flat-sphere`` - these are not
    measurements and the API marks them as such. It never invents a number.

``RasterTerrainProvider``
    Reads a real DEM (LOLA / SLDEM2015) through ``rasterio`` using windowed
    reads, so a 100+ GB global raster is never pulled into RAM. Enabled with
    ``USE_TERRAIN_DEM=true`` plus a raster at ``data/terrain/``.

Slope / aspect conventions
--------------------------

* **Slope** is the angle between the local terrain normal and the spherical
  radial (up) direction: 0 deg = flat, 90 deg = vertical cliff.
* **Aspect** is the compass azimuth of the downhill direction, measured
  clockwise from North (0 = N, 90 = E), matching the solar azimuth convention
  in :mod:`app.services.illumination`.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import MOON_RADIUS_M
from .geometry import Vector3, local_enu_basis, normalize, spherical_surface_normal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TerrainSample:
    """Terrain state at one lunar coordinate.

    Attributes:
        elevation_m: Height above the :data:`MOON_RADIUS_M` reference sphere.
            ``None`` when no DEM is loaded - never a fabricated value.
        slope_deg: Terrain slope in degrees, or ``None`` if unmeasured.
        aspect_deg: Downhill azimuth in degrees (0 = N, clockwise), or ``None``.
        surface_normal: Unit normal in the body-fixed frame. For the flat
            provider this equals the spherical radial direction.
        source: Provenance tag, e.g. ``"flat-sphere"`` or ``"SLDEM2015"``.
        is_measured: ``True`` only when values came from a real dataset.
    """

    elevation_m: float | None
    slope_deg: float | None
    aspect_deg: float | None
    surface_normal: Vector3
    source: str
    is_measured: bool


class TerrainProvider(ABC):
    """Interface for terrain data sources."""

    source: str = "unknown"
    #: ``True`` when this provider reads a real measured dataset.
    is_measured: bool = False
    #: ``True`` when :meth:`is_terrain_shadowed` performs a real computation.
    supports_shadows: bool = False

    @abstractmethod
    def get_elevation(self, latitude_deg: float, longitude_deg: float) -> float | None:
        """Elevation above the reference sphere [m], or ``None`` if unknown."""

    @abstractmethod
    def get_slope(self, latitude_deg: float, longitude_deg: float) -> float | None:
        """Terrain slope [deg], or ``None`` if unknown."""

    @abstractmethod
    def get_aspect(self, latitude_deg: float, longitude_deg: float) -> float | None:
        """Downhill azimuth [deg clockwise from North], or ``None`` if unknown."""

    @abstractmethod
    def get_surface_normal(self, latitude_deg: float, longitude_deg: float) -> Vector3:
        """Unit surface normal in the body-fixed frame."""

    @abstractmethod
    def is_terrain_shadowed(
        self,
        latitude_deg: float,
        longitude_deg: float,
        solar_azimuth_deg: float,
        solar_elevation_deg: float,
    ) -> bool | None:
        """Whether local topography blocks the Sun.

        Returns ``None`` when the provider cannot evaluate this (no DEM), which
        the API surfaces as "unknown" rather than a misleading ``false``.
        """

    def sample(self, latitude_deg: float, longitude_deg: float) -> TerrainSample:
        """Collect every terrain quantity for one coordinate in a single call."""
        return TerrainSample(
            elevation_m=self.get_elevation(latitude_deg, longitude_deg),
            slope_deg=self.get_slope(latitude_deg, longitude_deg),
            aspect_deg=self.get_aspect(latitude_deg, longitude_deg),
            surface_normal=self.get_surface_normal(latitude_deg, longitude_deg),
            source=self.source,
            is_measured=self.is_measured,
        )


# ---------------------------------------------------------------------------
# Flat / spherical provider (MVP default)
# ---------------------------------------------------------------------------


class FlatTerrainProvider(TerrainProvider):
    """Perfect-sphere Moon. Honest placeholder - measures nothing.

    Elevation and slope are reported as ``0.0`` with ``is_measured=False``,
    meaning "the spherical reference model says zero by construction", not "we
    measured zero". Shadow queries return ``None`` (unknown) because without a
    DEM there is no topography to ray-trace.
    """

    source = "flat-sphere"
    is_measured = False
    supports_shadows = False

    def get_elevation(self, latitude_deg: float, longitude_deg: float) -> float | None:
        # Zero by definition of the reference sphere, not a measurement.
        return 0.0

    def get_slope(self, latitude_deg: float, longitude_deg: float) -> float | None:
        return 0.0

    def get_aspect(self, latitude_deg: float, longitude_deg: float) -> float | None:
        # A flat surface has no downhill direction.
        return None

    def get_surface_normal(self, latitude_deg: float, longitude_deg: float) -> Vector3:
        return spherical_surface_normal(latitude_deg, longitude_deg)

    def is_terrain_shadowed(
        self,
        latitude_deg: float,
        longitude_deg: float,
        solar_azimuth_deg: float,
        solar_elevation_deg: float,
    ) -> bool | None:
        return None


# ---------------------------------------------------------------------------
# Raster DEM provider (Milestone 3 / 4)
# ---------------------------------------------------------------------------


class TerrainRasterError(RuntimeError):
    """Raised when a DEM is requested but cannot be opened or read."""


class RasterTerrainProvider(TerrainProvider):
    """DEM-backed terrain from a LOLA / SLDEM2015 GeoTIFF.

    Design notes:

    * The raster is opened once and kept open; ``rasterio`` reads are windowed
      so only the handful of pixels around the query are pulled from disk. A
      global 64 ppd SLDEM is ~90 GB - never call ``src.read()`` without a
      window.
    * Gradients are computed with central differences over a 3x3 window, with
      the east-west spacing scaled by ``cos(latitude)`` because a degree of
      longitude shrinks toward the poles.
    * Horizon shadowing marches outward along the solar azimuth and compares
      the maximum horizon angle against the solar elevation.
    """

    is_measured = True

    def __init__(
        self,
        raster_path: Path,
        *,
        source_label: str | None = None,
        enable_shadows: bool = False,
        horizon_max_distance_m: float = 20_000.0,
        horizon_samples: int = 64,
    ) -> None:
        self.raster_path = Path(raster_path)
        self.source = source_label or self.raster_path.stem
        self.supports_shadows = enable_shadows
        self.horizon_max_distance_m = horizon_max_distance_m
        self.horizon_samples = horizon_samples

        self._rasterio = _import_rasterio()
        if not self.raster_path.is_file():
            raise TerrainRasterError(f"DEM raster not found: {self.raster_path}")
        try:
            self._dataset = self._rasterio.open(self.raster_path)
        except Exception as error:  # noqa: BLE001 - re-raised as typed error
            raise TerrainRasterError(f"Could not open DEM {self.raster_path}: {error}") from error

        self._nodata = self._dataset.nodata
        logger.info(
            "Opened DEM %s (%dx%d, crs=%s)",
            self.raster_path.name,
            self._dataset.width,
            self._dataset.height,
            self._dataset.crs,
        )

    def close(self) -> None:
        """Close the underlying raster handle."""
        dataset = getattr(self, "_dataset", None)
        if dataset is not None and not dataset.closed:
            dataset.close()

    # -- Pixel addressing --------------------------------------------------

    def get_pixel_for_lat_lon(
        self, latitude_deg: float, longitude_deg: float
    ) -> tuple[int, int]:
        """Map lat/lon to ``(row, col)`` in the raster.

        Assumes a simple cylindrical (plate carree) lunar raster whose affine
        transform is expressed in degrees - the layout the USGS/PDS-distributed
        SLDEM2015 tiles use after standard preprocessing.
        """
        row, col = self._dataset.index(longitude_deg, latitude_deg)
        return int(row), int(col)

    def _read_window(self, row: int, col: int, half_size: int = 1) -> np.ndarray:
        """Read a ``(2h+1)`` square window centred on ``(row, col)``.

        Returns a float array with nodata replaced by ``np.nan``. Windows that
        run off the raster edge are clipped and padded with ``np.nan``.
        """
        from rasterio.windows import Window  # noqa: PLC0415 - lazy, optional dep

        size = 2 * half_size + 1
        row_start, col_start = row - half_size, col - half_size
        window = Window(col_start, row_start, size, size)
        data = self._dataset.read(
            1, window=window, boundless=True, fill_value=np.nan
        ).astype(np.float64)
        if self._nodata is not None:
            data[data == self._nodata] = np.nan
        return data

    # -- Elevation and gradients -------------------------------------------

    def get_elevation(self, latitude_deg: float, longitude_deg: float) -> float | None:
        try:
            row, col = self.get_pixel_for_lat_lon(latitude_deg, longitude_deg)
            value = float(self._read_window(row, col, half_size=0)[0, 0])
        except Exception as error:  # noqa: BLE001 - degrade to "unknown"
            logger.warning("DEM elevation read failed at %.4f/%.4f: %s", latitude_deg, longitude_deg, error)
            return None
        return None if math.isnan(value) else value

    def calculate_local_gradient(
        self, latitude_deg: float, longitude_deg: float
    ) -> tuple[float, float] | None:
        """Central-difference terrain gradient ``(d_h/d_east, d_h/d_north)``.

        Both components are dimensionless (metres of rise per metre of run).
        Returns ``None`` if the window contains nodata.
        """
        try:
            row, col = self.get_pixel_for_lat_lon(latitude_deg, longitude_deg)
            window = self._read_window(row, col, half_size=1)
        except Exception as error:  # noqa: BLE001
            logger.warning("DEM gradient read failed: %s", error)
            return None
        if np.isnan(window).any():
            return None

        # Ground spacing of one pixel, in metres, at this latitude.
        pixel_deg_lon = abs(self._dataset.transform.a)
        pixel_deg_lat = abs(self._dataset.transform.e)
        metres_per_deg = MOON_RADIUS_M * math.radians(1.0)
        spacing_east = pixel_deg_lon * metres_per_deg * max(
            math.cos(math.radians(latitude_deg)), 1e-6
        )
        spacing_north = pixel_deg_lat * metres_per_deg

        # window[row][col] with row 0 = north (raster rows increase southward).
        d_east = (window[1, 2] - window[1, 0]) / (2.0 * spacing_east)
        d_north = (window[0, 1] - window[2, 1]) / (2.0 * spacing_north)
        return float(d_east), float(d_north)

    def calculate_slope(self, latitude_deg: float, longitude_deg: float) -> float | None:
        """Slope magnitude [deg] from the local gradient.

        .. math:: \\text{slope} = \\arctan\\sqrt{(\\partial h/\\partial e)^2
                                                 + (\\partial h/\\partial n)^2}
        """
        gradient = self.calculate_local_gradient(latitude_deg, longitude_deg)
        if gradient is None:
            return None
        d_east, d_north = gradient
        return math.degrees(math.atan(math.hypot(d_east, d_north)))

    def calculate_aspect(self, latitude_deg: float, longitude_deg: float) -> float | None:
        """Downhill azimuth [deg clockwise from North].

        The gradient points uphill, so the downhill direction is its negation;
        azimuth is then ``atan2(east, north)`` of that downhill vector.
        """
        gradient = self.calculate_local_gradient(latitude_deg, longitude_deg)
        if gradient is None:
            return None
        d_east, d_north = gradient
        if abs(d_east) < 1e-12 and abs(d_north) < 1e-12:
            return None
        return math.degrees(math.atan2(-d_east, -d_north)) % 360.0

    def get_slope(self, latitude_deg: float, longitude_deg: float) -> float | None:
        return self.calculate_slope(latitude_deg, longitude_deg)

    def get_aspect(self, latitude_deg: float, longitude_deg: float) -> float | None:
        return self.calculate_aspect(latitude_deg, longitude_deg)

    def get_surface_normal(self, latitude_deg: float, longitude_deg: float) -> Vector3:
        """Terrain-tilted unit normal in the body-fixed frame.

        In the local ENU frame an unrotated normal is ``(0, 0, 1)``; a terrain
        with gradient ``(g_e, g_n)`` tilts it to ``(-g_e, -g_n, 1)`` normalised.
        That local vector is then expressed in the body frame via the ENU basis.
        Falls back to the spherical normal when the gradient is unavailable.
        """
        gradient = self.calculate_local_gradient(latitude_deg, longitude_deg)
        if gradient is None:
            return spherical_surface_normal(latitude_deg, longitude_deg)

        d_east, d_north = gradient
        east, north, up = local_enu_basis(latitude_deg, longitude_deg)
        tilted = (-d_east) * east + (-d_north) * north + up
        return normalize(tilted)

    # -- Horizon shadowing (Milestone 4) -----------------------------------

    def is_terrain_shadowed(
        self,
        latitude_deg: float,
        longitude_deg: float,
        solar_azimuth_deg: float,
        solar_elevation_deg: float,
    ) -> bool | None:
        """Horizon ray-march along the solar azimuth.

        Algorithm:

        1. Step outward from the point along ``solar_azimuth_deg`` at
           ``horizon_samples`` distances up to ``horizon_max_distance_m``.
        2. At each sample of distance ``d`` and elevation ``h_i`` compute
           ``terrain_angle = atan((h_i - h_0) / d)``.
        3. The local horizon angle is the maximum of those.
        4. The point is shadowed when
           ``solar_elevation <= terrain_horizon_angle``.

        Lunar surface curvature is accounted for by subtracting the drop-off
        ``d^2 / (2 R)`` from each neighbour's height.

        Returns ``None`` when shadows are disabled or the DEM cannot be read.
        """
        if not self.supports_shadows:
            return None
        if solar_elevation_deg <= 0.0:
            # Already below the geometric horizon; terrain is irrelevant.
            return False

        base_elevation = self.get_elevation(latitude_deg, longitude_deg)
        if base_elevation is None:
            return None

        azimuth = math.radians(solar_azimuth_deg)
        # Offsets per metre of ground distance, in degrees of lat/lon.
        metres_per_deg = MOON_RADIUS_M * math.radians(1.0)
        cos_latitude = max(math.cos(math.radians(latitude_deg)), 1e-6)

        max_horizon_angle = -90.0
        for index in range(1, self.horizon_samples + 1):
            distance_m = self.horizon_max_distance_m * index / self.horizon_samples
            north_offset_m = distance_m * math.cos(azimuth)
            east_offset_m = distance_m * math.sin(azimuth)

            sample_latitude = latitude_deg + north_offset_m / metres_per_deg
            sample_longitude = longitude_deg + east_offset_m / (metres_per_deg * cos_latitude)
            if not -90.0 <= sample_latitude <= 90.0:
                break

            sample_elevation = self.get_elevation(sample_latitude, sample_longitude)
            if sample_elevation is None:
                continue

            # Curvature correction: the surface falls away over distance d.
            curvature_drop = distance_m * distance_m / (2.0 * MOON_RADIUS_M)
            rise = sample_elevation - base_elevation - curvature_drop
            terrain_angle = math.degrees(math.atan2(rise, distance_m))
            max_horizon_angle = max(max_horizon_angle, terrain_angle)

        if max_horizon_angle <= -90.0:
            return None
        return solar_elevation_deg <= max_horizon_angle


class PolarRasterTerrainProvider(RasterTerrainProvider):
    """Terrain from a projected (e.g. polar stereographic) DEM such as LOLA LDEM_80S_80M.

    Coordinates are converted with GDAL/PROJ using the raster's own CRS, so no
    projection formulas are hard-coded. The grid is held in memory (about
    115 MB for LDEM_80S_80M) so horizon scans out to 150 km stay fast; polar
    shadows are cast by ridges tens of kilometres away.
    """

    def __init__(self, raster_path: Path, *, source_label: str | None = None, enable_shadows: bool = False,
                 horizon_max_distance_m: float = 150_000.0, horizon_samples: int = 300) -> None:
        super().__init__(raster_path, source_label=source_label, enable_shadows=enable_shadows,
                         horizon_max_distance_m=horizon_max_distance_m, horizon_samples=horizon_samples)
        from rasterio.crs import CRS  # noqa: PLC0415 - optional dependency

        ds = self._dataset
        self._crs = ds.crs
        radius = MOON_RADIUS_M
        self._geo_crs = CRS.from_proj4(f"+proj=longlat +R={radius:.0f} +no_defs")
        try:
            data = ds.read(1).astype(np.float32)
        except Exception as error:  # noqa: BLE001 - truncated or corrupt file
            raise TerrainRasterError(f"Could not read DEM {self.raster_path}: {error}") from error
        if self._nodata is not None:
            data[data == self._nodata] = np.nan
        scale = (ds.scales or (1.0,))[0] or 1.0
        offset = (ds.offsets or (0.0,))[0] or 0.0
        data = data * scale + offset
        units = ((ds.units or ("",))[0] or "").lower()
        # LOLA float LDEMs are heights in km above the 1737.4 km sphere.
        if units.startswith("k") or np.nanmax(np.abs(data)) < 100:
            data = data * 1000.0
        # Some LOLA products store planetary radius rather than height above the sphere.
        if np.nanmedian(data) > 1.0e6:
            data = data - MOON_RADIUS_M
        self._grid = data
        self._inverse = ~ds.transform
        self._pixel_m = abs(ds.transform.a)
        self.source = source_label or f"LOLA {self.raster_path.stem.upper()} ({self._pixel_m:.0f} m/px, polar)"

    # -- projection helpers --------------------------------------------------

    def _project(self, lats, lons):
        from rasterio.warp import transform  # noqa: PLC0415

        xs, ys = transform(self._geo_crs, self._crs, list(lons), list(lats))
        return np.asarray(xs), np.asarray(ys)

    def _value_at_xy(self, x: float, y: float) -> float | None:
        col_f, row_f = self._inverse * (x, y)
        col, row = col_f - 0.5, row_f - 0.5
        c0, r0 = int(math.floor(col)), int(math.floor(row))
        if r0 < 0 or c0 < 0 or r0 + 1 >= self._grid.shape[0] or c0 + 1 >= self._grid.shape[1]:
            return None
        fc, fr = col - c0, row - r0
        block = self._grid[r0:r0 + 2, c0:c0 + 2]
        if np.isnan(block).any():
            return None
        top = block[0, 0] * (1 - fc) + block[0, 1] * fc
        bottom = block[1, 0] * (1 - fc) + block[1, 1] * fc
        return float(top * (1 - fr) + bottom * fr)

    def _local_axes(self, lat: float, lon: float):
        """Unit vectors of local east and north expressed in projected x/y."""
        d = 1e-4
        xs, ys = self._project([lat, lat, min(lat + d, 89.99999)], [lon, lon + d, lon])
        east = np.array([xs[1] - xs[0], ys[1] - ys[0]])
        north = np.array([xs[2] - xs[0], ys[2] - ys[0]])
        if abs(lat) > 89.999:  # at the pole the meridian defines "north"
            north = np.array([math.sin(math.radians(lon)), math.cos(math.radians(lon))])
            east = np.array([north[1], -north[0]]) if lat < 0 else np.array([-north[1], north[0]])
        return east / np.linalg.norm(east), north / np.linalg.norm(north)

    # -- TerrainProvider ---------------------------------------------------------

    def get_pixel_for_lat_lon(self, latitude_deg: float, longitude_deg: float) -> tuple[int, int]:
        xs, ys = self._project([latitude_deg], [longitude_deg])
        row, col = self._dataset.index(float(xs[0]), float(ys[0]))
        return int(row), int(col)

    def get_elevation(self, latitude_deg: float, longitude_deg: float) -> float | None:
        xs, ys = self._project([latitude_deg], [longitude_deg])
        return self._value_at_xy(float(xs[0]), float(ys[0]))

    def calculate_local_gradient(self, latitude_deg: float, longitude_deg: float) -> tuple[float, float] | None:
        xs, ys = self._project([latitude_deg], [longitude_deg])
        x, y, h = float(xs[0]), float(ys[0]), self._pixel_m
        east, north = self._local_axes(latitude_deg, longitude_deg)
        samples = []
        for axis in (east, north):
            plus = self._value_at_xy(x + axis[0] * h, y + axis[1] * h)
            minus = self._value_at_xy(x - axis[0] * h, y - axis[1] * h)
            if plus is None or minus is None:
                return None
            samples.append((plus - minus) / (2 * h))
        return float(samples[0]), float(samples[1])

    def horizon_profile(self, latitude_deg: float, longitude_deg: float, azimuths_deg, observer_height_m: float = 2.0):
        """Terrain horizon elevation angle (deg) for each azimuth, curvature-corrected (vectorised)."""
        xs, ys = self._project([latitude_deg], [longitude_deg])
        x0, y0 = float(xs[0]), float(ys[0])
        base = self._value_at_xy(x0, y0)
        if base is None:
            return None
        base += observer_height_m
        east, north = self._local_axes(latitude_deg, longitude_deg)
        distances = np.geomspace(self._pixel_m, self.horizon_max_distance_m, self.horizon_samples)
        az = np.radians(np.asarray(list(azimuths_deg), dtype=float))
        dx = east[0] * np.sin(az) + north[0] * np.cos(az)
        dy = east[1] * np.sin(az) + north[1] * np.cos(az)
        px = x0 + dx[:, None] * distances[None, :]
        py = y0 + dy[:, None] * distances[None, :]
        inv = self._inverse
        col = inv.a * px + inv.b * py + inv.c - 0.5
        row = inv.d * px + inv.e * py + inv.f - 0.5
        c0 = np.floor(col).astype(int); r0 = np.floor(row).astype(int)
        h, w = self._grid.shape
        valid = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < w) & (r0 + 1 < h)
        c0c = np.clip(c0, 0, w - 2); r0c = np.clip(r0, 0, h - 2)
        fc = col - c0; fr = row - r0
        g = self._grid
        top = g[r0c, c0c] * (1 - fc) + g[r0c, c0c + 1] * fc
        bottom = g[r0c + 1, c0c] * (1 - fc) + g[r0c + 1, c0c + 1] * fc
        height = top * (1 - fr) + bottom * fr
        angle = np.degrees(np.arctan2(height - base - distances[None, :] ** 2 / (2 * MOON_RADIUS_M), distances[None, :]))
        angle = np.where(valid & ~np.isnan(height), angle, -np.inf)
        best = angle.max(axis=1)
        return [float(v) if np.isfinite(v) else None for v in best]

    def is_terrain_shadowed(self, latitude_deg: float, longitude_deg: float, solar_azimuth_deg: float, solar_elevation_deg: float) -> bool | None:
        if not self.supports_shadows:
            return None
        # The Sun's disk is 0.53° wide; treat its centre as the reference.
        if solar_elevation_deg <= -0.27:
            return True
        horizon = self.horizon_profile(latitude_deg, longitude_deg, [solar_azimuth_deg], observer_height_m=0.0)
        if not horizon or horizon[0] is None:
            return None
        return solar_elevation_deg <= horizon[0]


def _import_rasterio():  # noqa: ANN202 - third-party module object
    """Import ``rasterio`` lazily so the app runs without GDAL installed."""
    try:
        import rasterio  # noqa: PLC0415 - intentional lazy import
    except ImportError as error:  # pragma: no cover - depends on environment
        raise TerrainRasterError(
            "rasterio is not installed. Run `pip install rasterio` or set "
            "USE_TERRAIN_DEM=false."
        ) from error
    return rasterio


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_terrain_provider(
    *, use_dem: bool, raster_path: Path, enable_shadows: bool = False
) -> TerrainProvider:
    """Select the best available terrain provider, falling back to the sphere."""
    if not use_dem:
        return FlatTerrainProvider()

    try:
        provider = RasterTerrainProvider(raster_path, enable_shadows=enable_shadows)
        if provider._dataset.crs is not None and not provider._dataset.crs.is_geographic:
            provider.close()
            provider = PolarRasterTerrainProvider(raster_path, enable_shadows=enable_shadows)
    except TerrainRasterError as error:
        logger.warning(
            "USE_TERRAIN_DEM=true but the DEM is unavailable (%s). "
            "Falling back to the flat spherical model.",
            error,
        )
        return FlatTerrainProvider()

    logger.info("Using raster terrain provider: %s", provider.source)
    return provider


__all__ = [
    "TerrainProvider",
    "TerrainSample",
    "FlatTerrainProvider",
    "RasterTerrainProvider",
    "PolarRasterTerrainProvider",
    "TerrainRasterError",
    "create_terrain_provider",
]
