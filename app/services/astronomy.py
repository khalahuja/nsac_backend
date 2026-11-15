"""Sun / Earth / Moon geometry.

This module answers three questions for a given UTC instant:

1. Where is the Sun, as seen from the Moon's centre, in the lunar body-fixed
   frame?
2. Where is the Earth, likewise?
3. How is the Moon oriented with respect to the J2000 inertial frame?

Two interchangeable implementations are provided:

``SimpleAstronomyProvider``
    Self-contained analytic ephemeris. No external data files. Accurate to
    roughly a few arc-minutes in direction - good enough to tell day from
    night and to get solar elevation to within a few tenths of a degree, but
    **not** a NASA-precision result. Every value it returns is tagged
    ``simple-analytic-ephemeris``.

``SpiceAstronomyProvider``
    Uses JPL/NAIF SPICE kernels via ``spiceypy`` (Milestone 2). Selected
    automatically when ``USE_SPICE=true`` *and* a usable kernel set is present
    in ``data/spice``. Results are tagged ``jpl-spice``.

Nothing else in the codebase may assume which provider is active; consumers
read :attr:`AstronomyResult.source` for provenance.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
from threading import RLock

import numpy as np
from numpy.typing import NDArray

from ..constants import ASTRONOMICAL_UNIT_KM
from .geometry import Vector3, normalize, rotation_matrix_x, rotation_matrix_z

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Time scales
# ---------------------------------------------------------------------------

#: Julian Date of the J2000.0 epoch (2000-01-01T12:00:00 TT).
J2000_JD: float = 2451545.0

#: Julian Date of the Unix epoch (1970-01-01T00:00:00 UTC).
UNIX_EPOCH_JD: float = 2440587.5

SECONDS_PER_DAY: float = 86400.0
JULIAN_CENTURY_DAYS: float = 36525.0


def datetime_to_julian_date(moment: datetime) -> float:
    """Convert a timezone-aware UTC ``datetime`` to a Julian Date.

    Args:
        moment: Instant to convert. Naive datetimes are assumed to be UTC.

    Returns:
        Julian Date (UTC-based).
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    unix_seconds = moment.astimezone(timezone.utc).timestamp()
    return UNIX_EPOCH_JD + unix_seconds / SECONDS_PER_DAY


def days_since_j2000(moment: datetime) -> float:
    """Days elapsed since J2000.0.

    We deliberately ignore the UTC -> TT offset (~69 s in 2026). For the Moon,
    the fastest-moving body here, that is ~0.011 deg of orbital motion - well
    below this model's intrinsic error. ``SpiceAstronomyProvider`` handles the
    conversion rigorously via ``str2et``.
    """
    return datetime_to_julian_date(moment) - J2000_JD


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AstronomyResult:
    """Sun/Earth geometry relative to the Moon at one instant.

    Attributes:
        sun_direction_body: Unit vector to the Sun in the lunar body-fixed
            (IAU_MOON) frame.
        sun_distance_km: Moon-centre to Sun-centre distance.
        earth_direction_body: Unit vector to Earth in the body-fixed frame.
        earth_distance_km: Moon-centre to Earth-centre distance.
        sun_direction_j2000: Unit vector to the Sun in the J2000 inertial
            frame. Used by the frontend to orient the display light.
        subsolar_latitude_deg: Latitude of the sub-solar point.
        subsolar_longitude_deg: Longitude of the sub-solar point.
        source: Provenance tag, e.g. ``"simple-analytic-ephemeris"``.
        accuracy_note: Human-readable statement of the model's limitations.
    """

    sun_direction_body: Vector3
    sun_distance_km: float
    earth_direction_body: Vector3
    earth_distance_km: float
    sun_direction_j2000: Vector3
    subsolar_latitude_deg: float
    subsolar_longitude_deg: float
    source: str
    accuracy_note: str


@dataclass(frozen=True)
class MoonOrientation:
    """Rotation from the J2000 frame into the lunar body-fixed frame.

    Attributes:
        right_ascension_deg: Right ascension alpha0 of the lunar north pole.
        declination_deg: Declination delta0 of the lunar north pole.
        prime_meridian_deg: Rotation angle W of the prime meridian.
        j2000_to_body: 3x3 matrix mapping a J2000 vector to the body frame.
        source: Provenance tag.
    """

    right_ascension_deg: float
    declination_deg: float
    prime_meridian_deg: float
    j2000_to_body: NDArray[np.float64]
    source: str


# ---------------------------------------------------------------------------
# Abstraction
# ---------------------------------------------------------------------------


class AstronomyProvider(ABC):
    """Interface every ephemeris backend must implement."""

    #: Short provenance tag surfaced in API responses.
    source: str = "unknown"
    #: One-line statement of accuracy, surfaced next to results.
    accuracy_note: str = ""

    @abstractmethod
    def get_sun_vector_from_moon(self, moment: datetime) -> tuple[Vector3, float]:
        """Return ``(unit_vector_to_sun_body_frame, distance_km)``."""

    @abstractmethod
    def get_earth_vector_from_moon(self, moment: datetime) -> tuple[Vector3, float]:
        """Return ``(unit_vector_to_earth_body_frame, distance_km)``."""

    @abstractmethod
    def get_moon_orientation(self, moment: datetime) -> MoonOrientation:
        """Return the J2000 -> body-fixed rotation for the Moon."""

    @abstractmethod
    def _sun_vector_j2000(self, moment: datetime) -> tuple[Vector3, float]:
        """Return ``(unit_vector_to_sun_in_J2000, distance_km)``."""

    def get_geometry(self, moment: datetime) -> AstronomyResult:
        """Assemble a full :class:`AstronomyResult` for ``moment``."""
        sun_body, sun_distance = self.get_sun_vector_from_moon(moment)
        earth_body, earth_distance = self.get_earth_vector_from_moon(moment)
        sun_j2000, _ = self._sun_vector_j2000(moment)

        # The sub-solar point is simply where the Sun direction pierces the
        # surface, i.e. the lat/lon of the body-frame Sun unit vector.
        subsolar_latitude = math.degrees(
            math.asin(float(np.clip(sun_body[2], -1.0, 1.0)))
        )
        subsolar_longitude = math.degrees(math.atan2(float(sun_body[1]), float(sun_body[0])))

        return AstronomyResult(
            sun_direction_body=sun_body,
            sun_distance_km=sun_distance,
            earth_direction_body=earth_body,
            earth_distance_km=earth_distance,
            sun_direction_j2000=sun_j2000,
            subsolar_latitude_deg=subsolar_latitude,
            subsolar_longitude_deg=subsolar_longitude,
            source=self.source,
            accuracy_note=self.accuracy_note,
        )


# ---------------------------------------------------------------------------
# Simple analytic provider
# ---------------------------------------------------------------------------


class SimpleAstronomyProvider(AstronomyProvider):
    """Analytic Sun/Moon ephemeris requiring no external data.

    Models used (all classical, all cited in-line):

    * **Sun** - low-precision solar position from the *Astronomical Almanac*
      (the "Approximate Solar Coordinates" page). Quoted accuracy ~0.01 deg in
      ecliptic longitude for 1950-2050.
    * **Moon** - abbreviated ELP-2000/82 series as published in Meeus,
      *Astronomical Algorithms* ch. 47 (six longitude terms, four latitude
      terms, four distance terms). Typical error ~0.1-0.3 deg in longitude and
      a few hundred km in distance.
    * **Lunar orientation** - the IAU/IAG Working Group rotational elements for
      the Moon (Archinal et al. 2018), including all thirteen E1..E13 libration
      terms. This is the same analytic model that underpins the ``IAU_MOON``
      frame in NAIF's ``pck00010.tpc``, so the body-frame orientation is the
      *most* trustworthy part of this provider.

    The dominant error is therefore the Moon's orbital position, which shifts
    the sub-solar longitude by up to a few tenths of a degree. Near the
    terminator that can move a sunrise/sunset time by a few tens of minutes.
    Treat all outputs as **approximate**.
    """

    source = "simple-analytic-ephemeris"
    accuracy_note = (
        "Analytic Sun (Astronomical Almanac low-precision) + abbreviated "
        "ELP-2000/82 Moon series + IAU 2018 lunar rotation elements. "
        "Direction error of order 0.1-0.3 deg; not a JPL-precision ephemeris."
    )

    # -- Sun ---------------------------------------------------------------

    def _sun_vector_j2000(self, moment: datetime) -> tuple[Vector3, float]:
        """Geocentric Sun direction and distance in the J2000 equatorial frame.

        Astronomical Almanac low-precision formulae, with ``d`` in days from
        J2000.0::

            L = 280.460 + 0.9856474 d          (mean longitude, deg)
            g = 357.528 + 0.9856003 d          (mean anomaly, deg)
            lambda = L + 1.915 sin g + 0.020 sin 2g   (ecliptic longitude)
            R = 1.00014 - 0.01671 cos g - 0.00014 cos 2g   (AU)
            epsilon = 23.439 - 0.0000004 d     (obliquity of the ecliptic)
        """
        day = days_since_j2000(moment)

        mean_longitude = math.radians((280.460 + 0.9856474 * day) % 360.0)
        mean_anomaly = math.radians((357.528 + 0.9856003 * day) % 360.0)

        ecliptic_longitude = (
            mean_longitude
            + math.radians(1.915) * math.sin(mean_anomaly)
            + math.radians(0.020) * math.sin(2.0 * mean_anomaly)
        )
        distance_au = (
            1.00014
            - 0.01671 * math.cos(mean_anomaly)
            - 0.00014 * math.cos(2.0 * mean_anomaly)
        )
        obliquity = math.radians(23.439 - 0.0000004 * day)

        # Ecliptic (beta ~ 0) -> equatorial rectangular coordinates.
        direction = np.array(
            [
                math.cos(ecliptic_longitude),
                math.cos(obliquity) * math.sin(ecliptic_longitude),
                math.sin(obliquity) * math.sin(ecliptic_longitude),
            ],
            dtype=np.float64,
        )
        return normalize(direction), distance_au * ASTRONOMICAL_UNIT_KM

    # -- Moon --------------------------------------------------------------

    def _moon_position_j2000_km(self, moment: datetime) -> Vector3:
        """Geocentric Moon position vector in the J2000 equatorial frame [km].

        Abbreviated ELP-2000/82 (Meeus ch. 47). Arguments, ``d`` days from
        J2000.0, all in degrees::

            L' = 218.316 + 13.176396 d     mean longitude
            M  = 357.529 +  0.98560028 d   Sun's mean anomaly
            M' = 134.963 + 13.064993 d     Moon's mean anomaly
            D  = 297.850 + 12.190749 d     mean elongation
            F  =  93.272 + 13.229350 d     argument of latitude
        """
        day = days_since_j2000(moment)

        mean_longitude = math.radians((218.316 + 13.176396 * day) % 360.0)
        sun_anomaly = math.radians((357.529 + 0.98560028 * day) % 360.0)
        moon_anomaly = math.radians((134.963 + 13.064993 * day) % 360.0)
        elongation = math.radians((297.850 + 12.190749 * day) % 360.0)
        latitude_argument = math.radians((93.272 + 13.229350 * day) % 360.0)

        # Ecliptic longitude [rad] - six dominant periodic terms.
        ecliptic_longitude = mean_longitude + math.radians(
            6.289 * math.sin(moon_anomaly)
            + 1.274 * math.sin(2.0 * elongation - moon_anomaly)
            + 0.658 * math.sin(2.0 * elongation)
            + 0.214 * math.sin(2.0 * moon_anomaly)
            - 0.186 * math.sin(sun_anomaly)
            - 0.114 * math.sin(2.0 * latitude_argument)
        )

        # Ecliptic latitude [rad] - four dominant terms.
        ecliptic_latitude = math.radians(
            5.128 * math.sin(latitude_argument)
            + 0.280 * math.sin(moon_anomaly + latitude_argument)
            - 0.277 * math.sin(latitude_argument - moon_anomaly)
            - 0.173 * math.sin(2.0 * elongation - latitude_argument)
        )

        # Earth-Moon distance [km] - four dominant terms.
        distance_km = (
            385_000.56
            - 20_905.355 * math.cos(moon_anomaly)
            - 3_699.111 * math.cos(2.0 * elongation - moon_anomaly)
            - 2_955.968 * math.cos(2.0 * elongation)
            - 569.925 * math.cos(2.0 * moon_anomaly)
        )

        cos_latitude = math.cos(ecliptic_latitude)
        ecliptic_vector = np.array(
            [
                distance_km * cos_latitude * math.cos(ecliptic_longitude),
                distance_km * cos_latitude * math.sin(ecliptic_longitude),
                distance_km * math.sin(ecliptic_latitude),
            ],
            dtype=np.float64,
        )

        # Rotate ecliptic -> equatorial about the +X (vernal equinox) axis.
        obliquity = math.radians(23.439 - 0.0000004 * day)
        return rotation_matrix_x(obliquity) @ ecliptic_vector

    # -- Orientation -------------------------------------------------------

    def get_moon_orientation(self, moment: datetime) -> MoonOrientation:
        """IAU 2018 rotational elements for the Moon.

        Implements Archinal et al. (2018), *Report of the IAU Working Group on
        Cartographic Coordinates and Rotational Elements: 2015*, Table 2 - the
        full thirteen-term libration expansion. ``d`` is days and ``T``
        Julian centuries from J2000.0.

        The body-fixed axes follow the standard construction::

            Z_body = (cos d0 cos a0, cos d0 sin a0, sin d0)
            Q      = (-sin a0, cos a0, 0)          node of body equator
            X_body = Q cos W + (Z_body x Q) sin W  prime meridian direction
            Y_body = Z_body x X_body
        """
        day = days_since_j2000(moment)
        century = day / JULIAN_CENTURY_DAYS

        # Libration arguments E1..E13 [rad].
        e_args = [
            math.radians(125.045 - 0.0529921 * day),
            math.radians(250.089 - 0.1059842 * day),
            math.radians(260.008 + 13.0120009 * day),
            math.radians(176.625 + 13.3407154 * day),
            math.radians(357.529 + 0.9856003 * day),
            math.radians(311.589 + 26.4057084 * day),
            math.radians(134.963 + 13.0649930 * day),
            math.radians(276.617 + 0.3287146 * day),
            math.radians(34.226 + 1.7484877 * day),
            math.radians(15.134 - 0.1589763 * day),
            math.radians(119.743 + 0.0036096 * day),
            math.radians(239.961 + 0.1643573 * day),
            math.radians(25.053 + 12.9590088 * day),
        ]
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13 = e_args

        right_ascension = (
            269.9949
            + 0.0031 * century
            - 3.8787 * math.sin(e1)
            - 0.1204 * math.sin(e2)
            + 0.0700 * math.sin(e3)
            - 0.0172 * math.sin(e4)
            + 0.0072 * math.sin(e6)
            - 0.0052 * math.sin(e10)
            + 0.0043 * math.sin(e13)
        )
        declination = (
            66.5392
            + 0.0130 * century
            + 1.5419 * math.cos(e1)
            + 0.0239 * math.cos(e2)
            - 0.0278 * math.cos(e3)
            + 0.0068 * math.cos(e4)
            - 0.0029 * math.cos(e6)
            + 0.0009 * math.cos(e7)
            + 0.0008 * math.cos(e10)
            - 0.0009 * math.cos(e13)
        )
        prime_meridian = (
            38.3213
            + 13.17635815 * day
            - 1.4e-12 * day * day
            + 3.5610 * math.sin(e1)
            + 0.1208 * math.sin(e2)
            - 0.0642 * math.sin(e3)
            + 0.0158 * math.sin(e4)
            + 0.0252 * math.sin(e5)
            - 0.0066 * math.sin(e6)
            - 0.0047 * math.sin(e7)
            - 0.0046 * math.sin(e8)
            + 0.0028 * math.sin(e9)
            + 0.0052 * math.sin(e10)
            + 0.0040 * math.sin(e11)
            + 0.0019 * math.sin(e12)
            - 0.0044 * math.sin(e13)
        )

        matrix = _body_frame_matrix(right_ascension, declination, prime_meridian)
        return MoonOrientation(
            right_ascension_deg=right_ascension % 360.0,
            declination_deg=declination,
            prime_meridian_deg=prime_meridian % 360.0,
            j2000_to_body=matrix,
            source="iau-2018-rotational-elements",
        )

    # -- Public vectors ----------------------------------------------------

    def get_sun_vector_from_moon(self, moment: datetime) -> tuple[Vector3, float]:
        """Unit vector to the Sun in the lunar body-fixed frame, plus distance.

        The Sun's position is geocentric, so we subtract the Moon's geocentric
        position to get the selenocentric vector. This matters: the Earth-Moon
        separation subtends up to ~0.15 deg at the Sun.
        """
        sun_direction_j2000, sun_distance_km = self._sun_vector_j2000(moment)
        sun_from_earth = sun_direction_j2000 * sun_distance_km
        moon_from_earth = self._moon_position_j2000_km(moment)
        sun_from_moon = sun_from_earth - moon_from_earth

        distance = float(np.linalg.norm(sun_from_moon))
        orientation = self.get_moon_orientation(moment)
        body_vector = orientation.j2000_to_body @ normalize(sun_from_moon)
        return normalize(body_vector), distance

    def get_earth_vector_from_moon(self, moment: datetime) -> tuple[Vector3, float]:
        """Unit vector to Earth in the lunar body-fixed frame, plus distance."""
        moon_from_earth = self._moon_position_j2000_km(moment)
        earth_from_moon = -moon_from_earth
        distance = float(np.linalg.norm(earth_from_moon))
        orientation = self.get_moon_orientation(moment)
        body_vector = orientation.j2000_to_body @ normalize(earth_from_moon)
        return normalize(body_vector), distance


def _body_frame_matrix(
    right_ascension_deg: float, declination_deg: float, prime_meridian_deg: float
) -> NDArray[np.float64]:
    """Build the J2000 -> body-fixed rotation from IAU rotational elements.

    The body-to-inertial rotation is ``Rz(a0 + 90) Rx(90 - d0) Rz(W)``; we
    return its transpose, which is also its inverse since it is orthonormal.
    """
    alpha = math.radians(right_ascension_deg)
    delta = math.radians(declination_deg)
    spin = math.radians(prime_meridian_deg)

    body_to_j2000 = (
        rotation_matrix_z(alpha + math.pi / 2.0)
        @ rotation_matrix_x(math.pi / 2.0 - delta)
        @ rotation_matrix_z(spin)
    )
    return body_to_j2000.T


# ---------------------------------------------------------------------------
# SPICE provider (Milestone 2)
# ---------------------------------------------------------------------------


class SpiceKernelError(RuntimeError):
    """Raised when SPICE is requested but cannot be initialised."""


# CSPICE kernel pools and error/traceback state are process-global. All provider
# instances must share this lock, including during loading and error handling.
_SPICE_LOCK = RLock()


def _serialized_spice(method):
    @wraps(method)
    def locked(*args, **kwargs):
        with _SPICE_LOCK:
            return method(*args, **kwargs)

    return locked


class SpiceAstronomyProvider(AstronomyProvider):
    """JPL/NAIF SPICE-backed ephemeris.

    Requires ``spiceypy`` and a furnished kernel set in ``data/spice``. See
    ``backend/app/datasets/README.md`` for the exact kernels to download - we
    never fetch them automatically because they are large and their licence
    terms should be read by a human first.

    States are computed with ``LT+S`` aberration correction (light time plus
    stellar aberration), which is the correct choice for "where do I see the
    Sun from this surface point".
    """

    source = "jpl-spice"
    accuracy_note = (
        "JPL/NAIF SPICE ephemeris (DE440) and MOON_ME/IAU_MOON body frame with LT+S aberration "
        "correction. Accuracy limited by the furnished kernel set."
    )

    def __init__(self, kernel_directory: Path) -> None:
        self.kernel_directory = Path(kernel_directory)
        self._spice = _import_spiceypy()
        self._loaded_kernels: list[str] = []
        self._load_kernels()

    # -- Kernel management -------------------------------------------------

    @_serialized_spice
    def _load_kernels(self) -> None:
        """Furnish every kernel found in the configured directory.

        Ordering matters to SPICE only for conflicting data; sorting by name
        gives deterministic, reproducible behaviour.
        """
        if not self.kernel_directory.is_dir():
            raise SpiceKernelError(
                f"SPICE kernel directory does not exist: {self.kernel_directory}"
            )

        kernel_suffixes = {".tls", ".bsp", ".tpc", ".bpc", ".tf", ".tm", ".ti", ".tsc"}
        kernels = sorted(
            path
            for path in self.kernel_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in kernel_suffixes
        )
        if not kernels:
            raise SpiceKernelError(
                f"No SPICE kernels found in {self.kernel_directory}. "
                "See backend/app/datasets/README.md for the required files."
            )

        for kernel in kernels:
            self._spice.furnsh(str(kernel))
            self._loaded_kernels.append(kernel.name)

        logger.info("Furnished %d SPICE kernel(s): %s", len(kernels), self._loaded_kernels)
        self._self_test()

    def _self_test(self) -> None:
        """Verify the kernel set actually answers the queries we need.

        A directory can contain kernels that are individually valid but
        collectively insufficient (e.g. a leapseconds kernel with no ephemeris).
        Failing here lets the factory fall back cleanly instead of raising on
        the first user request.
        """
        probe = datetime(2026, 1, 1, tzinfo=timezone.utc)
        try:
            self.get_sun_vector_from_moon(probe)
            self.get_earth_vector_from_moon(probe)
            self.get_moon_orientation(probe)
        except Exception as error:  # noqa: BLE001 - re-raised as a typed error
            raise SpiceKernelError(
                f"SPICE kernels loaded but incomplete for Sun/Earth/Moon geometry: {error}"
            ) from error

    @property
    def loaded_kernels(self) -> list[str]:
        """Names of the kernels currently furnished."""
        return list(self._loaded_kernels)

    @_serialized_spice
    def unload(self) -> None:
        """Release all furnished kernels."""
        if self._loaded_kernels:
            self._spice.kclear()
            self._loaded_kernels.clear()

    # -- Ephemeris ---------------------------------------------------------

    @_serialized_spice
    def _ephemeris_time(self, moment: datetime) -> float:
        """Convert UTC to SPICE ephemeris time (TDB seconds past J2000)."""
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        utc_string = moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        return float(self._spice.str2et(utc_string))

    @_serialized_spice
    def _body_frame(self) -> str:
        """MOON_ME (the frame LOLA terrain uses) when its kernels are loaded, else IAU_MOON."""
        frame = getattr(self, "_frame_cache", None)
        if frame is None:
            try:
                self._spice.pxform("J2000", "MOON_ME", self._spice.str2et("2025-01-01T00:00:00"))
                frame = "MOON_ME"
            except Exception:  # noqa: BLE001 - frame kernels not installed
                frame = "IAU_MOON"
            self._frame_cache = frame
        return frame

    @_serialized_spice
    def _target_vector(self, target: str, moment: datetime) -> tuple[Vector3, float]:
        ephemeris_time = self._ephemeris_time(moment)
        state, _light_time = self._spice.spkpos(
            target, ephemeris_time, self._body_frame(), "LT+S", "MOON"
        )
        vector = np.asarray(state, dtype=np.float64)
        return normalize(vector), float(np.linalg.norm(vector))

    def get_sun_vector_from_moon(self, moment: datetime) -> tuple[Vector3, float]:
        return self._target_vector("SUN", moment)

    def get_earth_vector_from_moon(self, moment: datetime) -> tuple[Vector3, float]:
        return self._target_vector("EARTH", moment)

    @_serialized_spice
    def _sun_vector_j2000(self, moment: datetime) -> tuple[Vector3, float]:
        ephemeris_time = self._ephemeris_time(moment)
        state, _light_time = self._spice.spkpos(
            "SUN", ephemeris_time, "J2000", "LT+S", "MOON"
        )
        vector = np.asarray(state, dtype=np.float64)
        return normalize(vector), float(np.linalg.norm(vector))

    @_serialized_spice
    def get_moon_orientation(self, moment: datetime) -> MoonOrientation:
        """Read the IAU_MOON orientation straight out of the loaded PCK.

        ``pxform`` gives the rotation matrix directly; alpha0/delta0/W are then
        recovered from it so the response shape matches the simple provider.
        """
        ephemeris_time = self._ephemeris_time(moment)
        j2000_to_body = np.asarray(
            self._spice.pxform("J2000", self._body_frame(), ephemeris_time), dtype=np.float64
        )
        # Third row of J2000->body is the body +Z axis expressed in J2000.
        pole = j2000_to_body[2, :]
        declination = math.degrees(math.asin(float(np.clip(pole[2], -1.0, 1.0))))
        right_ascension = math.degrees(math.atan2(float(pole[1]), float(pole[0]))) % 360.0

        # Recover W from the prime-meridian axis relative to the equator node.
        prime_axis = j2000_to_body[0, :]
        node = np.array(
            [-math.sin(math.radians(right_ascension)), math.cos(math.radians(right_ascension)), 0.0]
        )
        node_cross = np.cross(pole, node)
        prime_meridian = math.degrees(
            math.atan2(float(np.dot(prime_axis, node_cross)), float(np.dot(prime_axis, node)))
        ) % 360.0

        return MoonOrientation(
            right_ascension_deg=right_ascension,
            declination_deg=declination,
            prime_meridian_deg=prime_meridian,
            j2000_to_body=j2000_to_body,
            source="jpl-spice-iau-moon",
        )


def _import_spiceypy():  # noqa: ANN202 - third-party module object
    """Import ``spiceypy`` lazily so the app runs without it installed."""
    try:
        import spiceypy  # noqa: PLC0415 - intentional lazy import
    except ImportError as error:  # pragma: no cover - depends on environment
        raise SpiceKernelError(
            "spiceypy is not installed. Run `pip install spiceypy` or set USE_SPICE=false."
        ) from error
    return spiceypy


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_astronomy_provider(
    *, use_spice: bool, kernel_directory: Path
) -> AstronomyProvider:
    """Select the best available astronomy provider.

    Falls back to :class:`SimpleAstronomyProvider` - with a logged warning -
    whenever SPICE is requested but unusable, so the API never goes down
    because a kernel is missing.
    """
    if not use_spice:
        return SimpleAstronomyProvider()

    try:
        provider = SpiceAstronomyProvider(kernel_directory)
    except SpiceKernelError as error:
        logger.warning(
            "USE_SPICE=true but SPICE is unavailable (%s). "
            "Falling back to the approximate analytic ephemeris.",
            error,
        )
        return SimpleAstronomyProvider()

    logger.info("Using SPICE astronomy provider with kernels: %s", provider.loaded_kernels)
    return provider


__all__ = [
    "AstronomyProvider",
    "AstronomyResult",
    "MoonOrientation",
    "SimpleAstronomyProvider",
    "SpiceAstronomyProvider",
    "SpiceKernelError",
    "create_astronomy_provider",
    "datetime_to_julian_date",
    "days_since_j2000",
    "J2000_JD",
]
