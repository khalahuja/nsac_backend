"""Pure geometric helpers shared by the science modules.

Coordinate conventions used *everywhere* in this backend
--------------------------------------------------------

**Body-fixed lunar frame** (the "scientific coordinate system"):

* Right-handed Cartesian frame centred on the Moon, equivalent to SPICE's
  ``IAU_MOON`` / Mean-Earth-Polar-Axis frame.
* ``+Z`` points at the lunar north pole.
* ``+X`` pierces the prime meridian at the equator (lat 0, lon 0) - i.e. the
  mean sub-Earth point.
* ``+Y`` completes the right-handed set (lat 0, lon +90 deg East).

**Angles**

* Latitude  phi   in [-90, +90], north positive (planetocentric).
* Longitude lambda in [-180, +180], **east positive** (the IAU standard for
  the Moon since 1961).
* Everything internal is radians; degrees appear only at API boundaries.

**Local topocentric ENU frame**

* East  = d(position)/d(longitude), normalised.
* North = d(position)/d(latitude), normalised.
* Up    = outward radial unit vector = the spherical surface normal.
* Azimuth is measured clockwise from North: 0 = N, 90 = E, 180 = S, 270 = W.

Note this is a *display-independent* frame. The Three.js scene uses a different
(Y-up, visually compressed) frame; conversion happens only in the frontend.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Vector3 = NDArray[np.float64]


def normalize(vector: Vector3) -> Vector3:
    """Return ``vector`` scaled to unit length.

    Raises:
        ValueError: if the vector has (near) zero length and no direction can
            be defined.
    """
    norm = float(np.linalg.norm(vector))
    if norm < 1e-15:
        raise ValueError("Cannot normalise a zero-length vector")
    return np.asarray(vector, dtype=np.float64) / norm


def lat_lon_to_cartesian(
    latitude_deg: float, longitude_deg: float, radius: float = 1.0
) -> Vector3:
    """Convert planetocentric lat/lon to a body-fixed Cartesian vector.

    .. math::

        \\mathbf{r} = r\\,(\\cos\\varphi\\cos\\lambda,\\;
                          \\cos\\varphi\\sin\\lambda,\\;
                          \\sin\\varphi)

    Args:
        latitude_deg: Latitude phi in degrees, north positive.
        longitude_deg: Longitude lambda in degrees, east positive.
        radius: Radial distance from the Moon's centre (any unit).

    Returns:
        ``[x, y, z]`` in the body-fixed frame, same unit as ``radius``.
    """
    phi = math.radians(latitude_deg)
    lam = math.radians(longitude_deg)
    cos_phi = math.cos(phi)
    return np.array(
        [
            radius * cos_phi * math.cos(lam),
            radius * cos_phi * math.sin(lam),
            radius * math.sin(phi),
        ],
        dtype=np.float64,
    )


def cartesian_to_lat_lon(vector: Vector3) -> tuple[float, float]:
    """Convert a body-fixed Cartesian vector to (latitude, longitude) degrees.

    Inverse of :func:`lat_lon_to_cartesian`. The returned longitude is
    normalised to [-180, +180]; at the poles (x = y = 0) longitude is
    degenerate and 0.0 is returned by the ``atan2(0, 0)`` convention.
    """
    x, y, z = (float(component) for component in np.asarray(vector, dtype=np.float64))
    radius = math.sqrt(x * x + y * y + z * z)
    if radius < 1e-15:
        raise ValueError("Cannot derive lat/lon from a zero-length vector")
    latitude = math.degrees(math.asin(max(-1.0, min(1.0, z / radius))))
    longitude = math.degrees(math.atan2(y, x))
    return latitude, normalize_longitude(longitude)


def normalize_longitude(longitude_deg: float) -> float:
    """Wrap a longitude into [-180, +180].

    ``+180`` is preserved rather than folded to ``-180`` so that a user who
    types exactly 180 sees 180 echoed back.
    """
    if longitude_deg == 180.0:
        return 180.0
    wrapped = (longitude_deg + 180.0) % 360.0 - 180.0
    return wrapped


def spherical_surface_normal(latitude_deg: float, longitude_deg: float) -> Vector3:
    """Outward unit normal of a *spherical* Moon at the given coordinate.

    .. math::

        \\mathbf{N} = (\\cos\\varphi\\cos\\lambda,\\;
                       \\cos\\varphi\\sin\\lambda,\\;
                       \\sin\\varphi)

    A terrain provider may replace this with a slope/aspect-tilted normal; see
    :meth:`app.services.terrain.TerrainProvider.get_surface_normal`.
    """
    return lat_lon_to_cartesian(latitude_deg, longitude_deg, radius=1.0)


def local_enu_basis(
    latitude_deg: float, longitude_deg: float
) -> tuple[Vector3, Vector3, Vector3]:
    """Return the local (East, North, Up) unit vectors in the body-fixed frame.

    .. math::

        \\hat{e} = (-\\sin\\lambda,\\; \\cos\\lambda,\\; 0) \\\\
        \\hat{n} = (-\\sin\\varphi\\cos\\lambda,\\;
                    -\\sin\\varphi\\sin\\lambda,\\; \\cos\\varphi) \\\\
        \\hat{u} = (\\cos\\varphi\\cos\\lambda,\\;
                    \\cos\\varphi\\sin\\lambda,\\; \\sin\\varphi)

    At the exact poles East/North are degenerate; the formulas still return an
    orthonormal pair (aligned with the prime-meridian direction), which keeps
    azimuth well defined and continuous in the limit.
    """
    phi = math.radians(latitude_deg)
    lam = math.radians(longitude_deg)
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)

    east = np.array([-sin_lam, cos_lam, 0.0], dtype=np.float64)
    north = np.array([-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi], dtype=np.float64)
    up = np.array([cos_phi * cos_lam, cos_phi * sin_lam, sin_phi], dtype=np.float64)
    return east, north, up


def angle_between_deg(first: Vector3, second: Vector3) -> float:
    """Angle between two vectors in degrees.

    Uses ``atan2(|a x b|, a . b)`` rather than ``acos(a . b)``. Near 0 and 180
    degrees the cosine is flat, so ``acos`` amplifies floating-point error to
    ~1e-6 deg; the ``atan2`` form stays accurate to machine precision across
    the whole range. This matters for the solar incidence angle, which is
    evaluated exactly at 0 deg at the sub-solar point.
    """
    unit_first = normalize(first)
    unit_second = normalize(second)
    cross_magnitude = float(np.linalg.norm(np.cross(unit_first, unit_second)))
    dot = float(np.dot(unit_first, unit_second))
    return math.degrees(math.atan2(cross_magnitude, dot))


def rotation_matrix_z(angle_rad: float) -> NDArray[np.float64]:
    """Right-handed rotation about +Z by ``angle_rad``."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def rotation_matrix_x(angle_rad: float) -> NDArray[np.float64]:
    """Right-handed rotation about +X by ``angle_rad``."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]], dtype=np.float64
    )


__all__ = [
    "Vector3",
    "normalize",
    "lat_lon_to_cartesian",
    "cartesian_to_lat_lon",
    "normalize_longitude",
    "spherical_surface_normal",
    "local_enu_basis",
    "angle_between_deg",
    "rotation_matrix_z",
    "rotation_matrix_x",
]
