"""Lunar surface temperature estimation.

MVP model: instantaneous radiative equilibrium
----------------------------------------------

The airless lunar surface absorbs a fraction ``(1 - A)`` of the incident
direct-beam irradiance and re-radiates as a grey body of emissivity
``epsilon``:

.. math::

    Q_\\text{absorbed} = (1 - A)\\,Q_\\text{solar} \\\\
    \\epsilon \\sigma T^4 = Q_\\text{absorbed} + Q_\\text{background} \\\\
    T = \\left(\\frac{Q_\\text{absorbed}
        + Q_\\text{background}}{\\epsilon\\sigma}\\right)^{1/4}

``Q_background`` is *not* a fudge factor to avoid 0 K. It represents the heat
the regolith releases from storage plus the small endogenic flux, and is
defined so that in complete darkness the model asymptotes exactly to
:data:`~app.constants.NIGHT_TEMPERATURE_FLOOR_K`:

.. math:: Q_\\text{background} = \\epsilon\\sigma T_\\text{floor}^4

Known limitations - these are surfaced to the user, never hidden:

* **No thermal inertia.** Real regolith lags the Sun; at sunset the surface
  cools over hours, not instantly. This model responds immediately, so it
  overestimates the cooling rate near the terminator.
* **No lateral or subsurface conduction.**
* **Single global albedo/emissivity.** Maria and highlands differ, and albedo
  rises sharply at high incidence angles.
* **No earthshine, no scattered light from surrounding terrain.**

For results that track Diviner observations the 1-D conduction model sketched
in :class:`RegolithThermalModel` is required; it is deliberately left
unimplemented rather than faked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import (
    DEFAULT_ALBEDO,
    DEFAULT_EMISSIVITY,
    KELVIN_OFFSET,
    NIGHT_TEMPERATURE_FLOOR_K,
    STEFAN_BOLTZMANN,
)

#: Tag identifying the MVP model in API responses.
EQUILIBRIUM_MODEL_NAME = "simplified-radiative-equilibrium"


@dataclass(frozen=True)
class ThermalResult:
    """Estimated surface temperature and its provenance.

    Attributes:
        temperature_k: Estimated surface temperature [K].
        temperature_c: Same value in degrees Celsius.
        model: Identifier of the model used.
        confidence: Qualitative reliability - always ``"experimental"`` for
            the simplified model.
        albedo: Albedo value actually used.
        emissivity: Emissivity value actually used.
        absorbed_flux_w_m2: ``(1 - A) * Q_solar``.
        is_measured: ``False`` for every modelled value; only a real Diviner
            observation would be ``True``.
        notes: Plain-language statement of the model's limitations.
    """

    temperature_k: float
    temperature_c: float
    model: str
    confidence: str
    albedo: float
    emissivity: float
    absorbed_flux_w_m2: float
    is_measured: bool
    notes: str


def kelvin_to_celsius(temperature_k: float) -> float:
    """Convert Kelvin to degrees Celsius."""
    return temperature_k - KELVIN_OFFSET


def background_flux_w_m2(
    *,
    emissivity: float = DEFAULT_EMISSIVITY,
    floor_temperature_k: float = NIGHT_TEMPERATURE_FLOOR_K,
) -> float:
    """Heat flux that sustains the night-time temperature floor.

    .. math:: Q_\\text{background} = \\epsilon \\sigma T_\\text{floor}^4

    Returns:
        Background flux [W/m^2]. With the defaults this is ~5.4 W/m^2.
    """
    return emissivity * STEFAN_BOLTZMANN * floor_temperature_k**4


def equilibrium_temperature_k(
    irradiance_w_m2: float,
    *,
    albedo: float = DEFAULT_ALBEDO,
    emissivity: float = DEFAULT_EMISSIVITY,
    floor_temperature_k: float = NIGHT_TEMPERATURE_FLOOR_K,
) -> float:
    """Radiative-equilibrium surface temperature [K].

    Args:
        irradiance_w_m2: Direct-beam solar irradiance on the surface, already
            reduced by ``cos(incidence)``. Must be non-negative.
        albedo: Bond albedo ``A`` in [0, 1).
        emissivity: Thermal emissivity ``epsilon`` in (0, 1].
        floor_temperature_k: Asymptotic darkness temperature.

    Returns:
        Temperature in Kelvin, never below ``floor_temperature_k``.

    Raises:
        ValueError: for out-of-range physical parameters.
    """
    if irradiance_w_m2 < 0.0:
        raise ValueError("Irradiance cannot be negative")
    if not 0.0 <= albedo < 1.0:
        raise ValueError("Albedo must be in [0, 1)")
    if not 0.0 < emissivity <= 1.0:
        raise ValueError("Emissivity must be in (0, 1]")

    absorbed = (1.0 - albedo) * irradiance_w_m2
    background = background_flux_w_m2(
        emissivity=emissivity, floor_temperature_k=floor_temperature_k
    )
    return ((absorbed + background) / (emissivity * STEFAN_BOLTZMANN)) ** 0.25


def estimate_temperature(
    irradiance_w_m2: float,
    *,
    albedo: float = DEFAULT_ALBEDO,
    emissivity: float = DEFAULT_EMISSIVITY,
    floor_temperature_k: float = NIGHT_TEMPERATURE_FLOOR_K,
) -> ThermalResult:
    """Full thermal estimate with provenance metadata attached."""
    temperature_k = equilibrium_temperature_k(
        irradiance_w_m2,
        albedo=albedo,
        emissivity=emissivity,
        floor_temperature_k=floor_temperature_k,
    )
    absorbed = (1.0 - albedo) * irradiance_w_m2

    return ThermalResult(
        temperature_k=temperature_k,
        temperature_c=kelvin_to_celsius(temperature_k),
        model=EQUILIBRIUM_MODEL_NAME,
        confidence="experimental",
        albedo=albedo,
        emissivity=emissivity,
        absorbed_flux_w_m2=absorbed,
        is_measured=False,
        notes=(
            "Instantaneous radiative equilibrium with a "
            f"{floor_temperature_k:.0f} K darkness floor. No thermal inertia, "
            "no subsurface conduction, single global albedo. Expect several "
            "tens of kelvin of error near the terminator. This is a model "
            "estimate, not an LRO Diviner measurement."
        ),
    )


class RegolithThermalModel:
    """Placeholder for a 1-D subsurface conduction model (future work).

    The physics to implement
    ------------------------

    Solve the heat equation through a depth-layered regolith column:

    .. math::

        \\rho(z)\\,c(T)\\,\\frac{\\partial T}{\\partial t}
        = \\frac{\\partial}{\\partial z}
          \\left(k(T, z)\\,\\frac{\\partial T}{\\partial z}\\right)

    with the standard lunar parameterisation (Hayne et al. 2017,
    *JGR Planets* 122):

    * **Upper boundary** - radiative balance at ``z = 0``::

          (1 - A(i)) Q_solar = epsilon sigma T_s^4 - k dT/dz |_(z=0)

      where ``A(i)`` grows with solar incidence angle ``i``.
    * **Lower boundary** - endogenic heat flux ~0.018 W/m^2 at a few metres.
    * **Density profile** - ``rho(z) = rho_d - (rho_d - rho_s) exp(-z/H)``
      with ``rho_s ~ 1100``, ``rho_d ~ 1800 kg/m^3``, ``H ~ 0.06 m``.
    * **Conductivity** - ``k(T, z) = k_c(z) [1 + chi (T/350)^3]``, the radiative
      term ``chi ~ 2.7`` dominating at high temperature.
    * **Heat capacity** - the fourth-order polynomial ``c(T)`` fit of Hayne
      et al. to Apollo sample measurements.

    Suggested implementation
    ------------------------

    1. Build a geometrically expanding depth grid: ~40 layers, first layer
       ~1 mm, to a few thermal skin depths (~0.5 m for a lunar diurnal cycle).
    2. Integrate with an implicit (Crank-Nicolson) scheme - the explicit
       stability limit on a millimetre first layer is punishingly small.
    3. Spin up over 3-5 full synodic months from an isothermal start until the
       diurnal cycle repeats to within ~0.1 K.
    4. Drive the upper boundary with the irradiance timeline already produced
       by :mod:`app.services.timeline`.
    5. Validate against LRO Diviner bolometric temperatures via
       :class:`app.services.diviner.DivinerProvider`.

    Until that exists this class raises rather than returning a plausible-
    looking number, because a fabricated temperature is worse than no
    temperature.
    """

    model_name = "regolith-1d-conduction"
    implemented = False

    def __init__(self, layer_count: int = 40, spin_up_cycles: int = 4) -> None:
        self.layer_count = layer_count
        self.spin_up_cycles = spin_up_cycles

    def simulate(self, *args: object, **kwargs: object) -> ThermalResult:
        """Not implemented. See the class docstring for the intended physics."""
        raise NotImplementedError(
            "RegolithThermalModel is a documented placeholder. Use "
            "estimate_temperature() for the simplified equilibrium model."
        )


def temperature_confidence_label(has_terrain: bool, has_diviner: bool) -> str:
    """Summarise thermal fidelity for display in the UI fidelity panel."""
    # Availability of observations alone is not evidence of model calibration.
    if has_terrain:
        return "experimental-with-terrain"
    return "experimental"


def _celsius_to_kelvin(temperature_c: float) -> float:
    """Inverse of :func:`kelvin_to_celsius` (used in tests)."""
    return temperature_c + KELVIN_OFFSET


def subsolar_peak_temperature_k(
    solar_constant_w_m2: float,
    *,
    albedo: float = DEFAULT_ALBEDO,
    emissivity: float = DEFAULT_EMISSIVITY,
) -> float:
    """Temperature at the sub-solar point - a useful sanity anchor.

    With the defaults this returns ~385-395 K, matching the ~390 K Diviner
    reports for the lunar noon equator. Used by the test suite to keep the
    model honest.
    """
    return equilibrium_temperature_k(
        solar_constant_w_m2, albedo=albedo, emissivity=emissivity
    )


def is_physically_plausible(temperature_k: float) -> bool:
    """Bounds check: the lunar surface spans roughly 20-400 K."""
    return 20.0 <= temperature_k <= 420.0 and not math.isnan(temperature_k)


__all__ = [
    "EQUILIBRIUM_MODEL_NAME",
    "RegolithThermalModel",
    "ThermalResult",
    "background_flux_w_m2",
    "equilibrium_temperature_k",
    "estimate_temperature",
    "is_physically_plausible",
    "kelvin_to_celsius",
    "subsolar_peak_temperature_k",
    "temperature_confidence_label",
]
