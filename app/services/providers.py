"""Dependency-injection container for the science providers.

One :class:`ScienceProviders` instance is built at application startup from the
:class:`~app.config.Settings` flags and then injected into request handlers.
Selection is automatic and always degrades gracefully:

.. code-block:: text

    USE_SPICE=true  + usable kernels -> SpiceAstronomyProvider
    otherwise                        -> SimpleAstronomyProvider

    USE_TERRAIN_DEM=true + readable raster -> RasterTerrainProvider
    otherwise                              -> FlatTerrainProvider

    USE_DIVINER=true + readable product -> GriddedDivinerProvider
    otherwise                           -> NullDivinerProvider

Every fallback is logged at WARNING level and reflected in the provenance
metadata returned to the client, so a silent downgrade is impossible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Settings
from .astronomy import AstronomyProvider, create_astronomy_provider
from .diviner import DivinerProvider, create_diviner_provider
from .terrain import TerrainProvider, create_terrain_provider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScienceProviders:
    """Bundle of the active data providers plus the settings that chose them."""

    astronomy: AstronomyProvider
    terrain: TerrainProvider
    diviner: DivinerProvider
    settings: Settings

    @property
    def albedo(self) -> float:
        """Albedo in effect, honouring any configuration override."""
        from ..constants import DEFAULT_ALBEDO  # noqa: PLC0415 - avoids a cycle

        return (
            self.settings.albedo_override
            if self.settings.albedo_override is not None
            else DEFAULT_ALBEDO
        )

    @property
    def emissivity(self) -> float:
        """Emissivity in effect, honouring any configuration override."""
        from ..constants import DEFAULT_EMISSIVITY  # noqa: PLC0415 - avoids a cycle

        return (
            self.settings.emissivity_override
            if self.settings.emissivity_override is not None
            else DEFAULT_EMISSIVITY
        )

    def describe(self) -> dict[str, str]:
        """Flat provenance map surfaced in API responses and ``/api/status``."""
        return {
            "astronomy": self.astronomy.source,
            "terrain": self.terrain.source,
            "temperature": "simplified-radiative-model",
            "diviner": self.diviner.source,
        }


def build_providers(settings: Settings) -> ScienceProviders:
    """Construct the provider bundle described by ``settings``."""
    astronomy = create_astronomy_provider(
        use_spice=settings.use_spice, kernel_directory=settings.spice_dir
    )
    terrain = create_terrain_provider(
        use_dem=settings.use_terrain_dem,
        raster_path=settings.terrain_raster_path,
        enable_shadows=settings.use_terrain_shadows,
    )
    diviner = create_diviner_provider(
        use_diviner=settings.use_diviner, data_directory=settings.diviner_dir, astronomy=astronomy
    )

    providers = ScienceProviders(
        astronomy=astronomy, terrain=terrain, diviner=diviner, settings=settings
    )
    logger.info("Science providers active: %s", providers.describe())
    return providers


__all__ = ["ScienceProviders", "build_providers"]
