"""
Photometry
==========

Define the photometric quantities computation related objects.

References
----------
-   :cite:`Wikipedia2003b` : Wikipedia. (2003). Luminosity function. Retrieved
    October 20, 2014, from
    https://en.wikipedia.org/wiki/Luminosity_function#Details
-   :cite:`Wikipedia2005c` : Wikipedia. (2005). Luminous Efficacy. Retrieved
    April 3, 2016, from https://en.wikipedia.org/wiki/Luminous_efficacy
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import (
    SDS_LEFS_PHOTOPIC,
    SpectralDistribution,
    reshape_sd,
)
from colour.constants import CONSTANT_K_M
from colour.utilities import as_float_scalar, optional

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "luminous_flux",
    "luminous_efficiency",
    "luminous_efficacy",
]


def luminous_flux(
    sd: SpectralDistribution,
    lef: SpectralDistribution | None = None,
    K_m: float = CONSTANT_K_M,
) -> float:
    """
    Return the *luminous flux* for given spectral distribution using given
    luminous efficiency function.

    Parameters
    ----------
    sd
        test spectral distribution
    lef
        :math:`V(\\lambda)` luminous efficiency function, default to the
        *CIE 1924 Photopic Standard Observer*.
    K_m
        :math:`lm\\cdot W^{-1}` maximum photopic luminous efficiency.

    Returns
    -------
    :class:`float`
        Luminous flux.

    References
    ----------
    :cite:`Wikipedia2003b`

    Examples
    --------
    >>> from colour import SDS_LIGHT_SOURCES
    >>> sd = SDS_LIGHT_SOURCES["Neodimium Incandescent"]
    >>> luminous_flux(sd)  # doctest: +ELLIPSIS
    23807.6555273...
    """

    lef = optional(lef, SDS_LEFS_PHOTOPIC["CIE 1924 Photopic Standard Observer"])

    lef = reshape_sd(
        lef,
        sd.shape,
        copy=False,
        extrapolator_kwargs={"method": "Constant", "left": 0, "right": 0},
    )

    flux = K_m * np.trapezoid(lef.values * sd.values, sd.wavelengths)  # pyright: ignore

    return as_float_scalar(flux)


def luminous_efficiency(
    sd: SpectralDistribution, lef: SpectralDistribution | None = None
) -> float:
    """
    Return the *luminous efficiency* of given spectral distribution using
    given luminous efficiency function.

    Parameters
    ----------
    sd
        test spectral distribution
    lef
        :math:`V(\\lambda)` luminous efficiency function, default to the
        *CIE 1924 Photopic Standard Observer*.

    Returns
    -------
    :class:`float`
        Luminous efficiency.

    References
    ----------
    :cite:`Wikipedia2003b`

    Examples
    --------
    >>> from colour import SDS_LIGHT_SOURCES
    >>> sd = SDS_LIGHT_SOURCES["Neodimium Incandescent"]
    >>> luminous_efficiency(sd)  # doctest: +ELLIPSIS
    0.1994393...
    """

    lef = optional(lef, SDS_LEFS_PHOTOPIC["CIE 1924 Photopic Standard Observer"])

    lef = reshape_sd(
        lef,
        sd.shape,
        copy=False,
        extrapolator_kwargs={"method": "Constant", "left": 0, "right": 0},
    )

    efficiency = np.trapezoid(  # pyright: ignore
        lef.values * sd.values, sd.wavelengths
    ) / np.trapezoid(  # pyright: ignore
        sd.values, sd.wavelengths
    )

    return as_float_scalar(efficiency)


def luminous_efficacy(
    sd: SpectralDistribution, lef: SpectralDistribution | None = None
) -> float:
    """
    Return the *luminous efficacy* in :math:`lm\\cdot W^{-1}` of given
    spectral distribution using given luminous efficiency function.

    Parameters
    ----------
    sd
        test spectral distribution
    lef
        :math:`V(\\lambda)` luminous efficiency function, default to the
        *CIE 1924 Photopic Standard Observer*.

    Returns
    -------
    :class:`float`
        Luminous efficacy in :math:`lm\\cdot W^{-1}`.

    References
    ----------
    :cite:`Wikipedia2005c`

    Examples
    --------
    >>> from colour import SDS_LIGHT_SOURCES
    >>> sd = SDS_LIGHT_SOURCES["Neodimium Incandescent"]
    >>> luminous_efficacy(sd)  # doctest: +ELLIPSIS
    136.2170803...
    """

    efficacy = CONSTANT_K_M * luminous_efficiency(sd, lef)

    return as_float_scalar(efficacy)
