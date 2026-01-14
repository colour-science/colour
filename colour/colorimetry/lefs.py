"""
Luminous Efficiency Functions Spectral Distributions
====================================================

Define the luminous efficiency functions and their computation objects for
photopic, scotopic, and mesopic vision conditions.

References
----------
-   :cite:`Wikipedia2005d` : Wikipedia. (2005). Mesopic weighting function.
    Retrieved June 20, 2014, from
    http://en.wikipedia.org/wiki/Mesopic_vision#Mesopic_weighting_function
"""

from __future__ import annotations

import typing

from colour.colorimetry import (
    SDS_LEFS_PHOTOPIC,
    SDS_LEFS_SCOTOPIC,
    SpectralDistribution,
    SpectralShape,
)
from colour.colorimetry.datasets.lefs import DATA_MESOPIC_X

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Literal, NDArrayFloat

from colour.utilities import closest, optional, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "mesopic_weighting_function",
    "sd_mesopic_luminous_efficiency_function",
]


def mesopic_weighting_function(
    wavelength: ArrayLike,
    L_p: float,
    source: Literal["Blue Heavy", "Red Heavy"] | str = "Blue Heavy",
    method: Literal["MOVE", "LRC"] | str = "MOVE",
    photopic_lef: SpectralDistribution | None = None,
    scotopic_lef: SpectralDistribution | None = None,
) -> NDArrayFloat:
    """
    Calculate the mesopic weighting function factor :math:`V_m` at the specified
    wavelength :math:`\\lambda` using the specified photopic luminance
    :math:`L_p`.

    The mesopic weighting function provides a transition between photopic and
    scotopic vision, accounting for the combined response of rods and cones
    under intermediate lighting conditions.

    Parameters
    ----------
    wavelength
        Wavelength :math:`\\lambda` to calculate the mesopic weighting function
        factor.
    L_p
        Photopic luminance :math:`L_p`.
    source
        Light source colour temperature.
    method
        Method to calculate the weighting factor.
    photopic_lef
        :math:`V(\\lambda)` photopic luminous efficiency function, defaults to
        the *CIE 1924 Photopic Standard Observer*.
    scotopic_lef
        :math:`V^\\prime(\\lambda)` scotopic luminous efficiency function,
        defaults to the *CIE 1951 Scotopic Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        Mesopic weighting function factor :math:`V_m`.

    References
    ----------
    :cite:`Wikipedia2005d`

    Examples
    --------
    >>> mesopic_weighting_function(500, 0.2)  # doctest: +ELLIPSIS
    np.float64(0.7052...)
    """

    photopic_lef = optional(
        photopic_lef,
        SDS_LEFS_PHOTOPIC["CIE 1924 Photopic Standard Observer"],
    )

    scotopic_lef = optional(
        scotopic_lef,
        SDS_LEFS_SCOTOPIC["CIE 1951 Scotopic Standard Observer"],
    )

    source = validate_method(
        source,
        ("Blue Heavy", "Red Heavy"),
        '"{0}" light source colour temperature is invalid, it must be one of {1}!',
    )
    method = validate_method(method, ("MOVE", "LRC"))

    mesopic_x_luminance_values = sorted(DATA_MESOPIC_X.keys())
    index = mesopic_x_luminance_values.index(closest(mesopic_x_luminance_values, L_p))
    x = DATA_MESOPIC_X[mesopic_x_luminance_values[index]][source][method]

    return (1 - x) * scotopic_lef[wavelength] + x * photopic_lef[wavelength]


def sd_mesopic_luminous_efficiency_function(
    L_p: float,
    source: Literal["Blue Heavy", "Red Heavy"] | str = "Blue Heavy",
    method: Literal["MOVE", "LRC"] | str = "MOVE",
    photopic_lef: SpectralDistribution | None = None,
    scotopic_lef: SpectralDistribution | None = None,
) -> SpectralDistribution:
    """
    Return the mesopic luminous efficiency function :math:`V_m(\\lambda)` for
    the specified photopic luminance :math:`L_p`.

    Parameters
    ----------
    L_p
        Photopic luminance :math:`L_p`.
    source
        Light source colour temperature.
    method
        Method to calculate the weighting factor.
    photopic_lef
        :math:`V(\\lambda)` photopic luminous efficiency function, default to
        the *CIE 1924 Photopic Standard Observer*.
    scotopic_lef
        :math:`V^\\prime(\\lambda)` scotopic luminous efficiency function,
        default to the *CIE 1951 Scotopic Standard Observer*.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Mesopic luminous efficiency function :math:`V_m(\\lambda)`.

    References
    ----------
    :cite:`Wikipedia2005d`

    Examples
    --------
    >>> sd = sd_mesopic_luminous_efficiency_function(0.2)
    >>> sd.shape
    SpectralShape(380.0, 780.0, 1.0)
    >>> sd[500]  # doctest: +ELLIPSIS
    np.float64(0.8352251...)
    """

    photopic_lef = optional(
        photopic_lef,
        SDS_LEFS_PHOTOPIC["CIE 1924 Photopic Standard Observer"],
    )

    scotopic_lef = optional(
        scotopic_lef,
        SDS_LEFS_SCOTOPIC["CIE 1951 Scotopic Standard Observer"],
    )

    shape = SpectralShape(
        max([photopic_lef.shape.start, scotopic_lef.shape.start]),
        min([photopic_lef.shape.end, scotopic_lef.shape.end]),
        max([photopic_lef.shape.interval, scotopic_lef.shape.interval]),
    )

    sd = SpectralDistribution(
        mesopic_weighting_function(
            shape.wavelengths, L_p, source, method, photopic_lef, scotopic_lef
        ),
        shape.wavelengths,
        name=f"{L_p} Lp Mesopic Luminous Efficiency Function",
    )

    return sd.normalise()
