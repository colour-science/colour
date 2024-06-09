"""
Blackbody - Planck (1900) - Correlated Colour Temperature
=========================================================

Define the *Planck (1900)* correlated colour temperature :math:`T_{cp}`
computations objects based on the spectral radiance of a planckian radiator:

-   :func:`colour.temperature.uv_to_CCT_Planck1900`
-   :func:`colour.temperature.CCT_to_uv_Planck1900`

References
----------
-   :cite:`CIETC1-482004i` : CIE TC 1-48. (2004). APPENDIX E. INFORMATION ON
    THE USE OF PLANCK'S EQUATION FOR STANDARD AIR. In CIE 015:2004 Colorimetry,
    3rd Edition (pp. 77-82). ISBN:978-3-901906-33-6
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from colour.colorimetry import (
    MultiSpectralDistributions,
    handle_spectral_arguments,
    msds_to_XYZ_integration,
    planck_law,
)
from colour.hints import ArrayLike, NDArrayFloat
from colour.models import UCS_to_uv, XYZ_to_UCS
from colour.utilities import as_float, as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "uv_to_CCT_Planck1900",
    "CCT_to_uv_Planck1900",
]


def uv_to_CCT_Planck1900(
    uv: ArrayLike,
    cmfs: MultiSpectralDistributions | None = None,
    optimisation_kwargs: dict | None = None,
) -> NDArrayFloat:
    """
    Return the correlated colour temperature :math:`T_{cp}` of a blackbody from
    given *CIE UCS* colourspace *uv* chromaticity coordinates and colour
    matching functions.

    Parameters
    ----------
    uv
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    Warnings
    --------
    The current implementation relies on optimisation using
    :func:`scipy.optimize.minimize` definition and thus has reduced precision
    and poor performance.

    References
    ----------
    :cite:`CIETC1-482004i`

    Examples
    --------
    >>> uv_to_CCT_Planck1900(np.array([0.20042808, 0.31033343]))
    ... # doctest: +ELLIPSIS
    6504.0000617...
    """

    uv = as_float_array(uv)
    cmfs, _illuminant = handle_spectral_arguments(cmfs)

    shape = uv.shape
    uv = np.atleast_1d(np.reshape(uv, (-1, 2)))

    def objective_function(CCT: NDArrayFloat, uv: NDArrayFloat) -> NDArrayFloat:
        """Objective function."""

        objective = np.linalg.norm(CCT_to_uv_Planck1900(CCT, cmfs) - uv)

        return as_float(objective)

    optimisation_settings = {
        "method": "Nelder-Mead",
        "options": {
            "fatol": 1e-10,
        },
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    CCT = as_float_array(
        [
            minimize(
                objective_function,
                x0=6500,
                args=(uv_i,),
                **optimisation_settings,
            ).x
            for uv_i in uv
        ]
    )

    return as_float(np.reshape(CCT, shape[:-1]))


def CCT_to_uv_Planck1900(
    CCT: ArrayLike, cmfs: MultiSpectralDistributions | None = None
) -> NDArrayFloat:
    """
    Return the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` and colour matching functions
    using the spectral radiance of a blackbody at the given thermodynamic
    temperature.

    Parameters
    ----------
    CCT
        Colour temperature :math:`T_{cp}`.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    References
    ----------
    :cite:`CIETC1-482004i`

    Examples
    --------
    >>> CCT_to_uv_Planck1900(6504)  # doctest: +ELLIPSIS
    array([ 0.2004280...,  0.3103334...])
    """

    CCT = as_float_array(CCT)
    cmfs, _illuminant = handle_spectral_arguments(cmfs)

    XYZ = msds_to_XYZ_integration(
        np.transpose(planck_law(cmfs.wavelengths * 1e-9, np.ravel(CCT)) * 1e-9),
        cmfs,
        shape=cmfs.shape,
    )

    UVW = XYZ_to_UCS(XYZ)
    uv = UCS_to_uv(UVW)

    return np.reshape(uv, [*list(CCT.shape), 2])
