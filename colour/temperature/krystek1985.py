"""
Krystek (1985) Correlated Colour Temperature
============================================

Define the *Krystek (1985)* correlated colour temperature :math:`T_{cp}`
computation objects.

-   :func:`colour.temperature.uv_to_CCT_Krystek1985`: Compute correlated
    colour temperature :math:`T_{cp}` from specified *CIE UCS* colourspace
    *uv* chromaticity coordinates using the *Krystek (1985)* method.
-   :func:`colour.temperature.CCT_to_uv_Krystek1985`: Compute *CIE UCS*
    colourspace *uv* chromaticity coordinates from specified correlated
    colour temperature :math:`T_{cp}` using the *Krystek (1985)* method.

References
----------
-   :cite:`Krystek1985b` : Krystek, M. (1985). An algorithm to calculate
    correlated colour temperature. Color Research & Application, 10(1), 38-40.
    doi:10.1002/col.5080100109
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, DTypeFloat, NDArrayFloat

from colour.utilities import as_float, as_float_array, required, tstack

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "uv_to_CCT_Krystek1985",
    "CCT_to_uv_Krystek1985",
]


@required("SciPy")
def uv_to_CCT_Krystek1985(
    uv: ArrayLike, optimisation_kwargs: dict | None = None
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` from the
    specified *CIE UCS* colourspace *uv* chromaticity coordinates using
    *Krystek (1985)* method.

    Parameters
    ----------
    uv
         *CIE UCS* colourspace *uv* chromaticity coordinates.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    Warnings
    --------
    *Krystek (1985)* does not provide an analytical inverse transformation
    to compute the correlated colour temperature :math:`T_{cp}` from the
    specified *CIE UCS* colourspace *uv* chromaticity coordinates. The
    current implementation relies on optimisation using
    :func:`scipy.optimize.minimize` definition and thus has reduced
    precision and poor performance.

    Notes
    -----
    -   *Krystek (1985)* method computations are valid for correlated
        colour temperature :math:`T_{cp}` normalised to domain
        [1000, 15000].

    References
    ----------
    :cite:`Krystek1985b`

    Examples
    --------
    >>> uv_to_CCT_Krystek1985(np.array([0.20047203, 0.31029290]))
    ... # doctest: +ELLIPSIS
    6504.3894290...
    """

    from scipy.optimize import minimize  # noqa: PLC0415

    uv = as_float_array(uv)
    shape = uv.shape
    uv = np.atleast_1d(np.reshape(uv, (-1, 2)))

    def objective_function(CCT: NDArrayFloat, uv: NDArrayFloat) -> DTypeFloat:
        """Objective function."""

        objective = np.linalg.norm(CCT_to_uv_Krystek1985(CCT) - uv)

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
                x0=[6500],
                args=(uv_i,),
                **optimisation_settings,
            ).x
            for uv_i in uv
        ]
    )

    return as_float(np.reshape(CCT, shape[:-1]))


def CCT_to_uv_Krystek1985(CCT: ArrayLike) -> NDArrayFloat:
    """
    Compute the *CIE UCS* colourspace *uv* chromaticity coordinates from the
    specified correlated colour temperature :math:`T_{cp}` using the
    *Krystek (1985)* method.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Notes
    -----
    -   *Krystek (1985)* method computations are valid for correlated colour
        temperature :math:`T_{cp}` normalised to domain [1000, 15000].

    References
    ----------
    :cite:`Krystek1985b`

    Examples
    --------
    >>> CCT_to_uv_Krystek1985(6504.38938305)  # doctest: +ELLIPSIS
    array([ 0.2004720...,  0.3102929...])
    """

    T = as_float_array(CCT)

    T_2 = T**2

    u = (0.860117757 + 1.54118254 * 10**-4 * T + 1.28641212 * 10**-7 * T_2) / (
        1 + 8.42420235 * 10**-4 * T + 7.08145163 * 10**-7 * T_2
    )
    v = (0.317398726 + 4.22806245 * 10**-5 * T + 4.20481691 * 10**-8 * T_2) / (
        1 - 2.89741816 * 10**-5 * T + 1.61456053 * 10**-7 * T_2
    )

    return tstack([u, v])
