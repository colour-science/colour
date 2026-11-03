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

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

from colour.temperature.common import (
    CCT_INVERSION_GRID_SAMPLES,
    solve_CCT_Newton,
    x0_CCT_grid,
)
from colour.utilities import (
    array_namespace,
    as_float,
    as_float_array,
    optional,
    tstack,
    usage_warning,
)

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
        Inversion parameters forwarded to
        :func:`colour.temperature.x0_CCT_grid` and
        :func:`colour.temperature.solve_CCT_Newton`. Accepted keys are
        ``samples`` (grid density for the initial guess, default
        :attr:`colour.temperature.CCT_INVERSION_GRID_SAMPLES`),
        ``newton_iterations``, ``backtrack_iterations`` and ``tolerance``
        (forwarded to :func:`solve_CCT_Newton`).

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    Warnings
    --------
    *Krystek (1985)* does not provide an analytical inverse transformation
    to compute the correlated colour temperature :math:`T_{cp}` from the
    specified *CIE UCS* colourspace *uv* chromaticity coordinates. The
    current implementation seeds a damped *Gauss-Newton* iteration with a
    nearest-neighbour lookup against a coarse grid sampled from the
    analytical forward, vectorised across all input samples.

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
    >>> uv_to_CCT_Krystek1985([0.20047203, 0.31029290])  # doctest: +ELLIPSIS
    np.float64(6504.389416...)
    """

    optimisation_kwargs = dict(optional(optimisation_kwargs, {}))

    uv = as_float_array(uv)

    x0 = x0_CCT_grid(
        CCT_to_uv_Krystek1985,
        uv,
        (1000.0, 15000.0),
        samples=optimisation_kwargs.pop("samples", CCT_INVERSION_GRID_SAMPLES),
    )

    return as_float(
        solve_CCT_Newton(CCT_to_uv_Krystek1985, uv, x0=x0, **optimisation_kwargs)
    )


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
        temperature :math:`T_{cp}` normalised to domain [1000, 15000]. The
        temperature must be finite.

    References
    ----------
    :cite:`Krystek1985b`

    Examples
    --------
    >>> CCT_to_uv_Krystek1985(6504.38938305)  # doctest: +ELLIPSIS
    array([0.2004720..., 0.3102929...])
    """

    T = as_float_array(CCT)

    xp = array_namespace(T)

    if xp.any(
        xp.logical_or(
            xp.logical_not(xp.isfinite(T)), xp.logical_or(T < 1000, T > 15000)
        )
    ):
        usage_warning(
            "Correlated colour temperature must be finite and in domain "
            "[1000, 15000] K, unpredictable results may occur!"
        )

    T_2 = T**2

    u = (0.860117757 + 1.54118254 * 10**-4 * T + 1.28641212 * 10**-7 * T_2) / (
        1 + 8.42420235 * 10**-4 * T + 7.08145163 * 10**-7 * T_2
    )
    v = (0.317398726 + 4.22806245 * 10**-5 * T + 4.20481691 * 10**-8 * T_2) / (
        1 - 2.89741816 * 10**-5 * T + 1.61456053 * 10**-7 * T_2
    )

    return tstack([u, v])
