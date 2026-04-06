"""
Kang, Moon, Hong, Lee, Cho and Kim (2002) Correlated Colour Temperature
=======================================================================

Define the *Kang et al. (2002)* correlated colour temperature :math:`T_{cp}`
computation objects.

-   :func:`colour.temperature.xy_to_CCT_Kang2002`: Compute correlated colour
    temperature :math:`T_{cp}` from specified *CIE xy* chromaticity
    coordinates using the *Kang, Moon, Hong, Lee, Cho and Kim (2002)* method.
-   :func:`colour.temperature.CCT_to_xy_Kang2002`: Compute *CIE xy*
    chromaticity coordinates from specified correlated colour temperature
    :math:`T_{cp}` using the *Kang, Moon, Hong, Lee, Cho and Kim (2002)*
    method.

References
----------
-   :cite:`Kang2002a` : Kang, B., Moon, O., Hong, C., Lee, H., Cho, B., & Kim,
    Y. (2002). Design of advanced color: Temperature control system for HDTV
    applications. Journal of the Korean Physical Society, 41(6), 865-871.
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
    xp_select,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "xy_to_CCT_Kang2002",
    "CCT_to_xy_Kang2002",
]


def xy_to_CCT_Kang2002(
    xy: ArrayLike, optimisation_kwargs: dict | None = None
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` from the
    specified *CIE xy* chromaticity coordinates using *Kang et al. (2002)*
    method.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates.
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
    The *Kang et al. (2002)* method does not provide an analytical inverse
    transformation to compute the correlated colour temperature
    :math:`T_{cp}` from the specified *CIE xy* chromaticity coordinates.
    The current implementation relies on a damped *Gauss-Newton* iteration
    seeded by nearest-neighbour lookup against a coarse grid sampled from
    the analytical forward over the [1667, 25000] domain. The lookup keeps
    the iteration in the correct basin near the domain edges where the
    polynomial is non-monotonic if extrapolated.

    References
    ----------
    :cite:`Kang2002a`

    Examples
    --------
    >>> xy_to_CCT_Kang2002([0.31342600, 0.32359597])  # doctest: +ELLIPSIS
    np.float64(6504.389303...)
    """

    optimisation_kwargs = dict(optional(optimisation_kwargs, {}))

    xy = as_float_array(xy)

    x0 = x0_CCT_grid(
        CCT_to_xy_Kang2002,
        xy,
        (1667.0, 25000.0),
        samples=optimisation_kwargs.pop("samples", CCT_INVERSION_GRID_SAMPLES),
    )

    return as_float(
        solve_CCT_Newton(CCT_to_xy_Kang2002, xy, x0=x0, **optimisation_kwargs)
    )


def CCT_to_xy_Kang2002(CCT: ArrayLike) -> NDArrayFloat:
    """
    Compute the *CIE xy* chromaticity coordinates from the specified
    correlated colour temperature :math:`T_{cp}` using *Kang et al. (2002)*
    method.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    Raises
    ------
    ValueError
        If the correlated colour temperature is not in appropriate domain.

    References
    ----------
    :cite:`Kang2002a`

    Examples
    --------
    >>> CCT_to_xy_Kang2002(6504.38938305)  # doctest: +ELLIPSIS
    array([0.313426..., 0.3235959...])
    """

    CCT = as_float_array(CCT)

    xp = array_namespace(CCT)

    if xp.any(xp.logical_or(CCT < 1667, CCT > 25000)):
        usage_warning(
            "Correlated colour temperature must be in domain "
            "[1667, 25000], unpredictable results may occur!"
        )

    CCT_3 = CCT**3
    CCT_2 = CCT**2

    x = xp.where(
        CCT <= 4000,
        -0.2661239 * 10**9 / CCT_3
        - 0.2343589 * 10**6 / CCT_2
        + 0.8776956 * 10**3 / CCT
        + 0.179910,
        -3.0258469 * 10**9 / CCT_3
        + 2.1070379 * 10**6 / CCT_2
        + 0.2226347 * 10**3 / CCT
        + 0.24039,
    )

    x_3 = x**3
    x_2 = x**2

    cnd_l = [CCT <= 2222, xp.logical_and(CCT > 2222, CCT <= 4000), CCT > 4000]
    i = -1.1063814 * x_3 - 1.34811020 * x_2 + 2.18555832 * x - 0.20219683
    j = -0.9549476 * x_3 - 1.37418593 * x_2 + 2.09137015 * x - 0.16748867
    k = 3.0817580 * x_3 - 5.8733867 * x_2 + 3.75112997 * x - 0.37001483
    y = xp_select(cnd_l, [i, j, k], xp=xp)

    return tstack([x, y])
