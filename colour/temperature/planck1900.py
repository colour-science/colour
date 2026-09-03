"""
Blackbody - Planck (1900) - Correlated Colour Temperature
=========================================================

Define the *Planck (1900)* correlated colour temperature :math:`T_{cp}`
computation objects based on the spectral radiance of a planckian
radiator:

-   :func:`colour.temperature.uv_to_CCT_Planck1900`
-   :func:`colour.temperature.CCT_to_uv_Planck1900`

References
----------
-   :cite:`CIETC1-482004i` : CIE TC 1-48. (2004). APPENDIX E. INFORMATION ON
    THE USE OF PLANCK'S EQUATION FOR STANDARD AIR. In CIE 015:2004 Colorimetry,
    3rd Edition (pp. 77-82). ISBN:978-3-901906-33-6
"""

from __future__ import annotations

import typing

from colour.colorimetry import (
    MultiSpectralDistributions,
    handle_spectral_arguments,
    msds_to_XYZ_integration,
    planck_law,
)

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

from colour.models import UCS_to_uv, XYZ_to_UCS
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
    usage_warning,
    xp_matrix_transpose,
    xp_reshape,
)

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


_CCT_MINIMAL_PLANCK1900: float = 1000
"""Minimum correlated colour temperature in kelvins used by the method."""


def uv_to_CCT_Planck1900(
    uv: ArrayLike,
    cmfs: MultiSpectralDistributions | None = None,
    optimisation_kwargs: dict | None = None,
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` of a blackbody
    from specified *CIE UCS* colourspace *uv* chromaticity coordinates using
    colour matching functions.

    Parameters
    ----------
    uv
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
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
    The current implementation seeds a damped *Gauss-Newton* iteration with
    a nearest-neighbour lookup against a coarse grid sampled from the
    analytical forward, vectorised across all input samples.

    References
    ----------
    :cite:`CIETC1-482004i`

    Examples
    --------
    >>> uv_to_CCT_Planck1900([0.20042808, 0.31033343])  # doctest: +ELLIPSIS
    np.float64(6504.000071...)
    """

    optimisation_kwargs = dict(optional(optimisation_kwargs, {}))

    cmfs, _illuminant = handle_spectral_arguments(cmfs)
    uv = as_float_array(uv)

    def forward(CCT: NDArrayFloat) -> NDArrayFloat:
        return CCT_to_uv_Planck1900(CCT, cmfs)

    x0 = x0_CCT_grid(
        forward,
        uv,
        (_CCT_MINIMAL_PLANCK1900, 25000.0),
        samples=optimisation_kwargs.pop("samples", CCT_INVERSION_GRID_SAMPLES),
    )

    return as_float(solve_CCT_Newton(forward, uv, x0=x0, **optimisation_kwargs))


def CCT_to_uv_Planck1900(
    CCT: ArrayLike, cmfs: MultiSpectralDistributions | None = None
) -> NDArrayFloat:
    """
    Compute the *CIE UCS* colourspace *uv* chromaticity coordinates from the
    specified correlated colour temperature :math:`T_{cp}` and colour
    matching functions using the spectral radiance of a blackbody at the
    specified thermodynamic temperature.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Notes
    -----
    -   Non-finite correlated colour temperatures and temperatures below 1000 K
        are outside the range used by this method and will produce a warning.

    References
    ----------
    :cite:`CIETC1-482004i`

    Examples
    --------
    >>> CCT_to_uv_Planck1900(6504)  # doctest: +ELLIPSIS
    array([0.2004280..., 0.3103334...])
    """

    CCT = as_float_array(CCT)

    xp = array_namespace(CCT)

    if xp.any(
        xp.logical_or(xp.logical_not(xp.isfinite(CCT)), CCT < _CCT_MINIMAL_PLANCK1900)
    ):
        usage_warning(
            "Correlated colour temperature must be finite and greater than or "
            "equal to 1000 K, unpredictable results may occur!"
        )

    cmfs, _illuminant = handle_spectral_arguments(cmfs)

    radiance = (
        planck_law(
            cmfs.wavelengths * 1e-9,
            xp_reshape(CCT, (-1,), xp=xp),
        )
        * 1e-9
    )
    if radiance.ndim >= 2:
        radiance = xp_matrix_transpose(radiance, xp=xp)

    XYZ = msds_to_XYZ_integration(
        radiance,
        cmfs,
        shape=cmfs.shape,
    )

    UVW = XYZ_to_UCS(XYZ)
    uv = UCS_to_uv(UVW)

    return xp_reshape(uv, [*list(CCT.shape), 2], xp=xp)
