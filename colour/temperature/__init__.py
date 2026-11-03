"""
References
----------
-   :cite:`AdobeSystems2013` : Adobe Systems. (2013). Adobe DNG Software
    Development Kit (SDK) - 1.3.0.0 -
    dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp::dng_temperature::\
Set_xy_coord. https://www.adobe.com/support/downloads/dng/dng_sdk.html
-   :cite:`CIETC1-482004i` : CIE TC 1-48. (2004). APPENDIX E. INFORMATION ON
    THE USE OF PLANCK'S EQUATION FOR STANDARD AIR. In CIE 015:2004 Colorimetry,
    3rd Edition (pp. 77-82). ISBN:978-3-901906-33-6
-   :cite:`Hernandez-Andres1999a` : Hernández-Andrés, J., Lee, R. L., &
    Romero, J. (1999). Calculating correlated color temperatures across the
    entire gamut of daylight and skylight chromaticities. Applied Optics,
    38(27),
    5703. doi:10.1364/AO.38.005703
-   :cite:`Kang2002a` : Kang, B., Moon, O., Hong, C., Lee, H., Cho, B., & Kim,
    Y. (2002). Design of advanced color: Temperature control system for HDTV
    applications. Journal of the Korean Physical Society, 41(6), 865-871.
-   :cite:`Krystek1985b` : Krystek, M. (1985). An algorithm to calculate
    correlated colour temperature. Color Research & Application, 10(1), 38-40.
    doi:10.1002/col.5080100109
-   :cite:`Ohno2014a` : Ohno, Yoshiro. (2014). Practical Use and Calculation of
    CCT and Duv. LEUKOS, 10(1), 47-55. doi:10.1080/15502724.2014.839020
-   :cite:`Wikipedia2001` : Wikipedia. (2001). Approximation. Retrieved June
    28, 2014, from http://en.wikipedia.org/wiki/Color_temperature#Approximation
-   :cite:`Wikipedia2001a` : Wikipedia. (2001). Color temperature. Retrieved
    June 28, 2014, from http://en.wikipedia.org/wiki/Color_temperature
-   :cite:`Wyszecki2000y` : Wyszecki, Günther, & Stiles, W. S. (2000).
    DISTRIBUTION TEMPERATURE, COLOR TEMPERATURE, AND CORRELATED COLOR
    TEMPERATURE. In Color Science: Concepts and Methods, Quantitative Data and
    Formulae (pp. 224-229). Wiley. ISBN:978-0-471-39918-6
-   :cite:`Wyszecki2000z` : Wyszecki, Günther, & Stiles, W. S. (2000). CIE
    Method of Calculating D-Illuminants. In Color Science: Concepts and
    Methods, Quantitative Data and Formulae (pp. 145-146). Wiley.
    ISBN:978-0-471-39918-6
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import Any, ArrayLike, NDArrayFloat, Literal

from colour.utilities import (
    CanonicalMapping,
    filter_kwargs,
    validate_method,
)

from .common import (
    CCT_INVERSION_GRID_SAMPLES,
    solve_CCT_Newton,
    solve_xy_Newton,
    x0_CCT_grid,
)

# isort: split

from .cie_d import CCT_to_xy_CIE_D, xy_to_CCT_CIE_D
from .hernandez1999 import CCT_to_xy_Hernandez1999, xy_to_CCT_Hernandez1999
from .kang2002 import CCT_to_xy_Kang2002, xy_to_CCT_Kang2002
from .krystek1985 import CCT_to_uv_Krystek1985, uv_to_CCT_Krystek1985
from .mccamy1992 import CCT_to_xy_McCamy1992, xy_to_CCT_McCamy1992
from .planck1900 import CCT_to_uv_Planck1900, uv_to_CCT_Planck1900

# isort: split

from .ohno2013 import (
    CCT_to_uv_Ohno2013,
    CCT_to_XYZ_Ohno2013,
    XYZ_to_CCT_Ohno2013,
    uv_to_CCT_Ohno2013,
)
from .robertson1968 import (
    CCT_to_mired,
    CCT_to_uv_Robertson1968,
    mired_to_CCT,
    uv_to_CCT_Robertson1968,
)

__all__ = [
    "CCT_INVERSION_GRID_SAMPLES",
    "solve_CCT_Newton",
    "solve_xy_Newton",
    "x0_CCT_grid",
]
__all__ += [
    "CCT_to_xy_CIE_D",
    "xy_to_CCT_CIE_D",
]
__all__ += [
    "CCT_to_xy_Hernandez1999",
    "xy_to_CCT_Hernandez1999",
]
__all__ += [
    "CCT_to_xy_Kang2002",
    "xy_to_CCT_Kang2002",
]
__all__ += [
    "CCT_to_uv_Krystek1985",
    "uv_to_CCT_Krystek1985",
]
__all__ += [
    "CCT_to_xy_McCamy1992",
    "xy_to_CCT_McCamy1992",
]
__all__ += [
    "CCT_to_uv_Planck1900",
    "uv_to_CCT_Planck1900",
]
__all__ += [
    "CCT_to_uv_Ohno2013",
    "CCT_to_XYZ_Ohno2013",
    "XYZ_to_CCT_Ohno2013",
    "uv_to_CCT_Ohno2013",
]
__all__ += [
    "CCT_to_mired",
    "CCT_to_uv_Robertson1968",
    "mired_to_CCT",
    "uv_to_CCT_Robertson1968",
]

UV_TO_CCT_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Krystek 1985": uv_to_CCT_Krystek1985,
        "Ohno 2013": uv_to_CCT_Ohno2013,
        "Planck 1900": uv_to_CCT_Planck1900,
        "Robertson 1968": uv_to_CCT_Robertson1968,
    }
)
UV_TO_CCT_METHODS.__doc__ = """
Supported *CIE UCS* colourspace *uv* chromaticity coordinates to correlated
colour temperature :math:`T_{cp}` computation methods.

References
----------
:cite:`AdobeSystems2013`,
:cite:`CIETC1-482004i`, :cite:`Krystek1985b`, :cite:`Ohno2014a`,
:cite:`Wyszecki2000y`

Aliases:

-   'ohno2013': 'Ohno 2013'
-   'robertson1968': 'Robertson 1968'
"""
UV_TO_CCT_METHODS["ohno2013"] = UV_TO_CCT_METHODS["Ohno 2013"]
UV_TO_CCT_METHODS["robertson1968"] = UV_TO_CCT_METHODS["Robertson 1968"]


def uv_to_CCT(
    uv: ArrayLike,
    method: (
        Literal["Krystek 1985", "Ohno 2013", "Planck 1900", "Robertson 1968"] | str
    ) = "Ohno 2013",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` and optionally
    :math:`\\Delta_{uv}` from the specified *CIE UCS* colourspace *uv*
    chromaticity coordinates using the specified method.

    Parameters
    ----------
    uv
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    method
        Computation method.

    Other Parameters
    ----------------
    cmfs
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`,
        :func:`colour.temperature.uv_to_CCT_Planck1900`},
        Standard observer colour matching functions.
    count
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Temperatures count in the planckian tables.
    end
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Temperature range end in kelvins.
    iterations
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Number of planckian tables to generate.
    optimisation_kwargs
        {:func:`colour.temperature.uv_to_CCT_Krystek1985`,
        :func:`colour.temperature.uv_to_CCT_Planck1900`},
        Inversion parameters forwarded to
        :func:`colour.temperature.x0_CCT_grid` and
        :func:`colour.temperature.solve_CCT_Newton`.
    start
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Temperature range start in kelvins.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`. The *Ohno (2013)* and
        *Robertson (1968)* methods also return :math:`\\Delta_{uv}` as the last
        dimension of the result.

    Notes
    -----
    -   The *Krystek (1985)* and *Planck (1900)* methods describe the
        Planckian locus and do not compute :math:`\\Delta_{uv}`.

    References
    ----------
    :cite:`AdobeSystems2013`,
    :cite:`CIETC1-482004i`, :cite:`Krystek1985b`, :cite:`Ohno2014a`,
    :cite:`Wyszecki2000y`

    Examples
    --------
    >>> import numpy as np
    >>> uv = np.array([0.1978, 0.3122])
    >>> uv_to_CCT(uv)  # doctest: +ELLIPSIS
    array([6.5074747...e+03, 3.2233463...e-03])
    """

    method = validate_method(method, tuple(UV_TO_CCT_METHODS))

    function = UV_TO_CCT_METHODS[method]

    return function(uv, **filter_kwargs(function, **kwargs))


CCT_TO_UV_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Krystek 1985": CCT_to_uv_Krystek1985,
        "Ohno 2013": CCT_to_uv_Ohno2013,
        "Planck 1900": CCT_to_uv_Planck1900,
        "Robertson 1968": CCT_to_uv_Robertson1968,
    }
)
CCT_TO_UV_METHODS.__doc__ = """
Supported correlated colour temperature :math:`T_{cp}` to *CIE UCS* colourspace
*uv* chromaticity coordinates computation methods.

References
----------
:cite:`AdobeSystems2013`,
:cite:`CIETC1-482004i`, :cite:`Krystek1985b`, :cite:`Ohno2014a`,
:cite:`Wyszecki2000y`

Aliases:

-   'ohno2013': 'Ohno 2013'
-   'robertson1968': 'Robertson 1968'
"""
CCT_TO_UV_METHODS["ohno2013"] = CCT_TO_UV_METHODS["Ohno 2013"]
CCT_TO_UV_METHODS["robertson1968"] = CCT_TO_UV_METHODS["Robertson 1968"]


def CCT_to_uv(
    CCT_D_uv: ArrayLike,
    method: (
        Literal["Krystek 1985", "Ohno 2013", "Planck 1900", "Robertson 1968"] | str
    ) = "Ohno 2013",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute the *CIE UCS* colourspace *uv* chromaticity coordinates from the
    specified correlated colour temperature :math:`T_{cp}` and optionally
    :math:`\\Delta_{uv}` using the specified method.

    Parameters
    ----------
    CCT_D_uv
        Correlated colour temperature :math:`T_{cp}`, or correlated colour
        temperature and :math:`\\Delta_{uv}`, depending on ``method``. The
        *Krystek (1985)* and *Planck (1900)* methods expect only
        :math:`T_{cp}`. The *Ohno (2013)* and *Robertson (1968)* methods expect
        final-axis :math:`[T_{cp}, \\Delta_{uv}]` pairs.
    method
        Computation method.

    Other Parameters
    ----------------
    cmfs
        {:func:`colour.temperature.CCT_to_uv_Ohno2013`,
        :func:`colour.temperature.CCT_to_uv_Planck1900`},
        Standard observer colour matching functions.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Notes
    -----
    -   For the *Krystek (1985)* and *Planck (1900)* methods, every element
        of ``CCT_D_uv`` is interpreted as an independent correlated colour
        temperature. A final axis of length two is not interpreted as a
        :math:`[T_{cp}, \\Delta_{uv}]` pair for those methods.

    References
    ----------
    :cite:`AdobeSystems2013`,
    :cite:`CIETC1-482004i`, :cite:`Krystek1985b`, :cite:`Ohno2014a`,
    :cite:`Wyszecki2000y`

    Examples
    --------
    >>> import numpy as np
    >>> CCT_D_uv = np.array([6507.47380460, 0.00322335])
    >>> CCT_to_uv(CCT_D_uv)  # doctest: +ELLIPSIS
    array([0.1977999..., 0.3121999...])
    """

    method = validate_method(method, tuple(CCT_TO_UV_METHODS))

    function = CCT_TO_UV_METHODS[method]

    return function(CCT_D_uv, **filter_kwargs(function, **kwargs))


__all__ += [
    "UV_TO_CCT_METHODS",
    "uv_to_CCT",
]
__all__ += [
    "CCT_TO_UV_METHODS",
    "CCT_to_uv",
]

XY_TO_CCT_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "CIE Illuminant D Series": xy_to_CCT_CIE_D,
        "Hernandez 1999": xy_to_CCT_Hernandez1999,
        "Kang 2002": xy_to_CCT_Kang2002,
        "McCamy 1992": xy_to_CCT_McCamy1992,
    }
)
XY_TO_CCT_METHODS.__doc__ = """
Supported *CIE xy* chromaticity coordinates to correlated colour temperature
:math:`T_{cp}` computation methods.

References
----------
:cite:`Hernandez-Andres1999a`, :cite:`Kang2002a`, :cite:`Wikipedia2001`,
:cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

Aliases:

-   'daylight': 'CIE Illuminant D Series'
-   'kang2002': 'Kang 2002'
-   'mccamy1992': 'McCamy 1992'
-   'hernandez1999': 'Hernandez 1999'
"""
XY_TO_CCT_METHODS["daylight"] = XY_TO_CCT_METHODS["CIE Illuminant D Series"]
XY_TO_CCT_METHODS["kang2002"] = XY_TO_CCT_METHODS["Kang 2002"]
XY_TO_CCT_METHODS["mccamy1992"] = XY_TO_CCT_METHODS["McCamy 1992"]
XY_TO_CCT_METHODS["hernandez1999"] = XY_TO_CCT_METHODS["Hernandez 1999"]


def xy_to_CCT(
    xy: ArrayLike,
    method: (
        Literal[
            "CIE Illuminant D Series",
            "Kang 2002",
            "Hernandez 1999",
            "McCamy 1992",
        ]
        | str
    ) = "CIE Illuminant D Series",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute the correlated colour temperature :math:`T_{cp}` from the
    specified *CIE xy* chromaticity coordinates using the specified method.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates.
    method
        Computation method.

    Other Parameters
    ----------------
    optimisation_kwargs
        {:func:`colour.temperature.xy_to_CCT_CIE_D`,
        :func:`colour.temperature.xy_to_CCT_Kang2002`},
        Inversion parameters forwarded to
        :func:`colour.temperature.x0_CCT_grid` and
        :func:`colour.temperature.solve_CCT_Newton`.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    :cite:`Hernandez-Andres1999a`, :cite:`Kang2002a`, :cite:`Wikipedia2001`,
    :cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

    Examples
    --------
    >>> import numpy as np
    >>> xy_to_CCT(np.array([0.31270, 0.32900]))  # doctest: +ELLIPSIS
    np.float64(6508.117542...)
    >>> xy_to_CCT(np.array([0.31270, 0.32900]), "Hernandez 1999")
    ... # doctest: +ELLIPSIS
    np.float64(6500.7420431...)
    """

    method = validate_method(method, tuple(XY_TO_CCT_METHODS))

    function = XY_TO_CCT_METHODS[method]

    return function(xy, **filter_kwargs(function, **kwargs))


CCT_TO_XY_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "CIE Illuminant D Series": CCT_to_xy_CIE_D,
        "Hernandez 1999": CCT_to_xy_Hernandez1999,
        "Kang 2002": CCT_to_xy_Kang2002,
        "McCamy 1992": CCT_to_xy_McCamy1992,
    }
)
CCT_TO_XY_METHODS.__doc__ = """
Supported correlated colour temperature :math:`T_{cp}` to *CIE xy* chromaticity
coordinates computation methods.

References
----------
:cite:`Hernandez-Andres1999a`, :cite:`Kang2002a`, :cite:`Wikipedia2001`,
:cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

Aliases:

-   'daylight': 'CIE Illuminant D Series'
-   'kang2002': 'Kang 2002'
-   'mccamy1992': 'McCamy 1992'
-   'hernandez1999': 'Hernandez 1999'
"""
CCT_TO_XY_METHODS["daylight"] = CCT_TO_XY_METHODS["CIE Illuminant D Series"]
CCT_TO_XY_METHODS["kang2002"] = CCT_TO_XY_METHODS["Kang 2002"]
CCT_TO_XY_METHODS["mccamy1992"] = CCT_TO_XY_METHODS["McCamy 1992"]
CCT_TO_XY_METHODS["hernandez1999"] = CCT_TO_XY_METHODS["Hernandez 1999"]


def CCT_to_xy(
    CCT: ArrayLike,
    method: (
        Literal[
            "CIE Illuminant D Series",
            "Kang 2002",
            "Hernandez 1999",
            "McCamy 1992",
        ]
        | str
    ) = "CIE Illuminant D Series",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute the *CIE xy* chromaticity coordinates from the specified
    correlated colour temperature :math:`T_{cp}` using the specified method.

    Parameters
    ----------
    CCT
        Correlated colour temperature :math:`T_{cp}`.
    method
        Computation method.

    Other Parameters
    ----------------
    optimisation_kwargs
        {:func:`colour.temperature.CCT_to_xy_Hernandez1999`,
        :func:`colour.temperature.CCT_to_xy_McCamy1992`},
        Inversion parameters forwarded to
        :func:`colour.temperature.x0_CCT_grid` and
        :func:`colour.temperature.solve_xy_Newton`.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Hernandez-Andres1999a`, :cite:`Kang2002a`, :cite:`Wikipedia2001`,
    :cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

    Examples
    --------
    >>> CCT_to_xy(6504.38938305)  # doctest: +ELLIPSIS
    array([0.3127077..., 0.3291128...])
    >>> CCT_to_xy(6504.38938305, "Kang 2002")
    ... # doctest: +ELLIPSIS
    array([0.313426..., 0.3235959...])
    """

    method = validate_method(method, tuple(CCT_TO_XY_METHODS))

    function = CCT_TO_XY_METHODS[method]

    return function(CCT, **filter_kwargs(function, **kwargs))


__all__ += [
    "XY_TO_CCT_METHODS",
    "xy_to_CCT",
]
__all__ += [
    "CCT_TO_XY_METHODS",
    "CCT_to_xy",
]
