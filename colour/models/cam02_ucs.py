"""
CAM02-LCD, CAM02-SCD, and CAM02-UCS Colourspaces - Luo, Cui and Li (2006)
=========================================================================

Define the *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, and *CAM02-UCS*
colourspaces transformations.

-   :func:`colour.JMh_CIECAM02_to_CAM02LCD`
-   :func:`colour.CAM02LCD_to_JMh_CIECAM02`
-   :func:`colour.JMh_CIECAM02_to_CAM02SCD`
-   :func:`colour.CAM02SCD_to_JMh_CIECAM02`
-   :func:`colour.JMh_CIECAM02_to_CAM02UCS`
-   :func:`colour.CAM02UCS_to_JMh_CIECAM02`
-   :func:`colour.XYZ_to_CAM02LCD`
-   :func:`colour.CAM02LCD_to_XYZ`
-   :func:`colour.XYZ_to_CAM02SCD`
-   :func:`colour.CAM02SCD_to_XYZ`
-   :func:`colour.XYZ_to_CAM02UCS`
-   :func:`colour.CAM02UCS_to_XYZ`

References
----------
-   :cite:`Luo2006b` : Luo, M. Ronnier, Cui, G., & Li, C. (2006). Uniform
    colour spaces based on CIECAM02 colour appearance model. Color Research &
    Application, 31(4), 320-330. doi:10.1002/col.20227
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from colour.algebra import cartesian_to_polar, polar_to_cartesian

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        Domain1,
        Domain100,
        Domain100_100_360,
        Range1,
        Range100,
        Range100_100_360,
    )

from colour.hints import NDArrayFloat, cast
from colour.utilities import (
    CanonicalMapping,
    MixinDataclassIterable,
    as_float_array,
    from_range_100,
    from_range_degrees,
    get_domain_range_scale,
    to_domain_100,
    to_domain_degrees,
    tsplit,
    tstack,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Coefficients_UCS_Luo2006",
    "COEFFICIENTS_UCS_LUO2006",
    "JMh_CIECAM02_to_UCS_Luo2006",
    "UCS_Luo2006_to_JMh_CIECAM02",
    "JMh_CIECAM02_to_CAM02LCD",
    "CAM02LCD_to_JMh_CIECAM02",
    "JMh_CIECAM02_to_CAM02SCD",
    "CAM02SCD_to_JMh_CIECAM02",
    "JMh_CIECAM02_to_CAM02UCS",
    "CAM02UCS_to_JMh_CIECAM02",
    "XYZ_to_UCS_Luo2006",
    "UCS_Luo2006_to_XYZ",
    "XYZ_to_CAM02LCD",
    "CAM02LCD_to_XYZ",
    "XYZ_to_CAM02SCD",
    "CAM02SCD_to_XYZ",
    "XYZ_to_CAM02UCS",
    "CAM02UCS_to_XYZ",
]


@dataclass(frozen=True)
class Coefficients_UCS_Luo2006(MixinDataclassIterable):
    """
    Store *Luo et al. (2006)* fitting coefficients for the *CAM02-LCD*,
    *CAM02-SCD*, and *CAM02-UCS* colourspaces.

    Attributes
    ----------
    K_L
        Lightness coefficient.
    c_1
        First chroma coefficient.
    c_2
        Second chroma coefficient.
    """

    K_L: float
    c_1: float
    c_2: float


COEFFICIENTS_UCS_LUO2006: CanonicalMapping = CanonicalMapping(
    {
        "CAM02-LCD": Coefficients_UCS_Luo2006(0.77, 0.007, 0.0053),
        "CAM02-SCD": Coefficients_UCS_Luo2006(1.24, 0.007, 0.0363),
        "CAM02-UCS": Coefficients_UCS_Luo2006(1.00, 0.007, 0.0228),
    }
)
"""
*Luo et al. (2006)* fitting coefficients for the *CAM02-LCD*, *CAM02-SCD*, and
*CAM02-UCS* colourspaces.
"""


def JMh_CIECAM02_to_UCS_Luo2006(
    JMh: Domain100_100_360, coefficients: Coefficients_UCS_Luo2006
) -> Range100:
    """
    Convert from *CIECAM02* :math:`JMh` correlates array to one of the
    *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
    colourspaces :math:`J'a'b'` array.

    The :math:`JMh` correlates array is constructed using the *CIECAM02*
    correlate of *Lightness* :math:`J`, the *CIECAM02* correlate of
    *colourfulness* :math:`M` and the *CIECAM02* *Hue* angle :math:`h`
    in degrees.

    Parameters
    ----------
    JMh
        *CIECAM02* correlates array :math:`JMh`.
    coefficients
        Coefficients of one of the *Luo et al. (2006)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
        colourspaces :math:`J'a'b'` array.

    Notes
    -----
    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``JMh``    | ``J`` : 100            | ``J`` : 1        |
    |            |                        |                  |
    |            | ``M`` : 100            | ``M`` : 1        |
    |            |                        |                  |
    |            | ``h`` : 360            | ``h`` : 1        |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    Examples
    --------
    >>> from colour.appearance import (
    ...     VIEWING_CONDITIONS_CIECAM02,
    ...     XYZ_to_CIECAM02,
    ... )
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_CIECAM02["Average"]
    >>> specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
    ... # doctest: +ELLIPSIS
    array([ 54.9043313...,  -0.0845039...,  -0.0685483...])
    """

    J, M, h = tsplit(JMh)
    J = to_domain_100(J)
    M = to_domain_100(M)
    h = to_domain_degrees(h)

    _K_L, c_1, c_2 = coefficients.values

    J_p = ((1 + 100 * c_1) * J) / (1 + c_1 * J)
    M_p = (1 / c_2) * np.log1p(c_2 * M)

    a_p, b_p = tsplit(polar_to_cartesian(tstack([M_p, np.radians(h)])))

    Jpapbp = tstack([J_p, a_p, b_p])

    return from_range_100(Jpapbp)


def UCS_Luo2006_to_JMh_CIECAM02(
    Jpapbp: Domain100, coefficients: Coefficients_UCS_Luo2006
) -> Range100_100_360:
    """
    Convert from one of the *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*,
    or *CAM02-UCS* colourspaces :math:`J'a'b'` array to *CIECAM02*
    :math:`JMh` correlates array.

    Parameters
    ----------
    Jpapbp
        *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
        colourspaces :math:`J'a'b'` array.
    coefficients
        Coefficients of one of the *Luo et al. (2006)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIECAM02* correlates array :math:`JMh`.

    Notes
    -----
    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``JMh``    | ``J`` : 100            | ``J`` : 1        |
    |            |                        |                  |
    |            | ``M`` : 100            | ``M`` : 1        |
    |            |                        |                  |
    |            | ``h`` : 360            | ``h`` : 1        |
    +------------+------------------------+------------------+

    Examples
    --------
    >>> Jpapbp = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> UCS_Luo2006_to_JMh_CIECAM02(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
    ... # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    J_p, a_p, b_p = tsplit(to_domain_100(Jpapbp))
    _K_L, c_1, c_2 = coefficients.values

    J = -J_p / (c_1 * J_p - 1 - 100 * c_1)

    M_p, h = tsplit(cartesian_to_polar(tstack([a_p, b_p])))

    M = np.expm1(M_p / (1 / c_2)) / c_2

    return tstack(
        [
            from_range_100(J),
            from_range_100(M),
            from_range_degrees(np.degrees(h) % 360),
        ]
    )


def JMh_CIECAM02_to_CAM02LCD(
    JMh: Domain100_100_360,
) -> Range100:
    """
    Convert from *CIECAM02* :math:`JMh` correlates array to *Luo et al.
    (2006)* *CAM02-LCD* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    JMh
        *CIECAM02* correlates array :math:`JMh`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'` array.

    Notes
    -----
    -   *LCD* in *CAM02-LCD* stands for *Large Colour Differences*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``JMh``    | ``J`` : 100            | ``J`` : 1        |
    |            |                        |                  |
    |            | ``M`` : 100            | ``M`` : 1        |
    |            |                        |                  |
    |            | ``h`` : 360            | ``h`` : 1        |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> from colour.appearance import (
    ...     VIEWING_CONDITIONS_CIECAM02,
    ...     XYZ_to_CIECAM02,
    ... )
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_CIECAM02["Average"]
    >>> specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> JMh_CIECAM02_to_CAM02LCD(JMh)  # doctest: +ELLIPSIS
    array([ 54.9043313...,  -0.0845039...,  -0.0685483...])
    """

    return JMh_CIECAM02_to_UCS_Luo2006(
        JMh, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]
    )


def CAM02LCD_to_JMh_CIECAM02(
    Jpapbp: Domain100,
) -> Range100_100_360:
    """
    Convert from *Luo et al. (2006)* *CAM02-LCD* colourspace
    :math:`J'a'b'` array to *CIECAM02* :math:`JMh` correlates array.

    Parameters
    ----------
    Jpapbp
        *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'` array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIECAM02* correlates array :math:`JMh`.

    Notes
    -----
    -   *LCD* in *CAM02-LCD* stands for *Large Colour Differences*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``JMh``    | ``J`` : 100            | ``J`` : 1        |
    |            |                        |                  |
    |            | ``M`` : 100            | ``M`` : 1        |
    |            |                        |                  |
    |            | ``h`` : 360            | ``h`` : 1        |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> CAM02LCD_to_JMh_CIECAM02(Jpapbp)  # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    return UCS_Luo2006_to_JMh_CIECAM02(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-LCD"]
    )


def JMh_CIECAM02_to_CAM02SCD(
    JMh: Domain100_100_360,
) -> Range100:
    """
    Convert from *CIECAM02* :math:`JMh` correlates array to *Luo et al.
    (2006)* *CAM02-SCD* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    JMh
        *CIECAM02* correlates array :math:`JMh`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'` array.

    Notes
    -----
    -   *SCD* in *CAM02-SCD* stands for *Small Colour Differences*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``JMh``    | ``J`` : 100            | ``J`` : 1        |
    |            |                        |                  |
    |            | ``M`` : 100            | ``M`` : 1        |
    |            |                        |                  |
    |            | ``h`` : 360            | ``h`` : 1        |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> from colour.appearance import (
    ...     VIEWING_CONDITIONS_CIECAM02,
    ...     XYZ_to_CIECAM02,
    ... )
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_CIECAM02["Average"]
    >>> specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> JMh_CIECAM02_to_CAM02SCD(JMh)  # doctest: +ELLIPSIS
    array([ 54.9043313...,  -0.0843617...,  -0.0684329...])
    """

    return JMh_CIECAM02_to_UCS_Luo2006(
        JMh, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-SCD"]
    )


def CAM02SCD_to_JMh_CIECAM02(
    Jpapbp: Domain100,
) -> Range100_100_360:
    """
    Convert from *Luo et al. (2006)* *CAM02-SCD* colourspace
    :math:`J'a'b'` array to *CIECAM02* :math:`JMh` correlates array.

    Parameters
    ----------
    Jpapbp
        *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'`
        array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIECAM02* correlates array :math:`JMh`.

    Notes
    -----
    -   *SCD* in *CAM02-SCD* stands for *Small Colour Differences*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``JMh``    | ``J`` : 100            | ``J`` : 1        |
    |            |                        |                  |
    |            | ``M`` : 100            | ``M`` : 1        |
    |            |                        |                  |
    |            | ``h`` : 360            | ``h`` : 1        |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([54.90433134, -0.08436178, -0.06843298])
    >>> CAM02SCD_to_JMh_CIECAM02(Jpapbp)  # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    return UCS_Luo2006_to_JMh_CIECAM02(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-SCD"]
    )


def JMh_CIECAM02_to_CAM02UCS(
    JMh: Domain100_100_360,
) -> Range100:
    """
    Convert from *CIECAM02* :math:`JMh` correlates array to *Luo et al.
    (2006)* *CAM02-UCS* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    JMh
        *CIECAM02* correlates array :math:`JMh`.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'` array.

    Notes
    -----
    -   *UCS* in *CAM02-UCS* stands for *Uniform Colour Space*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``JMh``    | ``J`` : 100            | ``J`` : 1        |
    |            |                        |                  |
    |            | ``M`` : 100            | ``M`` : 1        |
    |            |                        |                  |
    |            | ``h`` : 360            | ``h`` : 1        |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> from colour.appearance import (
    ...     VIEWING_CONDITIONS_CIECAM02,
    ...     XYZ_to_CIECAM02,
    ... )
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = VIEWING_CONDITIONS_CIECAM02["Average"]
    >>> specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> JMh_CIECAM02_to_CAM02UCS(JMh)  # doctest: +ELLIPSIS
    array([ 54.9043313...,  -0.0844236...,  -0.0684831...])
    """

    return JMh_CIECAM02_to_UCS_Luo2006(
        JMh, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-UCS"]
    )


def CAM02UCS_to_JMh_CIECAM02(
    Jpapbp: Domain100,
) -> Range100_100_360:
    """
    Convert from *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'`
    array to *CIECAM02* :math:`JMh` correlates array.

    Parameters
    ----------
    Jpapbp
        *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'` array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIECAM02* correlates array :math:`JMh`.

    Notes
    -----
    -   *UCS* in *CAM02-UCS* stands for *Uniform Colour Space*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``JMh``    | ``J`` : 100            | ``J`` : 1        |
    |            |                        |                  |
    |            | ``M`` : 100            | ``M`` : 1        |
    |            |                        |                  |
    |            | ``h`` : 360            | ``h`` : 1        |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([54.90433134, -0.08442362, -0.06848314])
    >>> CAM02UCS_to_JMh_CIECAM02(Jpapbp)  # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    return UCS_Luo2006_to_JMh_CIECAM02(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-UCS"]
    )


def XYZ_to_UCS_Luo2006(
    XYZ: Domain1,
    coefficients: Coefficients_UCS_Luo2006,
    **kwargs: Any,
) -> Range100:
    """
    Convert from *CIE XYZ* tristimulus values to one of the
    *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
    colourspaces :math:`J'a'b'` array.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    coefficients
        Coefficients of one of the *Luo et al. (2006)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.XYZ_to_CIECAM02`},
        See the documentation of the previously listed definition. The
        default viewing conditions are those of *IEC 61966-2-1:1999*,
        i.e., *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
        adapting field luminance about 20% of a white object in the
        scene.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
        colourspaces :math:`J'a'b'` array.

    Warnings
    --------
    The ``XYZ_w`` parameter for :func:`colour.XYZ_to_CAM16` definition must
    be specified in the same domain-range scale than the ``XYZ`` parameter.

    Notes
    -----
    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``XYZ``    | 1                      | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_UCS_Luo2006(XYZ, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
    ... # doctest: +ELLIPSIS
    array([ 46.6138615...,  39.3576023...,  15.9673043...])
    """

    from colour.appearance import (  # noqa: PLC0415
        CAM_KWARGS_CIECAM02_sRGB,
        XYZ_to_CIECAM02,
    )

    domain_range_reference = get_domain_range_scale() == "reference"

    settings = CAM_KWARGS_CIECAM02_sRGB.copy()
    settings.update(**kwargs)
    XYZ_w = kwargs.get("XYZ_w")

    if XYZ_w is not None and domain_range_reference:
        settings["XYZ_w"] = as_float_array(XYZ_w) * 100

    if domain_range_reference:
        XYZ = as_float_array(XYZ) * 100

    specification = XYZ_to_CIECAM02(XYZ, **settings)
    JMh = tstack(
        [
            cast("NDArrayFloat", specification.J),
            cast("NDArrayFloat", specification.M),
            cast("NDArrayFloat", specification.h),
        ]
    )

    return JMh_CIECAM02_to_UCS_Luo2006(JMh, coefficients)


def UCS_Luo2006_to_XYZ(
    Jpapbp: Domain100,
    coefficients: Coefficients_UCS_Luo2006,
    **kwargs: Any,
) -> Range1:
    """
    Convert from one of the *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or
    *CAM02-UCS* colourspaces :math:`J'a'b'` array to *CIE XYZ* tristimulus
    values.

    Parameters
    ----------
    Jpapbp
        *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
        colourspaces :math:`J'a'b'` array.
    coefficients
        Coefficients of one of the *Luo et al. (2006)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.CIECAM02_to_XYZ`},
        See the documentation of the previously listed definition. The
        default viewing conditions are those of *IEC 61966-2-1:1999*,
        i.e., *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
        adapting field luminance about 20% of a white object in the
        scene.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Warnings
    --------
    The ``XYZ_w`` parameter for :func:`colour.XYZ_to_CAM16` definition must
    be specified in the same domain-range scale as the ``XYZ`` parameter.

    Notes
    -----
    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``XYZ``    | 1                      | 1                |
    +------------+------------------------+------------------+

    Examples
    --------
    >>> Jpapbp = np.array([46.61386154, 39.35760236, 15.96730435])
    >>> UCS_Luo2006_to_XYZ(Jpapbp, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
    ... # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    from colour.appearance import (  # noqa: PLC0415
        CAM_KWARGS_CIECAM02_sRGB,
        CAM_Specification_CIECAM02,
        CIECAM02_to_XYZ,
    )

    domain_range_reference = get_domain_range_scale() == "reference"

    settings = CAM_KWARGS_CIECAM02_sRGB.copy()
    settings.update(**kwargs)
    XYZ_w = kwargs.get("XYZ_w")

    if XYZ_w is not None and domain_range_reference:
        settings["XYZ_w"] = as_float_array(XYZ_w) * 100

    J, M, h = tsplit(UCS_Luo2006_to_JMh_CIECAM02(Jpapbp, coefficients))

    specification = CAM_Specification_CIECAM02(J=J, M=M, h=h)

    XYZ = CIECAM02_to_XYZ(specification, **settings)

    if domain_range_reference:
        XYZ /= 100

    return XYZ


def XYZ_to_CAM02LCD(XYZ: Domain1, **kwargs: Any) -> Range100:
    """
    Convert from *CIE XYZ* tristimulus values to *Luo et al. (2006)*
    *CAM02-LCD* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.XYZ_to_CIECAM02`},
        See the documentation of the previously listed definition. The
        default viewing conditions are those of *IEC 61966-2-1:1999*,
        i.e., *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
        adapting field luminance about 20% of a white object in the
        scene.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'`
        array.

    Warnings
    --------
    The ``XYZ_w`` parameter for :func:`colour.XYZ_to_CAM16` definition
    must be specified in the same domain-range scale as the ``XYZ``
    parameter.

    Notes
    -----
    -   *LCD* in *CAM02-LCD* stands for *Large Colour Differences*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``XYZ``    | 1                      | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_CAM02LCD(XYZ)  # doctest: +ELLIPSIS
    array([ 46.6138615...,  39.3576023...,  15.9673043...])
    """

    return XYZ_to_UCS_Luo2006(
        XYZ, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-LCD"], **kwargs
    )


def CAM02LCD_to_XYZ(Jpapbp: Domain100, **kwargs: Any) -> Range1:
    """
    Convert from *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'`
    array to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Jpapbp
        *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'` array.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.CIECAM02_to_XYZ`},
        See the documentation of the previously listed definition. The
        default viewing conditions are those of *IEC 61966-2-1:1999*,
        i.e., *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
        adapting field luminance about 20% of a white object in the
        scene.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Warnings
    --------
    The ``XYZ_w`` parameter for :func:`colour.XYZ_to_CAM16` definition must
    be specified in the same domain-range scale as the ``XYZ`` parameter.

    Notes
    -----
    -   *LCD* in *CAM02-LCD* stands for *Large Colour Differences*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``XYZ``    | 1                      | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([46.61386154, 39.35760236, 15.96730435])
    >>> CAM02LCD_to_XYZ(Jpapbp)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    return UCS_Luo2006_to_XYZ(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-LCD"], **kwargs
    )


def XYZ_to_CAM02SCD(XYZ: Domain1, **kwargs: Any) -> Range100:
    """
    Convert from *CIE XYZ* tristimulus values to *Luo et al. (2006)*
    *CAM02-SCD* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.XYZ_to_CIECAM02`},
        See the documentation of the previously listed definition. The
        default viewing conditions are those of *IEC 61966-2-1:1999*,
        i.e., *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
        adapting field luminance about 20% of a white object in the
        scene.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'`
        array.

    Warnings
    --------
    The ``XYZ_w`` parameter for :func:`colour.XYZ_to_CAM16` definition
    must be specified in the same domain-range scale as the ``XYZ``
    parameter.

    Notes
    -----
    -   *SCD* in *CAM02-SCD* stands for *Small Colour Differences*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``XYZ``    | 1                      | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_CAM02SCD(XYZ)  # doctest: +ELLIPSIS
    array([ 46.6138615...,  25.6287988...,  10.3975548...])
    """

    return XYZ_to_UCS_Luo2006(
        XYZ, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-SCD"], **kwargs
    )


def CAM02SCD_to_XYZ(Jpapbp: Domain100, **kwargs: Any) -> Range1:
    """
    Convert from *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'`
    array to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Jpapbp
        *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'` array.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.CIECAM02_to_XYZ`},
        See the documentation of the previously listed definition. The
        default viewing conditions are those of *IEC 61966-2-1:1999*,
        i.e., *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
        adapting field luminance about 20% of a white object in the
        scene.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Warnings
    --------
    The ``XYZ_w`` parameter for :func:`colour.XYZ_to_CAM16` definition must be
    specified in the same domain-range scale as the ``XYZ`` parameter.

    Notes
    -----
    -   *SCD* in *CAM02-SCD* stands for *Small Colour Differences*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``XYZ``    | 1                      | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([46.61386154, 25.62879882, 10.39755489])
    >>> CAM02SCD_to_XYZ(Jpapbp)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    return UCS_Luo2006_to_XYZ(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-SCD"], **kwargs
    )


def XYZ_to_CAM02UCS(XYZ: Domain1, **kwargs: Any) -> Range100:
    """
    Convert from *CIE XYZ* tristimulus values to *Luo et al. (2006)*
    *CAM02-UCS* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.XYZ_to_CIECAM02`},
        See the documentation of the previously listed definition. The
        default viewing conditions are those of *IEC 61966-2-1:1999*,
        i.e., *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
        adapting field luminance about 20% of a white object in the
        scene.

    Returns
    -------
    :class:`numpy.ndarray`
        *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'`
        array.

    Warnings
    --------
    The ``XYZ_w`` parameter for :func:`colour.XYZ_to_CAM16` definition
    must be specified in the same domain-range scale as the ``XYZ``
    parameter.

    Notes
    -----
    -   *UCS* in *CAM02-UCS* stands for *Uniform Colour Space*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``XYZ``    | 1                      | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_CAM02UCS(XYZ)  # doctest: +ELLIPSIS
    array([ 46.6138615...,  29.8831001...,  12.1235168...])
    """

    return XYZ_to_UCS_Luo2006(
        XYZ, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-UCS"], **kwargs
    )


def CAM02UCS_to_XYZ(Jpapbp: Domain100, **kwargs: Any) -> Range1:
    """
    Convert from *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'`
    array to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Jpapbp
        *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'` array.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.CIECAM02_to_XYZ`},
        See the documentation of the previously listed definition. The
        default viewing conditions are those of *IEC 61966-2-1:1999*,
        i.e., *sRGB* 64 Lux ambient illumination, 80 :math:`cd/m^2`,
        adapting field luminance about 20% of a white object in the
        scene.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Warnings
    --------
    The ``XYZ_w`` parameter for :func:`colour.XYZ_to_CAM16` definition must
    be specified in the same domain-range scale as the ``XYZ`` parameter.

    Notes
    -----
    -   *UCS* in *CAM02-UCS* stands for *Uniform Colour Space*.

    +------------+------------------------+------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``Jpapbp`` | 100                    | 1                |
    +------------+------------------------+------------------+

    +------------+------------------------+------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**    |
    +============+========================+==================+
    | ``XYZ``    | 1                      | 1                |
    +------------+------------------------+------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([46.61386154, 29.88310013, 12.12351683])
    >>> CAM02UCS_to_XYZ(Jpapbp)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    return UCS_Luo2006_to_XYZ(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006["CAM02-UCS"], **kwargs
    )
