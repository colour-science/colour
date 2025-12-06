"""
sUCS Colourspace
================

Define the *sUCS* colourspace transformations.

-   :func:`colour.XYZ_to_sUCS`
-   :func:`colour.sUCS_to_XYZ`
-   :func:`colour.sUCS_chroma`
-   :func:`colour.sUCS_hue_angle`

The *sUCS* (Simple Uniform Colour Space) is designed for simplicity and
perceptual uniformity. This implementation is based on the work by
*Li & Luo (2024)*.

References
----------
-   :cite:`Li2024` : Li, M., & Luo, M. R. (2024). Simple color appearance model
    (sCAM) based on simple uniform color space (sUCS). Optics Express, 32(3),
    3100. doi:10.1364/OE.510196
"""

from __future__ import annotations

from functools import partial

import numpy as np

from colour.algebra import spow
from colour.hints import (  # noqa: TC001
    Domain1,
    Domain100,
    Domain100_100_360,
    NDArrayFloat,
    Range1,
    Range100,
    Range100_100_360,
    Range360,
)
from colour.models import Iab_to_XYZ, XYZ_to_Iab
from colour.utilities import (
    as_float,
    domain_range_scale,
    from_range_1,
    from_range_100,
    from_range_degrees,
    to_domain_1,
    to_domain_100,
    to_domain_degrees,
    tsplit,
    tstack,
)

__author__ = "UltraMo114(Molin Li), Colour Developers"
__copyright__ = "Copyright 2024 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_SUCS_XYZ_TO_LMS",
    "MATRIX_SUCS_LMS_TO_XYZ",
    "MATRIX_SUCS_LMS_P_TO_IAB",
    "MATRIX_SUCS_IAB_TO_LMS_P",
    "XYZ_to_sUCS",
    "sUCS_to_XYZ",
    "sUCS_chroma",
    "sUCS_hue_angle",
    "sUCS_Iab_to_sUCS_ICh",
    "sUCS_ICh_to_sUCS_Iab",
]

MATRIX_SUCS_XYZ_TO_LMS: NDArrayFloat = np.array(
    [
        [0.4002, 0.7075, -0.0807],
        [-0.2280, 1.1500, 0.0612],
        [0.0000, 0.0000, 0.9184],
    ]
)
"""
*CIE XYZ* tristimulus values (*CIE Standard Illuminant D Series* *D65*-adapted,
Y=1 for white) to LMS-like cone responses matrix.
"""

MATRIX_SUCS_LMS_TO_XYZ: NDArrayFloat = np.linalg.inv(MATRIX_SUCS_XYZ_TO_LMS)
"""
LMS-like cone responses to *CIE XYZ* tristimulus values
(*CIE Standard Illuminant D Series* *D65*-adapted, Y=1 for white) matrix.
"""

MATRIX_SUCS_LMS_P_TO_IAB: NDArrayFloat = np.array(
    [
        [200.0 / 3.05, 100.0 / 3.05, 5.0 / 3.05],
        [430.0, -470.0, 40.0],
        [49.0, 49.0, -98.0],
    ]
)
"""
Non-linear LMS-like responses :math:`LMS_p` to intermediate :math:`Iab`
colourspace matrix.
"""

MATRIX_SUCS_IAB_TO_LMS_P: NDArrayFloat = np.linalg.inv(MATRIX_SUCS_LMS_P_TO_IAB)
"""
Intermediate :math:`Iab` colourspace to non-linear LMS-like responses
:math:`LMS_p` matrix.
"""


def XYZ_to_sUCS(XYZ: Domain1) -> Range100:
    """
    Convert from *CIE XYZ* tristimulus values to *sUCS* colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values, adapted to
        *CIE Standard Illuminant D65* and in domain [0, 1] (where white
        :math:`Y` is 1.0).

    Returns
    -------
    :class:`numpy.ndarray`
        *sUCS* :math:`Iab` colourspace array.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | 1                     | 1               |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``Iab``    | 100                   | 1                |
    +------------+-----------------------+------------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_sUCS(XYZ)  # doctest: +ELLIPSIS
    array([ 42.6292365...,  36.9764683...,  14.1230135...])
    """

    XYZ = to_domain_1(XYZ)

    with domain_range_scale("ignore"):
        Iab = XYZ_to_Iab(
            XYZ,
            partial(spow, p=0.43),
            MATRIX_SUCS_XYZ_TO_LMS,
            MATRIX_SUCS_LMS_P_TO_IAB,
        )

    return from_range_100(Iab)


def sUCS_to_XYZ(Iab: Domain100) -> Range1:
    """
    Convert from *sUCS* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Iab
        *sUCS* :math:`Iab` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values, adapted to
        *CIE Standard Illuminant D65* and in domain [0, 1] (where white
        :math:`Y` is 1.0).

    Notes
    -----
    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``Iab``    | 100                   | 1                |
    +------------+-----------------------+------------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | 1                     | 1               |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> Iab = np.array([42.62923653, 36.97646831, 14.12301358])
    >>> sUCS_to_XYZ(Iab)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    Iab = to_domain_100(Iab)

    with domain_range_scale("ignore"):
        XYZ = Iab_to_XYZ(
            Iab,
            partial(spow, p=1 / 0.43),
            MATRIX_SUCS_IAB_TO_LMS_P,
            MATRIX_SUCS_LMS_TO_XYZ,
        )

    return from_range_1(XYZ)


def sUCS_chroma(Iab: Domain100) -> Range100:
    """
    Compute the chroma component from the *sUCS* colourspace.

    Parameters
    ----------
    Iab
        *sUCS* :math:`Iab` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        Chroma component.

    Notes
    -----
    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``Iab``    | 100                   | 1                |
    +------------+-----------------------+------------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``C``      | 100                   | 1               |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> Iab = np.array([42.62923653, 36.97646831, 14.12301358])
    >>> sUCS_chroma(Iab)  # doctest: +ELLIPSIS
    40.4205110...
    """

    _I, a, b = tsplit(to_domain_100(Iab))

    C = 1 / 0.0252 * np.log(1 + 0.0447 * np.hypot(a, b))

    return as_float(from_range_100(C))


def sUCS_hue_angle(Iab: Domain100) -> Range360:
    """
    Compute the hue angle in degrees from the *sUCS* colourspace.

    Parameters
    ----------
    Iab
        *sUCS* :math:`Iab` colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        Hue angle in degrees.

    Notes
    -----
    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``Iab``    | 100                   | 1                |
    +------------+-----------------------+------------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``hue``    | 360                   | 1               |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> Iab = np.array([42.62923653, 36.97646831, 14.12301358])
    >>> sUCS_hue_angle(Iab)  # doctest: +ELLIPSIS
    20.9041560...
    """

    _I, a, b = tsplit(to_domain_100(Iab))

    h = np.degrees(np.arctan2(b, a)) % 360

    return as_float(from_range_degrees(h))


def sUCS_Iab_to_sUCS_ICh(
    Iab: Domain100,
) -> Range100_100_360:
    """
    Convert from *sUCS* :math:`Iab` rectangular coordinates to *sUCS*
    :math:`ICh` cylindrical coordinates.

    Parameters
    ----------
    Iab
        *sUCS* :math:`Iab` rectangular coordinates array.

    Returns
    -------
    :class:`numpy.ndarray`
        *sUCS* :math:`ICh` cylindrical coordinates array.

    Notes
    -----
    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``Iab``    | 100                   | 1                |
    +------------+-----------------------+------------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``ICh``    | ``I`` : 100           | ``I`` : 1        |
    |            |                       |                  |
    |            | ``C`` : 100           | ``C`` : 1        |
    |            |                       |                  |
    |            | ``h`` : 360           | ``h`` : 1        |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> Iab = np.array([42.62923653, 36.97646831, 14.12301358])
    >>> sUCS_Iab_to_sUCS_ICh(Iab)  # doctest: +ELLIPSIS
    array([ 42.6292365...,  40.4205110...,  20.9041560...])
    """

    I, a, b = tsplit(to_domain_100(Iab))  # noqa: E741

    C = 1 / 0.0252 * np.log(1 + 0.0447 * np.hypot(a, b))

    h = np.degrees(np.arctan2(b, a)) % 360

    return tstack([from_range_100(I), from_range_100(C), from_range_degrees(h)])


def sUCS_ICh_to_sUCS_Iab(
    ICh: Domain100_100_360,
) -> Range100:
    """
    Convert from *sUCS* :math:`ICh` cylindrical coordinates to *sUCS*
    :math:`Iab` rectangular coordinates.

    Parameters
    ----------
    ICh
        *sUCS* :math:`ICh` cylindrical coordinates array.

    Returns
    -------
    :class:`numpy.ndarray`
        *sUCS* :math:`Iab` rectangular coordinates array.

    Notes
    -----
    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``ICh``    | ``I`` : 100           | ``I`` : 1        |
    |            |                       |                  |
    |            | ``C`` : 100           | ``C`` : 1        |
    |            |                       |                  |
    |            | ``h`` : 360           | ``h`` : 1        |
    +------------+-----------------------+------------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``Iab``    | 100                   | 1                |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`Li2024`

    Examples
    --------
    >>> ICh = np.array([42.62923653, 40.42051103, 20.90415604])
    >>> sUCS_ICh_to_sUCS_Iab(ICh)  # doctest: +ELLIPSIS
    array([ 42.6292365...,  36.9764682...,  14.1230135...])
    """

    I, C, h = tsplit(ICh)  # noqa: E741
    I = to_domain_100(I)  # noqa: E741
    C = to_domain_100(C)
    h = to_domain_degrees(h)

    C = (np.exp(0.0252 * C) - 1) / 0.0447

    a = C * np.cos(np.radians(h))
    b = C * np.sin(np.radians(h))

    return from_range_100(tstack([I, a, b]))
