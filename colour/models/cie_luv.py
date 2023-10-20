"""
CIE L*u*v* Colourspace
======================

Defines the *CIE L\\*u\\*v\\** colourspace transformations:

-   :func:`colour.XYZ_to_Luv`
-   :func:`colour.Luv_to_XYZ`
-   :func:`colour.Luv_to_uv`
-   :func:`colour.uv_to_Luv`
-   :func:`colour.Luv_uv_to_xy`
-   :func:`colour.xy_to_Luv_uv`
-   :func:`colour.Luv_to_LCHuv`
-   :func:`colour.LCHuv_to_Luv`

References
----------
-   :cite:`CIETC1-482004j` : CIE TC 1-48. (2004). CIE 1976 uniform
    chromaticity scale diagram (UCS diagram). In CIE 015:2004 Colorimetry, 3rd
    Edition (p. 24). ISBN:978-3-901906-33-6
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901906-33-6
-   :cite:`Wikipedia2007b` : Wikipedia. (2007). CIELUV. Retrieved February 24,
    2014, from http://en.wikipedia.org/wiki/CIELUV
-   :cite:`Wikipedia2007d` : Wikipedia. (2007). The reverse transformation.
    Retrieved February 24, 2014, from
    http://en.wikipedia.org/wiki/CIELUV#The_reverse_transformation
"""

from __future__ import annotations

from colour.algebra import sdiv, sdiv_mode
from colour.colorimetry import (
    CCS_ILLUMINANTS,
    lightness_CIE1976,
    luminance_CIE1976,
)
from colour.hints import ArrayLike, NDArrayFloat
from colour.models import Jab_to_JCh, JCh_to_Jab, xy_to_xyY, xyY_to_XYZ
from colour.utilities import (
    as_float_scalar,
    domain_range_scale,
    from_range_1,
    from_range_100,
    full,
    to_domain_1,
    to_domain_100,
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
    "XYZ_to_Luv",
    "Luv_to_XYZ",
    "Luv_to_uv",
    "uv_to_Luv",
    "Luv_uv_to_xy",
    "xy_to_Luv_uv",
    "Luv_to_LCHuv",
    "LCHuv_to_Luv",
]


def XYZ_to_Luv(
    XYZ: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
) -> NDArrayFloat:
    """
    Convert from *CIE XYZ* tristimulus values to *CIE L\\*u\\*v\\**
    colourspace.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE L\\*u\\*v\\** colourspace array.

    Notes
    -----
    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``XYZ``        | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+
    | ``illuminant`` | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

    +----------------+-----------------------+-----------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``Luv``        | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |                |                       |                 |
    |                | ``u`` : [-100, 100]   | ``u`` : [-1, 1] |
    |                |                       |                 |
    |                | ``v`` : [-100, 100]   | ``v`` : [-1, 1] |
    +----------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004m`, :cite:`Wikipedia2007b`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_Luv(XYZ)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  96.8362605...,  17.7521014...])
    """

    X, Y, Z = tsplit(to_domain_1(XYZ))

    X_r, Y_r, Z_r = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    with domain_range_scale("100"):
        L = lightness_CIE1976(Y, Y_r)

    X_Y_Z = X + 15 * Y + 3 * Z
    X_r_Y_r_Z_r = X_r + 15 * Y_r + 3 * Z_r

    with sdiv_mode():
        u = 13 * L * ((4 * sdiv(X, X_Y_Z)) - (4 * sdiv(X_r, X_r_Y_r_Z_r)))
        v = 13 * L * ((9 * sdiv(Y, X_Y_Z)) - (9 * sdiv(Y_r, X_r_Y_r_Z_r)))

    Luv = tstack([L, u, v])

    return from_range_100(Luv)


def Luv_to_XYZ(
    Luv: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
) -> NDArrayFloat:
    """
    Convert from *CIE L\\*u\\*v\\** colourspace to *CIE XYZ* tristimulus
    values.

    Parameters
    ----------
    Luv
        *CIE L\\*u\\*v\\** colourspace array.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----
    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``Luv``        | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |                |                       |                 |
    |                | ``u`` : [-100, 100]   | ``u`` : [-1, 1] |
    |                |                       |                 |
    |                | ``v`` : [-100, 100]   | ``v`` : [-1, 1] |
    +----------------+-----------------------+-----------------+
    | ``illuminant`` | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

    +----------------+-----------------------+-----------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``XYZ``        | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004m`, :cite:`Wikipedia2007b`

    Examples
    --------
    >>> import numpy as np
    >>> Luv = np.array([41.52787529, 96.83626054, 17.75210149])
    >>> Luv_to_XYZ(Luv)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    L, u, v = tsplit(to_domain_100(Luv))

    X_r, Y_r, Z_r = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    with domain_range_scale("100"):
        Y = luminance_CIE1976(L, Y_r)

    X_r_Y_r_Z_r = X_r + 15 * Y_r + 3 * Z_r

    with sdiv_mode():
        a_1 = u + 13 * L * (4 * sdiv(X_r, X_r_Y_r_Z_r))
        d_1 = v + 13 * L * (9 * sdiv(Y_r, X_r_Y_r_Z_r))

        a = 1 / 3 * (52 * sdiv(L, a_1) - 1)
        b = -5 * Y
        c = -1 / 3
        d = Y * (39 * sdiv(L, d_1) - 5)

        X = sdiv(d - b, a - c)

    Z = X * a + b

    XYZ = tstack([X, Y, Z])

    return from_range_1(XYZ)


def Luv_to_uv(
    Luv: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
) -> NDArrayFloat:
    """
    Return the :math:`uv^p` chromaticity coordinates from given
    *CIE L\\*u\\*v\\** colourspace array.

    Parameters
    ----------
    Luv
        *CIE L\\*u\\*v\\** colourspace array.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`uv^p` chromaticity coordinates.

    Notes
    -----
    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``Luv``        | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |                |                       |                 |
    |                | ``u`` : [-100, 100]   | ``u`` : [-1, 1] |
    |                |                       |                 |
    |                | ``v`` : [-100, 100]   | ``v`` : [-1, 1] |
    +----------------+-----------------------+-----------------+
    | ``illuminant`` | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004j`

    Examples
    --------
    >>> import numpy as np
    >>> Luv = np.array([41.52787529, 96.83626054, 17.75210149])
    >>> Luv_to_uv(Luv)  # doctest: +ELLIPSIS
    array([ 0.3772021...,  0.5012026...])
    """

    Luv = to_domain_100(Luv)

    X, Y, Z = tsplit(Luv_to_XYZ(Luv, illuminant))

    X_Y_Z = X + 15 * Y + 3 * Z

    with sdiv_mode():
        uv = tstack([4 * sdiv(X, X_Y_Z), 9 * sdiv(Y, X_Y_Z)])

    return uv


def uv_to_Luv(
    uv: ArrayLike,
    illuminant: ArrayLike = CCS_ILLUMINANTS[
        "CIE 1931 2 Degree Standard Observer"
    ]["D65"],
    Y: float = 1,
) -> NDArrayFloat:
    """
    Return the *CIE L\\*u\\*v\\** colourspace array from given :math:`uv^p`
    chromaticity coordinates by extending the array last dimension with given
    :math:`L` *Lightness*.

    Parameters
    ----------
    uv
        :math:`uv^p` chromaticity coordinates.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.
    Y
        Optional :math:`Y` *luminance* value used to construct the intermediate
        *CIE XYZ* colourspace array, the default :math:`Y` *luminance* value is
        1.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE L\\*u\\*v\\** colourspace array.

    Notes
    -----
    +----------------+-----------------------+-----------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``Luv``        | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |                |                       |                 |
    |                | ``u`` : [-100, 100]   | ``u`` : [-1, 1] |
    |                |                       |                 |
    |                | ``v`` : [-100, 100]   | ``v`` : [-1, 1] |
    +----------------+-----------------------+-----------------+
    | ``illuminant`` | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004j`

    Examples
    --------
    >>> import numpy as np
    >>> uv = np.array([0.37720213, 0.50120264])
    >>> uv_to_Luv(uv)  # doctest: +ELLIPSIS
    array([ 100.        ,  233.1837603...,   42.7474385...])
    """

    u, v = tsplit(uv)
    Y = as_float_scalar(to_domain_1(Y))

    with sdiv_mode():
        X = 9 * sdiv(u, 4 * v)
        Z = sdiv(-5 * Y * v - 3 * u / 4 + 3, v)

    XYZ = tstack([X, full(u.shape, Y), Z])

    return XYZ_to_Luv(from_range_1(XYZ), illuminant)


def Luv_uv_to_xy(uv: ArrayLike) -> NDArrayFloat:
    """
    Return the *CIE xy* chromaticity coordinates from given *CIE L\\*u\\*v\\**
    colourspace :math:`uv^p` chromaticity coordinates.

    Parameters
    ----------
    uv
        *CIE L\\*u\\*v\\* u"v"* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Wikipedia2007d`

    Examples
    --------
    >>> import numpy as np
    >>> uv = np.array([0.37720213, 0.50120264])
    >>> Luv_uv_to_xy(uv)  # doctest: +ELLIPSIS
    array([ 0.5436955...,  0.3210794...])
    """

    u, v = tsplit(uv)

    d = 6 * u - 16 * v + 12

    with sdiv_mode():
        xy = tstack([sdiv(9 * u, d), sdiv(4 * v, d)])

    return xy


def xy_to_Luv_uv(xy: ArrayLike) -> NDArrayFloat:
    """
    Return the *CIE L\\*u\\*v\\** colourspace :math:`uv^p` chromaticity
    coordinates from given *CIE xy* chromaticity coordinates.

    Parameters
    ----------
    xy
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE L\\*u\\*v\\* u"v"* chromaticity coordinates.

    References
    ----------
    :cite:`Wikipedia2007b`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.54369558, 0.32107944])
    >>> xy_to_Luv_uv(xy)  # doctest: +ELLIPSIS
    array([ 0.3772021...,  0.5012026...])
    """

    x, y = tsplit(xy)

    d = -2 * x + 12 * y + 3

    with sdiv_mode():
        uv = tstack([sdiv(4 * x, d), sdiv(9 * y, d)])

    return uv


def Luv_to_LCHuv(Luv: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE L\\*u\\*v\\** colourspace to *CIE L\\*C\\*Huv*
    colourspace.

    Parameters
    ----------
    Luv
        *CIE L\\*u\\*v\\** colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE L\\*C\\*Huv* colourspace array.

    Notes
    -----
    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Luv``    | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``u`` : [-100, 100]   | ``u`` : [-1, 1] |
    |            |                       |                 |
    |            | ``v`` : [-100, 100]   | ``v`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``LCHuv``  | ``L``   : [0, 100]    | ``L``   : [0, 1] |
    |            |                       |                  |
    |            | ``C``   : [0, 100]    | ``C``   : [0, 1] |
    |            |                       |                  |
    |            | ``Huv`` : [0, 360]    | ``Huv`` : [0, 1] |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> import numpy as np
    >>> Luv = np.array([41.52787529, 96.83626054, 17.75210149])
    >>> Luv_to_LCHuv(Luv)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  98.4499795...,  10.3881634...])
    """

    return Jab_to_JCh(Luv)


def LCHuv_to_Luv(LCHuv: ArrayLike) -> NDArrayFloat:
    """
    Convert from *CIE L\\*C\\*Huv* colourspace to *CIE L\\*u\\*v\\**
    colourspace.

    Parameters
    ----------
    LCHuv
        *CIE L\\*C\\*Huv* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE L\\*u\\*v\\** colourspace array.

    Notes
    -----
    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``LCHuv``  | ``L``   : [0, 100]    | ``L``   : [0, 1] |
    |            |                       |                  |
    |            | ``C``   : [0, 100]    | ``C``   : [0, 1] |
    |            |                       |                  |
    |            | ``Huv`` : [0, 360]    | ``Huv`` : [0, 1] |
    +------------+-----------------------+------------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Luv``    | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``u`` : [-100, 100]   | ``u`` : [-1, 1] |
    |            |                       |                 |
    |            | ``v`` : [-100, 100]   | ``v`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> import numpy as np
    >>> LCHuv = np.array([41.52787529, 98.44997950, 10.38816348])
    >>> LCHuv_to_Luv(LCHuv)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  96.8362605...,  17.7521014...])
    """

    return JCh_to_Jab(LCHuv)
