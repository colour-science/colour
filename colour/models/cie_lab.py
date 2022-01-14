# -*- coding: utf-8 -*-
"""
CIE L*a*b* Colourspace
======================

Defines the *CIE L\\*a\\*b\\** colourspace transformations:

-   :func:`colour.XYZ_to_Lab`
-   :func:`colour.Lab_to_XYZ`
-   :func:`colour.Lab_to_LCHab`
-   :func:`colour.LCHab_to_Lab`

References
----------
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901906-33-6
"""

from __future__ import annotations

from colour.colorimetry import (
    CCS_ILLUMINANTS,
    intermediate_lightness_function_CIE1976,
    intermediate_luminance_function_CIE1976,
)
from colour.hints import ArrayLike, NDArray
from colour.models import xy_to_xyY, xyY_to_XYZ, Jab_to_JCh, JCh_to_Jab
from colour.utilities import (
    from_range_1,
    from_range_100,
    to_domain_1,
    to_domain_100,
    tsplit,
    tstack,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'XYZ_to_Lab',
    'Lab_to_XYZ',
    'Lab_to_LCHab',
    'LCHab_to_Lab',
]


def XYZ_to_Lab(XYZ: ArrayLike,
               illuminant: ArrayLike = CCS_ILLUMINANTS[
                   'CIE 1931 2 Degree Standard Observer']['D65']) -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *CIE L\\*a\\*b\\**
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
        *CIE L\\*a\\*b\\** colourspace array.

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
    | ``Lab``        | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |                |                       |                 |
    |                | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |                |                       |                 |
    |                | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +----------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_Lab(XYZ)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  52.6385830...,  26.9231792...])
    """

    X, Y, Z = tsplit(to_domain_1(XYZ))
    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    f_X_X_n = intermediate_lightness_function_CIE1976(X, X_n)
    f_Y_Y_n = intermediate_lightness_function_CIE1976(Y, Y_n)
    f_Z_Z_n = intermediate_lightness_function_CIE1976(Z, Z_n)

    L = 116 * f_Y_Y_n - 16
    a = 500 * (f_X_X_n - f_Y_Y_n)
    b = 200 * (f_Y_Y_n - f_Z_Z_n)

    Lab = tstack([L, a, b])

    return from_range_100(Lab)


def Lab_to_XYZ(Lab: ArrayLike,
               illuminant: ArrayLike = CCS_ILLUMINANTS[
                   'CIE 1931 2 Degree Standard Observer']['D65']) -> NDArray:
    """
    Converts from *CIE L\\*a\\*b\\** colourspace to *CIE XYZ* tristimulus
    values.

    Parameters
    ----------
    Lab
        *CIE L\\*a\\*b\\** colourspace array.
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
    | ``Lab``        | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |                |                       |                 |
    |                | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |                |                       |                 |
    |                | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
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
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> import numpy as np
    >>> Lab = np.array([41.52787529, 52.63858304, 26.92317922])
    >>> Lab_to_XYZ(Lab)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    L, a, b = tsplit(to_domain_100(Lab))

    X_n, Y_n, Z_n = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    f_Y_Y_n = (L + 16) / 116
    f_X_X_n = a / 500 + f_Y_Y_n
    f_Z_Z_n = f_Y_Y_n - b / 200

    X = intermediate_luminance_function_CIE1976(f_X_X_n, X_n)
    Y = intermediate_luminance_function_CIE1976(f_Y_Y_n, Y_n)
    Z = intermediate_luminance_function_CIE1976(f_Z_Z_n, Z_n)

    XYZ = tstack([X, Y, Z])

    return from_range_1(XYZ)


def Lab_to_LCHab(Lab: ArrayLike) -> NDArray:
    """
    Converts from *CIE L\\*a\\*b\\** colourspace to *CIE L\\*C\\*Hab*
    colourspace.

    Parameters
    ----------
    Lab
        *CIE L\\*a\\*b\\** colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE L\\*C\\*Hab* colourspace array.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Lab``    | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``LCHab``  | ``L``   : [0, 100]    | ``L``   : [0, 1] |
    |            |                       |                  |
    |            | ``C``   : [0, 100]    | ``C``   : [0, 1] |
    |            |                       |                  |
    |            | ``Hab`` : [0, 360]    | ``Hab`` : [0, 1] |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> import numpy as np
    >>> Lab = np.array([41.52787529, 52.63858304, 26.92317922])
    >>> Lab_to_LCHab(Lab)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  59.1242590...,  27.0884878...])
    """

    return Jab_to_JCh(Lab)


def LCHab_to_Lab(LCHab: ArrayLike) -> NDArray:
    """
    Converts from *CIE L\\*C\\*Hab* colourspace to *CIE L\\*a\\*b\\**
    colourspace.

    Parameters
    ----------
    LCHab
        *CIE L\\*C\\*Hab* colourspace array.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE L\\*a\\*b\\** colourspace array.

    Notes
    -----

    +-------------+-----------------------+------------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1**    |
    +=============+=======================+==================+
    | ``LCHab``   | ``L``   : [0, 100]    | ``L``   : [0, 1] |
    |             |                       |                  |
    |             | ``C``   : [0, 100]    | ``C``   : [0, 1] |
    |             |                       |                  |
    |             | ``Hab`` : [0, 360]    | ``Hab`` : [0, 1] |
    +-------------+-----------------------+------------------+

    +-------------+-----------------------+-----------------+
    | **Range**   | **Scale - Reference** | **Scale - 1**   |
    +=============+=======================+=================+
    | ``Lab``     | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |             |                       |                 |
    |             | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |             |                       |                 |
    |             | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +-------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004m`

    Examples
    --------
    >>> import numpy as np
    >>> LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
    >>> LCHab_to_Lab(LCHab)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  52.6385830...,  26.9231792...])
    """

    return JCh_to_Jab(LCHab)
