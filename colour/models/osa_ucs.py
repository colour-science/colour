# -*- coding: utf-8 -*-
"""
Optical Society of America Uniform Colour Scales (OSA UCS)
==========================================================

Defines the *OSA UCS* colourspace:

-   :func:`colour.XYZ_to_OSA_UCS`
-   :func:`colour.OSA_UCS_to_XYZ`

References
----------
-   :cite:`Cao2013` : Cao, R., Trussell, H. J., & Shamey, R. (2013). Comparison
    of the performance of inverse transformation methods from OSA-UCS to
    CIEXYZ. Journal of the Optical Society of America A, 30(8), 1508.
    doi:10.1364/JOSAA.30.001508
-   :cite:`MacAdam1974` : MacAdam, D. L. (1974). Uniform color scales*. Journal
    of the Optical Society of America, 64(12), 1691. doi:10.1364/JOSA.64.001691
-   :cite:`Moroney2003` : Moroney, N. (2003). A Radial Sampling of the OSA
    Uniform Color Scales. Color and Imaging Conference, 2003(1), 175-180.
    ISSN:2166-9635
"""

import numpy as np

from colour.algebra import minimize_NewtonRaphson
from colour.models import XYZ_to_xyY
from colour.utilities import (from_range_100, to_domain_100, tsplit, tstack,
                              vector_dot)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['XYZ_to_OSA_UCS', 'OSA_UCS_to_XYZ']

MATRIX_XYZ_TO_RGB_OSA_UCS = np.array([
    [0.799, 0.4194, -0.1648],
    [-0.4493, 1.3265, 0.0927],
    [-0.1149, 0.3394, 0.717],
])
"""
*OSA UCS* matrix converting from *CIE XYZ* tristimulus values to *RGB*
colourspace.

MATRIX_XYZ_TO_RGB_OSA_UCS : array_like, (3, 3)
"""


def XYZ_to_OSA_UCS(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values under the
    *CIE 1964 10 Degree Standard Observer* to *OSA UCS* colourspace.

    The lightness axis, *L* is usually in range [-9, 5] and centered around
    middle gray (Munsell N/6). The yellow-blue axis, *j* is usually in range
    [-15, 15]. The red-green axis, *g* is usually in range [-20, 15].

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values under the
        *CIE 1964 10 Degree Standard Observer*.

    Returns
    -------
    ndarray
        *OSA UCS* :math:`Ljg` lightness, jaune (yellowness), and greenness.

    Notes
    -----

    +------------+-----------------------+--------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``XYZ``    | [0, 100]              | [0, 1]             |
    +------------+-----------------------+--------------------+

    +------------+-----------------------+--------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``Ljg``    | ``L`` : [-100, 100]   | ``L`` : [-1, 1]    |
    |            |                       |                    |
    |            | ``j`` : [-100, 100]   | ``j`` : [-1, 1]    |
    |            |                       |                    |
    |            | ``g`` : [-100, 100]   | ``g`` : [-1, 1]    |
    +------------+-----------------------+--------------------+

    -   *OSA UCS* uses the *CIE 1964 10 Degree Standard Observer*.

    References
    ----------
    :cite:`Cao2013`, :cite:`MacAdam1974`, :cite:`Moroney2003`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
    >>> XYZ_to_OSA_UCS(XYZ)  # doctest: +ELLIPSIS
    array([-3.0049979...,  2.9971369..., -9.6678423...])
    """

    XYZ = to_domain_100(XYZ)
    x, y, Y = tsplit(XYZ_to_xyY(XYZ))

    # NOTE: Cao et al. (2013) uses 4.27 instead of 4.276.
    Y_0 = Y * (4.4934 * x ** 2 + 4.3034 * y ** 2 - 4.276 * x * y - 1.3744 * x -
               2.5643 * y + 1.8103)

    Y_0_es = np.cbrt(Y_0) - 2 / 3
    L = 5.9 * (Y_0_es + 0.042 * np.cbrt(Y_0 - 30))

    RGB = np.cbrt(vector_dot(MATRIX_XYZ_TO_RGB_OSA_UCS, XYZ))
    C = L / (5.9 * Y_0_es)
    j = C * np.dot(RGB, np.array([1.7, 8, -9.7]))
    g = C * np.dot(RGB, np.array([-13.7, 17.7, -4]))

    # NOTE: MacAdam (1974), Cao et al. (2013) and Moroney (2003) use 14.4,
    # 14.3993 has been seen in the wild, e.g. Wikipedia.
    Ljg = tstack([(L - 14.4) / np.sqrt(2), j, g])

    return from_range_100(Ljg)


def OSA_UCS_to_XYZ(Ljg, optimisation_kwargs=None):
    """
    Converts from *OSA UCS* colourspace to *CIE XYZ* tristimulus values under
    the *CIE 1964 10 Degree Standard Observer*.

    Parameters
    ----------
    Ljg : array_like
        *OSA UCS* :math:`Ljg` lightness, jaune (yellowness), and greenness.
    optimisation_kwargs : dict_like, optional
        Parameters for :func:`scipy.optimize.fmin` definition.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values under the
        *CIE 1964 10 Degree Standard Observer*.

    Warnings
    --------
    There is no analytical inverse transformation from *OSA UCS* to :math:`Ljg`
    lightness, jaune (yellowness), and greenness to *CIE XYZ* tristimulus
    values, the current implementation relies on optimization using
    :func:`scipy.optimize.fmin` definition and thus has reduced precision and
    poor performance.

    Notes
    -----

    +------------+-----------------------+--------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``Ljg``    | ``L`` : [-100, 100]   | ``L`` : [-1, 1]    |
    |            |                       |                    |
    |            | ``j`` : [-100, 100]   | ``j`` : [-1, 1]    |
    |            |                       |                    |
    |            | ``g`` : [-100, 100]   | ``g`` : [-1, 1]    |
    +------------+-----------------------+--------------------+

    +------------+-----------------------+--------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**      |
    +============+=======================+====================+
    | ``XYZ``    | [0, 100]              | [0, 1]             |
    +------------+-----------------------+--------------------+

    -   *OSA UCS* uses the *CIE 1964 10 Degree Standard Observer*.

    References
    ----------
    :cite:`Cao2013`, :cite:`Moroney2003`

    Examples
    --------
    >>> import numpy as np
    >>> Ljg = np.array([-3.00499790, 2.99713697, -9.66784231])
    >>> OSA_UCS_to_XYZ(Ljg)  # doctest: +ELLIPSIS
    array([ 20.6540240...,  12.1972369...,   5.1369372...])
    """

    Ljg = to_domain_100(Ljg)

    XYZ = minimize_NewtonRaphson(Ljg, 1, 0.01, XYZ_to_OSA_UCS)

    return from_range_100(XYZ)


if __name__ == '__main__':

    a = [1, 2, 3]
    np.testing.assert_allclose(
        OSA_UCS_to_XYZ(XYZ_to_OSA_UCS(a)), a, rtol=0.0001, atol=0.0001)

    a = [[1, 2, 3]]
    np.testing.assert_allclose(
        OSA_UCS_to_XYZ(XYZ_to_OSA_UCS(a)), a, rtol=0.0001, atol=0.0001)

    a = [[1, 2, 3], [3, 1.04, 0.05]]
    np.testing.assert_allclose(
        OSA_UCS_to_XYZ(XYZ_to_OSA_UCS(a)), a, rtol=0.0001, atol=0.0001)

    a = [[[1, 2, 3], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]
    np.testing.assert_allclose(
        OSA_UCS_to_XYZ(XYZ_to_OSA_UCS(a)), a, rtol=0.0001, atol=0.0001)
