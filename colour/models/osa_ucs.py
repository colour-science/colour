# -*- coding: utf-8 -*-
"""
Optical Society of America Uniform Colour Scales (OSA UCS)
==========================================================

Defines the *OSA UCS* colourspace:

-   :func:`colour.XYZ_to_OSA_UCS`
-   :func:`colour.OSA_UCS_to_XYZ`

See Also
--------
`Optical Society of America Uniform Color Scales Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/osa_ucs.ipynb>`_

References
----------
-   :cite:`Cao2013` : Cao, R., Trussell, H. J., & Shamey, R. (2013). Comparison
    of the performance of inverse transformation methods from OSA-UCS to
    CIEXYZ. Journal of the Optical Society of America A, 30(8), 1508.
    doi:10.1364/JOSAA.30.001508
-   :cite:`Moroney2003` : Moroney, N. (2003). A radial sampling of the OSA
    uniform color scales. Color and Imaging Conference, 1-14. Retrieved from
    http://www.ingentaconnect.com/content/\
ist/cic/2003/00002003/00000001/art00031
"""

from __future__ import division, unicode_literals

import numpy as np
from scipy.optimize import fmin

from colour.algebra import spow
from colour.models import XYZ_to_xyY
from colour.utilities import (as_float_array, domain_range_scale, dot_vector,
                              from_range_100, to_domain_100, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['XYZ_to_OSA_UCS', 'OSA_UCS_to_XYZ']

M_XYZ_TO_RGB_OSA_UCS = np.array([
    [0.799, 0.4194, -0.1648],
    [-0.4493, 1.3265, 0.0927],
    [-0.1149, 0.3394, 0.717],
])
"""
*OSA UCS* matrix converting from *CIE XYZ* tristimulus values to *RGB*
colourspace.

M_XYZ_TO_RGB_OSA_UCS : array_like, (3, 3)
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
    :cite:`Cao2013`, :cite:`Moroney2003`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
    >>> XYZ_to_OSA_UCS(XYZ)  # doctest: +ELLIPSIS
    array([-3.0049979...,  2.9971369..., -9.6678423...])
    """

    XYZ = to_domain_100(XYZ)
    x, y, Y = tsplit(XYZ_to_xyY(XYZ))

    Y_0 = Y * (4.4934 * x ** 2 + 4.3034 * y ** 2 - 4.276 * x * y - 1.3744 * x -
               2.5643 * y + 1.8103)

    o_3 = 1 / 3
    Y_0_es = spow(Y_0, o_3) - 2 / 3
    # Gracefully handles Y_0 < 30.
    Y_0_s = Y_0 - 30
    Lambda = 5.9 * (Y_0_es + 0.042 * spow(Y_0_s, o_3))

    RGB = dot_vector(M_XYZ_TO_RGB_OSA_UCS, XYZ)
    RGB_3 = spow(RGB, 1 / 3)

    C = Lambda / (5.9 * Y_0_es)
    L = (Lambda - 14.4) / spow(2, 1 / 2)
    j = C * np.dot(RGB_3, np.array([1.7, 8, -9.7]))
    g = C * np.dot(RGB_3, np.array([-13.7, 17.7, -4]))

    Ljg = tstack([L, j, g])

    return from_range_100(Ljg)


def OSA_UCS_to_XYZ(Ljg, optimisation_parameters=None):
    """
    Converts from *OSA UCS* colourspace to *CIE XYZ* tristimulus values under
    the *CIE 1964 10 Degree Standard Observer*.

    Parameters
    ----------
    Ljg : array_like
        *OSA UCS* :math:`Ljg` lightness, jaune (yellowness), and greenness.
    optimisation_parameters : dict_like, optional
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
    shape = Ljg.shape
    Ljg = np.atleast_1d(Ljg.reshape([-1, 3]))

    optimisation_settings = {'disp': False}
    if optimisation_parameters is not None:
        optimisation_settings.update(optimisation_parameters)

    def error_function(XYZ, Ljg):
        """
        Error function.
        """

        # Error must be computed in "reference" domain and range.
        with domain_range_scale('ignore'):
            error = np.linalg.norm(XYZ_to_OSA_UCS(XYZ) - Ljg)

        return error

    x_0 = np.array([30, 30, 30])
    XYZ = as_float_array([
        fmin(error_function, x_0, (Ljg_i, ), **optimisation_settings)
        for Ljg_i in Ljg
    ])

    return from_range_100(XYZ.reshape(shape))
