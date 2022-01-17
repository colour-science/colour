# -*- coding: utf-8 -*-
"""
:math:`IC_AC_B` Colourspace
===========================

Defines the :math:`IC_AC_B` colourspace transformations:

-   :func:`colour.XYZ_to_ICaCb`
-   :func:`colour.ICaCb_to_XYZ`

References
----------
-   :cite:`Frohlich2017` : Fröhlich, J. (2017). Encoding high dynamic range
    and wide color gamut imagery. doi:10.18419/OPUS-9664
"""
import numpy as np

from colour.algebra import vector_dot
from colour.models.rgb.transfer_functions import (
    eotf_ST2084,
    eotf_inverse_ST2084,
)
from colour.utilities import (
    domain_range_scale,
    from_range_1,
    to_domain_1,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_ICACB_XYZ_TO_LMS',
    'MATRIX_ICACB_XYZ_TO_LMS_2',
    'MATRIX_ICACB_LMS_TO_XYZ',
    'MATRIX_ICACB_LMS_TO_XYZ_2',
    'XYZ_to_ICaCb',
    'ICaCb_to_XYZ',
]

MATRIX_ICACB_XYZ_TO_LMS = np.array([
    [0.37613, 0.70431, -0.05675],
    [-0.21649, 1.14744, 0.05356],
    [0.02567, 0.16713, 0.74235],
])
"""
*CIE XYZ* tristimulus values to normalised cone responses matrix.

MATRIX_ICACB_XYZ_TO_LMS : array_like, (3, 3)
"""

MATRIX_ICACB_XYZ_TO_LMS_2 = np.array([
    [0.4949, 0.5037, 0.0015],
    [4.2854, -4.5462, 0.2609],
    [0.3605, 1.1499, -1.5105],
])
"""
MATRIX_ICACB_XYZ_TO_LMS_2 : array_like, (3, 3)
"""

MATRIX_ICACB_LMS_TO_XYZ = np.linalg.inv(MATRIX_ICACB_XYZ_TO_LMS)
"""
Normalised cone responses to *CIE XYZ* tristimulus values matrix.

MATRIX_ICACB_LMS_TO_XYZ : array_like, (3, 3)
"""

MATRIX_ICACB_LMS_TO_XYZ_2 = np.linalg.inv(MATRIX_ICACB_XYZ_TO_LMS_2)
"""
Normalised cone responses to *CIE XYZ* tristimulus values matrix.

MATRIX_ICACB_LMS_TO_XYZ : array_like, (3, 3)
"""


def XYZ_to_ICaCb(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to :math:`IC_AC_B` colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        :math:`IC_AC_B` colourspace array.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``ICaCb``  | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``Ca`` : [-1, 1]      | ``Ca``: [-1, 1] |
    |            |                       |                 |
    |            | ``Cb`` : [-1, 1]      | ``Cb``: [-1, 1] |
    +------------+-----------------------+-----------------+

    -   Input *CIE XYZ* tristimulus values must be adapted to
        *CIE Standard Illuminant D Series* *D65*.

    References
    ----------
    :cite:`Frohlich2017`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_ICaCb(XYZ)
    array([ 0.06875297,  0.05753352,  0.02081548])
    """

    XYZ = to_domain_1(XYZ)
    LMS = vector_dot(MATRIX_ICACB_XYZ_TO_LMS, XYZ)

    with domain_range_scale('ignore'):
        LMS_prime = eotf_inverse_ST2084(LMS)

    return from_range_1(vector_dot(MATRIX_ICACB_XYZ_TO_LMS_2, LMS_prime))


def ICaCb_to_XYZ(ICaCb):
    """
    Converts from :math:`IC_AC_B` tristimulus values to *CIE XYZ* colourspace.

    Parameters
    ----------
    ICaCb : array_like
        :math:`IC_AC_B` tristimulus values.

    Returns
    -------
    ndarray
        *CIE XYZ* colourspace array.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``ICaCb``  | ``I`` : [0, 1]        | ``I`` : [0, 1]  |
    |            |                       |                 |
    |            | ``Ca`` : [-1, 1]      | ``Ca``: [-1, 1] |
    |            |                       |                 |
    |            | ``Cb`` : [-1, 1]      | ``Cb``: [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Frohlich2017`

    Examples
    --------
    >>> XYZ = np.array([0.06875297, 0.05753352, 0.02081548])
    >>> ICaCb_to_XYZ(XYZ)
    array([ 0.20654008,  0.12197225,  0.05136951])
    """

    ICaCb = to_domain_1(ICaCb)
    LMS_prime = vector_dot(MATRIX_ICACB_LMS_TO_XYZ_2, ICaCb)

    with domain_range_scale('ignore'):
        LMS = eotf_ST2084(LMS_prime)

    return from_range_1(vector_dot(MATRIX_ICACB_LMS_TO_XYZ, LMS))
