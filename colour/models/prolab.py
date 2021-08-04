# -*- coding: utf-8 -*-
"""
ProLab Colourspace
==================

Defines the *ProLab* colourspace transformations:

-   :func:`colour.XYZ_to_ProLab`
-   :func:`colour.ProLab_to_XYZ`

References
----------
-   :cite:`Ivan2021` : Ivan A. Konovalenko, Anna A. Smagina,
    Dmitry P. Nikolaev, Petr P. Nikolaev.
    ProLab: perceptually uniform projective colour coordinate
    system. doi:10.1109/ACCESS.2017
"""

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'D_65', 'MATRIX_1_XYZ_to_ProLab', 'MATRIX_2_ProLab_to_XYZ', 'MATRIX_3',
    'ProLab_to_XYZ', 'XYZ_to_ProLab'
]

MATRIX_1_XYZ_to_ProLab = np.array([
    [75.5362, 486.661, 167.387],
    [617.7141, -595.4477, -22.2664],
    [48.3433, 194.9377, -243.281],
])
"""
Normalised cone responses to *CIE XYZ* tristimulus values matrix.

MATRIX_1_XYZ_to_ProLab : array_like, (3, 3)
"""

MATRIX_2_ProLab_to_XYZ = np.linalg.inv(MATRIX_1_XYZ_to_ProLab)
"""
Normalised cone responses to *ProLab* colourspace matrix.

MATRIX_2_ProLab_to_XYZ : array_like, (3, 3)
"""

MATRIX_3 = np.array([0.7554, 3.8666, 1.6739])

D_65 = np.array([95.047, 100, 108.883])
"""
*CIE Standard Illuminant D Series* *D65*

D_65: array_like, (1, 3)

"""


def XYZ_to_ProLab(XYZ):
    """
    Converts from *CIE XYZ* tristimulus values to *ProLab* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.

    Returns
    -------
    ndarray
        *ProLab* colourspace array.

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
    | ``ProLab`` | ``L`` : [0, 1]        | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-1, 1]       | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-1, 1]       | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Ivan2021`
    """

    XYZ = np.asarray(XYZ)
    XYZ_ = (XYZ.T / D_65).T

    return np.dot(MATRIX_1_XYZ_to_ProLab, XYZ_) / (np.dot(MATRIX_3, XYZ_) + 1)


def ProLab_to_XYZ(ProLab):
    """
    Converts from *ProLab* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Lab : array_like
        *ProLab* colourspace array.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Lab``    | ``L`` : [0, 1]        | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-1, 1]       | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-1, 1]       | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``XYZ``    | [0, 1]                | [0, 1]          |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`Ivan2021`
    """

    XYZ_ = np.dot(MATRIX_2_ProLab_to_XYZ, ProLab)
    XYZ_ = XYZ_ / (1 - np.dot(MATRIX_3, XYZ_))

    return (XYZ_.T * D_65).T
