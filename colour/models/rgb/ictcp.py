# -*- coding: utf-8 -*-
"""
:math:`IC_TC_P` Colour Encoding
===============================

Defines the :math:`IC_TC_P` colour encoding related transformations:

-   :func:`colour.RGB_to_ICTCP`
-   :func:`colour.ICTCP_to_RGB`

References
----------
-   :cite:`Dolby2016a` : Dolby. (2016). WHAT IS ICTCP? - INTRODUCTION.
    https://www.dolby.com/us/en/technologies/dolby-vision/ICtCp-white-paper.pdf
-   :cite:`Lu2016c` : Lu, T., Pu, F., Yin, P., Chen, T., Husak, W., Pytlarz,
    J., Atkins, R., Froehlich, J., & Su, G.-M. (2016). ITP Colour Space and Its
    Compression Performance for High Dynamic Range and Wide Colour Gamut Video
    Distribution. ZTE Communications, 14(1), 32-38.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import (eotf_inverse_ST2084,
                                                  eotf_ST2084)
from colour.utilities import (domain_range_scale, vector_dot, from_range_1,
                              to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_ICTCP_RGB_TO_LMS', 'MATRIX_ICTCP_LMS_TO_RGB',
    'MATRIX_ICTCP_LMS_P_TO_ICTCP', 'MATRIX_ICTCP_ICTCP_TO_LMS_P',
    'RGB_to_ICTCP', 'ICTCP_to_RGB'
]

MATRIX_ICTCP_RGB_TO_LMS = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688],
]) / 4096
"""
*ITU-R BT.2020* colourspace to normalised cone responses matrix.

MATRIX_ICTCP_RGB_TO_LMS : array_like, (3, 3)
"""

MATRIX_ICTCP_LMS_TO_RGB = np.linalg.inv(MATRIX_ICTCP_RGB_TO_LMS)
"""
:math:`IC_TC_P` colourspace normalised cone responses to *ITU-R BT.2020*
colourspace matrix.

MATRIX_ICTCP_LMS_TO_RGB : array_like, (3, 3)
"""

MATRIX_ICTCP_LMS_P_TO_ICTCP = np.array([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543],
]) / 4096
"""
:math:`LMS_p` *SMPTE ST 2084:2014* encoded normalised cone responses to
:math:`IC_TC_P` colour encoding matrix.

MATRIX_ICTCP_LMS_P_TO_ICTCP : array_like, (3, 3)
"""

MATRIX_ICTCP_ICTCP_TO_LMS_P = np.linalg.inv(MATRIX_ICTCP_LMS_P_TO_ICTCP)
"""
:math:`IC_TC_P` colour encoding to :math:`LMS_p` *SMPTE ST 2084:2014* encoded
normalised cone responses matrix.

MATRIX_ICTCP_ICTCP_TO_LMS_P : array_like, (3, 3)
"""


def RGB_to_ICTCP(RGB, L_p=10000):
    """
    Converts from *ITU-R BT.2020* colourspace to :math:`IC_TC_P` colour
    encoding.

    Parameters
    ----------
    RGB : array_like
        *ITU-R BT.2020* colourspace array.
    L_p : numeric, optional
        Display peak luminance :math:`cd/m^2` for *SMPTE ST 2084:2014*
        non-linear encoding. This parameter should stay at its default
        :math:`10000 cd/m^2` value for practical applications. It is exposed so
        that the definition can be used as a fitting function.

    Returns
    -------
    ndarray
        :math:`IC_TC_P` colour encoding array.

    Warnings
    --------
    The underlying *SMPTE ST 2084:2014* transfer function is an absolute
    transfer function.

    Notes
    -----

    -   The underlying *SMPTE ST 2084:2014* transfer function is an absolute
        transfer function, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations. The effective domain of *SMPTE ST 2084:2014*
        inverse electro-optical transfer function (EOTF / EOCF) is
        [0.0001, 10000].

    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``RGB``    | ``UN``                | ``UN``           |
    +------------+-----------------------+------------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``ICTCP``  | ``I``  : [0, 1]       | ``I``  : [0, 1]  |
    |            |                       |                  |
    |            | ``CT`` : [-1, 1]      | ``CT`` : [-1, 1] |
    |            |                       |                  |
    |            | ``CP`` : [-1, 1]      | ``CP`` : [-1, 1] |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`Dolby2016a`, :cite:`Lu2016c`

    Examples
    --------
    >>> RGB = np.array([0.45620519, 0.03081071, 0.04091952])
    >>> RGB_to_ICTCP(RGB)  # doctest: +ELLIPSIS
    array([ 0.0735136...,  0.0047525...,  0.0935159...])
    """

    RGB = to_domain_1(RGB)

    LMS = vector_dot(MATRIX_ICTCP_RGB_TO_LMS, RGB)

    with domain_range_scale('ignore'):
        LMS_p = eotf_inverse_ST2084(LMS, L_p)

    ICTCP = vector_dot(MATRIX_ICTCP_LMS_P_TO_ICTCP, LMS_p)

    return from_range_1(ICTCP)


def ICTCP_to_RGB(ICTCP, L_p=10000):
    """
    Converts from :math:`IC_TC_P` colour encoding to *ITU-R BT.2020*
    colourspace.

    Parameters
    ----------
    ICTCP : array_like
        :math:`IC_TC_P` colour encoding array.
    L_p : numeric, optional
        Display peak luminance :math:`cd/m^2` for *SMPTE ST 2084:2014*
        non-linear encoding. This parameter should stay at its default
        :math:`10000 cd/m^2` value for practical applications. It is exposed so
        that the definition can be used as a fitting function.

    Returns
    -------
    ndarray
        *ITU-R BT.2020* colourspace array.

    Warnings
    --------
    The underlying *SMPTE ST 2084:2014* transfer function is an absolute
    transfer function.

    Notes
    -----

    -   The underlying *SMPTE ST 2084:2014* transfer function is an absolute
        transfer function, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations.

    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``ICTCP``  | ``I``  : [0, 1]       | ``I``  : [0, 1]  |
    |            |                       |                  |
    |            | ``CT`` : [-1, 1]      | ``CT`` : [-1, 1] |
    |            |                       |                  |
    |            | ``CP`` : [-1, 1]      | ``CP`` : [-1, 1] |
    +------------+-----------------------+------------------+

    +------------+-----------------------+------------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``RGB``    | ``UN``                | ``UN``           |
    +------------+-----------------------+------------------+

    References
    ----------
    :cite:`Dolby2016a`, :cite:`Lu2016c`

    Examples
    --------
    >>> ICTCP = np.array([0.07351364, 0.00475253, 0.09351596])
    >>> ICTCP_to_RGB(ICTCP)  # doctest: +ELLIPSIS
    array([ 0.4562052...,  0.0308107...,  0.0409195...])
    """

    ICTCP = to_domain_1(ICTCP)

    LMS_p = vector_dot(MATRIX_ICTCP_ICTCP_TO_LMS_P, ICTCP)

    with domain_range_scale('ignore'):
        LMS = eotf_ST2084(LMS_p, L_p)

    RGB = vector_dot(MATRIX_ICTCP_LMS_TO_RGB, LMS)

    return from_range_1(RGB)
