# -*- coding: utf-8 -*-
"""
:math:`IC_TC_P` Colour Encoding
===============================

Defines the :math:`IC_TC_P` colour encoding related transformations:

-   :func:`colour.RGB_to_ICTCP`
-   :func:`colour.ICTCP_to_RGB`

See Also
--------
`ICTCP Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/ictcp.ipynb>`_

References
----------
-   :cite:`Dolby2016a` : Dolby. (2016). WHAT IS ICTCP? - INTRODUCTION.
    Retrieved from https://www.dolby.com/us/en/technologies/dolby-vision/\
ICtCp-white-paper.pdf
-   :cite:`Lu2016c` : Lu, T., Pu, F., Yin, P., Chen, T., Husak, W.,
    Pytlarz, J.,  Su, G.-M. (2016). ITP Colour Space and Its Compression
    Performance for High Dynamic Range and Wide Colour Gamut Video
    Distribution. ZTE Communications, 14(1), 32-38. Retrieved from
    http://www.cnki.net/kcms/detail/34.1294.TN.20160205.1903.006.html
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import oetf_ST2084, eotf_ST2084
from colour.utilities import (domain_range_scale, dot_vector, from_range_1,
                              to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ICTCP_RGB_TO_LMS_MATRIX', 'ICTCP_LMS_TO_RGB_MATRIX',
    'ICTCP_LMS_P_TO_ICTCP_MATRIX', 'ICTCP_ICTCP_TO_LMS_P_MATRIX',
    'RGB_to_ICTCP', 'ICTCP_to_RGB'
]

ICTCP_RGB_TO_LMS_MATRIX = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688],
]) / 4096
"""
*ITU-R BT.2020* colourspace to normalised cone responses matrix.

ICTCP_RGB_TO_LMS_MATRIX : array_like, (3, 3)
"""

ICTCP_LMS_TO_RGB_MATRIX = np.linalg.inv(ICTCP_RGB_TO_LMS_MATRIX)
"""
:math:`IC_TC_P` colourspace normalised cone responses to *ITU-R BT.2020*
colourspace matrix.

ICTCP_LMS_TO_RGB_MATRIX : array_like, (3, 3)
"""

ICTCP_LMS_P_TO_ICTCP_MATRIX = np.array([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543],
]) / 4096
"""
:math:`LMS_p` *SMPTE ST 2084:2014* encoded normalised cone responses to
:math:`IC_TC_P` colour encoding matrix.

ICTCP_LMS_P_TO_ICTCP_MATRIX : array_like, (3, 3)
"""

ICTCP_ICTCP_TO_LMS_P_MATRIX = np.linalg.inv(ICTCP_LMS_P_TO_ICTCP_MATRIX)
"""
:math:`IC_TC_P` colour encoding to :math:`LMS_p` *SMPTE ST 2084:2014* encoded
normalised cone responses matrix.

ICTCP_ICTCP_TO_LMS_P_MATRIX : array_like, (3, 3)
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
        non-linear encoding.

    Returns
    -------
    ndarray
        :math:`IC_TC_P` colour encoding array.

    Notes
    -----

    +------------+-----------------------+------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**    |
    +============+=======================+==================+
    | ``RGB``    | [0, 1]                | [0, 1]           |
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

    LMS = dot_vector(ICTCP_RGB_TO_LMS_MATRIX, RGB)

    with domain_range_scale('ignore'):
        LMS_p = oetf_ST2084(LMS, L_p)

    ICTCP = dot_vector(ICTCP_LMS_P_TO_ICTCP_MATRIX, LMS_p)

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
        non-linear encoding.

    Returns
    -------
    ndarray
        *ITU-R BT.2020* colourspace array.

    Notes
    -----

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
    | ``RGB``    | [0, 1]                | [0, 1]           |
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

    LMS_p = dot_vector(ICTCP_ICTCP_TO_LMS_P_MATRIX, ICTCP)

    with domain_range_scale('ignore'):
        LMS = eotf_ST2084(LMS_p, L_p)

    RGB = dot_vector(ICTCP_LMS_TO_RGB_MATRIX, LMS)

    return from_range_1(RGB)
