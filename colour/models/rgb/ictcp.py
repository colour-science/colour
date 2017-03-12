#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:math:`IC_TC_P` Colour Encoding
===============================

Defines the :math:`IC_TC_P` colour encoding related transformations:

-   :func:`RGB_to_ICTCP`
-   :func:`ICTCP_to_RGB`

See Also
--------
`ICTCP Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/ictcp.ipynb>`_

References
----------
.. [1]  Dolby. (2016). WHAT IS ICTCP? - INTRODUCTION. Retrieved from
        https://www.dolby.com/us/en/technologies/dolby-vision/\
ICtCp-white-paper.pdf
.. [2]  Lu, T., Pu, F., Yin, P., Chen, T., Husak, W., Pytlarz, J., … Su, G.-M.
        (2016). ICTCP Colour Space and Its Compression Performance for High
        Dynamic Range and Wide Colour Gamut Video Distribution. ZTE
        Communications, 14(1), 32–38. doi:10.3969/j.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import oetf_ST2084, eotf_ST2084
from colour.utilities import dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ICTCP_RGB_TO_LMS_MATRIX',
           'ICTCP_LMS_TO_RGB_MATRIX',
           'ICTCP_LMS_P_TO_ICTCP_MATRIX',
           'ICTCP_ICTCP_TO_LMS_P_MATRIX',
           'RGB_to_ICTCP',
           'ICTCP_to_RGB']

ICTCP_RGB_TO_LMS_MATRIX = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688]]) / 4096
"""
*Rec. 2020* colourspace to normalised cone responses matrix.

ICTCP_RGB_TO_LMS_MATRIX : array_like, (3, 3)
"""

ICTCP_LMS_TO_RGB_MATRIX = np.linalg.inv(ICTCP_RGB_TO_LMS_MATRIX)
"""
:math:`IC_TC_P` colourspace normalised cone responses to *Rec. 2020*
colourspace matrix.

ICTCP_LMS_TO_RGB_MATRIX : array_like, (3, 3)
"""

ICTCP_LMS_P_TO_ICTCP_MATRIX = np.array([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543]]) / 4096
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
    Converts from *Rec. 2020* colourspace to :math:`IC_TC_P` colour encoding.

    Parameters
    ----------
    RGB : array_like
        *Rec. 2020* colourspace array.
    L_p : numeric, optional
        Display peak luminance :math:`cd/m^2` for *SMPTE ST 2084:2014*
        non-linear encoding.

    Returns
    -------
    ndarray
        :math:`IC_TC_P` colour encoding array.

    Examples
    --------
    >>> RGB = np.array([0.35181454, 0.26934757, 0.21288023])
    >>> RGB_to_ICTCP(RGB)  # doctest: +ELLIPSIS
    array([ 0.0955407..., -0.0089063...,  0.0138928...])
    """

    LMS = dot_vector(ICTCP_RGB_TO_LMS_MATRIX, RGB)
    LMS_p = oetf_ST2084(LMS, L_p)
    ICTCP = dot_vector(ICTCP_LMS_P_TO_ICTCP_MATRIX, LMS_p)

    return ICTCP


def ICTCP_to_RGB(ICTCP, L_p=10000):
    """
    Converts from :math:`IC_TC_P` colour encoding to *Rec. 2020* colourspace.

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
        *Rec. 2020* colourspace array.

    Examples
    --------
    >>> ICTCP = np.array([0.09554079, -0.00890639, 0.01389286])
    >>> ICTCP_to_RGB(ICTCP)  # doctest: +ELLIPSIS
    array([ 0.3518145...,  0.2693475...,  0.2128802...])
    """

    LMS_p = dot_vector(ICTCP_ICTCP_TO_LMS_P_MATRIX, ICTCP)
    LMS = eotf_ST2084(LMS_p, L_p)
    RGB = dot_vector(ICTCP_LMS_TO_RGB_MATRIX, LMS)

    return RGB
