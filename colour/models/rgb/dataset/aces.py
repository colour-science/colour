#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Academy Color Encoding System
=============================

Defines the *Academy Color Encoding System* (ACES) related encodings:

-   :attr:`ACES_2065_1_COLOURSPACE`
-   :attr:`ACES_CG_COLOURSPACE`
-   :attr:`ACES_CC_COLOURSPACE`
-   :attr:'ACES_CCT_COLOURSPACE'
-   :attr:`ACES_PROXY_COLOURSPACE`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (n.d.). Academy Color Encoding System. Retrieved
        February 24, 2014, from
        http://www.oscars.org/science-technology/council/projects/aces.html
.. [2]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2014). Technical Bulletin TB-2014-004 - Informative
        Notes on SMPTE ST 2065-1 – Academy Color Encoding Specification
        (ACES). Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [3]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2015). Specification S-2014-004 - ACEScg –
        A Working Space for CGI Render and Compositing, 1–9.  Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [4]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2014). Specification S-2014-003 - ACEScc , A
        Logarithmic Encoding of ACES Data for use within Color Grading
        Systems. Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [5]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2014). Specification S-2013-001 - ACESproxy , an
        Integer Log Encoding of ACES Image Data. Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [6]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2014). Technical Bulletin TB-2014-012 - Academy Color
        Encoding System Version 1.0 Component Names. Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [7]  The Academy of Motion Picture Arts and Sciences. (2016).
        Specification S-2016-001 - ACEScct, A Quasi-Logarithmic
        Encoding of ACES Data for use within Color Grading Systems.
        Retrieved October 10, 2016, from
        https://github.com/ampas/aces-dev/tree/v1.0.3/documents
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    normalised_primary_matrix,
    linear_function,
    log_encoding_ACEScc,
    log_decoding_ACEScc,
    log_encoding_ACEScct,
    log_decoding_ACEScct,
    log_encoding_ACESproxy,
    log_decoding_ACESproxy)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['AP0',
           'AP1',
           'ACES_ILLUMINANT',
           'ACES_WHITEPOINT',
           'AP0_TO_XYZ_MATRIX',
           'XYZ_TO_AP0_MATRIX',
           'AP1_TO_XYZ_MATRIX',
           'XYZ_TO_AP1_MATRIX',
           'ACES_2065_1_COLOURSPACE',
           'ACES_CG_COLOURSPACE',
           'ACES_CC_COLOURSPACE',
           'ACES_CCT_COLOURSPACE',
           'ACES_PROXY_COLOURSPACE']

AP0 = np.array(
    [[0.73470, 0.26530],
     [0.00000, 1.00000],
     [0.00010, -0.07700]])
"""
*ACES Primaries 0* or *AP0* primaries.

AP0 : ndarray, (3, 2)
"""

AP1 = np.array(
    [[0.71300, 0.29300],
     [0.16500, 0.83000],
     [0.12800, 0.04400]])
"""
*ACES Primaries 1* or *AP1* primaries (known as *Rec. 2020+* primaries prior
to *ACES* 1.0 release).

AP1 : ndarray, (3, 2)
"""

ACES_ILLUMINANT = 'D60'
"""
*ACES2065-1* colourspace whitepoint name as illuminant.

ACES_ILLUMINANT : unicode
"""

ACES_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][ACES_ILLUMINANT])
"""
*ACES2065-1* colourspace whitepoint.

ACES_WHITEPOINT : ndarray
"""

AP0_TO_XYZ_MATRIX = np.array(
    [[0.9525523959, 0.0000000000, 0.0000936786],
     [0.3439664498, 0.7281660966, -0.0721325464],
     [0.0000000000, 0.0000000000, 1.0088251844]])
"""
*ACES Primaries 0* to *CIE XYZ* tristimulus values matrix defined as per [2].
AP0_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_AP0_MATRIX = np.array(
    [[1.0498110175, 0.0000000000, -0.0000974845],
     [- 0.4959030231, 1.3733130458, 0.0982400361],
     [0.0000000000, 0.0000000000, 0.9912520182]])
"""
*CIE XYZ* tristimulus values to *ACES Primaries 0* matrix.

XYZ_TO_AP0_MATRIX : array_like, (3, 3)
"""

AP1_TO_XYZ_MATRIX = normalised_primary_matrix(AP1, ACES_WHITEPOINT)
"""
*ACES Primaries 1* to *CIE XYZ* tristimulus values matrix.

AP1_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_AP1_MATRIX = np.linalg.inv(AP1_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *ACES Primaries 1* matrix.

XYZ_TO_AP1_MATRIX : array_like, (3, 3)
"""

ACES_2065_1_COLOURSPACE = RGB_Colourspace(
    'ACES2065-1',
    AP0,
    ACES_WHITEPOINT,
    ACES_ILLUMINANT,
    AP0_TO_XYZ_MATRIX,
    XYZ_TO_AP0_MATRIX,
    linear_function,
    linear_function)
"""
*ACES2065-1* colourspace, base encoding, used for exchange of full fidelity
images and archiving.

ACES_2065_1_COLOURSPACE : RGB_Colourspace
"""

ACES_CG_COLOURSPACE = RGB_Colourspace(
    'ACEScg',
    AP1,
    ACES_WHITEPOINT,
    ACES_ILLUMINANT,
    AP1_TO_XYZ_MATRIX,
    XYZ_TO_AP1_MATRIX,
    linear_function,
    linear_function)
"""
*ACEScg* colourspace, a working space for paint/compositor applications that
don’t support ACES2065-1 or ACEScc.

ACES_CG_COLOURSPACE : RGB_Colourspace
"""

ACES_CC_COLOURSPACE = RGB_Colourspace(
    'ACEScc',
    AP1,
    ACES_WHITEPOINT,
    ACES_ILLUMINANT,
    AP1_TO_XYZ_MATRIX,
    XYZ_TO_AP1_MATRIX,
    log_encoding_ACEScc,
    log_decoding_ACEScc)
"""
*ACEScc* colourspace, a working space for color correctors, target for ASC-CDL
values created on-set.

ACES_CC_COLOURSPACE : RGB_Colourspace
"""

ACES_CCT_COLOURSPACE = RGB_Colourspace(
    'ACEScct',
    AP1,
    ACES_WHITEPOINT,
    ACES_ILLUMINANT,
    AP1_TO_XYZ_MATRIX,
    XYZ_TO_AP1_MATRIX,
    log_encoding_ACEScct,
    log_decoding_ACEScct)
"""
*ACEScct* colourspace, an alternative working space for colour correctors,
intended to be transient and internal to software or hardware systems,
and is specifically not intended for interchange or archiving.

ACES_CCT_COLOURSPACE : RGB_Colourspace
"""

ACES_PROXY_COLOURSPACE = RGB_Colourspace(
    'ACESproxy',
    AP1,
    ACES_WHITEPOINT,
    ACES_ILLUMINANT,
    AP1_TO_XYZ_MATRIX,
    XYZ_TO_AP1_MATRIX,
    log_encoding_ACESproxy,
    log_decoding_ACESproxy)
"""
*ACESproxy* colourspace, a lightweight encoding for transmission over HD-SDI
(or other production transmission schemes), onset look management. Not
intended to be stored or used in production imagery or for final colour
grading / mastering.

ACES_PROXY_COLOURSPACE : RGB_Colourspace
"""
