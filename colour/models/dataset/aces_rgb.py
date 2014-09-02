#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ACES RGB Colourspace
====================

Defines the *ACES RGB* colourspace and its variations:

-   :attr:`ACES_RGB_COLOURSPACE`
-   :attr:`ACES_RGB_LOG_COLOURSPACE`
-   :attr:`ACES_RGB_PROXY_10_COLOURSPACE`
-   :attr:`ACES_RGB_PROXY_12_COLOURSPACE`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/models/rgb.ipynb>`_  # noqa

References
----------
.. [1]  http://www.oscars.org/science-technology/council/projects/aces.html
        (Last accessed 24 February 2014)
.. [2]  `Academy Color Encoding Specification (ACES)
        <https://www.dropbox.com/sh/nt9z9m6utzvkc5m/AACBum5OdkLPCZ3d6trfVeU8a/ACES_v1.0.1.pdf>`_  # noqa
        (Last accessed 24 February 2014)
"""

from __future__ import division, unicode_literals

import math
import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.utilities import Structure
from colour.models import RGB_Colourspace
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ACES_RGB_PRIMARIES',
           'ACES_RGB_WHITEPOINT',
           'ACES_RGB_TO_XYZ_MATRIX',
           'XYZ_TO_ACES_RGB_MATRIX',
           'ACES_RGB_TRANSFER_FUNCTION',
           'ACES_RGB_INVERSE_TRANSFER_FUNCTION',
           'ACES_RGB_COLOURSPACE',
           'ACES_RGB_LOG_CONSTANTS',
           'ACES_RGB_LOG_TRANSFER_FUNCTION',
           'ACES_RGB_LOG_INVERSE_TRANSFER_FUNCTION',
           'ACES_RGB_LOG_COLOURSPACE',
           'ACES_RGB_PROXY_10_CONSTANTS',
           'ACES_RGB_PROXY_12_CONSTANTS',
           'ACES_RGB_PROXY_CONSTANTS',
           'ACES_RGB_PROXY_10_TRANSFER_FUNCTION',
           'ACES_RGB_PROXY_10_INVERSE_TRANSFER_FUNCTION',
           'ACES_RGB_PROXY_12_TRANSFER_FUNCTION',
           'ACES_RGB_PROXY_12_INVERSE_TRANSFER_FUNCTION',
           'ACES_RGB_PROXY_10_COLOURSPACE',
           'ACES_RGB_PROXY_12_COLOURSPACE']

ACES_RGB_PRIMARIES = np.array(
    [[0.73470, 0.26530],
     [0.00000, 1.00000],
     [0.00010, -0.07700]])
"""
*ACES RGB* colourspace primaries.

ACES_RGB_PRIMARIES : ndarray, (3, 2)

References
----------
.. [3]  `Academy Color Encoding Specification (ACES)
        <https://www.dropbox.com/sh/nt9z9m6utzvkc5m/AACBum5OdkLPCZ3d6trfVeU8a/ACES_v1.0.1.pdf>`_  # noqa
        4.1.2 Color space chromaticities (Last accessed 24 February 2014)
"""

ACES_RGB_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get('D60')
"""
*ACES RGB* colourspace whitepoint.

ACES_RGB_WHITEPOINT : tuple
"""

ACES_RGB_TO_XYZ_MATRIX = np.array(
    [[9.52552396e-01, 0.00000000e+00, 9.36786317e-05],
     [3.43966450e-01, 7.28166097e-01, -7.21325464e-02],
     [0.00000000e+00, 0.00000000e+00, 1.00882518e+00]])
"""
*ACES RGB* colourspace to *CIE XYZ* colourspace matrix.

ACES_RGB_TO_XYZ_MATRIX : array_like, (3, 3)

References
----------
.. [4]  http://www.dropbox.com/sh/iwd09buudm3lfod/gyjDF-k7oC/ACES_v1.0.1.pdf:
        4.1.4 Converting ACES RGB values to CIE XYZ values
        (Last accessed 24 February 2014)
"""

XYZ_TO_ACES_RGB_MATRIX = np.linalg.inv(ACES_RGB_TO_XYZ_MATRIX)
"""
*CIE XYZ* colourspace to *ACES RGB* colourspace matrix.

XYZ_TO_ACES_RGB_MATRIX : array_like, (3, 3)
"""

ACES_RGB_TRANSFER_FUNCTION = lambda x: x
"""
Transfer function from linear to *ACES RGB* colourspace.

ACES_RGB_TRANSFER_FUNCTION : object
"""

ACES_RGB_INVERSE_TRANSFER_FUNCTION = lambda x: x
"""
Inverse transfer function from *ACES RGB* colourspace to linear.

ACES_RGB_INVERSE_TRANSFER_FUNCTION : object
"""

ACES_RGB_COLOURSPACE = RGB_Colourspace(
    'ACES RGB',
    ACES_RGB_PRIMARIES,
    ACES_RGB_WHITEPOINT,
    ACES_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ACES_RGB_MATRIX,
    ACES_RGB_TRANSFER_FUNCTION,
    ACES_RGB_INVERSE_TRANSFER_FUNCTION)
"""
*ACES RGB* colourspace.

ACES_RGB_COLOURSPACE : RGB_Colourspace
"""

ACES_RGB_LOG_CONSTANTS = Structure(
    log_unity=32768,
    log_xperstop=2048,
    denorm_trans=math.pow(2, -15),
    denorm_fake0=math.pow(2, -16))
"""
*ACES RGB Log* colourspace constants.

ACES_RGB_LOG_CONSTANTS : Structure
"""


def _aces_rgb_log_transfer_function(value, is_16_bit_integer=False):
    """
    Defines the *ACES RGB Log* colourspace transfer function.

    Parameters
    ----------
    value : numeric
        value.
    is_16_bit_integer : bool
        Is value 16 bit integer.

    Returns
    -------
    numeric
        Companded value.
    """

    if value < 0:
        return 0

    if value < ACES_RGB_LOG_CONSTANTS.denorm_trans:
        value = ACES_RGB_LOG_CONSTANTS.denorm_fake0 + (value / 2)

    value = ((math.log10(value) / math.log10(2)) *
             ACES_RGB_LOG_CONSTANTS.log_xperstop +
             ACES_RGB_LOG_CONSTANTS.log_unity)

    if is_16_bit_integer:
        value = min(math.floor(value) + 0.5, 65535)

    return value


def _aces_rgb_log_inverse_transfer_function(value):
    """
    Defines the *ACES RGB Log* colourspace inverse transfer function.

    Parameters
    ----------
    value : numeric
        value.

    Returns
    -------
    numeric
        Companded value.
    """

    value = (math.pow(2,
                      (value - ACES_RGB_LOG_CONSTANTS.log_unity) /
                      ACES_RGB_LOG_CONSTANTS.log_xperstop))
    if value < ACES_RGB_LOG_CONSTANTS.denorm_trans:
        value = (value - ACES_RGB_LOG_CONSTANTS.denorm_fake0) * 2

    return value


ACES_RGB_LOG_TRANSFER_FUNCTION = _aces_rgb_log_transfer_function
"""
Transfer function from linear to *ACES RGB Log* colourspace.

ACES_RGB_LOG_TRANSFER_FUNCTION : object
"""

ACES_RGB_LOG_INVERSE_TRANSFER_FUNCTION = (
    _aces_rgb_log_inverse_transfer_function)
"""
Inverse transfer function from *ACES RGB Log* colourspace to linear.

ACES_RGB_LOG_INVERSE_TRANSFER_FUNCTION : object
"""

ACES_RGB_LOG_COLOURSPACE = RGB_Colourspace(
    'ACES RGB Log',
    ACES_RGB_PRIMARIES,
    ACES_RGB_WHITEPOINT,
    ACES_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ACES_RGB_MATRIX,
    ACES_RGB_LOG_TRANSFER_FUNCTION,
    ACES_RGB_LOG_INVERSE_TRANSFER_FUNCTION)
"""
*ACES RGB Log* colourspace.

ACES_RGB_LOG_COLOURSPACE : RGB_Colourspace

References
----------
.. [5]  `Logarithmic Encoding of ACES Data for use within Color Grading Systems
        <http://www.dropbox.com/sh/iwd09buudm3lfod/AAA-X1nVs_XLjWlzNhfhqiIna/ACESlog_v1.0.pdf>`_  # noqa
        (Last accessed 17 May 2014)
"""

ACES_RGB_PROXY_10_CONSTANTS = Structure(
    CV_min=0,
    CV_max=1023,
    steps_per_stop=50,
    mid_CV_offset=425,
    mid_log_offset=-2.5)
"""
*ACES RGB Proxy 10* colourspace constants.

ACES_RGB_PROXY_10_CONSTANTS : Structure
"""

ACES_RGB_PROXY_12_CONSTANTS = Structure(
    CV_min=0,
    CV_max=4095,
    steps_per_stop=200,
    mid_CV_offset=1700,
    mid_log_offset=-2.5)
"""
*ACES RGB Proxy 12* colourspace constants.

ACES_RGB_PROXY_12_CONSTANTS : Structure
"""

ACES_RGB_PROXY_CONSTANTS = CaseInsensitiveMapping(
    {'10 Bit': ACES_RGB_PROXY_10_CONSTANTS,
     '12 Bit': ACES_RGB_PROXY_12_CONSTANTS})
"""
Aggregated *ACES RGB Proxy* colourspace constants.

ACES_RGB_PROXY_CONSTANTS : dict
    ('10 Bit', '12 Bit')
"""


def _aces_rgb_proxy_transfer_function(value, bit_depth='10 Bit'):
    """
    Defines the *ACES RGB Proxy* colourspace transfer function.

    Parameters
    ----------
    value : numeric
        value.
    bit_depth : unicode ('10 Bit', '12 Bit')
        *ACES RGB Proxy* bit depth.

    Returns
    -------
    numeric
        Companded value.
    """

    constants = ACES_RGB_PROXY_CONSTANTS.get(bit_depth)

    if value > 0:
        return max(constants.CV_min,
                   min(constants.CV_max,
                       ((math.log10(value) / (math.log10(2)) -
                         constants.mid_log_offset) * constants.steps_per_stop +
                        constants.mid_CV_offset)) + 0.5)
    else:
        return constants.CV_min


def _aces_rgb_proxy_inverse_transfer_function(value, bit_depth='10 Bit'):
    """
    Defines the *ACES RGB Proxy* colourspace inverse transfer function.

    Parameters
    ----------
    value : numeric
        value.
    bit_depth : str ('10 Bit', '12 Bit')
        *ACES RGB Proxy* bit depth.

    Returns
    -------
    numeric
        Companded value.
    """

    constants = ACES_RGB_PROXY_CONSTANTS.get(bit_depth)

    return math.pow(2,
                    ((((value - constants.mid_CV_offset) /
                       constants.steps_per_stop) + constants.mid_log_offset)))


ACES_RGB_PROXY_10_TRANSFER_FUNCTION = lambda x: (
    _aces_rgb_proxy_transfer_function(x))
"""
Transfer function from linear to *ACES RGB Proxy 10* colourspace.

ACES_RGB_PROXY_10_TRANSFER_FUNCTION : object
"""

ACES_RGB_PROXY_10_INVERSE_TRANSFER_FUNCTION = lambda x: (
    _aces_rgb_proxy_inverse_transfer_function(x))
"""
Inverse transfer function from *ACES RGB Proxy 10* colourspace to linear.

ACES_RGB_PROXY_10_INVERSE_TRANSFER_FUNCTION : object
"""

ACES_RGB_PROXY_12_TRANSFER_FUNCTION = lambda x: (
    _aces_rgb_proxy_transfer_function(x, bit_depth='12 Bit'))
"""
Transfer function from linear to *ACES RGB Proxy 12* colourspace.

ACES_RGB_PROXY_12_TRANSFER_FUNCTION : object
"""

ACES_RGB_PROXY_12_INVERSE_TRANSFER_FUNCTION = lambda x: (
    _aces_rgb_proxy_inverse_transfer_function(x, bit_depth='12 Bit'))
"""
Inverse transfer function from *ACES RGB Proxy 12* colourspace to linear.

ACES_RGB_PROXY_12_INVERSE_TRANSFER_FUNCTION : object
"""

ACES_RGB_PROXY_10_COLOURSPACE = RGB_Colourspace(
    'ACES RGB Proxy 10',
    ACES_RGB_PRIMARIES,
    ACES_RGB_WHITEPOINT,
    ACES_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ACES_RGB_MATRIX,
    ACES_RGB_PROXY_10_TRANSFER_FUNCTION,
    ACES_RGB_PROXY_10_INVERSE_TRANSFER_FUNCTION)
"""
*ACES RGB Proxy 10* colourspace.

ACES_RGB_PROXY_10_COLOURSPACE : RGB_Colourspace

References
----------
.. [6]  `ACESproxy, an Integer Log Encoding of ACES Image Data
        <http://www.dropbox.com/sh/iwd09buudm3lfod/AAAsl8WskbNNAJXh1r0dPlp2a/ACESproxy_v1.1.pdf>`_  # noqa
        (Last accessed 17 May 2014)
"""

ACES_RGB_PROXY_12_COLOURSPACE = RGB_Colourspace(
    'ACES RGB Proxy 12',
    ACES_RGB_PRIMARIES,
    ACES_RGB_WHITEPOINT,
    ACES_RGB_TO_XYZ_MATRIX,
    XYZ_TO_ACES_RGB_MATRIX,
    ACES_RGB_PROXY_12_TRANSFER_FUNCTION,
    ACES_RGB_PROXY_12_INVERSE_TRANSFER_FUNCTION)
"""
*ACES RGB Proxy 12* colourspace.

ACES_RGB_PROXY_12_COLOURSPACE : RGB_Colourspace

References
----------
.. [7]  `ACESproxy, an Integer Log Encoding of ACES Image Data
        <http://www.dropbox.com/sh/iwd09buudm3lfod/AAAsl8WskbNNAJXh1r0dPlp2a/ACESproxy_v1.1.pdf>`_  # noqa
        (Last accessed 17 May 2014)
"""
