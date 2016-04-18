#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Academy Color Encoding System - Log Encodings
=============================================

Defines the *Academy Color Encoding System* (ACES) log encodings:

-   :def:`log_encoding_ACESproxy`
-   :def:`log_decoding_ACESproxy`
-   :def:`log_encoding_ACEScc`
-   :def:`log_decoding_ACEScc`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
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
        Subcommittee. (2014). Specification S-2014-003 - ACEScc , A
        Logarithmic Encoding of ACES Data for use within Color Grading
        Systems. Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [4]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2014). Specification S-2013-001 - ACESproxy , an
        Integer Log Encoding of ACES Image Data. Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping, Structure, as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ACES_PROXY_10_CONSTANTS',
           'ACES_PROXY_12_CONSTANTS',
           'ACES_PROXY_CONSTANTS',
           'log_encoding_ACESproxy',
           'log_decoding_ACESproxy',
           'log_encoding_ACEScc',
           'log_decoding_ACEScc']

ACES_PROXY_10_CONSTANTS = Structure(
    CV_min=64,
    CV_max=940,
    steps_per_stop=50,
    mid_CV_offset=425,
    mid_log_offset=2.5)
"""
*ACESproxy* 10 bit colourspace constants.

ACES_PROXY_10_CONSTANTS : Structure
"""

ACES_PROXY_12_CONSTANTS = Structure(
    CV_min=256,
    CV_max=3760,
    steps_per_stop=200,
    mid_CV_offset=1700,
    mid_log_offset=2.5)
"""
*ACESproxy* 12 bit colourspace constants.

ACES_PROXY_12_CONSTANTS : Structure
"""

ACES_PROXY_CONSTANTS = CaseInsensitiveMapping(
    {'10 Bit': ACES_PROXY_10_CONSTANTS,
     '12 Bit': ACES_PROXY_12_CONSTANTS})
"""
Aggregated *ACESproxy* colourspace constants.

ACES_PROXY_CONSTANTS : CaseInsensitiveMapping
    **{'10 Bit', '12 Bit'}**
"""


def log_encoding_ACESproxy(value, bit_depth='10 Bit', **kwargs):
    """
    Defines the *ACESproxy* colourspace log encoding curve / opto-electronic
    conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    bit_depth : unicode, optional
        **{'10 Bit', '12 Bit'}**,
        *ACESproxy* bit depth.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* encoding curves.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> log_encoding_ACESproxy(0.18)
    426
    """

    value = np.asarray(value)

    constants = ACES_PROXY_CONSTANTS.get(bit_depth)

    CV_min = np.resize(constants.CV_min, value.shape)
    CV_max = np.resize(constants.CV_max, value.shape)

    def float_2_cv(x):
        """
        Converts given numeric to code value.
        """

        return np.maximum(CV_min, np.minimum(CV_max, np.round(x)))

    output = np.where(value > 2 ** -9.72,
                      float_2_cv((np.log2(value) + constants.mid_log_offset) *
                                 constants.steps_per_stop +
                                 constants.mid_CV_offset),
                      np.resize(CV_min, value.shape))

    return as_numeric(output, int)


def log_decoding_ACESproxy(value, bit_depth='10 Bit', **kwargs):
    """
    Defines the *ACESproxy* colourspace log decoding curve / electro-optical
    conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    bit_depth : unicode, optional
        **{'10 Bit', '12 Bit'}**,
        *ACESproxy* bit depth.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* decoding curves.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_ACESproxy(426)  # doctest: +ELLIPSIS
    0.1792444...
    """

    value = np.asarray(value).astype(np.int)

    constants = ACES_PROXY_CONSTANTS.get(bit_depth)

    return (2 ** (((value - constants.mid_CV_offset) /
                   constants.steps_per_stop - constants.mid_log_offset)))


def log_encoding_ACEScc(value, **kwargs):
    """
    Defines the *ACEScc* colourspace log encoding / opto-electronic conversion
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* encoding curves.

    Returns
    -------
    numeric or ndarray
        Encoded value.

    Examples
    --------
    >>> log_encoding_ACEScc(0.18)  # doctest: +ELLIPSIS
    0.4135884...
    """

    value = np.asarray(value)

    output = np.where(value < 0,
                      (np.log2(2 ** -15 * 0.5) + 9.72) / 17.52,
                      (np.log2(2 ** -16 + value * 0.5) + 9.72) / 17.52)
    output = np.where(value >= 2 ** -15,
                      (np.log2(value) + 9.72) / 17.52,
                      output)

    return as_numeric(output)


def log_decoding_ACEScc(value, **kwargs):
    """
    Defines the *ACEScc* colourspace log decoding / electro-optical conversion
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Decoded value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* decoding curves.

    Examples
    --------
    >>> log_decoding_ACEScc(0.4135884024924423)  # doctest: +ELLIPSIS
    0.18...
    """

    value = np.asarray(value)

    output = np.where(value < (9.72 - 15) / 17.52,
                      (2 ** (value * 17.52 - 9.72) - 2 ** -16) * 2,
                      2 ** (value * 17.52 - 9.72))
    output = np.where(value >= (np.log2(65504) + 9.72) / 17.52,
                      65504,
                      output)

    return as_numeric(output)
