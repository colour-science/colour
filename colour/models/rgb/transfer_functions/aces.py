#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Academy Color Encoding System - Log Encodings
=============================================

Defines the *Academy Color Encoding System* (ACES) log encodings:

-   :func:`log_encoding_ACESproxy`
-   :func:`log_decoding_ACESproxy`
-   :func:`log_encoding_ACEScc`
-   :func:`log_decoding_ACEScc`
-   :func:'log_encoding_ACEScct'
-   :func:'log_decoding_ACEScct'

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
        Subcommittee. (2014). Specification S-2014-003 - ACEScc , A
        Logarithmic Encoding of ACES Data for use within Color Grading
        Systems. Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [4]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2014). Specification S-2013-001 - ACESproxy , an
        Integer Log Encoding of ACES Image Data. Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [5]  The Academy of Motion Picture Arts and Sciences. (2016).
        Specification S-2016-001 - ACEScct, A Quasi-Logarithmic
        Encoding of ACES Data for use within Color Grading Systems.
        Retrieved October 10, 2016, from
        https://github.com/ampas/aces-dev/tree/v1.0.3/documents
"""


from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping, Structure, as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ACES_PROXY_10_CONSTANTS',
           'ACES_PROXY_12_CONSTANTS',
           'ACES_PROXY_CONSTANTS',
           'ACES_CCT_CONSTANTS',
           'log_encoding_ACESproxy',
           'log_decoding_ACESproxy',
           'log_encoding_ACEScc',
           'log_decoding_ACEScc',
           'log_encoding_ACEScct',
           'log_decoding_ACEScct']

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

ACES_CCT_CONSTANTS = Structure(
    X_BRK=0.0078125,
    Y_BRK=0.155251141552511,
    A=10.5402377416545,
    B=0.0729055341958355)
"""
*ACEScct* colourspace constants.

ACES_CCT_CONSTANTS : Structure
"""


def log_encoding_ACESproxy(lin_AP1, bit_depth='10 Bit'):
    """
    Defines the *ACESproxy* colourspace log encoding curve / opto-electronic
    transfer function.

    Parameters
    ----------
    lin_AP1 : numeric or array_like
        *lin_AP1* value.
    bit_depth : unicode, optional
        **{'10 Bit', '12 Bit'}**,
        *ACESproxy* bit depth.

    Returns
    -------
    numeric or ndarray
        *ACESproxy* non-linear value.

    Examples
    --------
    >>> log_encoding_ACESproxy(0.18)
    426
    """

    lin_AP1 = np.asarray(lin_AP1)

    constants = ACES_PROXY_CONSTANTS[bit_depth]

    CV_min = np.resize(constants.CV_min, lin_AP1.shape)
    CV_max = np.resize(constants.CV_max, lin_AP1.shape)

    def float_2_cv(x):
        """
        Converts given numeric to code value.
        """

        return np.maximum(CV_min, np.minimum(CV_max, np.round(x)))

    output = np.where(lin_AP1 > 2 ** -9.72,
                      float_2_cv((np.log2(lin_AP1) +
                                  constants.mid_log_offset) *
                                 constants.steps_per_stop +
                                 constants.mid_CV_offset),
                      np.resize(CV_min, lin_AP1.shape))

    return as_numeric(output, int)


def log_decoding_ACESproxy(ACESproxy, bit_depth='10 Bit'):
    """
    Defines the *ACESproxy* colourspace log decoding curve / electro-optical
    transfer function.

    Parameters
    ----------
    ACESproxy : numeric or array_like
        *ACESproxy* non-linear value.
    bit_depth : unicode, optional
        **{'10 Bit', '12 Bit'}**,
        *ACESproxy* bit depth.

    Returns
    -------
    numeric or ndarray
        *lin_AP1* value.

    Examples
    --------
    >>> log_decoding_ACESproxy(426)  # doctest: +ELLIPSIS
    0.1792444...
    """

    ACESproxy = np.asarray(ACESproxy).astype(np.int)

    constants = ACES_PROXY_CONSTANTS[bit_depth]

    return (2 ** (((ACESproxy - constants.mid_CV_offset) /
                   constants.steps_per_stop - constants.mid_log_offset)))


def log_encoding_ACEScc(lin_AP1):
    """
    Defines the *ACEScc* colourspace log encoding / opto-electronic transfer
    function.

    Parameters
    ----------
    lin_AP1 : numeric or array_like
        *lin_AP1* value.

    Returns
    -------
    numeric or ndarray
        *ACEScc* non-linear value.

    Examples
    --------
    >>> log_encoding_ACEScc(0.18)  # doctest: +ELLIPSIS
    0.4135884...
    """

    lin_AP1 = np.asarray(lin_AP1)

    output = np.where(lin_AP1 < 0,
                      (np.log2(2 ** -15 * 0.5) + 9.72) / 17.52,
                      (np.log2(2 ** -16 + lin_AP1 * 0.5) + 9.72) / 17.52)
    output = np.where(lin_AP1 >= 2 ** -15,
                      (np.log2(lin_AP1) + 9.72) / 17.52,
                      output)

    return as_numeric(output)


def log_decoding_ACEScc(ACEScc):
    """
    Defines the *ACEScc* colourspace log decoding / electro-optical transfer
    function.

    Parameters
    ----------
    ACEScc : numeric or array_like
        *ACEScc* non-linear value.

    Returns
    -------
    numeric or ndarray
        *lin_AP1* value.

    Examples
    --------
    >>> log_decoding_ACEScc(0.413588402492442)  # doctest: +ELLIPSIS
    0.1799999...
    """

    ACEScc = np.asarray(ACEScc)

    output = np.where(ACEScc < (9.72 - 15) / 17.52,
                      (2 ** (ACEScc * 17.52 - 9.72) - 2 ** -16) * 2,
                      2 ** (ACEScc * 17.52 - 9.72))
    output = np.where(ACEScc >= (np.log2(65504) + 9.72) / 17.52,
                      65504,
                      output)

    return as_numeric(output)


def log_encoding_ACEScct(lin_AP1):
    """
    Defines the *ACEScct* colourspace log encoding / opto-electronic transfer
    function.

    Parameters
    ----------
    lin_AP1 : numeric or array_like
        *lin_AP1* value.

    Returns
    -------
    numeric or ndarray
        *ACEScct* non-linear value.

    Examples
    --------
    >>> log_encoding_ACEScct(0.18)  # doctest: +ELLIPSIS
    0.4135884...
    """

    constants = ACES_CCT_CONSTANTS

    lin_AP1 = np.asarray(lin_AP1)

    output = np.where(lin_AP1 <= constants.X_BRK,
                      constants.A * lin_AP1 + constants.B,
                      (np.log2(lin_AP1) + 9.72) / 17.52)

    return as_numeric(output)


def log_decoding_ACEScct(ACEScct):
    """
    Defines the *ACEScct* colourspace log decoding / electro-optical transfer
    function.

    Parameters
    ----------
    ACEScct : numeric or array_like
        *ACEScct* non-linear value.

    Returns
    -------
    numeric or ndarray
        *lin_AP1* value.

    Examples
    --------
    >>> log_decoding_ACEScct(0.413588402492442)  # doctest: +ELLIPSIS
    0.1799999...
    """

    constants = ACES_CCT_CONSTANTS

    ACEScct = np.asarray(ACEScct)

    output = np.where(ACEScct > constants.Y_BRK,
                      2 ** (ACEScct * 17.52 - 9.72),
                      (ACEScct - constants.B) / constants.A)

    return as_numeric(output)
