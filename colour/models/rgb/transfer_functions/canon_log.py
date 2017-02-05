#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canon Log Encodings
===================

Defines the *Canon Log* encodings:

-   :func:`log_encoding_CanonLog`
-   :func:`log_decoding_CanonLog`
-   :func:`log_encoding_CanonLog2`
-   :func:`log_decoding_CanonLog2`
-   :func:`log_encoding_CanonLog3`
-   :func:`log_decoding_CanonLog3`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Thorpe, L. (2012). CANON-LOG TRANSFER CHARACTERISTIC. Retrieved
        from http://downloads.canon.com/CDLC/\
Canon-Log_Transfer_Characteristic_6-20-2012.pdf
.. [2]  Canon. (n.d.). EOS C300 Mark II - EOS C300 Mark II Input Transform
        Version 2.0 (for Cinema Gamut / BT.2020). Retrieved August 23, 2016,
        from https://www.usa.canon.com/internet/portal/us/home/support/\
details/cameras/cinema-eos/eos-c300-mark-ii

Notes
-----
-   [2]_ is available as a *Drivers & Downloads* *Software* for Windows 10
    (x64) *Operating System*, a copy of the archive is hosted at this url:
    https://drive.google.com/open?id=0B_IQZQdc4Vy8ZGYyY29pMEVwZU0
"""

from __future__ import division, unicode_literals

import numpy as np
from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_CanonLog',
           'log_decoding_CanonLog',
           'log_encoding_CanonLog2',
           'log_decoding_CanonLog2',
           'log_encoding_CanonLog3',
           'log_decoding_CanonLog3']


def log_encoding_CanonLog(x):
    """
    Defines the *Canon Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.

    Returns
    -------
    numeric or ndarray
        *Canon Log* non-linear *IRE* data.
    Notes
    -----
    -   Output *Canon Log* non-linear *IRE* data should be converted to code
        value *CV* as follows: `CV = IRE * (940 - 64) + 64`.

    Examples
    --------
    >>> log_encoding_CanonLog(0.20) * 100  # doctest: +ELLIPSIS
    32.7953896...
    """

    x = np.asarray(x)

    clog_ire = np.where(
        x < log_decoding_CanonLog(0.0730597),
        -(0.529136 * (np.log10(-x * 10.1596 + 1)) - 0.0730597),
        0.529136 * np.log10(10.1596 * x + 1) + 0.0730597)

    return as_numeric(clog_ire)


def log_decoding_CanonLog(clog_ire):
    """
    Defines the *Canon Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog_ire : numeric or array_like
        *Canon Log* non-linear *IRE* data.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Notes
    -----
    -   Input *Canon Log* non-linear *IRE* data should be converted from code
        value *CV* to *IRE* as follows: `IRE = (CV - 64) / (940 - 64)`.

    Examples
    --------
    >>> log_decoding_CanonLog(32.795389693580908 / 100)  # doctest: +ELLIPSIS
    0.19999999...
    """

    clog_ire = np.asarray(clog_ire)

    x = np.where(
        clog_ire < 0.0730597,
        -(10 ** ((0.0730597 - clog_ire) / 0.529136) - 1) / 10.1596,
        (10 ** ((clog_ire - 0.0730597) / 0.529136) - 1) / 10.1596)

    return as_numeric(x)


def log_encoding_CanonLog2(x):
    """
    Defines the *Canon Log 2* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.

    Returns
    -------
    numeric or ndarray
        *Canon Log 2* non-linear *IRE* data.
    Notes
    -----
    -   Output *Canon Log 2* non-linear *IRE* data should be converted to code
        value *CV* as follows: `CV = IRE * (940 - 64) + 64`.

    Examples
    --------
    >>> log_encoding_CanonLog2(0.20) * 100  # doctest: +ELLIPSIS
    39.2025745...
    """

    x = np.asarray(x)

    clog2_ire = np.where(
        x < log_decoding_CanonLog2(0.035388128),
        -(0.281863093 * (np.log10(-x * 87.09937546 + 1)) - 0.035388128),
        0.281863093 * np.log10(x * 87.09937546 + 1) + 0.035388128)

    return as_numeric(clog2_ire)


def log_decoding_CanonLog2(clog2_ire):
    """
    Defines the *Canon Log 2* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog2_ire : numeric or array_like
        *Canon Log 2* non-linear *IRE* data.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Notes
    -----
    -   Input *Canon Log 2* non-linear *IRE* data should be converted from code
        value *CV* to *IRE* as follows: `IRE = (CV - 64) / (940 - 64)`.

    Examples
    --------
    >>> log_decoding_CanonLog2(39.202574539700947 / 100)  # doctest: +ELLIPSIS
    0.2000000...
    """

    clog2_ire = np.asarray(clog2_ire)

    x = np.where(
        clog2_ire < 0.035388128,
        -(10 ** ((0.035388128 - clog2_ire) / 0.281863093) - 1) / 87.09937546,
        (10 ** ((clog2_ire - 0.035388128) / 0.281863093) - 1) / 87.09937546)

    return as_numeric(x)


def log_encoding_CanonLog3(x):
    """
    Defines the *Canon Log 3* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.

    Returns
    -------
    numeric or ndarray
        *Canon Log 3* non-linear *IRE* data.
    Notes
    -----
    -   Output *Canon Log 3* non-linear *IRE* data should be converted to code
        value *CV* as follows: `CV = IRE * (940 - 64) + 64`.

    Examples
    --------
    >>> log_encoding_CanonLog3(0.20) * 100  # doctest: +ELLIPSIS
    32.7953567...
    """

    x = np.asarray(x)

    clog3_ire = np.select(
        (x < log_decoding_CanonLog3(0.04076162),
         x <= log_decoding_CanonLog3(0.105357102),
         x > log_decoding_CanonLog3(0.105357102)),
        (-(0.42889912 * (np.log10(-x * 14.98325 + 1)) - 0.069886632),
         2.3069815 * x + 0.073059361,
         0.42889912 * np.log10(x * 14.98325 + 1) + 0.069886632))

    return as_numeric(clog3_ire)


def log_decoding_CanonLog3(clog3_ire):
    """
    Defines the *Canon Log 3* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog3_ire : numeric or array_like
        *Canon Log 3* non-linear *IRE* data.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Notes
    -----
    -   Input *Canon Log 3* non-linear *IRE* data should be converted from code
        value *CV* to *IRE* as follows: `IRE = (CV - 64) / (940 - 64)`.

    Examples
    --------
    >>> log_decoding_CanonLog3(32.795356721989336 / 100)  # doctest: +ELLIPSIS
    0.2000000...
    """

    clog3_ire = np.asarray(clog3_ire)

    x = np.select(
        (clog3_ire < 0.04076162,
         clog3_ire <= 0.105357102,
         clog3_ire > 0.105357102),
        (-(10 ** ((0.069886632 - clog3_ire) / 0.42889912) - 1) / 14.98325,
         (clog3_ire - 0.073059361) / 2.3069815,
         (10 ** ((clog3_ire - 0.069886632) / 0.42889912) - 1) / 14.98325))

    return as_numeric(x)
