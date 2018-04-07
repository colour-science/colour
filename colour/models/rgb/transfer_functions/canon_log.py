# -*- coding: utf-8 -*-
"""
Canon Log Encodings
===================

Defines the *Canon Log* encodings:

-   :func:`colour.models.log_encoding_CanonLog`
-   :func:`colour.models.log_decoding_CanonLog`
-   :func:`colour.models.log_encoding_CanonLog2`
-   :func:`colour.models.log_decoding_CanonLog2`
-   :func:`colour.models.log_encoding_CanonLog3`
-   :func:`colour.models.log_decoding_CanonLog3`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

Notes
-----
-   :cite:`Canona` is available as a *Drivers & Downloads* *Software* for
    Windows 10 (x64) *Operating System*, a copy of the archive is hosted at
    this url: https://drive.google.com/open?id=0B_IQZQdc4Vy8ZGYyY29pMEVwZU0

References
----------
-   :cite:`Canona` : Canon. (n.d.). EOS C300 Mark II - EOS C300 Mark II Input
    Transform Version 2.0 (for Cinema Gamut / BT.2020). Retrieved August 23,
    2016, from https://www.usa.canon.com/internet/portal/us/home/support/\
details/cameras/cinema-eos/eos-c300-mark-ii
-   :cite:`Thorpe2012a` : Thorpe, L. (2012). CANON-LOG TRANSFER CHARACTERISTIC.
    Retrieved from http://downloads.canon.com/CDLC/\
Canon-Log_Transfer_Characteristic_6-20-2012.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full
from colour.utilities import (as_numeric, domain_range_scale, from_range_1,
                              to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'log_encoding_CanonLog', 'log_decoding_CanonLog', 'log_encoding_CanonLog2',
    'log_decoding_CanonLog2', 'log_encoding_CanonLog3',
    'log_decoding_CanonLog3'
]


def log_encoding_CanonLog(x, bit_depth=10, out_legal=True, in_reflection=True):
    """
    Defines the *Canon Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_legal : bool, optional
        Whether the *Canon Log* non-linear data is encoded in legal
        range.
    in_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    numeric or ndarray
        *Canon Log* non-linear data.

    References
    ----------
    -   :cite:`Thorpe2012a`

    Examples
    --------
    >>> log_encoding_CanonLog(0.18) * 100  # doctest: +ELLIPSIS
    34.3389651...
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale('ignore'):
        clog = np.where(x < log_decoding_CanonLog(0.0730597, bit_depth, False),
                        -(0.529136 * (np.log10(-x * 10.1596 + 1)) - 0.0730597),
                        0.529136 * np.log10(10.1596 * x + 1) + 0.0730597)

    clog = full_to_legal(clog, bit_depth) if out_legal else clog

    return as_numeric(from_range_1(clog))


def log_decoding_CanonLog(clog,
                          bit_depth=10,
                          in_legal=True,
                          out_reflection=True):
    """
    Defines the *Canon Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog : numeric or array_like
        *Canon Log* non-linear data.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_legal : bool, optional
        Whether the *Canon Log* non-linear data is encoded in legal
        range.
    out_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    References
    ----------
    -   :cite:`Thorpe2012a`

    Examples
    --------
    >>> log_decoding_CanonLog(34.338965172606912 / 100)  # doctest: +ELLIPSIS
    0.17999999...
    """

    clog = to_domain_1(clog)

    clog = legal_to_full(clog, bit_depth) if in_legal else clog

    x = np.where(clog < 0.0730597,
                 -(10 ** ((0.0730597 - clog) / 0.529136) - 1) / 10.1596,
                 (10 ** ((clog - 0.0730597) / 0.529136) - 1) / 10.1596)

    if out_reflection:
        x = x * 0.9

    return as_numeric(from_range_1(x))


def log_encoding_CanonLog2(x, bit_depth=10, out_legal=True,
                           in_reflection=True):
    """
    Defines the *Canon Log 2* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_legal : bool, optional
        Whether the *Canon Log 2* non-linear data is encoded in legal
        range.
    in_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    numeric or ndarray
        *Canon Log 2* non-linear data.

    References
    ----------
    -   :cite:`Canona`

    Examples
    --------
    >>> log_encoding_CanonLog2(0.18) * 100  # doctest: +ELLIPSIS
    39.8254694...
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale('ignore'):
        clog2 = np.where(
            x < log_decoding_CanonLog2(0.035388128, bit_depth, False),
            -(0.281863093 * (np.log10(-x * 87.09937546 + 1)) - 0.035388128),
            0.281863093 * np.log10(x * 87.09937546 + 1) + 0.035388128)

    clog2 = full_to_legal(clog2, bit_depth) if out_legal else clog2

    return as_numeric(from_range_1(clog2))


def log_decoding_CanonLog2(clog2,
                           bit_depth=10,
                           in_legal=True,
                           out_reflection=True):
    """
    Defines the *Canon Log 2* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog2 : numeric or array_like
        *Canon Log 2* non-linear data.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_legal : bool, optional
        Whether the *Canon Log 2* non-linear data is encoded in legal
        range.
    out_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    References
    ----------
    -   :cite:`Canona`

    Examples
    --------
    >>> log_decoding_CanonLog2(39.825469498316735 / 100)  # doctest: +ELLIPSIS
    0.1799999...
    """

    clog2 = to_domain_1(clog2)

    clog2 = legal_to_full(clog2, bit_depth) if in_legal else clog2

    x = np.where(clog2 < 0.035388128,
                 -(10 **
                   ((0.035388128 - clog2) / 0.281863093) - 1) / 87.09937546,
                 (10 **
                  ((clog2 - 0.035388128) / 0.281863093) - 1) / 87.09937546)

    if out_reflection:
        x = x * 0.9

    return as_numeric(from_range_1(x))


def log_encoding_CanonLog3(x, bit_depth=10, out_legal=True,
                           in_reflection=True):
    """
    Defines the *Canon Log 3* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_legal : bool, optional
        Whether the *Canon Log 3* non-linear data is encoded in legal
        range.
    in_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    numeric or ndarray
        *Canon Log 3* non-linear data.

    Notes
    -----
    -   Introspection of the grafting points by Shaw, N. (2018) shows that the
        *Canon Log 3* IDT was likely derived from its encoding curve as the
        later is grafted at *+/-0.014*::

            >>> clog3 = 0.04076162
            >>> (clog3 - 0.073059361) / 2.3069815
            -0.014000000000000002
            >>> clog3 = 0.105357102
            >>> (clog3 - 0.073059361) / 2.3069815
            0.013999999999999997

    References
    ----------
    -   :cite:`Canona`

    Examples
    --------
    >>> log_encoding_CanonLog3(0.18) * 100  # doctest: +ELLIPSIS
    34.3389369...
    """

    x = to_domain_1(x)

    if in_reflection:
        x = x / 0.9

    with domain_range_scale('ignore'):
        clog3 = np.select(
            (x < log_decoding_CanonLog3(0.04076162, bit_depth, False, False),
             x <= log_decoding_CanonLog3(0.105357102, bit_depth, False, False),
             x > log_decoding_CanonLog3(0.105357102, bit_depth, False, False)),
            (-0.42889912 * np.log10(-x * 14.98325 + 1) + 0.07623209,
             2.3069815 * x + 0.073059361,
             0.42889912 * np.log10(x * 14.98325 + 1) + 0.069886632))

    clog3 = full_to_legal(clog3, bit_depth) if out_legal else clog3

    return as_numeric(from_range_1(clog3))


def log_decoding_CanonLog3(clog3,
                           bit_depth=10,
                           in_legal=True,
                           out_reflection=True):
    """
    Defines the *Canon Log 3* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    clog3 : numeric or array_like
        *Canon Log 3* non-linear data.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_legal : bool, optional
        Whether the *Canon Log 3* non-linear data is encoded in legal
        range.
    out_reflection : bool, optional
        Whether the light level :math:`x` to a camera is reflection.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    References
    ----------
    -   :cite:`Canona`

    Examples
    --------
    >>> log_decoding_CanonLog3(34.338936938868677 / 100)  # doctest: +ELLIPSIS
    0.1800000...
    """

    clog3 = to_domain_1(clog3)

    clog3 = legal_to_full(clog3, bit_depth) if in_legal else clog3

    x = np.select(
        (clog3 < 0.04076162, clog3 <= 0.105357102, clog3 > 0.105357102),
        (-(10 ** ((0.07623209 - clog3) / 0.42889912) - 1) / 14.98325,
         (clog3 - 0.073059361) / 2.3069815,
         (10 ** ((clog3 - 0.069886632) / 0.42889912) - 1) / 14.98325))

    if out_reflection:
        x = x * 0.9

    return as_numeric(from_range_1(x))
