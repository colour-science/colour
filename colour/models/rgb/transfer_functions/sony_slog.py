#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sony S-Log Encodings
====================

Defines the *Sony S-Log* log encodings:

-   :func:`log_encoding_SLog`
-   :func:`log_decoding_SLog`
-   :func:`log_encoding_SLog2`
-   :func:`log_decoding_SLog2`
-   :func:`log_encoding_SLog3`
-   :func:`log_decoding_SLog3`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Sony Corporation. (2012). S-Log2 Technical Paper. Retrieved from
        https://pro.sony.com/bbsccms/assets/files/micro/dmpc/training/\
S-Log2_Technical_PaperV1_0.pdf
.. [2]  Sony Corporation. (n.d.). Technical Summary for
        S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3. Retrieved from
        http://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/2/\
TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf
"""

from __future__ import division, unicode_literals

import numpy as np
from colour.utilities import as_numeric
from colour.models.rgb.transfer_functions import full_to_legal, legal_to_full

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'log_encoding_SLog', 'log_decoding_SLog', 'log_encoding_SLog2',
    'log_decoding_SLog2', 'log_encoding_SLog3', 'log_decoding_SLog3'
]


def log_encoding_SLog(x, bit_depth=10, out_legal=False, in_reflection=True):
    """
    Defines the *Sony S-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        :math:`IRE` in Scene-Linear space.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_legal : bool, optional
        Whether the :math`IRE` in S-Log space are legal.
    in_reflection : bool, optional
        Whether the :math:`IRE` in Scene-Linear space are reflection.

    Returns
    -------
    numeric or ndarray
        :math`IRE` in S-Log space.

    Examples
    --------
    >>> log_encoding_SLog(0.18)  # doctest: +ELLIPSIS
    0.3765127...
    >>> log_encoding_SLog(0.18, out_legal=True)  # doctest: +ELLIPSIS
    0.3849708...
    >>> log_encoding_SLog(0.18, in_reflection=False)  # doctest: +ELLIPSIS
    0.3599878...
    """

    x = np.asarray(x)

    if in_reflection:
        x = x / 0.9

    y = np.where(x >= 0,
                 ((0.432699 * np.log10(x + 0.037584) + 0.616596) + 0.03),
                 x * 5 + 0.030001222851889303)

    y = full_to_legal(y, bit_depth) if out_legal else y

    return as_numeric(y)


def log_decoding_SLog(y, bit_depth=10, in_legal=False, out_reflection=True):
    """
    Defines the *Sony S-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        :math`IRE` in S-Log space.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_legal : bool, optional
        Whether the :math`IRE` in S-Log space space are legal.
    out_reflection : bool, optional
        Whether the :math:`IRE` in Scene-Linear space are reflection.

    Returns
    -------
    numeric or ndarray
        :math:`IRE` in Scene-Linear space.

    Examples
    --------
    >>> log_decoding_SLog(0.37651272225459997)  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_SLog(
    ...     0.38497081592867027, in_legal=True)  # doctest: +ELLIPSIS
    0.1...
    >>> log_decoding_SLog(
    ...     0.35998784642215442, out_reflection=True)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    x = legal_to_full(y, bit_depth) if in_legal else y

    x = np.where(y >= log_encoding_SLog(0.0, bit_depth, in_legal),
                 10 ** ((x - 0.616596 - 0.03) / 0.432699) - 0.037584,
                 (x - 0.030001222851889303) / 5.0)

    if out_reflection:
        x = x * 0.9

    return as_numeric(x)


def log_encoding_SLog2(x, bit_depth=10, out_legal=False, in_reflection=True):
    """
    Defines the *Sony S-Log2* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        :math:`IRE` in Scene-Linear space.
    bit_depth : int, optional
        Bit depth used for conversion.
    out_legal : bool, optional
        Whether the :math`IRE` in S-Log2 space are legal.
    in_reflection : bool, optional
        Whether the :math:`IRE` in Scene-Linear space are reflection.

    Returns
    -------
    numeric or ndarray
        :math`IRE` in S-Log2 space.

    Examples
    --------
    >>> log_encoding_SLog2(0.18)  # doctest: +ELLIPSIS
    0.3234495...
    """

    return log_encoding_SLog(x * 155 / 219, bit_depth, out_legal,
                             in_reflection)


def log_decoding_SLog2(y, bit_depth=10, in_legal=False, as_reflection=True):
    """
    Defines the *Sony S-Log2* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        :math`IRE` in S-Log2 space.
    bit_depth : int, optional
        Bit depth used for conversion.
    in_legal : bool, optional
        Whether the :math`IRE` in S-Log2 space space are legal.
    out_reflection : bool, optional
        Whether the :math:`IRE` in Scene-Linear space are reflection.

    Returns
    -------
    numeric or ndarray
        :math:`IRE` in Scene-Linear space.

    Examples
    --------
    >>> log_decoding_SLog2(0.32344951221501261)  # doctest: +ELLIPSIS
    0.1...
    """

    return 219 * log_decoding_SLog(y, bit_depth, in_legal, as_reflection) / 155


def log_encoding_SLog3(x):
    """
    Defines the *Sony S-Log3* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        :math:`IRE` in Scene-Linear space.

    Returns
    -------
    numeric or ndarray
        :math`IRE` in S-Log3 space.

    Examples
    --------
    >>> log_encoding_SLog3(0.18)  # doctest: +ELLIPSIS
    0.4105571...
    """

    x = np.asarray(x)

    y = np.where(x >= 0.01125000, (420 + np.log10(
        (x + 0.01) / (0.18 + 0.01)) * 261.5) / 1023,
                 (x * (171.2102946929 - 95) / 0.01125000 + 95) / 1023)

    return as_numeric(y)


def log_decoding_SLog3(y):
    """
    Defines the *Sony S-Log3* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        :math`IRE` in S-Log3 space.

    Returns
    -------
    numeric or ndarray
        :math:`IRE` in Scene-Linear space.

    Examples
    --------
    >>> log_decoding_SLog3(0.410557184750733)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    x = np.where(y >= 171.2102946929 / 1023,
                 ((10 ** ((y * 1023 - 420) / 261.5)) * (0.18 + 0.01) - 0.01),
                 (y * 1023 - 95) * 0.01125000 / (171.2102946929 - 95))

    return as_numeric(x)
