#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sony S-Log Encodings
====================

Defines the *Sony S-Log* log encodings:

-   :def:`log_encoding_SLog`
-   :def:`log_decoding_SLog`

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Gaggioni, H., Dhanendra, P., Yamashita, J., Kawada, N., Endo, K., &
        Clark, C. (n.d.). S-Log: A new LUT for digital production mastering
        and interchange applications. Retrieved from
        http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/\
slog_manual.pdf
.. [2]  Sony Corporation. (n.d.). S-Log Whitepaper. Retrieved from
        http://www.theodoropoulos.info/attachments/076_on S-Log.pdf
.. [3]  Sony Corporation. (n.d.). Technical Summary for
        S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3. Retrieved from
        http://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/2/\
TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf
"""

from __future__ import division, unicode_literals

import numpy as np
from colour.utilities import as_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['log_encoding_SLog',
           'log_decoding_SLog',
           'log_encoding_SLog2',
           'log_decoding_SLog2',
           'log_encoding_SLog3',
           'log_decoding_SLog3']


def log_encoding_SLog(value, **kwargs):
    """
    Defines the *Sony S-Log* log encoding curve / opto-electronic conversion
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
    >>> log_encoding_SLog(0.18)  # doctest: +ELLIPSIS
    0.3599878...
    """

    value = np.asarray(value)

    return (0.432699 * np.log10(value + 0.037584) + 0.616596) + 0.03


def log_decoding_SLog(value, **kwargs):
    """
    Defines the *Sony S-Log* log decoding curve / electro-optical conversion
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* decoding curves.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_SLog(0.35998784642215442)  # doctest: +ELLIPSIS
    0.1...
    """

    value = np.asarray(value)

    return 10 ** ((value - 0.616596 - 0.03) / 0.432699) - 0.037584


def log_encoding_SLog2(value, **kwargs):
    """
    Defines the *Sony S-Log2* log encoding curve / opto-electronic conversion
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
    >>> log_encoding_SLog2(0.18)  # doctest: +ELLIPSIS
    0.3849708...
    """

    value = np.asarray(value)

    return ((4 * (16 + 219 * (0.616596 + 0.03 + 0.432699 *
                              (np.log10(0.037584 + value / 0.9))))) / 1023)


def log_decoding_SLog2(value, **kwargs):
    """
    Defines the *Sony S-Log2* log decoding curve / electro-optical conversion
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* decoding curves.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_SLog2(0.38497081592867027)  # doctest: +ELLIPSIS
    0.1...
    """

    value = np.asarray(value)

    return ((10 ** (((((value * 1023 / 4 - 16) / 219) - 0.616596 - 0.03) /
                     0.432699)) - 0.037584) * 0.9)


def log_encoding_SLog3(value, **kwargs):
    """
    Defines the *Sony S-Log3* log encoding curve / opto-electronic conversion
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
    >>> log_encoding_SLog3(0.18)  # doctest: +ELLIPSIS
    0.4105571...
    """

    value = np.asarray(value)

    return as_numeric(
        np.where(value >= 0.01125000,
                 (420 + np.log10((value + 0.01) /
                                 (0.18 + 0.01)) * 261.5) / 1023,
                 (value * (171.2102946929 - 95) / 0.01125000 + 95) / 1023))


def log_decoding_SLog3(value, **kwargs):
    """
    Defines the *Sony S-Log3* log decoding curve / electro-optical conversion
    function.

    Parameters
    ----------
    value : numeric or array_like
        Value.
    \**kwargs : dict, optional
        Unused parameter provided for signature compatibility with other
        *log* decoding curves.

    Returns
    -------
    numeric or ndarray
        Decoded value.

    Examples
    --------
    >>> log_decoding_SLog3(0.41055718475073316)  # doctest: +ELLIPSIS
    0.1...
    """

    value = np.asarray(value)

    return as_numeric(
        np.where(value >= 171.2102946929 / 1023,
                 ((10 ** ((value * 1023 - 420) / 261.5)) *
                  (0.18 + 0.01) - 0.01),
                 (value * 1023 - 95) * 0.01125000 / (171.2102946929 - 95)))
