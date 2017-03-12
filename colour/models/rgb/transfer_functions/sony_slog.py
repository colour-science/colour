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
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
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


def log_encoding_SLog(t):
    """
    Defines the *Sony S-Log* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    t : numeric or array_like
        Input light level :math:`t` to a camera.

    Returns
    -------
    numeric or ndarray
        Camera output code :math:`y`.

    Examples
    --------
    >>> log_encoding_SLog(0.18)  # doctest: +ELLIPSIS
    0.3599878...
    """

    t = np.asarray(t)

    return (0.432699 * np.log10(t + 0.037584) + 0.616596) + 0.03


def log_decoding_SLog(y):
    """
    Defines the *Sony S-Log* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Camera output code :math:`y`.

    Returns
    -------
    numeric or ndarray
        Input light level :math:`t` to a camera.

    Examples
    --------
    >>> log_decoding_SLog(0.359987846422154)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    return 10 ** ((y - 0.616596 - 0.03) / 0.432699) - 0.037584


def log_encoding_SLog2(t):
    """
    Defines the *Sony S-Log2* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    t : numeric or array_like
        Input light level :math:`t` to a camera.

    Returns
    -------
    numeric or ndarray
        Camera output code :math:`y`.

    Examples
    --------
    >>> log_encoding_SLog2(0.18)  # doctest: +ELLIPSIS
    0.3849708...
    """

    t = np.asarray(t)

    return ((4 * (16 + 219 * (0.616596 + 0.03 + 0.432699 *
                              (np.log10(0.037584 + t / 0.9))))) / 1023)


def log_decoding_SLog2(y):
    """
    Defines the *Sony S-Log2* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Camera output code :math:`y`.

    Returns
    -------
    numeric or ndarray
        Input light level :math:`t` to a camera.

    Examples
    --------
    >>> log_decoding_SLog2(0.384970815928670)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    return ((10 ** (((((y * 1023 / 4 - 16) / 219) - 0.616596 - 0.03) /
                     0.432699)) - 0.037584) * 0.9)


def log_encoding_SLog3(t):
    """
    Defines the *Sony S-Log3* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    t : numeric or array_like
        Input light level :math:`t` to a camera.

    Returns
    -------
    numeric or ndarray
        Camera output code :math:`y`.

    Examples
    --------
    >>> log_encoding_SLog3(0.18)  # doctest: +ELLIPSIS
    0.4105571...
    """

    t = np.asarray(t)

    return as_numeric(
        np.where(t >= 0.01125000,
                 (420 + np.log10((t + 0.01) /
                                 (0.18 + 0.01)) * 261.5) / 1023,
                 (t * (171.2102946929 - 95) / 0.01125000 + 95) / 1023))


def log_decoding_SLog3(y):
    """
    Defines the *Sony S-Log3* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Camera output code :math:`y`.

    Returns
    -------
    numeric or ndarray
        Input light level :math:`t` to a camera.

    Examples
    --------
    >>> log_decoding_SLog3(0.410557184750733)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    return as_numeric(
        np.where(y >= 171.2102946929 / 1023,
                 ((10 ** ((y * 1023 - 420) / 261.5)) *
                  (0.18 + 0.01) - 0.01),
                 (y * 1023 - 95) * 0.01125000 / (171.2102946929 - 95)))
