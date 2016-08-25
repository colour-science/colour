#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RED Log Encodings
=================

Defines the *RED* log encodings:

-   :func:`log_encoding_REDLog`
-   :func:`log_decoding_REDLog`
-   :func:`log_encoding_REDLogFilm`
-   :func:`log_decoding_REDLogFilm`
-   :func:`log_encoding_Log3G10`
-   :func:`log_decoding_Log3G10`

Notes
-----
-   The intent of *Log3G10* is that zero maps to zero, 0.18 maps to 1/3, and
    10 stops above 0.18 maps to 1.0. The name indicates this in a similar way
    to the naming conventions of Sony HyperGamma curves.

    The constants used in the functions do not in fact quite hit these values,
    but rather than use corrected constants, the functions here use the
    official Red values, in order to match the output of the Red SDK.

    For those interested, solving for constants which exactly hit 1/3 and 1.0
    yields the following values:

    B = 25.0 * (np.sqrt(4093.0) - 3) / 9.0
    A = 1.0 / np.log10(B * 184.32 + 1.0)

    where the function takes the form:

    Log3G10(x) = A * np.log10(B * x + 1)

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Sony Imageworks. (2012). make.py. Retrieved November 27, 2014, from
        https://github.com/imageworks/OpenColorIO-Configs/\
blob/master/nuke-default/make.py
.. [2]  Nattress, G. (2016). Private Discussion with Shaw, N.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import (
    log_encoding_Cineon,
    log_decoding_Cineon)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['log_encoding_REDLog',
           'log_decoding_REDLog',
           'log_encoding_REDLogFilm',
           'log_decoding_REDLogFilm',
           'log_encoding_Log3G10',
           'log_decoding_Log3G10']


def log_encoding_REDLog(x,
                        black_offset=10 ** ((0 - 1023) / 511)):
    """
    Defines the *REDLog* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Examples
    --------
    >>> log_encoding_REDLog(0.18)  # doctest: +ELLIPSIS
    0.6376218...
    """

    x = np.asarray(x)

    return ((1023 +
             511 * np.log10(x * (1 - black_offset) + black_offset)) / 1023)


def log_decoding_REDLog(y,
                        black_offset=10 ** ((0 - 1023) / 511)):
    """
    Defines the *REDLog* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Examples
    --------
    >>> log_decoding_REDLog(0.637621845988175)  # doctest: +ELLIPSIS
    0.1...
    """

    y = np.asarray(y)

    return (((10 **
              ((1023 * y - 1023) / 511)) - black_offset) /
            (1 - black_offset))


def log_encoding_REDLogFilm(x,
                            black_offset=10 ** ((95 - 685) / 300)):
    """
    Defines the *REDLogFilm* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Examples
    --------
    >>> log_encoding_REDLogFilm(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    return log_encoding_Cineon(x, black_offset)


def log_decoding_REDLogFilm(y,
                            black_offset=10 ** ((95 - 685) / 300)):
    """
    Defines the *REDLogFilm* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.
    black_offset : numeric or array_like
        Black offset.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Examples
    --------
    >>> log_decoding_REDLogFilm(0.457319613085418)  # doctest: +ELLIPSIS
    0.1799999...
    """

    return log_decoding_Cineon(y, black_offset)


def log_encoding_Log3G10(x):
    """
    Defines the *Log3G10* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Examples
    --------
    >>> log_encoding_Log3G10(0.18)  # doctest: +ELLIPSIS
    0.3333336...
    """

    x = np.asarray(x)

    return np.sign(x) * 0.222497 * np.log10((np.abs(x) * 169.379333) + 1)


def log_decoding_Log3G10(y):
    """
    Defines the *Log3G10* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Examples
    --------
    >>> log_decoding_Log3G10(1.0 / 3)  # doctest: +ELLIPSIS
    0.1799994...
    """

    y = np.asarray(y)

    return (np.sign(y) *
            (np.power(10.0, np.abs(y) / 0.222497) - 1) / 169.379333)
