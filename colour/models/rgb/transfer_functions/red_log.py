# -*- coding: utf-8 -*-
"""
RED Log Encodings
=================

Defines the *RED* log encodings:

-   :func:`colour.models.log_encoding_REDLog`
-   :func:`colour.models.log_decoding_REDLog`
-   :func:`colour.models.log_encoding_REDLogFilm`
-   :func:`colour.models.log_decoding_REDLogFilm`
-   :func:`colour.models.log_encoding_Log3G10`
-   :func:`colour.models.log_decoding_Log3G10`
-   :func:`colour.models.log_encoding_Log3G12`
-   :func:`colour.models.log_decoding_Log3G12`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Nattress2016a` : Nattress, G. (2016). Private Discussion with
    Shaw, N.
-   :cite:`SonyImageworks2012a` : Sony Imageworks. (2012). make.py. Retrieved
    November 27, 2014, from https://github.com/imageworks/OpenColorIO-Configs/\
blob/master/nuke-default/make.py
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import (log_encoding_Cineon,
                                                  log_decoding_Cineon)

from colour.utilities import from_range_1, to_domain_1

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'log_encoding_REDLog', 'log_decoding_REDLog', 'log_encoding_REDLogFilm',
    'log_decoding_REDLogFilm', 'log_encoding_Log3G10', 'log_decoding_Log3G10',
    'log_encoding_Log3G12', 'log_decoding_Log3G12'
]


def log_encoding_REDLog(x, black_offset=10 ** ((0 - 1023) / 511)):
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

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_encoding_REDLog(0.18)  # doctest: +ELLIPSIS
    0.6376218...
    """

    x = to_domain_1(x)

    y = (1023 + 511 * np.log10(x * (1 - black_offset) + black_offset)) / 1023

    return from_range_1(y)


def log_decoding_REDLog(y, black_offset=10 ** ((0 - 1023) / 511)):
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

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_decoding_REDLog(0.637621845988175)  # doctest: +ELLIPSIS
    0.1...
    """

    y = to_domain_1(y)

    x = ((10 ** ((1023 * y - 1023) / 511)) - black_offset) / (1 - black_offset)

    return from_range_1(x)


def log_encoding_REDLogFilm(x, black_offset=10 ** ((95 - 685) / 300)):
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

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_encoding_REDLogFilm(0.18)  # doctest: +ELLIPSIS
    0.4573196...
    """

    return log_encoding_Cineon(x, black_offset)


def log_decoding_REDLogFilm(y, black_offset=10 ** ((95 - 685) / 300)):
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

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`SonyImageworks2012a`

    Examples
    --------
    >>> log_decoding_REDLogFilm(0.457319613085418)  # doctest: +ELLIPSIS
    0.1799999...
    """

    return log_decoding_Cineon(y, black_offset)


def log_encoding_Log3G10(x, legacy_curve=False):
    """
    Defines the *Log3G10* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.
    legacy_curve : bool, optional
        Whether to use the v1 *Log3G10* log encoding curve. Default is *False*.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The v1 *Log3G10* log encoding curve is the one used in *REDCINE-X beta
        42*. *Resolve 12.5.2* also uses the v1 curve. *RED* is planning to use
        v2 *Log3G10* log encoding curve in the release version of the
        *RED SDK*.
        Use the `legacy_curve=True` argument to switch to the v1 curve for
        compatibility with the current (as of September 21, 2016) *RED SDK*.
    -   The intent of the v1 *Log3G10* log encoding curve is that zero maps to
        zero, 0.18 maps to 1/3, and 10 stops above 0.18 maps to 1.0.
        The name indicates this in a similar way to the naming conventions of
        *Sony HyperGamma* curves.

        The constants used in the functions do not in fact quite hit these
        values, but rather than use corrected constants, the functions here
        use the official *RED* values, in order to match the output of the
        *RED SDK*.

        For those interested, solving for constants which exactly hit 1/3
        and 1.0 yields the following values::

            B = 25 * (np.sqrt(4093.0) - 3) / 9
            A = 1 / np.log10(B * 184.32 + 1)

        where the function takes the form::

            Log3G10(x) = A * np.log10(B * x + 1)

        Similarly for *Log3G12*, the values which hit exactly 1/3 and 1.0
        are::

            B = 25 * (np.sqrt(16381.0) - 3) / 9
            A = 1 / np.log10(B * 737.28 + 1)

    References
    ----------
    :cite:`Nattress2016a`

    Examples
    --------
    >>> log_encoding_Log3G10(0.18, legacy_curve=True)  # doctest: +ELLIPSIS
    0.3333336...
    >>> log_encoding_Log3G10(0.0)  # doctest: +ELLIPSIS
    0.0915514...
    """

    x = to_domain_1(x)

    if legacy_curve:
        y = np.sign(x) * 0.222497 * np.log10((np.abs(x) * 169.379333) + 1)
    else:
        y = (np.sign(x + 0.01) * 0.224282 *
             np.log10((np.abs(x + 0.01) * 155.975327) + 1))

    return from_range_1(y)


def log_decoding_Log3G10(y, legacy_curve=False):
    """
    Defines the *Log3G10* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.
    legacy_curve : bool, optional
        Whether to use the v1 *Log3G10* log encoding curve. Default is *False*.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Nattress2016a`

    Examples
    --------
    >>> log_decoding_Log3G10(1.0 / 3, legacy_curve=True)  # doctest: +ELLIPSIS
    0.1799994...
    >>> log_decoding_Log3G10(1.0)  # doctest: +ELLIPSIS
    184.3223476...
    """

    y = to_domain_1(y)

    if legacy_curve:
        x = (np.sign(y) * (10.0 ** (np.abs(y) / 0.222497) - 1) / 169.379333)
    else:
        x = (np.sign(y) * (10.0 **
                           (np.abs(y) / 0.224282) - 1) / 155.975327) - 0.01

    return from_range_1(x)


def log_encoding_Log3G12(x):
    """
    Defines the *Log3G12* log encoding curve / opto-electronic transfer
    function.

    Parameters
    ----------
    x : numeric or array_like
        Linear data :math:`x`.

    Returns
    -------
    numeric or ndarray
        Non-linear data :math:`y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Nattress2016a`

    Examples
    --------
    >>> log_encoding_Log3G12(0.18)  # doctest: +ELLIPSIS
    0.3333326...
    """

    x = to_domain_1(x)

    y = np.sign(x) * 0.184904 * np.log10((np.abs(x) * 347.189667) + 1)

    return from_range_1(y)


def log_decoding_Log3G12(y):
    """
    Defines the *Log3G12* log decoding curve / electro-optical transfer
    function.

    Parameters
    ----------
    y : numeric or array_like
        Non-linear data :math:`y`.

    Returns
    -------
    numeric or ndarray
        Linear data :math:`x`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``x``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Nattress2016a`

    Examples
    --------
    >>> log_decoding_Log3G12(1.0 / 3)  # doctest: +ELLIPSIS
    0.1800015...
    """

    y = to_domain_1(y)

    x = np.sign(y) * (10.0 ** (np.abs(y) / 0.184904) - 1) / 347.189667

    return from_range_1(x)
