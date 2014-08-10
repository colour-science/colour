#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lightness :math:`L*`
====================

Defines *Lightness* :math:`L*` computation objects.

The following methods are available:

-   :func:`lightness_glasser1958`: *Lightness* :math:`L^*` computation of given
    *luminance* :math:`Y` using *Glasser et al. (1958)* method.
-   :func:`lightness_wyszecki1964`: *Lightness* :math:`W^*` computation of
    given *luminance* :math:`Y` using *Wyszecki (1964)* method.
-   :func:`lightness_1976`: *Lightness* :math:`L^*` computation of given
    *luminance* :math:`Y` as per *CIE Lab* implementation.
"""

from __future__ import unicode_literals

from colour.constants import CIE_E, CIE_K
from colour.utilities import warning

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["lightness_glasser1958",
           "lightness_wyszecki1964",
           "lightness_1976",
           "LIGHTNESS_FUNCTIONS",
           "get_lightness"]


def lightness_glasser1958(Y):
    """
    Returns the *Lightness* :math:`L^*` of given *luminance* :math:`Y` using
    *Glasser et al. (1958)* method.

    Parameters
    ----------
    Y : float
        *luminance* :math:`Y`.

    Returns
    -------
    float
        *Lightness* :math:`L^*`.

    Notes
    -----
    -   Input *luminance* :math:`Y` is in domain [0, 100].
    -   Output *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [1]  http://en.wikipedia.org/wiki/Lightness
            (Last accessed 13 April 2014)

    Examples
    --------
    >>> lightness_glasser1958(10.08)
    36.2505626458
    """

    L_star = 25.29 * (Y ** (1. / 3.)) - 18.38

    return L_star


def lightness_wyszecki1964(Y):
    """
    Returns the *Lightness* :math:`W^*` of given *luminance* :math:`Y` using
    *Wyszecki (1964)* method.


    Parameters
    ----------
    Y : float
        *luminance* :math:`Y`.

    Returns
    -------
    float
        *Lightness* :math:`W^*`.

    Notes
    -----
    -   Input *luminance* :math:`Y` is in domain [0, 100].
    -   Output *Lightness* :math:`W^*` is in domain [0, 100].

    References
    ----------
    .. [1]  http://en.wikipedia.org/wiki/Lightness
            (Last accessed 13 April 2014)

    Examples
    --------
    >>> lightness_wyszecki1964(10.08)
    37.0041149128
    """

    if not 1. < Y < 98.:
        warning("'W*' Lightness computation is only applicable for \
1% < 'Y' < 98%, unpredictable results may occur!")

    W = 25. * (Y ** (1. / 3.)) - 17.

    return W


def lightness_1976(Y, Yn=100.):
    """
    Returns the *Lightness* :math:`L^*` of given *luminance* :math:`Y` using
    given reference white *luminance* :math:`Y_n` as per *CIE Lab*
    implementation.

    Parameters
    ----------
    Y : float
        *luminance* :math:`Y`.
    Yn : float, optional
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    float
        *Lightness* :math:`L^*`.

    Notes
    -----
    -   Input *luminance* :math:`Y` and :math:`Y_n` are in domain [0, 100].
    -   Output *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [2]  http://www.poynton.com/PDFs/GammaFAQ.pdf
            (Last accessed 12 April 2014)

    Examples
    --------
    >>> lightness_1976(10.08, 100.)
    37.9856290977
    """

    ratio = Y / Yn
    L = CIE_K * ratio if ratio <= CIE_E else 116. * ratio ** (1. / 3.) - 16

    return L


LIGHTNESS_FUNCTIONS = {"Lightness Glasser 1958": lightness_glasser1958,
                       "Lightness Wyszecki 1964": lightness_wyszecki1964,
                       "Lightness 1976": lightness_1976}
"""
Supported *Lightness* computations methods.

LIGHTNESS_FUNCTIONS : dict
    ("Lightness Glasser 1958", "Lightness Wyszecki 1964", "Lightness 1976")
"""


def get_lightness(Y, Yn=100., method="Lightness 1976"):
    """
    Returns the *Lightness* :math:`L^*` using given method.

    Parameters
    ----------
    Y : float
        *luminance* :math:`Y`.
    Yn : float, optional
        White reference *luminance*.
    method : unicode, optional
        ("Lightness Glasser 1958", "Lightness Wyszecki 1964",
        "Lightness 1976"),
        Computation method.

    Returns
    -------
    float
        *Lightness* :math:`L^*`.

    Notes
    -----
    -   Input *luminance* :math:`Y` and :math:`Y_n` are in domain [0, 100].
    -   Output *Lightness* :math:`L^*` is in domain [0, 100].

    Examples
    --------
    >>> get_lightness(10.08, 100)
    37.9856290977
    """

    if Yn is None or method is not None:
        return LIGHTNESS_FUNCTIONS.get(method)(Y)
    else:
        return lightness_1976(Y, Yn)