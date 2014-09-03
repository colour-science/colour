#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lightness :math:`L*`
====================

Defines *Lightness* :math:`L*` computation objects.

The following methods are available:

-   :func:`lightness_glasser1958`: *Lightness* :math:`L^*` computation of given
    *luminance* :math:`Y` using *Glasser et al. (1958)* method.
-   :func:`lightness_wyszecki1964`: *Lightness* :math:`W` computation of
    given *luminance* :math:`Y` using *Wyszecki (1964)* method.
-   :func:`lightness_1976`: *Lightness* :math:`L^*` computation of given
    *luminance* :math:`Y` as per *CIE Lab* implementation.

See Also
--------
`Lightness IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/colorimetry/lightness.ipynb>`_  # noqa

References
----------
.. [1]  http://en.wikipedia.org/wiki/Lightness
        (Last accessed 13 April 2014)
"""

from __future__ import division, unicode_literals

from colour.constants import CIE_E, CIE_K
from colour.utilities import CaseInsensitiveMapping, warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['lightness_glasser1958',
           'lightness_wyszecki1964',
           'lightness_1976',
           'LIGHTNESS_METHODS',
           'lightness']


def lightness_glasser1958(Y, **kwargs):
    """
    Returns the *Lightness* :math:`L` of given *luminance* :math:`Y` using
    *Glasser et al. (1958)* method.

    Parameters
    ----------
    Y : numeric
        *luminance* :math:`Y`.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *Lightness* computation objects.

    Returns
    -------
    numeric
        *Lightness* :math:`L`.

    Notes
    -----
    -   Input *luminance* :math:`Y` is in domain [0, 100].
    -   Output *Lightness* :math:`L` is in domain [0, 100].

    References
    ----------
    .. [2]  **Glasser et al.**, *Cube-Root Color Coordinate System*,
            *JOSA, Vol. 48, Issue 10, pp. 736-740 (1958)*,
            DOI: http://dx.doi.org/10.1364/JOSA.48.000736

    Examples
    --------
    >>> lightness_glasser1958(10.08)  # doctest: +ELLIPSIS
    36.2505626...
    """

    L = 25.29 * (Y ** (1 / 3)) - 18.38

    return L


def lightness_wyszecki1964(Y, **kwargs):
    """
    Returns the *Lightness* :math:`W` of given *luminance* :math:`Y` using
    *Wyszecki (1964)* method.


    Parameters
    ----------
    Y : numeric
        *luminance* :math:`Y`.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *Lightness* computation objects.

    Returns
    -------
    numeric
        *Lightness* :math:`W`.

    Notes
    -----
    -   Input *luminance* :math:`Y` is in domain [0, 100].
    -   Output *Lightness* :math:`W` is in domain [0, 100].

    References
    ----------
    .. [3]  **G. Wyszecki**, *Proposal for a New Color-Difference Formula*,
            *JOSA, Vol. 53, Issue 11, pp. 1318-1319 (1963)*,
            DOI: http://dx.doi.org/10.1364/JOSA.53.001318

    Examples
    --------
    >>> lightness_wyszecki1964(10.08)  # doctest: +ELLIPSIS
    37.0041149...
    """

    if not 1 < Y < 98:
        warning(('"W*" Lightness computation is only applicable for '
                 '1% < "Y" < 98%, unpredictable results may occur!'))

    W = 25 * (Y ** (1 / 3)) - 17

    return W


def lightness_1976(Y, Y_n=100):
    """
    Returns the *Lightness* :math:`L^*` of given *luminance* :math:`Y` using
    given reference white *luminance* :math:`Y_n` as per *CIE Lab*
    implementation.

    Parameters
    ----------
    Y : numeric
        *luminance* :math:`Y`.
    Y_n : numeric, optional
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    numeric
        *Lightness* :math:`L^*`.

    Notes
    -----
    -   Input *luminance* :math:`Y` and :math:`Y_n` are in domain [0, 100].
    -   Output *Lightness* :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [4]  **Wyszecki & Stiles**,
            *Color Science - Concepts and Methods Data and Formulae -
            Second Edition*,
            Wiley Classics Library Edition, published 2000,
            ISBN-10: 0-471-39918-3,
            page 167.
    .. [5]  http://brucelindbloom.com/index.html?LContinuity.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> lightness_1976(10.08)  # doctest: +ELLIPSIS
    37.9856290...
    """

    ratio = Y / Y_n
    Lstar = CIE_K * ratio if ratio <= CIE_E else 116 * ratio ** (1 / 3) - 16

    return Lstar


LIGHTNESS_METHODS = CaseInsensitiveMapping(
    {'Glasser 1958': lightness_glasser1958,
     'Wyszecki 1964': lightness_wyszecki1964,
     'CIE 1976': lightness_1976})
"""
Supported *Lightness* computations methods.

LIGHTNESS_METHODS : dict
    ('Glasser 1958', 'Wyszecki 1964', 'CIE 1976')

Aliases:

-   'Lstar1976': 'CIE 1976'
"""
LIGHTNESS_METHODS['Lstar1976'] = LIGHTNESS_METHODS['CIE 1976']


def lightness(Y, method='CIE 1976', **kwargs):
    """
    Returns the *Lightness* :math:`L^*` using given method.

    Parameters
    ----------
    Y : numeric
        *luminance* :math:`Y`.
    method : unicode, optional
        ('Glasser 1958', 'Wyszecki 1964', 'CIE 1976'),
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    numeric
        *Lightness* :math:`L^*`.

    Notes
    -----
    -   Input *luminance* :math:`Y` and optional :math:`Y_n` are in domain
        [0, 100].
    -   Output *Lightness* :math:`L^*` is in domain [0, 100].

    Examples
    --------
    >>> lightness(10.08)  # doctest: +ELLIPSIS
    37.9856290...
    >>> lightness(10.08, Y_n=100)  # doctest: +ELLIPSIS
    37.9856290...
    >>> lightness(10.08, Y_n=95)  # doctest: +ELLIPSIS
    38.9165987...
    >>> lightness(10.08, method='Glasser 1958')  # doctest: +ELLIPSIS
    36.2505626...
    >>> lightness(10.08, method='Wyszecki 1964')  # doctest: +ELLIPSIS
    37.0041149...
    """

    return LIGHTNESS_METHODS.get(method)(Y, **kwargs)
