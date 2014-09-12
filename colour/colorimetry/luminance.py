#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Luminance :math:`Y`
===================

Defines *luminance* :math:`Y` computation objects.

The following methods are available:

-   :func:`luminance_newhall1943`: *luminance* :math:`Y` computation of given
    *Munsell* value :math:`V` using *Newhall, Nickerson, and Judd (1943)*
    method.
-   :func:`luminance_ASTM_D1535_08`: *luminance* :math:`Y` computation of given
    *Munsell* value :math:`V` using *ASTM D1535-08e1 (2008)* method.
-   :func:`luminance_1976`: *luminance* :math:`Y` computation of given
    *Lightness* :math:`L^*` as per *CIE Lab* implementation.

See Also
--------
`Luminance IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/colorimetry/luminance.ipynb>`_  # noqa
"""

from __future__ import division, unicode_literals

from colour.constants import CIE_E, CIE_K
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['luminance_newhall1943',
           'luminance_ASTM_D1535_08',
           'luminance_1976',
           'LUMINANCE_METHODS',
           'luminance']


def luminance_newhall1943(V, **kwargs):
    """
    Returns the *luminance* :math:`R_Y` of given *Munsell* value :math:`V`
    using *Sidney M. Newhall, Dorothy Nickerson, and Deane B. Judd (1943)*
    method.

    Parameters
    ----------
    V : numeric
        *Munsell* value :math:`V`.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *luminance* computation objects.

    Returns
    -------
    numeric
        *luminance* :math:`R_Y`.

    Notes
    -----
    -   Input *Munsell* value :math:`V` is in domain [0, 10].
    -   Output *luminance* :math:`R_Y` is in domain [0, 100].

    References
    ----------
    .. [1]  **Sidney M. Newhall, Dorothy Nickerson, and Deane B. Judd**,
            *Final Report of the O.S.A. Subcommittee on the Spacing of the
            Munsell Colors*,
            *JOSA, Vol. 33, Issue 7, pp. 385-411 (1943)*,
            DOI: http://dx.doi.org/10.1364/JOSA.33.000385

    Examples
    --------
    >>> luminance_newhall1943(3.74629715382)  # doctest: +ELLIPSIS
    10.4089874...
    """

    R_Y = 1.2219 * V - 0.23111 * (V * V) + 0.23951 * (V ** 3) - 0.021009 * (
        V ** 4) + 0.0008404 * (V ** 5)

    return R_Y


def luminance_ASTM_D1535_08(V, **kwargs):
    """
    Returns the *luminance* :math:`Y` of given *Munsell* value :math:`V` using
    *ASTM D1535-08e1 (2008)* method.

    Parameters
    ----------
    V : numeric
        *Munsell* value :math:`V`.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        *luminance* computation objects.

    Returns
    -------
    numeric
        *luminance* :math:`Y`.

    Notes
    -----
    -   Input *Munsell* value :math:`V` is in domain [0, 10].
    -   Output *luminance* :math:`Y` is in domain [0, 100].

    References
    ----------
    .. [4]  `ASTM D1535-08e1 - Standard Practice for Specifying Color by the
            Munsell System
            <http://www.scribd.com/doc/89648322/ASTM-D1535-08e1-Standard-Practice-for-Specifying-Color-by-the-Munsell-System>`_,  # noqa
            DOI: http://dx.doi.org/10.1520/D1535-13

    Examples
    --------
    >>> luminance_ASTM_D1535_08(3.74629715382)  # doctest: +ELLIPSIS
    10.1488096...
    """

    Y = (1.1914 * V - 0.22533 * (V ** 2) + 0.23352 * (V ** 3) - 0.020484 *
         (V ** 4) + 0.00081939 * (V ** 5))

    return Y


def luminance_1976(Lstar, Y_n=100):
    """
    Returns the *luminance* :math:`Y` of given *Lightness* :math:`L^*` with
    given reference white *luminance* :math:`Y_n`.

    Parameters
    ----------
    L : numeric
        *Lightness* :math:`L^*`
    Yn : numeric
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    numeric
        *luminance* :math:`Y`.

    Notes
    -----
    -   Input *Lightness* :math:`L^*` and reference white *luminance*
        :math:`Y_n` are in domain [0, 100].
    -   Output *luminance* :math:`Y` is in domain [0, 100].

    References
    ----------
    .. [2]  **Wyszecki & Stiles**,
            *Color Science - Concepts and Methods Data and Formulae -
            Second Edition*,
            Wiley Classics Library Edition, published 2000,
            ISBN-10: 0-471-39918-3,
            page 167.
    .. [3]  http://brucelindbloom.com/index.html?LContinuity.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> luminance_1976(37.9856290977)  # doctest: +ELLIPSIS
    10.0800000...
    >>> luminance_1976(37.9856290977, 95)  # doctest: +ELLIPSIS
    9.5760000...
    """

    Y = (Y_n * ((Lstar + 16) / 116) ** 3
         if Lstar > CIE_K * CIE_E else
         Y_n * (Lstar / CIE_K))

    return Y


LUMINANCE_METHODS = CaseInsensitiveMapping(
    {'Newhall 1943': luminance_newhall1943,
     'ASTM D1535-08': luminance_ASTM_D1535_08,
     'CIE 1976': luminance_1976})
"""
Supported *luminance* computations methods.

LUMINANCE_METHODS : CaseInsensitiveMapping
    {'Newhall 1943', 'ASTM D1535-08', 'CIE 1976'}

Aliases:

-   'astm2008': 'ASTM D1535-08'
-   'cie1976': 'CIE 1976'
"""
LUMINANCE_METHODS['astm2008'] = (
    LUMINANCE_METHODS['ASTM D1535-08'])
LUMINANCE_METHODS['cie1976'] = (
    LUMINANCE_METHODS['CIE 1976'])


def luminance(LV, method='CIE 1976', **kwargs):
    """
    Returns the *luminance* :math:`Y` of given *Lightness* :math:`L^*` or given
    *Munsell* value :math:`V`.

    Parameters
    ----------
    LV : numeric
        *Lightness* :math:`L^*` or *Munsell* value :math:`V`.
    method : unicode, optional
        {'CIE 1976', 'Newhall 1943', 'ASTM D1535-08'}
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    numeric
        *luminance* :math:`Y`.

    Notes
    -----
    -   Input *LV* is in domain [0, 100] or [0, 10] and optional *luminance*
        :math:`Y_n` is in domain [0, 100].
    -   Output *luminance* :math:`Y` is in domain [0, 100].

    Examples
    --------
    >>> luminance(37.9856290977)  # doctest: +ELLIPSIS
    10.0800000...
    >>> luminance(37.9856290977, Y_n=100)  # doctest: +ELLIPSIS
    10.0800000...
    >>> luminance(37.9856290977, Y_n=95)  # doctest: +ELLIPSIS
    9.5760000...
    >>> luminance(3.74629715382, method='Newhall 1943')  # doctest: +ELLIPSIS
    10.4089874...
    >>> luminance(3.74629715382, method='ASTM D1535-08')  # doctest: +ELLIPSIS
    10.1488096...
    """

    return LUMINANCE_METHODS.get(method)(LV, **kwargs)
