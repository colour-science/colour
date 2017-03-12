#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Luminance :math:`Y`
===================

Defines *luminance* :math:`Y` computation objects.

The following methods are available:

-   :func:`luminance_Newhall1943`: *luminance* :math:`Y` computation of given
    *Munsell* value :math:`V` using *Newhall, Nickerson and Judd (1943)*
    method.
-   :func:`luminance_ASTMD153508`: *luminance* :math:`Y` computation of given
    *Munsell* value :math:`V` using *ASTM D1535-08e1* method.
-   :func:`luminance_CIE1976`: *luminance* :math:`Y` computation of given
    *Lightness* :math:`L^*` as per *CIE 1976* recommendation.

See Also
--------
`Luminance Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/luminance.ipynb>`_
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import CIE_E, CIE_K
from colour.utilities import CaseInsensitiveMapping, filter_kwargs

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['luminance_Newhall1943',
           'luminance_ASTMD153508',
           'luminance_CIE1976',
           'LUMINANCE_METHODS',
           'luminance']


def luminance_Newhall1943(V):
    """
    Returns the *luminance* :math:`R_Y` of given *Munsell* value :math:`V`
    using *Newhall et al. (1943)* method.

    Parameters
    ----------
    V : numeric or array_like
        *Munsell* value :math:`V`.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`R_Y`.

    Notes
    -----
    -   Input *Munsell* value :math:`V` is in domain [0, 10].
    -   Output *luminance* :math:`R_Y` is in range [0, 100].

    References
    ----------
    .. [1]  Newhall, S. M., Nickerson, D., & Judd, D. B. (1943). Final report
            of the OSA subcommittee on the spacing of the munsell colors. JOSA,
            33(7), 385. doi:10.1364/JOSA.33.000385

    Examples
    --------
    >>> luminance_Newhall1943(3.74629715382)  # doctest: +ELLIPSIS
    10.4089874...
    """

    V = np.asarray(V)

    R_Y = (1.2219 * V - 0.23111 * (V * V) + 0.23951 * (V ** 3) - 0.021009 *
           (V ** 4) + 0.0008404 * (V ** 5))

    return R_Y


def luminance_ASTMD153508(V):
    """
    Returns the *luminance* :math:`Y` of given *Munsell* value :math:`V` using
    *ASTM D1535-08e1* method.

    Parameters
    ----------
    V : numeric or array_like
        *Munsell* value :math:`V`.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`Y`.

    Notes
    -----
    -   Input *Munsell* value :math:`V` is in domain [0, 10].
    -   Output *luminance* :math:`Y` is in range [0, 100].

    References
    ----------
    .. [4]  ASTM International. (n.d.). ASTM D1535-08e1 Standard Practice for
            Specifying Color by the Munsell System. doi:10.1520/D1535-08E01

    Examples
    --------
    >>> luminance_ASTMD153508(3.74629715382)  # doctest: +ELLIPSIS
    10.1488096...
    """

    V = np.asarray(V)

    Y = (1.1914 * V - 0.22533 * (V ** 2) + 0.23352 * (V ** 3) - 0.020484 *
         (V ** 4) + 0.00081939 * (V ** 5))

    return Y


def luminance_CIE1976(Lstar, Y_n=100):
    """
    Returns the *luminance* :math:`Y` of given *Lightness* :math:`L^*` with
    given reference white *luminance* :math:`Y_n`.

    Parameters
    ----------
    Lstar : numeric or array_like
        *Lightness* :math:`L^*`
    Y_n : numeric or array_like
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`Y`.

    Notes
    -----
    -   Input *Lightness* :math:`L^*` and reference white *luminance*
        :math:`Y_n` are in domain [0, 100].
    -   Output *luminance* :math:`Y` is in range [0, 100].

    References
    ----------
    .. [2]  Wyszecki, G., & Stiles, W. S. (2000). CIE 1976 (L*u*v*)-Space and
            Color-Difference Formula. In Color Science: Concepts and Methods,
            Quantitative Data and Formulae (p. 167). Wiley. ISBN:978-0471399186
    .. [3]  Lindbloom, B. (2003). A Continuity Study of the CIE L* Function.
            Retrieved February 24, 2014, from
            http://brucelindbloom.com/LContinuity.html

    Examples
    --------
    >>> luminance_CIE1976(37.98562910)  # doctest: +ELLIPSIS
    array(10.0800000...)
    >>> luminance_CIE1976(37.98562910, 95)  # doctest: +ELLIPSIS
    array(9.5760000...)
    """

    Lstar = np.asarray(Lstar)
    Y_n = np.asarray(Y_n)

    Y = np.where(Lstar > CIE_K * CIE_E,
                 Y_n * ((Lstar + 16) / 116) ** 3,
                 Y_n * (Lstar / CIE_K))

    return Y


LUMINANCE_METHODS = CaseInsensitiveMapping(
    {'Newhall 1943': luminance_Newhall1943,
     'ASTM D1535-08': luminance_ASTMD153508,
     'CIE 1976': luminance_CIE1976})
"""
Supported *luminance* computations methods.

LUMINANCE_METHODS : CaseInsensitiveMapping
    **{'Newhall 1943', 'ASTM D1535-08', 'CIE 1976'}**

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
    LV : numeric or array_like
        *Lightness* :math:`L^*` or *Munsell* value :math:`V`.
    method : unicode, optional
        **{'CIE 1976', 'Newhall 1943', 'ASTM D1535-08'}**,
        Computation method.

    Other Parameters
    ----------------
    Y_n : numeric or array_like, optional
        {:func:`luminance_CIE1976`},
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`Y`.

    Notes
    -----
    -   Input *LV* is in domain [0, 100] or [0, 10] and optional *luminance*
        :math:`Y_n` is in domain [0, 100].
    -   Output *luminance* :math:`Y` is in range [0, 100].

    Examples
    --------
    >>> luminance(37.98562910)  # doctest: +ELLIPSIS
    array(10.0800000...)
    >>> luminance(37.98562910, Y_n=100)  # doctest: +ELLIPSIS
    array(10.0800000...)
    >>> luminance(37.98562910, Y_n=95)  # doctest: +ELLIPSIS
    array(9.5760000...)
    >>> luminance(3.74629715, method='Newhall 1943')  # doctest: +ELLIPSIS
    10.4089874...
    >>> luminance(3.74629715, method='ASTM D1535-08')  # doctest: +ELLIPSIS
    10.1488096...
    """

    function = LUMINANCE_METHODS[method]

    filter_kwargs(function, **kwargs)

    return function(LV, **kwargs)
