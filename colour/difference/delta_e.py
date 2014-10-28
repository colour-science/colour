#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:math:`\Delta E_{ab}` - Delta E Colour Difference
=================================================

Defines :math:`\Delta E_{ab}` colour difference computation objects:

The following methods are available:

-   :func:`delta_E_CIE1976`
-   :func:`delta_E_CIE1994`
-   :func:`delta_E_CIE2000`
-   :func:`delta_E_CMC`

See Also
--------
`Delta E - Colour Difference IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/difference/delta_e.ipynb>`_  # noqa

References
----------
.. [1]  Wikipedia. (n.d.). Color difference. Retrieved August 29, 2014, from
        http://en.wikipedia.org/wiki/Color_difference
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['delta_E_CIE1976',
           'delta_E_CIE1994',
           'delta_E_CIE2000',
           'delta_E_CMC',
           'DELTA_E_METHODS',
           'delta_E']


def delta_E_CIE1976(Lab1, Lab2, **kwargs):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given
    *CIE Lab* *array_like* colours using CIE 1976 recommendation.

    Parameters
    ----------
    Lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    Lab2 : array_like, (3,)
        *CIE Lab* *array_like* colour 2.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        :math:`\Delta E_{ab}` computation objects.

    Returns
    -------
    numeric
        Colour difference :math:`\Delta E_{ab}`.

    References
    ----------
    .. [2]  Lindbloom, B. (2003). Delta E (CIE 1976). Retrieved February 24,
            2014, from http://brucelindbloom.com/Eqn_DeltaE_CIE76.html

    Examples
    --------
    >>> Lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> Lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E_CIE1976(Lab1, Lab2)  # doctest: +ELLIPSIS
    451.7133019...
    """
    return np.linalg.norm(np.array(Lab1) - np.array(Lab2))


def delta_E_CIE1994(Lab1, Lab2, textiles=True, **kwargs):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colours using CIE 1994 recommendation.

    Parameters
    ----------
    Lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    Lab2 : array_like, (3,)
        *CIE Lab* *array_like* colour 2.
    textiles : bool, optional
        Application specific weights.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        :math:`\Delta E_{ab}` computation objects.

    Returns
    -------
    numeric
        Colour difference :math:`\Delta E_{ab}`.

    References
    ----------
    .. [3]  Lindbloom, B. (2011). Delta E (CIE 1994). Retrieved February 24,
            2014, from http://brucelindbloom.com/Eqn_DeltaE_CIE94.html

    Examples
    --------
    >>> Lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> Lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E_CIE1994(Lab1, Lab2)  # doctest: +ELLIPSIS
    88.3355530...
    >>> delta_E_CIE1994(Lab1, Lab2, textiles=False)  # doctest: +ELLIPSIS
    83.7792255...
    """

    k1 = 0.048 if textiles else 0.045
    k2 = 0.014 if textiles else 0.015
    kL = 2 if textiles else 1
    kC = 1
    kH = 1

    L1, a1, b1 = np.ravel(Lab1)
    L2, a2, b2 = np.ravel(Lab2)

    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)

    sL = 1
    sC = 1 + k1 * C1
    sH = 1 + k2 * C1

    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_A = a1 - a2
    delta_B = b1 - b2

    try:
        delta_H = np.sqrt(delta_A ** 2 + delta_B ** 2 - delta_C ** 2)
    except ValueError:
        delta_H = 0.0

    L = (delta_L / (kL * sL)) ** 2
    C = (delta_C / (kC * sC)) ** 2
    H = (delta_H / (kH * sH)) ** 2

    return np.sqrt(L + C + H)


def delta_E_CIE2000(Lab1, Lab2, **kwargs):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colours using CIE 2000 recommendation.

    Parameters
    ----------
    Lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    Lab2 : array_like, (3,)
        *CIE Lab* *array_like* colour 2.
    \*\*kwargs : \*\*, optional
        Unused parameter provided for signature compatibility with other
        :math:`\Delta E_{ab}` computation objects.

    Returns
    -------
    numeric
        Colour difference :math:`\Delta E_{ab}`.

    References
    ----------

    .. [4]  Lindbloom, B. (2009). Delta E (CIE 2000). Retrieved February 24,
            2014, from http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html

    Examples
    --------
    >>> Lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> Lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E_CIE2000(Lab1, Lab2)  # doctest: +ELLIPSIS
    94.0356490...
    """

    L1, a1, b1 = np.ravel(Lab1)
    L2, a2, b2 = np.ravel(Lab2)

    kL = 1
    kC = 1
    kH = 1

    l_bar_prime = 0.5 * (L1 + L2)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)

    c_bar = 0.5 * (c1 + c2)
    c_bar7 = np.power(c_bar, 7)

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a1_prime = a1 * (1 + g)
    a2_prime = a2 * (1 + g)
    c1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    h1_prime = (np.arctan2(b1, a1_prime) * 180) / np.pi
    if h1_prime < 0:
        h1_prime += 360

    h2_prime = (np.arctan2(b2, a2_prime) * 180) / np.pi
    if h2_prime < 0.0:
        h2_prime += 360

    h_bar_prime = (0.5 * (h1_prime + h2_prime + 360)
                   if np.fabs(h1_prime - h2_prime) > 180 else
                   0.5 * (h1_prime + h2_prime))

    t = (1 - 0.17 * np.cos(np.pi * (h_bar_prime - 30) / 180) +
         0.24 * np.cos(np.pi * (2 * h_bar_prime) / 180) +
         0.32 * np.cos(np.pi * (3 * h_bar_prime + 6) / 180) -
         0.20 * np.cos(np.pi * (4 * h_bar_prime - 63) / 180))

    if np.fabs(h2_prime - h1_prime) <= 180:
        delta_h_prime = h2_prime - h1_prime
    else:
        delta_h_prime = (h2_prime - h1_prime + 360
                         if h2_prime <= h1_prime else
                         h2_prime - h1_prime - 360)

    delta_L_prime = L2 - L1
    delta_C_prime = c2_prime - c1_prime
    delta_H_prime = (2 * np.sqrt(c1_prime * c2_prime) *
                     np.sin(np.pi * (0.5 * delta_h_prime) / 180))

    sL = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
              np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    sC = 1 + 0.045 * c_bar_prime
    sH = 1 + 0.015 * c_bar_prime * t

    delta_theta = (30 * np.exp(-((h_bar_prime - 275) / 25) *
                               ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    rC = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    rT = -2 * rC * np.sin(np.pi * (2 * delta_theta) / 180)

    return np.sqrt(
        (delta_L_prime / (kL * sL)) * (delta_L_prime / (kL * sL)) +
        (delta_C_prime / (kC * sC)) * (delta_C_prime / (kC * sC)) +
        (delta_H_prime / (kH * sH)) * (delta_H_prime / (kH * sH)) +
        (delta_C_prime / (kC * sC)) * (delta_H_prime / (kH * sH)) * rT)


def delta_E_CMC(Lab1, Lab2, l=2, c=1):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colours using *Colour Measurement Committee* recommendation.

    The quasimetric has two parameters: *Lightness* (l) and *chroma* (c),
    allowing the users to weight the difference based on the ratio of l:c.
    Commonly used values are 2:1 for acceptability and 1:1 for the threshold of
    imperceptibility.

    Parameters
    ----------
    Lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    Lab2 : array_like, (3,)
        *CIE Lab* *array_like* colour 2.
    l : numeric, optional
        Lightness weighting factor.
    c : numeric, optional
        Chroma weighting factor.

    Returns
    -------
    numeric
        Colour difference :math:`\Delta E_{ab}`.

    References
    ----------
    .. [5]  Lindbloom, B. (2009). Delta E (CMC). Retrieved February 24, 2014,
            from http://brucelindbloom.com/Eqn_DeltaE_CMC.html

    Examples
    --------
    >>> Lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> Lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E_CMC(Lab1, Lab2)  # doctest: +ELLIPSIS
    172.7047712...
    """

    L1, a1, b1 = np.ravel(Lab1)
    L2, a2, b2 = np.ravel(Lab2)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    sl = 0.511 if L1 < 16 else (0.040975 * L1) / (1 + 0.01765 * L1)
    sc = 0.0638 * c1 / (1 + 0.0131 * c1) + 0.638
    h1 = 0 if c1 < 0.000001 else (np.arctan2(b1, a1) * 180) / np.pi

    while h1 < 0:
        h1 += 360

    while h1 >= 360:
        h1 -= 360

    t = (0.56 + np.fabs(0.2 * np.cos((np.pi * (h1 + 168)) / 180))
         if 164 <= h1 <= 345 else
         0.36 + np.fabs(0.4 * np.cos((np.pi * (h1 + 35)) / 180)))
    c4 = c1 * c1 * c1 * c1
    f = np.sqrt(c4 / (c4 + 1900))
    sh = sc * (f * t + 1 - f)

    delta_L = L1 - L2
    delta_C = c1 - c2
    delta_A = a1 - a2
    delta_B = b1 - b2
    delta_H2 = delta_A * delta_A + delta_B * delta_B - delta_C * delta_C

    v1 = delta_L / (l * sl)
    v2 = delta_C / (c * sc)
    v3 = sh

    return np.sqrt(v1 * v1 + v2 * v2 + (delta_H2 / (v3 * v3)))


DELTA_E_METHODS = CaseInsensitiveMapping(
    {'CIE 1976': delta_E_CIE1976,
     'CIE 1994': delta_E_CIE1994,
     'CIE 2000': delta_E_CIE2000,
     'CMC': delta_E_CMC})
"""
Supported *Delta E* computations methods.

DELTA_E_METHODS : CaseInsensitiveMapping
    {'CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC'}

Aliases:

-   'cie1976': 'CIE 1976'
-   'cie1994': 'CIE 1994'
-   'cie2000': 'CIE 2000'
"""
DELTA_E_METHODS['cie1976'] = DELTA_E_METHODS['CIE 1976']
DELTA_E_METHODS['cie1994'] = DELTA_E_METHODS['CIE 1994']
DELTA_E_METHODS['cie2000'] = DELTA_E_METHODS['CIE 2000']


def delta_E(Lab1, Lab2, method='CMC', **kwargs):
    """
    Returns the *Lightness* :math:`L^*` using given method.

    Parameters
    ----------
    Lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    Lab2 : array_like, (3,)
        *CIE Lab* *array_like* colour 2.
    method : unicode, optional
        {'CMC', 'CIE 1976', 'CIE 1994', 'CIE 2000'}
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    numeric
        Colour difference :math:`\Delta E_{ab}`.

    Examples
    --------
    >>> Lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> Lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E(Lab1, Lab2)  # doctest: +ELLIPSIS
    172.7047712...
    >>> delta_E(Lab1, Lab2, method='CIE 1976')  # doctest: +ELLIPSIS
    451.7133019...
    >>> delta_E(Lab1, Lab2, method='CIE 1994')  # doctest: +ELLIPSIS
    88.3355530...
    >>> delta_E(Lab1, Lab2, method='CIE 1994', textiles=False)  # noqa  # doctest: +ELLIPSIS
    83.7792255...
    >>> delta_E(Lab1, Lab2, method='CIE 2000')  # doctest: +ELLIPSIS
    94.0356490...
    """

    return DELTA_E_METHODS.get(method)(Lab1, Lab2, **kwargs)
