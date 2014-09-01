#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:math:`\Delta E_{ab}` - Delta E Colour Difference
=================================================

Defines :math:`\Delta E_{ab}` colour difference computation objects:

The following methods are available:

-   :func:`delta_E_CIE_1976`
-   :func:`delta_E_CIE_1994`
-   :func:`delta_E_CIE_2000`
-   :func:`delta_E_CMC`

See Also
--------
`Delta E - Colour Difference IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/difference/delta_e.ipynb>`_  # noqa

References
----------
.. [1]  http://en.wikipedia.org/wiki/Color_difference
        (Last accessed 29 August 2014)
"""

from __future__ import division, unicode_literals

import math
import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['delta_E_CIE_1976',
           'delta_E_CIE_1994',
           'delta_E_CIE_2000',
           'delta_E_CMC',
           'DELTA_E_METHODS',
           'delta_E']


def delta_E_CIE_1976(lab1, lab2, **kwargs):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given
    *CIE Lab* *array_like* colours using *CIE 1976* recommendation.

    Parameters
    ----------
    lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    lab2 : array_like, (3,)
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
    .. [2]  http://brucelindbloom.com/Eqn_DeltaE_CIE76.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E_CIE_1976(lab1, lab2)  # doctest: +ELLIPSIS
    451.7133019...
    """
    return np.linalg.norm(np.array(lab1) - np.array(lab2))


def delta_E_CIE_1994(lab1, lab2, textiles=True, **kwargs):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colours using *CIE 1994* recommendation.

    Parameters
    ----------
    lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    lab2 : array_like, (3,)
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
    .. [3]  http://brucelindbloom.com/Eqn_DeltaE_CIE94.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E_CIE_1994(lab1, lab2)  # doctest: +ELLIPSIS
    88.3355530...
    >>> delta_E_CIE_1994(lab1, lab2, textiles=False)  # doctest: +ELLIPSIS
    83.7792255...
    """

    k1 = 0.048 if textiles else 0.045
    k2 = 0.014 if textiles else 0.015
    kL = 2 if textiles else 1
    kC = 1
    kH = 1

    L1, a1, b1 = np.ravel(lab1)
    L2, a2, b2 = np.ravel(lab2)

    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)

    sL = 1
    sC = 1 + k1 * C1
    sH = 1 + k2 * C1

    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_A = a1 - a2
    delta_B = b1 - b2

    try:
        delta_H = math.sqrt(delta_A ** 2 + delta_B ** 2 - delta_C ** 2)
    except ValueError:
        delta_H = 0.0

    L = (delta_L / (kL * sL)) ** 2
    C = (delta_C / (kC * sC)) ** 2
    H = (delta_H / (kH * sH)) ** 2

    return math.sqrt(L + C + H)


def delta_E_CIE_2000(lab1, lab2, **kwargs):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colours using *CIE 2000* recommendation.

    Parameters
    ----------
    lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    lab2 : array_like, (3,)
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

    .. [4]  http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E_CIE_2000(lab1, lab2)  # doctest: +ELLIPSIS
    94.0356490...
    """

    L1, a1, b1 = np.ravel(lab1)
    L2, a2, b2 = np.ravel(lab2)

    kL = 1
    kC = 1
    kH = 1

    l_bar_prime = 0.5 * (L1 + L2)

    c1 = math.sqrt(a1 * a1 + b1 * b1)
    c2 = math.sqrt(a2 * a2 + b2 * b2)

    c_bar = 0.5 * (c1 + c2)
    c_bar7 = math.pow(c_bar, 7)

    g = 0.5 * (1 - math.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a1_prime = a1 * (1 + g)
    a2_prime = a2 * (1 + g)
    c1_prime = math.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = math.sqrt(a2_prime * a2_prime + b2 * b2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    h1_prime = (math.atan2(b1, a1_prime) * 180) / math.pi
    if h1_prime < 0:
        h1_prime += 360

    h2_prime = (math.atan2(b2, a2_prime) * 180) / math.pi
    if h2_prime < 0.0:
        h2_prime += 360

    h_bar_prime = (0.5 * (h1_prime + h2_prime + 360)
                   if math.fabs(h1_prime - h2_prime) > 180 else
                   0.5 * (h1_prime + h2_prime))

    t = (1 - 0.17 * math.cos(math.pi * (h_bar_prime - 30) / 180) +
         0.24 * math.cos(math.pi * (2 * h_bar_prime) / 180) +
         0.32 * math.cos(math.pi * (3 * h_bar_prime + 6) / 180) -
         0.20 * math.cos(math.pi * (4 * h_bar_prime - 63) / 180))

    if math.fabs(h2_prime - h1_prime) <= 180:
        delta_h_prime = h2_prime - h1_prime
    else:
        delta_h_prime = (h2_prime - h1_prime + 360
                         if h2_prime <= h1_prime else
                         h2_prime - h1_prime - 360)

    delta_L_prime = L2 - L1
    delta_C_prime = c2_prime - c1_prime
    delta_H_prime = (2 * math.sqrt(c1_prime * c2_prime) *
                     math.sin(math.pi * (0.5 * delta_h_prime) / 180))

    sL = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
              math.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    sC = 1 + 0.045 * c_bar_prime
    sH = 1 + 0.015 * c_bar_prime * t

    delta_theta = (30 * math.exp(-((h_bar_prime - 275) / 25) *
                                 ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    rC = math.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    rT = -2 * rC * math.sin(math.pi * (2 * delta_theta) / 180)

    return math.sqrt(
        (delta_L_prime / (kL * sL)) * (delta_L_prime / (kL * sL)) +
        (delta_C_prime / (kC * sC)) * (delta_C_prime / (kC * sC)) +
        (delta_H_prime / (kH * sH)) * (delta_H_prime / (kH * sH)) +
        (delta_C_prime / (kC * sC)) * (delta_H_prime / (kH * sH)) * rT)


def delta_E_CMC(lab1, lab2, l=2, c=1):
    """
    Returns the difference :math:`\Delta E_{ab}` between two given *CIE Lab*
    *array_like* colours using *Colour Measurement Committee* recommendation.

    The quasimetric has two parameters: *Lightness* (l) and *chroma* (c),
    allowing the users to weight the difference based on the ratio of l:c.
    Commonly used values are 2:1 for acceptability and 1:1 for the threshold of
    imperceptibility.

    Parameters
    ----------
    lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    lab2 : array_like, (3,)
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
    .. [5]  http://brucelindbloom.com/Eqn_DeltaE_CMC.html
            (Last accessed 24 February 2014)

    Examples
    --------
    >>> lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E_CMC(lab1, lab2)  # doctest: +ELLIPSIS
    172.7047712...
    """

    L1, a1, b1 = np.ravel(lab1)
    L2, a2, b2 = np.ravel(lab2)

    c1 = math.sqrt(a1 * a1 + b1 * b1)
    c2 = math.sqrt(a2 * a2 + b2 * b2)
    sl = 0.511 if L1 < 16 else (0.040975 * L1) / (1 + 0.01765 * L1)
    sc = 0.0638 * c1 / (1 + 0.0131 * c1) + 0.638
    h1 = 0 if c1 < 0.000001 else (math.atan2(b1, a1) * 180) / math.pi

    while h1 < 0:
        h1 += 360

    while h1 >= 360:
        h1 -= 360

    t = (0.56 + math.fabs(0.2 * math.cos((math.pi * (h1 + 168)) / 180))
         if 164 <= h1 <= 345 else
         0.36 + math.fabs(0.4 * math.cos((math.pi * (h1 + 35)) / 180)))
    c4 = c1 * c1 * c1 * c1
    f = math.sqrt(c4 / (c4 + 1900))
    sh = sc * (f * t + 1 - f)

    delta_L = L1 - L2
    delta_C = c1 - c2
    delta_A = a1 - a2
    delta_B = b1 - b2
    delta_H2 = delta_A * delta_A + delta_B * delta_B - delta_C * delta_C

    v1 = delta_L / (l * sl)
    v2 = delta_C / (c * sc)
    v3 = sh

    return math.sqrt(v1 * v1 + v2 * v2 + (delta_H2 / (v3 * v3)))


DELTA_E_METHODS = {
    'CIE 1976': delta_E_CIE_1976,
    'CIE 1994': delta_E_CIE_1994,
    'CIE 2000': delta_E_CIE_2000,
    'CMC': delta_E_CMC,
}
"""
Supported *Delta E* computations methods.

DELTA_E_METHODS : dict
    ('CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC')

Aliases:

-   'cie1976': 'CIE 1976'
-   'cie1994': 'CIE 1994'
-   'cie2000': 'CIE 2000'
"""
DELTA_E_METHODS['cie1976'] = DELTA_E_METHODS['CIE 1976']
DELTA_E_METHODS['cie1994'] = DELTA_E_METHODS['CIE 1994']
DELTA_E_METHODS['cie2000'] = DELTA_E_METHODS['CIE 2000']


def delta_E(lab1, lab2, method='CMC', **kwargs):
    """
    Returns the *Lightness* :math:`L^*` using given method.

    Parameters
    ----------
    lab1 : array_like, (3,)
        *CIE Lab* *array_like* colour 1.
    lab2 : array_like, (3,)
        *CIE Lab* *array_like* colour 2.
    method : unicode, optional
        ('CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC')
        Computation method.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    numeric
        Colour difference :math:`\Delta E_{ab}`.

    Examples
    --------
    >>> lab1 = np.array([100, 21.57210357, 272.2281935])
    >>> lab2 = np.array([100, 426.67945353, 72.39590835])
    >>> delta_E(lab1, lab2)  # doctest: +ELLIPSIS
    172.7047712...
    >>> delta_E(lab1, lab2, method='CIE 1976')  # doctest: +ELLIPSIS
    451.7133019...
    >>> delta_E(lab1, lab2, method='CIE 1994')  # doctest: +ELLIPSIS
    88.3355530...
    >>> delta_E(lab1, lab2, method='CIE 1994', textiles=False)  # noqa  # doctest: +ELLIPSIS
    83.7792255...
    >>> delta_E(lab1, lab2, method='CIE 2000')  # doctest: +ELLIPSIS
    94.0356490...
    """

    return DELTA_E_METHODS.get(method)(lab1, lab2, **kwargs)
