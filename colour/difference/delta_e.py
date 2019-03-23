# -*- coding: utf-8 -*-
"""
:math:`\\Delta E_{ab}` - Delta E Colour Difference
=================================================

Defines :math:`\\Delta E_{ab}` colour difference computation objects:

The following methods are available:

-   :func:`colour.difference.delta_E_CIE1976`
-   :func:`colour.difference.delta_E_CIE1994`
-   :func:`colour.difference.delta_E_CIE2000`
-   :func:`colour.difference.delta_E_CMC`

See Also
--------
`Delta E - Colour Difference Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/difference/delta_e.ipynb>`_

References
----------
-   :cite:`Lindbloom2003c` : Lindbloom, B. (2003). Delta E (CIE 1976).
    Retrieved February 24, 2014, from
    http://brucelindbloom.com/Eqn_DeltaE_CIE76.html
-   :cite:`Lindbloom2009e` : Lindbloom, B. (2009). Delta E (CIE 2000).
    Retrieved February 24, 2014, from
    http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html
-   :cite:`Lindbloom2009f` : Lindbloom, B. (2009). Delta E (CMC). Retrieved
    February 24, 2014, from http://brucelindbloom.com/Eqn_DeltaE_CMC.html
-   :cite:`Lindbloom2011a` : Lindbloom, B. (2011). Delta E (CIE 1994).
    Retrieved February 24, 2014, from
    http://brucelindbloom.com/Eqn_DeltaE_CIE94.html
-   :cite:`Melgosa2013b` : Melgosa, M. (2013). CIE / ISO new standard:
    CIEDE2000. Retrieved from http://www.color.org/events/colorimetry/\
Melgosa_CIEDE2000_Workshop-July4.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import euclidean_distance
from colour.utilities import to_domain_100, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'delta_E_CIE1976', 'delta_E_CIE1994', 'delta_E_CIE2000', 'delta_E_CMC'
]


def delta_E_CIE1976(Lab_1, Lab_2):
    """
    Returns the difference :math:`\\Delta E_{76}` between two given
    *CIE L\\*a\\*b\\** colourspace arrays using *CIE 1976* recommendation.

    Parameters
    ----------
    Lab_1 : array_like
        *CIE L\\*a\\*b\\** colourspace array 1.
    Lab_2 : array_like
        *CIE L\\*a\\*b\\** colourspace array 2.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E_{76}`.

    Notes
    -----

    +------------+-----------------------+-------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**     |
    +============+=======================+===================+
    | ``Lab_1``  | ``L_1`` : [0, 100]    | ``L_1`` : [0, 1]  |
    |            |                       |                   |
    |            | ``a_1`` : [-100, 100] | ``a_1`` : [-1, 1] |
    |            |                       |                   |
    |            | ``b_1`` : [-100, 100] | ``b_1`` : [-1, 1] |
    +------------+-----------------------+-------------------+
    | ``Lab_2``  | ``L_2`` : [0, 100]    | ``L_2`` : [0, 1]  |
    |            |                       |                   |
    |            | ``a_2`` : [-100, 100] | ``a_2`` : [-1, 1] |
    |            |                       |                   |
    |            | ``b_2`` : [-100, 100] | ``b_2`` : [-1, 1] |
    +------------+-----------------------+-------------------+

    References
    ----------
    :cite:`Lindbloom2003c`

    Examples
    --------
    >>> Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> delta_E_CIE1976(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    451.7133019...
    """

    d_E = euclidean_distance(to_domain_100(Lab_1), to_domain_100(Lab_2))

    return d_E


def delta_E_CIE1994(Lab_1, Lab_2, textiles=False):
    """
    Returns the difference :math:`\\Delta E_{94}` between two given
    *CIE L\\*a\\*b\\** colourspace arrays using *CIE 1994* recommendation.

    Parameters
    ----------
    Lab_1 : array_like
        *CIE L\\*a\\*b\\** colourspace array 1.
    Lab_2 : array_like
        *CIE L\\*a\\*b\\** colourspace array 2.
    textiles : bool, optional
        Textiles application specific parametric factors
        :math:`k_L=2,\\ k_C=k_H=1,\\ k_1=0.048,\\ k_2=0.014` weights are used
        instead of :math:`k_L=k_C=k_H=1,\\ k_1=0.045,\\ k_2=0.015`.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E_{94}`.

    Notes
    -----

    +------------+-----------------------+-------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**     |
    +============+=======================+===================+
    | ``Lab_1``  | ``L_1`` : [0, 100]    | ``L_1`` : [0, 1]  |
    |            |                       |                   |
    |            | ``a_1`` : [-100, 100] | ``a_1`` : [-1, 1] |
    |            |                       |                   |
    |            | ``b_1`` : [-100, 100] | ``b_1`` : [-1, 1] |
    +------------+-----------------------+-------------------+
    | ``Lab_2``  | ``L_2`` : [0, 100]    | ``L_2`` : [0, 1]  |
    |            |                       |                   |
    |            | ``a_2`` : [-100, 100] | ``a_2`` : [-1, 1] |
    |            |                       |                   |
    |            | ``b_2`` : [-100, 100] | ``b_2`` : [-1, 1] |
    +------------+-----------------------+-------------------+

    -   *CIE 1994* colour differences are not symmetrical: difference between
        ``Lab_1`` and ``Lab_2`` may not be the same as difference between
        ``Lab_2`` and ``Lab_1`` thus one colour must be understood to be the
        reference against which a sample colour is compared.

    References
    ----------
    :cite:`Lindbloom2011a`

    Examples
    --------
    >>> Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> delta_E_CIE1994(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    83.7792255...
    >>> delta_E_CIE1994(Lab_1, Lab_2, textiles=True)  # doctest: +ELLIPSIS
    88.3355530...
    """

    L_1, a_1, b_1 = tsplit(to_domain_100(Lab_1))
    L_2, a_2, b_2 = tsplit(to_domain_100(Lab_2))

    k_1 = 0.048 if textiles else 0.045
    k_2 = 0.014 if textiles else 0.015
    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    C_1 = np.hypot(a_1, b_1)
    C_2 = np.hypot(a_2, b_2)

    s_L = 1
    s_C = 1 + k_1 * C_1
    s_H = 1 + k_2 * C_1

    delta_L = L_1 - L_2
    delta_C = C_1 - C_2
    delta_A = a_1 - a_2
    delta_B = b_1 - b_2

    delta_H = np.sqrt(delta_A ** 2 + delta_B ** 2 - delta_C ** 2)

    L = (delta_L / (k_L * s_L)) ** 2
    C = (delta_C / (k_C * s_C)) ** 2
    H = (delta_H / (k_H * s_H)) ** 2

    d_E = np.sqrt(L + C + H)

    return d_E


def delta_E_CIE2000(Lab_1, Lab_2, textiles=False):
    """
    Returns the difference :math:`\\Delta E_{00}` between two given
    *CIE L\\*a\\*b\\** colourspace arrays using *CIE 2000* recommendation.

    Parameters
    ----------
    Lab_1 : array_like
        *CIE L\\*a\\*b\\** colourspace array 1.
    Lab_2 : array_like
        *CIE L\\*a\\*b\\** colourspace array 2.
    textiles : bool, optional
        Textiles application specific parametric factors
        :math:`k_L=2,\\ k_C=k_H=1` weights are used instead of
        :math:`k_L=k_C=k_H=1`.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E_{00}`.

    Notes
    -----

    +------------+-----------------------+-------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**     |
    +============+=======================+===================+
    | ``Lab_1``  | ``L_1`` : [0, 100]    | ``L_1`` : [0, 1]  |
    |            |                       |                   |
    |            | ``a_1`` : [-100, 100] | ``a_1`` : [-1, 1] |
    |            |                       |                   |
    |            | ``b_1`` : [-100, 100] | ``b_1`` : [-1, 1] |
    +------------+-----------------------+-------------------+
    | ``Lab_2``  | ``L_2`` : [0, 100]    | ``L_2`` : [0, 1]  |
    |            |                       |                   |
    |            | ``a_2`` : [-100, 100] | ``a_2`` : [-1, 1] |
    |            |                       |                   |
    |            | ``b_2`` : [-100, 100] | ``b_2`` : [-1, 1] |
    +------------+-----------------------+-------------------+

    -   *CIE 2000* colour differences are not symmetrical: difference between
        ``Lab_1`` and ``Lab_2`` may not be the same as difference between
        ``Lab_2`` and ``Lab_1`` thus one colour must be understood to be the
        reference against which a sample colour is compared.
    -   Parametric factors :math:`k_L=k_C=k_H=1` weights under
        *reference conditions*:

        -   Illumination: D65 source
        -   Illuminance: 1000 lx
        -   Observer: Normal colour vision
        -   Background field: Uniform, neutral gray with :math:`L^*=50`
        -   Viewing mode: Object
        -   Sample size: Greater than 4 degrees
        -   Sample separation: Direct edge contact
        -   Sample colour-difference magnitude: Lower than 5.0
            :math:`\\Delta E_{00}`
        -   Sample structure: Homogeneous (without texture)

    References
    ----------
    :cite:`Lindbloom2009e`, :cite:`Melgosa2013b`

    Examples
    --------
    >>> Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> delta_E_CIE2000(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    94.0356490...
    >>> Lab_2 = np.array([50.00000000, 426.67945353, 72.39590835])
    >>> delta_E_CIE2000(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    100.8779470...
    >>> delta_E_CIE2000(Lab_1, Lab_2, textiles=True)  # doctest: +ELLIPSIS
    95.7920535...
    """

    L_1, a_1, b_1 = tsplit(to_domain_100(Lab_1))
    L_2, a_2, b_2 = tsplit(to_domain_100(Lab_2))

    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    l_bar_prime = 0.5 * (L_1 + L_2)

    c_1 = np.hypot(a_1, b_1)
    c_2 = np.hypot(a_2, b_2)

    c_bar = 0.5 * (c_1 + c_2)
    c_bar7 = c_bar ** 7

    g = 0.5 * (1 - np.sqrt(c_bar7 / (c_bar7 + 25 ** 7)))

    a_1_prime = a_1 * (1 + g)
    a_2_prime = a_2 * (1 + g)
    c_1_prime = np.hypot(a_1_prime, b_1)
    c_2_prime = np.hypot(a_2_prime, b_2)
    c_bar_prime = 0.5 * (c_1_prime + c_2_prime)

    h_1_prime = np.degrees(np.arctan2(b_1, a_1_prime)) % 360
    h_2_prime = np.degrees(np.arctan2(b_2, a_2_prime)) % 360

    h_bar_prime = np.where(
        np.fabs(h_1_prime - h_2_prime) <= 180,
        0.5 * (h_1_prime + h_2_prime),
        (0.5 * (h_1_prime + h_2_prime + 360)),
    )

    t = (1 - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30)) +
         0.24 * np.cos(np.deg2rad(2 * h_bar_prime)) +
         0.32 * np.cos(np.deg2rad(3 * h_bar_prime + 6)) -
         0.20 * np.cos(np.deg2rad(4 * h_bar_prime - 63)))

    h = h_2_prime - h_1_prime
    delta_h_prime = np.where(h_2_prime <= h_1_prime, h - 360, h + 360)
    delta_h_prime = np.where(np.fabs(h) <= 180, h, delta_h_prime)

    delta_L_prime = L_2 - L_1
    delta_C_prime = c_2_prime - c_1_prime
    delta_H_prime = (2 * np.sqrt(c_1_prime * c_2_prime) * np.sin(
        np.deg2rad(0.5 * delta_h_prime)))

    s_L = 1 + ((0.015 * (l_bar_prime - 50) * (l_bar_prime - 50)) /
               np.sqrt(20 + (l_bar_prime - 50) * (l_bar_prime - 50)))
    s_C = 1 + 0.045 * c_bar_prime
    s_H = 1 + 0.015 * c_bar_prime * t

    delta_theta = (
        30 * np.exp(-((h_bar_prime - 275) / 25) * ((h_bar_prime - 275) / 25)))

    c_bar_prime7 = c_bar_prime ** 7

    r_C = np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25 ** 7))
    r_T = -2 * r_C * np.sin(np.deg2rad(2 * delta_theta))

    d_E = np.sqrt((delta_L_prime / (k_L * s_L)) ** 2 +
                  (delta_C_prime / (k_C * s_C)) ** 2 +
                  (delta_H_prime / (k_H * s_H)) ** 2 +
                  (delta_C_prime / (k_C * s_C)) * (delta_H_prime /
                                                   (k_H * s_H)) * r_T)

    return d_E


def delta_E_CMC(Lab_1, Lab_2, l=2, c=1):  # noqa
    """
    Returns the difference :math:`\\Delta E_{CMC}` between two given
    *CIE L\\*a\\*b\\** colourspace arrays using *Colour Measurement Committee*
    recommendation.

    The quasimetric has two parameters: *Lightness* (l) and *chroma* (c),
    allowing the users to weight the difference based on the ratio of l:c.
    Commonly used values are 2:1 for acceptability and 1:1 for the threshold of
    imperceptibility.

    Parameters
    ----------
    Lab_1 : array_like
        *CIE L\\*a\\*b\\** colourspace array 1.
    Lab_2 : array_like
        *CIE L\\*a\\*b\\** colourspace array 2.
    l : numeric, optional
        Lightness weighting factor.
    c : numeric, optional
        Chroma weighting factor.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E_{CMC}`.

    Notes
    -----

    +------------+-----------------------+-------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**     |
    +============+=======================+===================+
    | ``Lab_1``  | ``L_1`` : [0, 100]    | ``L_1`` : [0, 1]  |
    |            |                       |                   |
    |            | ``a_1`` : [-100, 100] | ``a_1`` : [-1, 1] |
    |            |                       |                   |
    |            | ``b_1`` : [-100, 100] | ``b_1`` : [-1, 1] |
    +------------+-----------------------+-------------------+
    | ``Lab_2``  | ``L_2`` : [0, 100]    | ``L_2`` : [0, 1]  |
    |            |                       |                   |
    |            | ``a_2`` : [-100, 100] | ``a_2`` : [-1, 1] |
    |            |                       |                   |
    |            | ``b_2`` : [-100, 100] | ``b_2`` : [-1, 1] |
    +------------+-----------------------+-------------------+

    References
    ----------
    :cite:`Lindbloom2009f`

    Examples
    --------
    >>> Lab_1 = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> Lab_2 = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> delta_E_CMC(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    172.7047712...
    """

    L_1, a_1, b_1 = tsplit(to_domain_100(Lab_1))
    L_2, a_2, b_2 = tsplit(to_domain_100(Lab_2))

    c_1 = np.hypot(a_1, b_1)
    c_2 = np.hypot(a_2, b_2)
    s_l = np.where(L_1 < 16, 0.511, (0.040975 * L_1) / (1 + 0.01765 * L_1))
    s_c = 0.0638 * c_1 / (1 + 0.0131 * c_1) + 0.638
    h_1 = np.degrees(np.arctan2(b_1, a_1)) % 360

    t = np.where(
        np.logical_and(h_1 >= 164, h_1 <= 345),
        0.56 + np.fabs(0.2 * np.cos(np.deg2rad(h_1 + 168))),
        0.36 + np.fabs(0.4 * np.cos(np.deg2rad(h_1 + 35))),
    )

    c_4 = c_1 * c_1 * c_1 * c_1
    f = np.sqrt(c_4 / (c_4 + 1900))
    s_h = s_c * (f * t + 1 - f)

    delta_L = L_1 - L_2
    delta_C = c_1 - c_2
    delta_A = a_1 - a_2
    delta_B = b_1 - b_2
    delta_H2 = delta_A ** 2 + delta_B ** 2 - delta_C ** 2

    v_1 = delta_L / (l * s_l)
    v_2 = delta_C / (c * s_c)
    v_3 = s_h

    d_E = np.sqrt(v_1 ** 2 + v_2 ** 2 + (delta_H2 / (v_3 * v_3)))

    return d_E
