#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SMPTE ST 2084:2014
==================

Defines *SMPTE ST 2084:2014* opto-electrical transfer function (OETF / OECF)
and electro-optical transfer function (EOTF / EOCF):

-   :func:`eotf_ST2084`
-   :func:`oetf_ST2084`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Society of Motion Picture and Television Engineers. (2014). SMPTE
        ST 2084:2014 - Dynamic Range Electro-Optical Transfer Function of
        Mastering Reference Displays. doi:10.5594/SMPTE.ST2084.2014
.. [2]  Miller, S., & Dolby Laboratories. (2014). A Perceptual EOTF for
        Extended Dynamic Range Imagery, 1–17. Retrieved from
        https://www.smpte.org/sites/default/files/\
2014-05-06-EOTF-Miller-1-2-handout.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import Structure

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ST2084_CONSTANTS',
           'oetf_ST2084',
           'eotf_ST2084']

ST2084_CONSTANTS = Structure(m_1=2610 / 4096 * (1 / 4),
                             m_2=2523 / 4096 * 128,
                             c_1=3424 / 4096,
                             c_2=2413 / 4096 * 32,
                             c_3=2392 / 4096 * 32)
"""
*SMPTE ST 2084:2014* opto-electrical transfer function (OETF / OECF) and
electro-optical transfer function (EOTF / EOCF) constants.

ST2084_CONSTANTS : Structure
"""


def oetf_ST2084(C, L_p=10000):
    """
    Defines *SMPTE ST 2084:2014* optimised perceptual opto-electronic transfer
    function (OETF / OECF).

    Parameters
    ----------
    C : numeric or array_like
        Target optical output :math:`C` in :math:`cd/m^2` of the ideal
        reference display.
    L_p : numeric, optional
        Display peak luminance :math:`cd/m^2`.

    Returns
    -------
    numeric or ndarray
        Color value abbreviated as :math:`N`, normalized to the range [0, 1],
        that is directly proportional to the encoded signal representation,
        and which is not directly proportional to the optical output of a
        display device.

    Examples
    --------
    >>> oetf_ST2084(0.18)  # doctest: +ELLIPSIS
    0.0794209...
    """

    C = np.asarray(C)

    Y_p = (C / L_p) ** ST2084_CONSTANTS.m_1

    N = ((ST2084_CONSTANTS.c_1 + ST2084_CONSTANTS.c_2 * Y_p) /
         (ST2084_CONSTANTS.c_3 * Y_p + 1)) ** ST2084_CONSTANTS.m_2

    return N


def eotf_ST2084(N, L_p=10000):
    """
    Defines *SMPTE ST 2084:2014* optimised perceptual electro-optical transfer
    function (EOTF / EOCF).

    This perceptual quantizer (PQ) has been modeled by Dolby Laboratories
    using *Barten (1999)* contrast sensitivity function.

    Parameters
    ----------
    N : numeric or array_like
        Color value abbreviated as :math:`N`, normalized to the range [0, 1],
        that is directly proportional to the encoded signal representation,
        and which is not directly proportional to the optical output of a
        display device.
    L_p : numeric, optional
        Display peak luminance :math:`cd/m^2`.

    Returns
    -------
    numeric or ndarray
          Target optical output :math:`C` in :math:`cd/m^2` of the ideal
          reference display.

    Examples
    --------
    >>> eotf_ST2084(0.079420969944927)  # doctest: +ELLIPSIS
    0.1...
    """

    N = np.asarray(N)

    m_1_d = 1 / ST2084_CONSTANTS.m_1
    m_2_d = 1 / ST2084_CONSTANTS.m_2

    V_p = N ** m_2_d

    n = V_p - ST2084_CONSTANTS.c_1
    # Preventing *nan*.
    n = np.where(n < 0, 0, n)

    L = (n / (ST2084_CONSTANTS.c_2 - ST2084_CONSTANTS.c_3 * V_p)) ** m_1_d
    C = L_p * L

    return C
