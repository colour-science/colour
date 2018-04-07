# -*- coding: utf-8 -*-
"""
DIN99 Colourspace
=================

Defines the *DIN99* colourspace transformations:

-   :func:`colour.Lab_to_DIN99`

See Also
--------
`DIN99 Colourspace Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/din99.ipynb>`_

References
----------
-   :cite:`ASTMInternational2007` : ASTM International. (2007). ASTM D2244-07 -
    Standard Practice for Calculation of Color Tolerances and Color Differences
    from Instrumentally Measured Color Coordinates, i, 1-10.
    doi:10.1520/D2244-07
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import from_range_100, tsplit, tstack, to_domain_100

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Lab_to_DIN99']


def Lab_to_DIN99(Lab, k_E=1, k_CH=1):
    """
    Converts from *CIE L\*a\*b\** colourspace to *DIN99* colourspace.

    Parameters
    ----------
    Lab : array_like
        *CIE L\*a\*b\** colourspace array.
    k_E : numeric, optional
        Parametric factor :math:`K_E` used to compensate for texture and other
        specimen presentation effects.
    k_CH : numeric, optional
        Parametric factor :math:`K_{CH}` used to compensate for texture and
        other specimen presentation effects.

    Returns
    -------
    ndarray
        *DIN99* colourspace array.

    Notes
    -----
    -   Input *Lightness* :math:`L^*` is normalised to domain [0, 100].

    References
    ----------
    -   :cite:`ASTMInternational2007`

    Examples
    --------
    >>> import numpy as np
    >>> Lab = np.array([37.98562910, -23.62907688, -4.41746615])
    >>> Lab_to_DIN99(Lab)  # doctest: +ELLIPSIS
    array([ 49.6010164..., -16.2314573...,   1.0761812...])
    """

    L, a, b = tsplit(to_domain_100(Lab))

    cos_16 = np.cos(np.radians(16))
    sin_16 = np.sin(np.radians(16))

    e = cos_16 * a + sin_16 * b
    f = 0.7 * (-sin_16 * a + cos_16 * b)
    G = (e ** 2 + f ** 2) ** 0.5
    h_ef = np.arctan2(f, e)

    C_99 = (np.log(1 + 0.045 * G)) / (0.045 * k_CH * k_E)
    # Hue angle is unused currently.
    # h_99 = np.degrees(h_ef)
    a_99 = C_99 * np.cos(h_ef)
    b_99 = C_99 * np.sin(h_ef)
    L_99 = 105.509 * (np.log(1 + 0.0158 * L)) * k_E

    Lab_99 = tstack([L_99, a_99, b_99])

    return from_range_100(Lab_99)
