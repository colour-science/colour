# -*- coding: utf-8 -*-
"""
DIN99 Colourspace
=================

Defines the *DIN99* colourspace transformations:

-   :func:`colour.Lab_to_DIN99`
-   :func:`colour.DIN99_to_Lab`

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

from colour.algebra import spow
from colour.utilities import from_range_100, tsplit, tstack, to_domain_100

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Lab_to_DIN99', 'DIN99_to_Lab']


def Lab_to_DIN99(Lab, k_E=1, k_CH=1):
    """
    Converts from *CIE L\\*a\\*b\\** colourspace to *DIN99* colourspace.

    Parameters
    ----------
    Lab : array_like
        *CIE L\\*a\\*b\\** colourspace array.
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

    +------------+------------------------+--------------------+
    | **Domain** | **Scale - Reference**  | **Scale - 1**      |
    +============+========================+====================+
    | ``Lab``    | ``L`` : [0, 100]       | ``L`` : [0, 1]     |
    |            |                        |                    |
    |            | ``a`` : [-100, 100]    | ``a`` : [-1, 1]    |
    |            |                        |                    |
    |            | ``b`` : [-100, 100]    | ``b`` : [-1, 1]    |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  | **Scale - Reference**  | **Scale - 1**      |
    +============+========================+====================+
    | ``Lab_99`` | ``L_99`` : [0, 100]    | ``L_99`` : [0, 1]  |
    |            |                        |                    |
    |            | ``a_99`` : [-100, 100] | ``a_99`` : [-1, 1] |
    |            |                        |                    |
    |            | ``b_99`` : [-100, 100] | ``b_99`` : [-1, 1] |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`ASTMInternational2007`

    Examples
    --------
    >>> import numpy as np
    >>> Lab = np.array([41.52787529, 52.63858304, 26.92317922])
    >>> Lab_to_DIN99(Lab)  # doctest: +ELLIPSIS
    array([ 53.2282198...,  28.4163465...,   3.8983955...])
    """

    L, a, b = tsplit(to_domain_100(Lab))

    cos_16 = np.cos(np.radians(16))
    sin_16 = np.sin(np.radians(16))

    e = cos_16 * a + sin_16 * b
    f = 0.7 * (-sin_16 * a + cos_16 * b)
    G = spow(e ** 2 + f ** 2, 0.5)
    h_ef = np.arctan2(f, e)

    C_99 = (np.log(1 + 0.045 * G)) / (0.045 * k_CH * k_E)
    # Hue angle is unused currently.
    # h_99 = np.degrees(h_ef)
    a_99 = C_99 * np.cos(h_ef)
    b_99 = C_99 * np.sin(h_ef)
    L_99 = 105.509 * (np.log(1 + 0.0158 * L)) * k_E

    Lab_99 = tstack([L_99, a_99, b_99])

    return from_range_100(Lab_99)


def DIN99_to_Lab(Lab_99, k_E=1, k_CH=1):
    """
    Converts from *DIN99* colourspace to *CIE L\\*a\\*b\\** colourspace.

    Parameters
    ----------
    Lab_99 : array_like
        *DIN99* colourspace array.
    k_E : numeric, optional
        Parametric factor :math:`K_E` used to compensate for texture and other
        specimen presentation effects.
    k_CH : numeric, optional
        Parametric factor :math:`K_{CH}` used to compensate for texture and
        other specimen presentation effects.

    Returns
    -------
    ndarray
        *CIE L\\*a\\*b\\** colourspace array.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** | **Scale - Reference**  | **Scale - 1**      |
    +============+========================+====================+
    | ``Lab_99`` | ``L_99`` : [0, 100]    | ``L_99`` : [0, 1]  |
    |            |                        |                    |
    |            | ``a_99`` : [-100, 100] | ``a_99`` : [-1, 1] |
    |            |                        |                    |
    |            | ``b_99`` : [-100, 100] | ``b_99`` : [-1, 1] |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  | **Scale - Reference**  | **Scale - 1**      |
    +============+========================+====================+
    | ``Lab``    | ``L`` : [0, 100]       | ``L`` : [0, 1]     |
    |            |                        |                    |
    |            | ``a`` : [-100, 100]    | ``a`` : [-1, 1]    |
    |            |                        |                    |
    |            | ``b`` : [-100, 100]    | ``b`` : [-1, 1]    |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`ASTMInternational2007`

    Examples
    --------
    >>> import numpy as np
    >>> Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
    >>> DIN99_to_Lab(Lab_99)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  52.6385830...,  26.9231792...])
    """

    L_99, a_99, b_99 = tsplit(to_domain_100(Lab_99))

    cos_16 = np.cos(np.radians(16))
    sin_16 = np.sin(np.radians(16))

    h_99 = np.arctan2(b_99, a_99)

    C_99 = np.sqrt(a_99 ** 2 + b_99 ** 2)
    G = (np.exp(0.045 * C_99 * k_CH * k_E) - 1) / 0.045

    e = G * np.cos(h_99)
    f = G * np.sin(h_99)

    a = e * cos_16 - (f / 0.7) * sin_16
    b = e * sin_16 + (f / 0.7) * cos_16
    L = (np.exp(L_99 * k_E / 105.509) - 1) / 0.0158

    Lab = tstack([L, a, b])

    return from_range_100(Lab)
