# -*- coding: utf-8 -*-
"""
:math:`\\Delta E_{99}` DIN99 - Colour Difference Formula
=======================================================

Defines the :math:`\\Delta E_{99}` *DIN99* colour difference formula:

-   :func:`colour.difference.delta_E_DIN99`

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

from colour.algebra import euclidean_distance
from colour.models import Lab_to_DIN99
from colour.utilities import get_domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['delta_E_DIN99']


def delta_E_DIN99(Lab_1, Lab_2, textiles=False):
    """
    Returns the difference :math:`\\Delta E_{DIN99}` between two given
    *CIE L\\*a\\*b\\** colourspace arrays using *DIN99* formula.

    Parameters
    ----------
    Lab_1 : array_like
        *CIE L\\*a\\*b\\** colourspace array 1.
    Lab_2 : array_like
        *CIE L\\*a\\*b\\** colourspace array 2.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E_{DIN99}`.

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
    :cite:`ASTMInternational2007`

    Examples
    --------
    >>> import numpy as np
    >>> Lab_1 = np.array([60.2574, -34.0099, 36.2677])
    >>> Lab_2 = np.array([60.4626, -34.1751, 39.4387])
    >>> delta_E_DIN99(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    1.1772166...
    """

    k_E = 2 if textiles else 1
    k_CH = 0.5 if textiles else 1

    factor = 100 if get_domain_range_scale() == '1' else 1

    d_E = euclidean_distance(
        Lab_to_DIN99(Lab_1, k_E, k_CH) * factor,
        Lab_to_DIN99(Lab_2, k_E, k_CH) * factor)

    return d_E
