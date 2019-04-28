# -*- coding: utf-8 -*-
"""
CIE Illuminant D Series Correlated Colour Temperature
=====================================================

Defines *CIE Illuminant D Series* correlated colour temperature :math:`T_{cp}
computations objects:

-   :func:`colour.temperature.CCT_to_xy_CIE_D`: *CIE XYZ* tristimulus values
    *CIE xy* chromaticity coordinates computation of *CIE Illuminant D Series*
    from given correlated colour temperature :math:`T_{cp}` of that
    *CIE Illuminant D Series*.

See Also
--------
`Colour Temperature & Correlated Colour Temperature Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/temperature/cct.ipynb>`_

References
----------
-   :cite:`Wyszecki2000z` : Wyszecki, G., & Stiles, W. S. (2000). CIE Method of
    Calculating D-Illuminants. In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (pp. 145-146). Wiley. ISBN:978-0471399186
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import daylight_locus_function
from colour.utilities import as_float_array, tstack, usage_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CCT_to_xy_CIE_D']


def CCT_to_xy_CIE_D(CCT):
    """
    Converts from the correlated colour temperature :math:`T_{cp}` of a
    *CIE Illuminant D Series* to the chromaticity of that
    *CIE Illuminant D Series* illuminant.

    Parameters
    ----------
    CCT : numeric or array_like
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    ndarray
        *CIE xy* chromaticity coordinates.

    Raises
    ------
    ValueError
        If the correlated colour temperature is not in appropriate domain.

    References
    ----------
    :cite:`Wyszecki2000z`

    Examples
    --------
    >>> CCT_to_xy_CIE_D(6504.38938305)  # doctest: +ELLIPSIS
    array([ 0.3127077...,  0.3291128...])
    """

    CCT = as_float_array(CCT)

    if np.any(CCT[np.asarray(np.logical_or(CCT < 4000, CCT > 25000))]):
        usage_warning(('Correlated colour temperature must be in domain '
                       '[4000, 25000], unpredictable results may occur!'))

    x = np.where(
        CCT <= 7000,
        -4.607 * 10 ** 9 / CCT ** 3 + 2.9678 * 10 ** 6 / CCT ** 2 +
        0.09911 * 10 ** 3 / CCT + 0.244063,
        -2.0064 * 10 ** 9 / CCT ** 3 + 1.9018 * 10 ** 6 / CCT ** 2 +
        0.24748 * 10 ** 3 / CCT + 0.23704,
    )

    y = daylight_locus_function(x)

    xy = tstack([x, y])

    return xy
