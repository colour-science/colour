# -*- coding: utf-8 -*-
"""
McCamy (1992) Correlated Colour Temperature
===========================================

Defines *McCamy (1992)* correlated colour temperature :math:`T_{cp}`
computations objects:

-   :func:`colour.temperature.xy_to_CCT_McCamy1992`: Correlated colour
    temperature :math:`T_{cp}` computation of given *CIE XYZ* tristimulus
    values *CIE xy* chromaticity coordinates using *McCamy (1992)* method.

See Also
--------
`Colour Temperature & Correlated Colour Temperature Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/temperature/cct.ipynb>`_

References
----------
-   :cite:`Wikipedia2001` : Wikipedia. (2001). Approximation. Retrieved June
    28, 2014, from http://en.wikipedia.org/wiki/Color_temperature#Approximation
"""

from __future__ import division, unicode_literals

from colour.utilities import tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['xy_to_CCT_McCamy1992']


def xy_to_CCT_McCamy1992(xy):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE XYZ* tristimulus values *CIE xy* chromaticity coordinates using
    *McCamy (1992)* method.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    numeric or ndarray
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    :cite:`Wikipedia2001`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.31270, 0.32900])
    >>> xy_to_CCT_McCamy1992(xy)  # doctest: +ELLIPSIS
    6505.0805913...
    """

    x, y = tsplit(xy)

    n = (x - 0.3320) / (y - 0.1858)
    CCT = -449 * n ** 3 + 3525 * n ** 2 - 6823.3 * n + 5520.33

    return CCT
