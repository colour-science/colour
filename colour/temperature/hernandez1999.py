# -*- coding: utf-8 -*-
"""
Hernandez-Andres, Lee and Romero (1999) Correlated Colour Temperature
=====================================================================

Defines *Hernandez-Andres et al. (1999)* correlated colour temperature
:math:`T_{cp}` computations objects:

-   :func:`colour.temperature.xy_to_CCT_Hernandez1999`: Correlated colour
    temperature :math:`T_{cp}` computation of given *CIE XYZ* tristimulus
    values *CIE xy* chromaticity coordinates using
    *Hernandez-Andres, Lee and Romero (1999)* method.

See Also
--------
`Colour Temperature & Correlated Colour Temperature Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/temperature/cct.ipynb>`_

References
----------
-   :cite:`Hernandez-Andres1999a` : Hernandez-Andres, J., Lee, R. L., &
    Romero, J. (1999). Calculating correlated color temperatures across the
    entire gamut of daylight and skylight chromaticities. Applied Optics,
    38(27), 5703. doi:10.1364/AO.38.005703
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_float, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['xy_to_CCT_Hernandez1999']


def xy_to_CCT_Hernandez1999(xy):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE XYZ* tristimulus values *CIE xy* chromaticity coordinates using
    *Hernandez-Andres et al. (1999)* method.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.

    Returns
    -------
    numeric
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    :cite:`Hernandez-Andres1999a`

    Examples
    --------
    >>> xy = np.array([0.31270, 0.32900])
    >>> xy_to_CCT_Hernandez1999(xy)  # doctest: +ELLIPSIS
    6500.7420431...
    """

    x, y = tsplit(xy)

    n = (x - 0.3366) / (y - 0.1735)
    CCT = (-949.86315 + 6253.80338 * np.exp(-n / 0.92159) +
           28.70599 * np.exp(-n / 0.20039) + 0.00004 * np.exp(-n / 0.07125))

    n = np.where(CCT > 50000, (x - 0.3356) / (y - 0.1691), n)

    CCT = np.where(
        CCT > 50000,
        36284.48953 + 0.00228 * np.exp(-n / 0.07861) +
        5.4535e-36 * np.exp(-n / 0.01543),
        CCT,
    )

    return as_float(CCT)
