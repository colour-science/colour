# -*- coding: utf-8 -*-
"""
Krystek (1985) Correlated Colour Temperature
============================================

Defines *Krystek (1985)* correlated colour temperature :math:`T_{cp}`
computations objects:

-   :func:`colour.temperature.CCT_to_uv_Krystek1985`: *CIE UCS* colourspace
    *uv* chromaticity coordinates computation of given correlated colour
    temperature :math:`T_{cp}` using *Krystek (1985)* method.

See Also
--------
`Colour Temperature & Correlated Colour Temperature Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/temperature/cct.ipynb>`_

References
----------
-   :cite:`Krystek1985b` : Krystek, M. (1985). An algorithm to calculate
    correlated colour temperature. Color Research & Application, 10(1),
    38-40. doi:10.1002/col.5080100109
"""

from __future__ import division, unicode_literals

from colour.utilities import as_float_array, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['CCT_to_uv_Krystek1985']


def CCT_to_uv_Krystek1985(CCT):
    """
    Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` using *Krystek (1985)* method.

    Parameters
    ----------
    CCT : numeric
        Correlated colour temperature :math:`T_{cp}`.

    Returns
    -------
    ndarray
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    Notes
    -----
    -   *Krystek (1985)* method computations are valid for correlated colour
        temperature :math:`T_{cp}` normalised to domain [1000, 15000].

    References
    ----------
    :cite:`Krystek1985b`

    Examples
    --------
    >>> CCT_to_uv_Krystek1985(6504.38938305)  # doctest: +ELLIPSIS
    array([ 0.1837669...,  0.3093443...])
    """

    T = as_float_array(CCT)

    u = ((0.860117757 + 1.54118254 * 10e-4 * T + 1.28641212 * 10e-7 * T ** 2) /
         (1 + 8.42420235 * 10e-4 * T + 7.08145163 * 10e-7 * T ** 2))
    v = ((0.317398726 + 4.22806245 * 10e-5 * T + 4.20481691 * 10e-8 * T ** 2) /
         (1 - 2.89741816 * 10e-5 * T + 1.61456053 * 10e-7 * T ** 2))

    return tstack([u, v])
