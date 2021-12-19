# -*- coding: utf-8 -*-
"""
Huang et al. (2015) Power-Functions
===================================

Defines the *Huang, Cui, Melgosa, Sanchez-Maranon, Li, Luo and Liu (2015)*
power-functions improving the performance of colour-difference formulas:

-   :func:`colour.difference.power_function_Huang2015`

References
----------
-   :cite:`Huang2015` : Huang, M., Cui, G., Melgosa, M., Sanchez-Maranon, M.,
    Li, C., Luo, M. R., & Liu, H. (2015). Power functions improving the
    performance of color-difference formulas. Optical Society of America,
    23(1), 597-610. doi:10.1364/OE.23.000597
-   :cite:`Li2017` : Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G.,
    Melgosa, M., Brill, M. H., & Pointer, M. (2017). Comprehensive color
    solutions: CAM16, CAT16, and CAM16-UCS. Color Research & Application,
    42(6), 703-718. doi:10.1002/col.22131
"""

from __future__ import annotations

import numpy as np

from colour.hints import FloatingOrArrayLike, FloatingOrNDArray, Literal, Union
from colour.utilities import CaseInsensitiveMapping, tsplit, validate_method

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'power_function_Huang2015',
]

COEFFICIENTS_HUANG2015: CaseInsensitiveMapping = CaseInsensitiveMapping({
    'CIE 1976': np.array([1.26, 0.55]),
    'CIE 1994': np.array([1.41, 0.70]),
    'CIE 2000': np.array([1.43, 0.70]),
    'CMC': np.array([1.34, 0.66]),
    'CAM02-LCD': np.array([1.00, 0.85]),
    'CAM02-SCD': np.array([1.45, 0.75]),
    'CAM02-UCS': np.array([1.30, 0.75]),
    'CAM16-UCS': np.array([1.41, 0.63]),
    'DIN99d': np.array([1.28, 0.74]),
    'OSA': np.array([3.32, 0.62]),
    'OSA-GP-Euclidean': np.array([1.52, 0.76]),
    'ULAB': np.array([1.17, 0.69]),
})
COEFFICIENTS_HUANG2015.__doc__ = """
*Huang et al. (2015)* power-functions coefficients.

References
----------
:cite:`Huang2015`, :cite:`Li2017`

Notes
-----
-   :cite:`Li2017` does not give the coefficients for the *CAM16-LCD* and
    *CAM16-SCD* colourspaces. *Ronnie Luo* has been contacted to know if they
    have been computed.

Aliases:

-   'cie1976': 'CIE 1976'
-   'cie1994': 'CIE 1994'
-   'cie2000': 'CIE 2000'
"""
COEFFICIENTS_HUANG2015['cie1976'] = COEFFICIENTS_HUANG2015['CIE 1976']
COEFFICIENTS_HUANG2015['cie1994'] = COEFFICIENTS_HUANG2015['CIE 1994']
COEFFICIENTS_HUANG2015['cie2000'] = COEFFICIENTS_HUANG2015['CIE 2000']


def power_function_Huang2015(
        d_E: FloatingOrArrayLike,
        coefficients: Union[Literal[
            'CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC', 'CAM02-LCD',
            'CAM02-SCD', 'CAM16-UCS', 'DIN99d', 'OSA', 'OSA-GP-Euclidean',
            'ULAB'], str] = 'CIE 2000') -> FloatingOrNDArray:
    """
    Improves the performance of the :math:`\\Delta E` value for given
    coefficients using
    *Huang, Cui, Melgosa, Sanchez-Maranon, Li, Luo and Liu (2015)*
    power-function: :math:`d_E^{\\prime}=a*d_{E^b}`.

    Parameters
    ----------
    d_E
        Computed colour difference array :math:`\\Delta E`.
    coefficients
        Coefficients for the power-function.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Improved math:`\\Delta E` value.

    References
    ----------
    :cite:`Huang2015`, :cite:`Li2017`

    Examples
    --------
    >>> d_E = np.array([2.0425, 2.8615, 3.4412])
    >>> power_function_Huang2015(d_E)  # doctest: +ELLIPSIS
    array([ 2.3574879...,  2.9850503...,  3.3965106...])
    """

    coefficients = validate_method(
        coefficients, COEFFICIENTS_HUANG2015,
        '"{0}" coefficients are invalid, '
        'they must be one of {1}!')

    a, b = tsplit(COEFFICIENTS_HUANG2015[coefficients])

    d_E_p = a * d_E ** b

    return d_E_p
