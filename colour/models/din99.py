# -*- coding: utf-8 -*-
"""
DIN99 Colourspace and DIN99b, DIN99c, DIN99d Refined Formulas
=============================================================

Defines the *DIN99* colourspace and *DIN99b*, *DIN99c*, *DIN99d* refined
formulas transformations:

-   :func:`colour.Lab_to_DIN99`
-   :func:`colour.DIN99_to_Lab`
-   :func:`colour.XYZ_to_DIN99`
-   :func:`colour.DIN99_to_XYZ`

References
----------
-   :cite:`ASTMInternational2007` : ASTM International. (2007). ASTM D2244-07 -
    Standard Practice for Calculation of Color Tolerances and Color Differences
    from Instrumentally Measured Color Coordinates: Vol. i (pp. 1-10).
    doi:10.1520/D2244-16
-   :cite:`Cui2002` :  Cui, G., Luo, M. R., Rigg, B., Roesler, G., & Witt, K.
    (2002). Uniform colour spaces based on the DIN99 colour-difference formula.
    Color Research & Application, 27(4), 282-290. doi:10.1002/col.10066
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import ArrayLike, Floating, Literal, NDArray, Union
from colour.models import Lab_to_XYZ, XYZ_to_Lab
from colour.utilities import (
    CaseInsensitiveMapping,
    from_range_100,
    tsplit,
    tstack,
    to_domain_100,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'DIN99_METHODS',
    'Lab_to_DIN99',
    'DIN99_to_Lab',
    'XYZ_to_DIN99',
    'DIN99_to_XYZ',
]

DIN99_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping({
    'ASTMD2244-07':
        np.array([105.509, 0.0158, 16.0, 0.7, 1, 9 / 200, 0.0, 9 / 200]),
    'DIN99':
        np.array([105.509, 0.0158, 16.0, 0.7, 1, 9 / 200, 0.0, 9 / 200]),
    'DIN99b':
        np.array([303.67, 0.0039, 26.0, 0.83, 23.0, 0.075, 26.0, 1]),
    'DIN99c':
        np.array([317.65, 0.0037, 0.0, 0.94, 23.0, 0.066, 0.0, 1]),
    'DIN99d':
        np.array([325.22, 0.0036, 50.0, 1.14, 22.5, 0.06, 50.0, 1]),
})
"""
*DIN99* colourspace methods, i.e. the coefficients for the *DIN99b*, *DIN99c*,
and *DIN99d* refined formulas according to *Cui et al. (2002)*.

References
----------
:cite:`ASTMInternational2007`, :cite:`Cui2002`
"""


def Lab_to_DIN99(
        Lab: ArrayLike,
        k_E: Floating = 1,
        k_CH: Floating = 1,
        method: Union[Literal['ASTMD2244-07', 'DIN99', 'DIN99b', 'DIN99c',
                              'DIN99d'], str] = 'DIN99') -> NDArray:
    """
    Converts from *CIE L\\*a\\*b\\** colourspace to *DIN99* colourspace or
    one of the *DIN99b*, *DIN99c*, *DIN99d* refined formulas according
    to *Cui et al. (2002)*.

    Parameters
    ----------
    Lab
        *CIE L\\*a\\*b\\** colourspace array.
    k_E
        Parametric factor :math:`K_E` used to compensate for texture and other
        specimen presentation effects.
    k_CH
        Parametric factor :math:`K_{CH}` used to compensate for texture and
        other specimen presentation effects.
    method
        Computation method to choose between the :cite:`ASTMInternational2007`
        formula and the refined formulas according to *Cui et al. (2002)*.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`ASTMInternational2007`, :cite:`Cui2002`

    Examples
    --------
    >>> import numpy as np
    >>> Lab = np.array([41.52787529, 52.63858304, 26.92317922])
    >>> Lab_to_DIN99(Lab)  # doctest: +ELLIPSIS
    array([ 53.2282198...,  28.4163465...,   3.8983955...])
    """

    c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = DIN99_METHODS[validate_method(
        str(method), DIN99_METHODS)]

    L, a, b = tsplit(to_domain_100(Lab))

    cos_c = np.cos(np.radians(c_3))
    sin_c = np.sin(np.radians(c_3))

    e = cos_c * a + sin_c * b
    f = c_4 * (-sin_c * a + cos_c * b)
    G = spow(e ** 2 + f ** 2, 0.5)
    h_ef = np.arctan2(f, e) + np.radians(c_7)

    C_99 = c_5 * (np.log(1 + c_6 * G)) / (c_8 * k_CH * k_E)
    # Hue angle is unused currently.
    # h_99 = np.degrees(h_ef)
    a_99 = C_99 * np.cos(h_ef)
    b_99 = C_99 * np.sin(h_ef)
    L_99 = c_1 * (np.log(1 + c_2 * L)) * k_E

    Lab_99 = tstack([L_99, a_99, b_99])

    return from_range_100(Lab_99)


def DIN99_to_Lab(
        Lab_99: ArrayLike,
        k_E: Floating = 1,
        k_CH: Floating = 1,
        method: Union[Literal['ASTMD2244-07', 'DIN99', 'DIN99b', 'DIN99c',
                              'DIN99d'], str] = 'DIN99') -> NDArray:
    """
    Converts from *DIN99* colourspace or one of the *DIN99b*, *DIN99c*,
    *DIN99d* refined formulas according to *Cui et al. (2002)* to
    *CIE L\\*a\\*b\\** colourspace.

    Parameters
    ----------
    Lab_99
        *DIN99* colourspace array.
    k_E
        Parametric factor :math:`K_E` used to compensate for texture and other
        specimen presentation effects.
    k_CH
        Parametric factor :math:`K_{CH}` used to compensate for texture and
        other specimen presentation effects.
    method
        Computation method to choose between the :cite:`ASTMInternational2007`
        formula and the refined formulas according to *Cui et al. (2002)*.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`ASTMInternational2007`, :cite:`Cui2002`

    Examples
    --------
    >>> import numpy as np
    >>> Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
    >>> DIN99_to_Lab(Lab_99)  # doctest: +ELLIPSIS
    array([ 41.5278752...,  52.6385830...,  26.9231792...])
    """

    c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8 = DIN99_METHODS[validate_method(
        str(method), DIN99_METHODS)]

    L_99, a_99, b_99 = tsplit(to_domain_100(Lab_99))

    cos = np.cos(np.radians(c_3))
    sin = np.sin(np.radians(c_3))

    h_99 = np.arctan2(b_99, a_99) - np.radians(c_7)

    C_99 = np.sqrt(a_99 ** 2 + b_99 ** 2)
    G = (np.exp((c_8 / c_5) * (C_99) * k_CH * k_E) - 1) / c_6

    e = G * np.cos(h_99)
    f = G * np.sin(h_99)

    a = e * cos - (f / c_4) * sin
    b = e * sin + (f / c_4) * cos
    L = (np.exp(L_99 * k_E / c_1) - 1) / c_2

    Lab = tstack([L, a, b])

    return from_range_100(Lab)


def XYZ_to_DIN99(
        XYZ: ArrayLike,
        illuminant: ArrayLike = CCS_ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D65'],
        k_E: Floating = 1,
        k_CH: Floating = 1,
        method: Union[Literal['ASTMD2244-07', 'DIN99', 'DIN99b', 'DIN99c',
                              'DIN99d'], str] = 'DIN99') -> NDArray:
    """
    Converts from *CIE XYZ* tristimulus values to *DIN99* colourspace or
    one of the *DIN99b*, *DIN99c*, *DIN99d* refined formulas according
    to *Cui et al. (2002)*.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.
    k_E
        Parametric factor :math:`K_E` used to compensate for texture and other
        specimen presentation effects.
    k_CH
        Parametric factor :math:`K_{CH}` used to compensate for texture and
        other specimen presentation effects.
    method
        Computation method to choose between the :cite:`ASTMInternational2007`
        formula and the refined formulas according to *Cui et al. (2002)*.

    Returns
    -------
    :class:`numpy.ndarray`
        *DIN99* colourspace array.

    Notes
    -----

    +----------------+-----------------------+-----------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**   |
    +================+=======================+=================+
    | ``XYZ``        | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+
    | ``illuminant`` | [0, 1]                | [0, 1]          |
    +----------------+-----------------------+-----------------+

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
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> XYZ_to_DIN99(XYZ)  # doctest: +ELLIPSIS
    array([ 53.2282198...,  28.4163465...,   3.8983955...])
    """

    Lab = XYZ_to_Lab(XYZ, illuminant)

    return Lab_to_DIN99(Lab, k_E, k_CH, method)


def DIN99_to_XYZ(
        Lab_99: ArrayLike,
        illuminant: ArrayLike = CCS_ILLUMINANTS[
            'CIE 1931 2 Degree Standard Observer']['D65'],
        k_E: Floating = 1,
        k_CH: Floating = 1,
        method: Union[Literal['ASTMD2244-07', 'DIN99', 'DIN99b', 'DIN99c',
                              'DIN99d'], str] = 'DIN99') -> NDArray:
    """
    Converts from *DIN99* colourspace or one of the *DIN99b*, *DIN99c*,
    *DIN99d* refined formulas according to *Cui et al. (2002)* to *CIE XYZ*
    tristimulus values.

    Parameters
    ----------
    Lab_99
        *DIN99* colourspace array.
    illuminant
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array.
    k_E
        Parametric factor :math:`K_E` used to compensate for texture and other
        specimen presentation effects.
    k_CH
        Parametric factor :math:`K_{CH}` used to compensate for texture and
        other specimen presentation effects.
    method
        Computation method to choose between the :cite:`ASTMInternational2007`
        formula and the refined formulas according to *Cui et al. (2002)*.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE XYZ* tristimulus values.

    Notes
    -----

    +----------------+------------------------+--------------------+
    | **Domain**     | **Scale - Reference**  | **Scale - 1**      |
    +================+========================+====================+
    | ``Lab_99``     | ``L_99`` : [0, 100]    | ``L_99`` : [0, 1]  |
    |                |                        |                    |
    |                | ``a_99`` : [-100, 100] | ``a_99`` : [-1, 1] |
    |                |                        |                    |
    |                | ``b_99`` : [-100, 100] | ``b_99`` : [-1, 1] |
    +----------------+------------------------+--------------------+
    | ``illuminant`` | [0, 1]                 | [0, 1]             |
    +----------------+------------------------+--------------------+

    +----------------+-----------------------+---------------------+
    | **Range**      | **Scale - Reference** | **Scale - 1**       |
    +================+=======================+=====================+
    | ``XYZ``        | [0, 1]                | [0, 1]              |
    +----------------+-----------------------+---------------------+

    References
    ----------
    :cite:`ASTMInternational2007`

    Examples
    --------
    >>> import numpy as np
    >>> Lab_99 = np.array([53.22821989, 28.41634656, 3.89839552])
    >>> DIN99_to_XYZ(Lab_99)  # doctest: +ELLIPSIS
    array([ 0.2065400...,  0.1219722...,  0.0513695...])
    """

    Lab = DIN99_to_Lab(Lab_99, k_E, k_CH, method)

    return Lab_to_XYZ(Lab, illuminant)
