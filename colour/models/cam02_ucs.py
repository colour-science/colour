# -*- coding: utf-8 -*-
"""
CAM02-LCD, CAM02-SCD, and CAM02-UCS Colourspaces - Luo, Cui and Li (2006)
=========================================================================

Defines the *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, and *CAM02-UCS*
colourspaces transformations:

-   :func:`colour.JMh_CIECAM02_to_CAM02LCD`
-   :func:`colour.CAM02LCD_to_JMh_CIECAM02`
-   :func:`colour.JMh_CIECAM02_to_CAM02SCD`
-   :func:`colour.CAM02SCD_to_JMh_CIECAM02`
-   :func:`colour.JMh_CIECAM02_to_CAM02UCS`
-   :func:`colour.CAM02UCS_to_JMh_CIECAM02`

See Also
--------
`CAM02-LCD, CAM02-SCD, and CAM02-UCS Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/cam02_ucs.ipynb>`_

References
----------
-   :cite:`Luo2006b` : Luo, M. R., Cui, G., & Li, C. (2006). Uniform colour
    spaces based on CIECAM02 colour appearance model. Color Research &
    Application, 31(4), 320-330. doi:10.1002/col.20227
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.algebra import cartesian_to_polar, polar_to_cartesian
from colour.utilities import (CaseInsensitiveMapping, from_range_100,
                              from_range_degrees, to_domain_100,
                              to_domain_degrees, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'Coefficients_UCS_Luo2006', 'COEFFICIENTS_UCS_LUO2006',
    'JMh_CIECAM02_to_UCS_Luo2006', 'UCS_Luo2006_to_JMh_CIECAM02',
    'JMh_CIECAM02_to_CAM02LCD', 'CAM02LCD_to_JMh_CIECAM02',
    'JMh_CIECAM02_to_CAM02SCD', 'CAM02SCD_to_JMh_CIECAM02',
    'JMh_CIECAM02_to_CAM02UCS', 'CAM02UCS_to_JMh_CIECAM02'
]


class Coefficients_UCS_Luo2006(
        namedtuple('Coefficients_UCS_Luo2006', ('K_L', 'c_1', 'c_2'))):
    """
    Defines the the class storing *Luo et al. (2006)* fitting coefficients for
    the *CAM02-LCD*, *CAM02-SCD*, and *CAM02-UCS* colourspaces.
    """


COEFFICIENTS_UCS_LUO2006 = CaseInsensitiveMapping({
    'CAM02-LCD': Coefficients_UCS_Luo2006(0.77, 0.007, 0.0053),
    'CAM02-SCD': Coefficients_UCS_Luo2006(1.24, 0.007, 0.0363),
    'CAM02-UCS': Coefficients_UCS_Luo2006(1.00, 0.007, 0.0228)
})
"""
*Luo et al. (2006)* fitting coefficients for the *CAM02-LCD*, *CAM02-SCD*, and
*CAM02-UCS* colourspaces.

COEFFICIENTS_UCS_LUO2006 : CaseInsensitiveMapping
    **{'CAM02-LCD', 'CAM02-SCD', 'CAM02-UCS'}**
"""


def JMh_CIECAM02_to_UCS_Luo2006(JMh, coefficients):
    """
    Converts from *CIECAM02* :math:`JMh` correlates array to one of the
    *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS* colourspaces
    :math:`J'a'b'` array.

    The :math:`JMh` correlates array is constructed using the CIECAM02
    correlate of *Lightness* :math:`J`, the *CIECAM02* correlate of
    *colourfulness* :math:`M` and the *CIECAM02* *Hue* angle :math:`h` in
    degrees.

    Parameters
    ----------
    JMh : array_like
        *CIECAM02* correlates array :math:`JMh`.
    coefficients : array_like
        Coefficients of one of the *Luo et al. (2006)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.

    Returns
    -------
    ndarray
        *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
        colourspaces :math:`J'a'b'` array.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``JMh``    | ``J`` : [0, 100]       | ``J`` : [0, 1]     |
    |            |                        |                    |
    |            | ``M`` : [0, 100]       | ``M`` : [0, 1]     |
    |            |                        |                    |
    |            | ``h`` : [0, 360]       | ``h`` : [0, 1]     |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``Jpapbp`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |            |                        |                    |
    |            | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |            |                        |                    |
    |            | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +------------+------------------------+--------------------+

    Examples
    --------
    >>> from colour.appearance import (
    ...     CIECAM02_VIEWING_CONDITIONS,
    ...     XYZ_to_CIECAM02)
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = CIECAM02_VIEWING_CONDITIONS['Average']
    >>> specification = XYZ_to_CIECAM02(
    ...     XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> JMh_CIECAM02_to_UCS_Luo2006(JMh, COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])
    ... # doctest: +ELLIPSIS
    array([ 54.9043313...,  -0.0845039...,  -0.0685483...])
    """

    J, M, h = tsplit(JMh)
    J = to_domain_100(J)
    M = to_domain_100(M)
    h = to_domain_degrees(h)

    _K_L, c_1, c_2 = tsplit(coefficients)

    J_p = ((1 + 100 * c_1) * J) / (1 + c_1 * J)
    M_p = (1 / c_2) * np.log(1 + c_2 * M)

    a_p, b_p = tsplit(polar_to_cartesian(tstack([M_p, np.radians(h)])))

    Jpapbp = tstack([J_p, a_p, b_p])

    return from_range_100(Jpapbp)


def UCS_Luo2006_to_JMh_CIECAM02(Jpapbp, coefficients):
    """
    Converts from one of the *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or
    *CAM02-UCS* colourspaces :math:`J'a'b'` array to *CIECAM02* :math:`JMh`
    correlates array.

    Parameters
    ----------
    Jpapbp : array_like
        *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
        colourspaces :math:`J'a'b'` array.
    coefficients : array_like
        Coefficients of one of the *Luo et al. (2006)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.

    Returns
    -------
    ndarray
        *CIECAM02* correlates array :math:`JMh`.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``Jpapbp`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |            |                        |                    |
    |            | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |            |                        |                    |
    |            | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``JMh``    | ``J`` : [0, 100]       | ``J`` : [0, 1]     |
    |            |                        |                    |
    |            | ``M`` : [0, 100]       | ``M`` : [0, 1]     |
    |            |                        |                    |
    |            | ``h`` : [0, 360]       | ``h`` : [0, 1]     |
    +------------+------------------------+--------------------+

    Examples
    --------
    >>> Jpapbp = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> UCS_Luo2006_to_JMh_CIECAM02(
    ...     Jpapbp, COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])
    ... # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    J_p, a_p, b_p = tsplit(to_domain_100(Jpapbp))
    _K_L, c_1, c_2 = tsplit(coefficients)

    J = -J_p / (c_1 * J_p - 1 - 100 * c_1)

    M_p, h = tsplit(cartesian_to_polar(tstack([a_p, b_p])))

    M = (np.exp(M_p / (1 / c_2)) - 1) / c_2

    JMh = tstack([
        from_range_100(J),
        from_range_100(M),
        from_range_degrees(np.degrees(h) % 360)
    ])

    return JMh


def JMh_CIECAM02_to_CAM02LCD(JMh):
    """
    Converts from *CIECAM02* :math:`JMh` correlates array to
    *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    JMh : array_like
        *CIECAM02* correlates array :math:`JMh`.

    Returns
    -------
    ndarray
        *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'` array.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``JMh``    | ``J`` : [0, 100]       | ``J`` : [0, 1]     |
    |            |                        |                    |
    |            | ``M`` : [0, 100]       | ``M`` : [0, 1]     |
    |            |                        |                    |
    |            | ``h`` : [0, 360]       | ``h`` : [0, 1]     |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``Jpapbp`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |            |                        |                    |
    |            | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |            |                        |                    |
    |            | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> from colour.appearance import (
    ...     CIECAM02_VIEWING_CONDITIONS,
    ...     XYZ_to_CIECAM02)
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = CIECAM02_VIEWING_CONDITIONS['Average']
    >>> specification = XYZ_to_CIECAM02(
    ...     XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> JMh_CIECAM02_to_CAM02LCD(JMh)  # doctest: +ELLIPSIS
    array([ 54.9043313...,  -0.0845039...,  -0.0685483...])
    """

    return JMh_CIECAM02_to_UCS_Luo2006(
        JMh, coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])


def CAM02LCD_to_JMh_CIECAM02(Jpapbp):
    """
    Converts from *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'`
    array to *CIECAM02* :math:`JMh` correlates array.

    Parameters
    ----------
    Jpapbp : array_like
        *Luo et al. (2006)* *CAM02-LCD* colourspace :math:`J'a'b'` array.

    Returns
    -------
    ndarray
        *CIECAM02* correlates array :math:`JMh`.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``Jpapbp`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |            |                        |                    |
    |            | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |            |                        |                    |
    |            | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``JMh``    | ``J`` : [0, 100]       | ``J`` : [0, 1]     |
    |            |                        |                    |
    |            | ``M`` : [0, 100]       | ``M`` : [0, 1]     |
    |            |                        |                    |
    |            | ``h`` : [0, 360]       | ``h`` : [0, 1]     |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> CAM02LCD_to_JMh_CIECAM02(Jpapbp)  # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    return UCS_Luo2006_to_JMh_CIECAM02(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])


def JMh_CIECAM02_to_CAM02SCD(JMh):
    """
    Converts from *CIECAM02* :math:`JMh` correlates array to
    *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    JMh : array_like
        *CIECAM02* correlates array :math:`JMh`.

    Returns
    -------
    ndarray
        *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'` array.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``JMh``    | ``J`` : [0, 100]       | ``J`` : [0, 1]     |
    |            |                        |                    |
    |            | ``M`` : [0, 100]       | ``M`` : [0, 1]     |
    |            |                        |                    |
    |            | ``h`` : [0, 360]       | ``h`` : [0, 1]     |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``Jpapbp`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |            |                        |                    |
    |            | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |            |                        |                    |
    |            | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> from colour.appearance import (
    ...     CIECAM02_VIEWING_CONDITIONS,
    ...     XYZ_to_CIECAM02)
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = CIECAM02_VIEWING_CONDITIONS['Average']
    >>> specification = XYZ_to_CIECAM02(
    ...     XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> JMh_CIECAM02_to_CAM02SCD(JMh)  # doctest: +ELLIPSIS
    array([ 54.9043313...,  -0.0843617...,  -0.0684329...])
    """

    return JMh_CIECAM02_to_UCS_Luo2006(
        JMh, coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-SCD'])


def CAM02SCD_to_JMh_CIECAM02(Jpapbp):
    """
    Converts from *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'`
    array to *CIECAM02* :math:`JMh` correlates array.

    Parameters
    ----------
    Jpapbp : array_like
        *Luo et al. (2006)* *CAM02-SCD* colourspace :math:`J'a'b'` array.

    Returns
    -------
    ndarray
        *CIECAM02* correlates array :math:`JMh`.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``Jpapbp`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |            |                        |                    |
    |            | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |            |                        |                    |
    |            | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``JMh``    | ``J`` : [0, 100]       | ``J`` : [0, 1]     |
    |            |                        |                    |
    |            | ``M`` : [0, 100]       | ``M`` : [0, 1]     |
    |            |                        |                    |
    |            | ``h`` : [0, 360]       | ``h`` : [0, 1]     |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([54.90433134, -0.08436178, -0.06843298])
    >>> CAM02SCD_to_JMh_CIECAM02(Jpapbp)  # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    return UCS_Luo2006_to_JMh_CIECAM02(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-SCD'])


def JMh_CIECAM02_to_CAM02UCS(JMh):
    """
    Converts from *CIECAM02* :math:`JMh` correlates array to
    *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'` array.

    Parameters
    ----------
    JMh : array_like
        *CIECAM02* correlates array :math:`JMh`.

    Returns
    -------
    ndarray
        *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'` array.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``JMh``    | ``J`` : [0, 100]       | ``J`` : [0, 1]     |
    |            |                        |                    |
    |            | ``M`` : [0, 100]       | ``M`` : [0, 1]     |
    |            |                        |                    |
    |            | ``h`` : [0, 360]       | ``h`` : [0, 1]     |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``Jpapbp`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |            |                        |                    |
    |            | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |            |                        |                    |
    |            | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> from colour.appearance import (
    ...     CIECAM02_VIEWING_CONDITIONS,
    ...     XYZ_to_CIECAM02)
    >>> XYZ = np.array([19.01, 20.00, 21.78])
    >>> XYZ_w = np.array([95.05, 100.00, 108.88])
    >>> L_A = 318.31
    >>> Y_b = 20.0
    >>> surround = CIECAM02_VIEWING_CONDITIONS['Average']
    >>> specification = XYZ_to_CIECAM02(
    ...     XYZ, XYZ_w, L_A, Y_b, surround)
    >>> JMh = (specification.J, specification.M, specification.h)
    >>> JMh_CIECAM02_to_CAM02UCS(JMh)  # doctest: +ELLIPSIS
    array([ 54.9043313...,  -0.0844236...,  -0.0684831...])
    """

    return JMh_CIECAM02_to_UCS_Luo2006(
        JMh, coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-UCS'])


def CAM02UCS_to_JMh_CIECAM02(Jpapbp):
    """
    Converts from *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'`
    array to *CIECAM02* :math:`JMh` correlates array.

    Parameters
    ----------
    Jpapbp : array_like
        *Luo et al. (2006)* *CAM02-UCS* colourspace :math:`J'a'b'` array.

    Returns
    -------
    ndarray
        *CIECAM02* correlates array :math:`JMh`.

    Notes
    -----

    +------------+------------------------+--------------------+
    | **Domain** |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``Jpapbp`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |            |                        |                    |
    |            | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |            |                        |                    |
    |            | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +------------+------------------------+--------------------+

    +------------+------------------------+--------------------+
    | **Range**  |  **Scale - Reference** | **Scale - 1**      |
    +============+========================+====================+
    | ``JMh``    | ``J`` : [0, 100]       | ``J`` : [0, 1]     |
    |            |                        |                    |
    |            | ``M`` : [0, 100]       | ``M`` : [0, 1]     |
    |            |                        |                    |
    |            | ``h`` : [0, 360]       | ``h`` : [0, 1]     |
    +------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp = np.array([54.90433134, -0.08442362, -0.06848314])
    >>> CAM02UCS_to_JMh_CIECAM02(Jpapbp)  # doctest: +ELLIPSIS
    array([  4.1731091...e+01,   1.0884217...e-01,   2.1904843...e+02])
    """

    return UCS_Luo2006_to_JMh_CIECAM02(
        Jpapbp, coefficients=COEFFICIENTS_UCS_LUO2006['CAM02-UCS'])
