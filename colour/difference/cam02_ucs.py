# -*- coding: utf-8 -*-
"""
:math:`\\Delta E'` - Delta E Colour Difference - Luo, Cui and Li (2006)
======================================================================

Defines :math:`\\Delta E'` colour difference computation objects based on
*Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, and *CAM02-UCS* colourspaces:

-   :func:`colour.difference.delta_E_CAM02LCD`
-   :func:`colour.difference.delta_E_CAM02SCD`
-   :func:`colour.difference.delta_E_CAM02UCS`

See Also
--------
`CAM02-LCD, CAM02-SCD, and CAM02-UCS Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/ucs_luo2006.ipynb>`_

References
----------
-   :cite:`Luo2006b` : Luo, M. R., Cui, G., & Li, C. (2006). Uniform colour
    spaces based on CIECAM02 colour appearance model. Color Research &
    Application, 31(4), 320-330. doi:10.1002/col.20227
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import tsplit
from colour.models.cam02_ucs import COEFFICIENTS_UCS_LUO2006

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'delta_E_Luo2006', 'delta_E_CAM02LCD', 'delta_E_CAM02SCD',
    'delta_E_CAM02UCS'
]


def delta_E_Luo2006(Jpapbp_1, Jpapbp_2, coefficients):
    """
    Returns the difference :math:`\\Delta E'` between two given
    *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS* colourspaces
    :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1 : array_like
        Standard / reference *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or
        *CAM02-UCS* colourspaces :math:`J'a'b'` array.
    Jpapbp_2 : array_like
        Sample / test *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or
        *CAM02-UCS* colourspaces :math:`J'a'b'` array.
    coefficients : array_like
        Coefficients of one of the *Luo et al. (2006)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E'`.

    Notes
    -----

    +--------------+------------------------+--------------------+
    | **Domain**   |  **Scale - Reference** | **Scale - 1**      |
    +==============+========================+====================+
    | ``Jpapbp_1`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |              |                        |                    |
    |              | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |              |                        |                    |
    |              | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +--------------+------------------------+--------------------+
    | ``Jpapbp_2`` | ``Jp_2`` : [0, 100]    | ``Jp_2`` : [0, 1]  |
    |              |                        |                    |
    |              | ``ap_2`` : [-100, 100] | ``ap_2`` : [-1, 1] |
    |              |                        |                    |
    |              | ``bp_2`` : [-100, 100] | ``bp_2`` : [-1, 1] |
    +--------------+------------------------+--------------------+

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
    >>> delta_E_Luo2006(Jpapbp_1, Jpapbp_2,
    ...                 COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])
    ... # doctest: +ELLIPSIS
    0.0001034...
    """

    J_p_1, a_p_1, b_p_1 = tsplit(Jpapbp_1)
    J_p_2, a_p_2, b_p_2 = tsplit(Jpapbp_2)
    K_L, _c_1, _c_2 = tsplit(coefficients)

    d_E = np.sqrt(((J_p_1 - J_p_2) / K_L) ** 2 + (a_p_1 - a_p_2) ** 2 +
                  (b_p_1 - b_p_2) ** 2)
    return d_E


def delta_E_CAM02LCD(Jpapbp_1, Jpapbp_2):
    """
    Returns the difference :math:`\\Delta E'` between two given
    *Luo et al. (2006)* *CAM02-LCD* colourspaces :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1 : array_like
        Standard / reference *Luo et al. (2006)* *CAM02-LCD* colourspaces
        :math:`J'a'b'` array.
    Jpapbp_2 : array_like
        Sample / test *Luo et al. (2006)* *CAM02-LCD* colourspaces
        :math:`J'a'b'` array.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E'`.

    Notes
    -----

    +--------------+------------------------+--------------------+
    | **Domain**   |  **Scale - Reference** | **Scale - 1**      |
    +==============+========================+====================+
    | ``Jpapbp_1`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |              |                        |                    |
    |              | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |              |                        |                    |
    |              | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +--------------+------------------------+--------------------+
    | ``Jpapbp_2`` | ``Jp_2`` : [0, 100]    | ``Jp_2`` : [0, 1]  |
    |              |                        |                    |
    |              | ``ap_2`` : [-100, 100] | ``ap_2`` : [-1, 1] |
    |              |                        |                    |
    |              | ``bp_2`` : [-100, 100] | ``bp_2`` : [-1, 1] |
    +--------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
    >>> delta_E_CAM02LCD(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    14.0555464...
    """
    return delta_E_Luo2006(Jpapbp_1, Jpapbp_2,
                           COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])


def delta_E_CAM02SCD(Jpapbp_1, Jpapbp_2):
    """
    Returns the difference :math:`\\Delta E'` between two given
    *Luo et al. (2006)* *CAM02-SCD* colourspaces :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1 : array_like
        Standard / reference *Luo et al. (2006)* *CAM02-SCD* colourspaces
        :math:`J'a'b'` array.
    Jpapbp_2 : array_like
        Sample / test *Luo et al. (2006)* *CAM02-SCD* colourspaces
        :math:`J'a'b'` array.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E'`.

    Notes
    -----

    +--------------+------------------------+--------------------+
    | **Domain**   |  **Scale - Reference** | **Scale - 1**      |
    +==============+========================+====================+
    | ``Jpapbp_1`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |              |                        |                    |
    |              | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |              |                        |                    |
    |              | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +--------------+------------------------+--------------------+
    | ``Jpapbp_2`` | ``Jp_2`` : [0, 100]    | ``Jp_2`` : [0, 1]  |
    |              |                        |                    |
    |              | ``ap_2`` : [-100, 100] | ``ap_2`` : [-1, 1] |
    |              |                        |                    |
    |              | ``bp_2`` : [-100, 100] | ``bp_2`` : [-1, 1] |
    +--------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
    >>> delta_E_CAM02SCD(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    14.0551718...
    """
    return delta_E_Luo2006(Jpapbp_1, Jpapbp_2,
                           COEFFICIENTS_UCS_LUO2006['CAM02-SCD'])


def delta_E_CAM02UCS(Jpapbp_1, Jpapbp_2):
    """
    Returns the difference :math:`\\Delta E'` between two given
    *Luo et al. (2006)* *CAM02-UCS* colourspaces :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1 : array_like
        Standard / reference *Luo et al. (2006)* *CAM02-UCS* colourspaces
        :math:`J'a'b'` array.
    Jpapbp_2 : array_like
        Sample / test *Luo et al. (2006)* *CAM02-UCS* colourspaces
        :math:`J'a'b'` array.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\\Delta E'`.

    Notes
    -----

    +--------------+------------------------+--------------------+
    | **Domain**   |  **Scale - Reference** | **Scale - 1**      |
    +==============+========================+====================+
    | ``Jpapbp_1`` | ``Jp_1`` : [0, 100]    | ``Jp_1`` : [0, 1]  |
    |              |                        |                    |
    |              | ``ap_1`` : [-100, 100] | ``ap_1`` : [-1, 1] |
    |              |                        |                    |
    |              | ``bp_1`` : [-100, 100] | ``bp_1`` : [-1, 1] |
    +--------------+------------------------+--------------------+
    | ``Jpapbp_2`` | ``Jp_2`` : [0, 100]    | ``Jp_2`` : [0, 1]  |
    |              |                        |                    |
    |              | ``ap_2`` : [-100, 100] | ``ap_2`` : [-1, 1] |
    |              |                        |                    |
    |              | ``bp_2`` : [-100, 100] | ``bp_2`` : [-1, 1] |
    +--------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
    >>> delta_E_CAM02UCS(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    14.0552982...
    """
    return delta_E_Luo2006(Jpapbp_1, Jpapbp_2,
                           COEFFICIENTS_UCS_LUO2006['CAM02-UCS'])
