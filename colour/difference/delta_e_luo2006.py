#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:math:`\Delta E'` - Delta E Colour Difference - Luo, Cui and Li (2006)
======================================================================

Defines :math:`\Delta E'` colour difference computation objects based on
*Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, and *CAM02-UCS* colourspaces:

The following objects are available:

-   :func:`delta_E_CAM02LCD`
-   :func:`delta_E_CAM02SCD`
-   :func:`delta_E_CAM02UCS`

See Also
--------
`CAM02-LCD, CAM02-SCD, and CAM02-UCS Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/ucs_luo2006.ipynb>`_

References
----------
.. [1]  Luo, R. M., Cui, G., & Li, C. (2006). Uniform Colour Spaces Based on
        CIECAM02 Colour Appearance Model. Color Research and Application,
        31(4), 320–330. doi:10.1002/col.20227
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import tsplit
from colour.models.ucs_luo2006 import COEFFICIENTS_UCS_LUO2006

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['delta_E_Luo2006',
           'delta_E_CAM02LCD',
           'delta_E_CAM02SCD',
           'delta_E_CAM02UCS']


def delta_E_Luo2006(Jpapbp_1, Jpapbp_2, coefficients):
    """
    Returns the difference :math:`\Delta E'` between two given
    *Luo et al. (2016)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS* colourspaces
    :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1 : array_like
        Standard / reference *Luo et al.* (2016) *CAM02-LCD*, *CAM02-SCD*, or
        *CAM02-UCS* colourspaces :math:`J'a'b'` array.
    Jpapbp_2 : array_like
        Sample / test *Luo et al. (2016)* *CAM02-LCD*, *CAM02-SCD*, or
        *CAM02-UCS* colourspaces :math:`J'a'b'` array.
    coefficients : array_like
        Coefficients of one of the *Luo et al. (2016)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\Delta E'`.

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.90433134, -0.08442362, -0.06848314])
    >>> delta_E_Luo2006(  # doctest: +ELLIPSIS
    ...     Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])
    0.0001034...
    """

    J_p_1, a_p_1, b_p_1 = tsplit(Jpapbp_1)
    J_p_2, a_p_2, b_p_2 = tsplit(Jpapbp_2)
    K_L, c_1_, c_2_ = tsplit(coefficients)

    d_E = np.sqrt(((J_p_1 - J_p_2) / K_L) ** 2 +
                  (a_p_1 - a_p_2) ** 2 +
                  (b_p_1 - b_p_2) ** 2)
    return d_E


def delta_E_CAM02LCD(Jpapbp_1, Jpapbp_2):
    """
    Returns the difference :math:`\Delta E'` between two given
    *Luo et al. (2016)* *CAM02-LCD* colourspaces :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1 : array_like
        Standard / reference *Luo et al.* (2016) *CAM02-LCD* colourspaces
        :math:`J'a'b'` array.
    Jpapbp_2 : array_like
        Sample / test *Luo et al.* (2016) *CAM02-LCD* colourspaces
        :math:`J'a'b'` array.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\Delta E'`.

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.90433134, -0.08442362, -0.06848314])
    >>> delta_E_CAM02LCD(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    0.0001034...
    """
    return delta_E_Luo2006(
        Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006['CAM02-LCD'])


def delta_E_CAM02SCD(Jpapbp_1, Jpapbp_2):
    """
    Returns the difference :math:`\Delta E'` between two given
    *Luo et al. (2016)* *CAM02-SCD* colourspaces :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1 : array_like
        Standard / reference *Luo et al.* (2016) *CAM02-SCD* colourspaces
        :math:`J'a'b'` array.
    Jpapbp_2 : array_like
        Sample / test *Luo et al.* (2016) *CAM02-SCD* colourspaces
        :math:`J'a'b'` array.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\Delta E'`.

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.90433134, -0.08442362, -0.06848314])
    >>> delta_E_CAM02SCD(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    0.0001034...
    """
    return delta_E_Luo2006(
        Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006['CAM02-SCD'])


def delta_E_CAM02UCS(Jpapbp_1, Jpapbp_2):
    """
    Returns the difference :math:`\Delta E'` between two given
    *Luo et al. (2016)* *CAM02-UCS* colourspaces :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1 : array_like
        Standard / reference *Luo et al.* (2016) *CAM02-UCS* colourspaces
        :math:`J'a'b'` array.
    Jpapbp_2 : array_like
        Sample / test *Luo et al.* (2016) *CAM02-UCS* colourspaces
        :math:`J'a'b'` array.

    Returns
    -------
    numeric or ndarray
        Colour difference :math:`\Delta E'`.

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.90433134, -0.08442362, -0.06848314])
    >>> delta_E_CAM02UCS(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    0.0001034...
    """
    return delta_E_Luo2006(
        Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006['CAM02-UCS'])
