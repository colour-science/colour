# -*- coding: utf-8 -*-
"""
Helmholtz-Kohlrausch effect
===========================

Defines the following methods for estimating helmholtz-Kohlrausch effect:

-   :func:`colour.hke_object_Nayatani1997`:
    *Nayatani (1997)* Helmholtz-Kohlrausch effect estimation for object
    colours.
-   :func:`colour.hke_luminous_VAC_Nayatani1997`:
    *Nayatani (1997)* Helmholtz-Kohlrausch effect estimation for luminous
    colours.
-   :func:`colour.colorimetry.coefficient_q_Nayatani1997`:
    Calculates :math:`WI` coefficient for *Nayatani 1997* HKE estimation.
-   :func:`colour.colorimetry.coefficient_K_B_r_Nayatani1997`:
    Calculates :math:`K_B_r` coefficient for *Nayatani 1997* HKE estimation.
-   :attr:`colour.HKE_NAYATANI1997_METHODS`: Nayatani HKE computation methods,
    choice between variable achromatic colour ('VAC') and variable chromatic
    colour ('VCC').

References
-   :cite:`nayatani1997` : Nayatani, Y. (1997). Simple Estimation Methods for
    the Helmholtz-Kohlrausch Effect. Color Research and Application, 22(6),
    385–401. doi:10.1002/(SICI)1520-6378(199712)22:6<385::AID-COL6>3.0.CO;2-R
"""

import numpy as np

from colour.utilities import (CaseInsensitiveMapping, tsplit, as_float_array)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'hke_object_Nayatani1997', 'hke_luminous_Nayatani1997',
    'coefficient_q_Nayatani1997', 'coefficient_K_B_r_Nayatani1997',
    'HKE_METHODS_NAYATANI'
]

HKE_NAYATANI1997_METHODS = CaseInsensitiveMapping({
    'VAC': -0.1340,
    'VCC': -0.8660,
})
HKE_NAYATANI1997_METHODS.__doc__ = """
Nayatani HKE computation methods, choice between variable achromatic colour
('VAC') and variable chromatic colour ('VCC')

References
----------
:cite:`nayatani1997`

HKE_NAYATANI1997_METHODS : CaseInsensitiveMapping
    **{'VAC', 'VCC'}**
"""

def hke_object_Nayatani1997(uv, uv_c, L_a, method='VCC'):
    """
    Returns the HKE value for object colours using *Nayatani (1997)* method.

    Parameters
    ----------
    uv : array_like
        *CIE uv* chromaticity coordinates of samples.
    uv_c : array_like
        *CIE uv* chromaticity coordinates of reference white.
    L_a : numeric
        Adapting luminance in :math:`cd/m^2`.
    method: unicode, optional
        **{'VAC', 'VCC'}**
        Which estimation method to use, VCC or VAC.

    Returns
    -------
    numeric or ndarray
        Luminance factor (:math:`Gamma`) value(s) computed through Nayatani
        object colour estimation method.

    References
    ----------
    :cite:`nayatani1997`

    Examples
    --------
    >>> import colour
    >>> from colour import *
    >>> white = xy_to_Luv_uv(colour.temperature.CCT_to_xy_CIE_D(6504))
    >>> colours = XYZ_to_xy([wavelength_to_XYZ(430+i*50) for i in range(5)])
    >>> L_adapting = 65
    >>> hke_object_Nayatani1997(xy_to_Luv_uv(colours), white, L_adapting)
    array([ 2.24683833,  1.46197995,  1.18016586,  0.90313188,  1.79993762])
    """

    u, v = tsplit(uv)
    u_c, v_c = tsplit(uv_c)

    K_B_r = coefficient_K_B_r_Nayatani1997(L_a)
    q = coefficient_q_Nayatani1997(np.arctan2(v-v_c, u-u_c))
    s = np.sqrt((u-u_c) ** 2 + (v-v_c) ** 2) * 13

    return 1 + (HKE_NAYATANI1997_METHODS[method] * q + 0.0872 * K_B_r) * s

def hke_luminous_Nayatani1997(uv, uv_c, L_a, method='VCC'):
    """
    Returns the HKE factor for luminous colours using *Nayatani (1997)* method.

    Parameters
    ----------
    uv : array_like
        *CIE uv* chromaticity coordinates of samples.
    uv_c : array_like
        *CIE uv* chromaticity coordinates of reference white.
    L_a : numeric or array_like
        Adapting luminance in :math:`cd/m^2`.
    method: unicode, optional
        **{'VAC', 'VCC'}**
        Which estimation method to use, VCC or VAC.

    Returns
    -------
    numeric or ndarray
        Luminance factor (:math:`gamma`) value(s) computed through Nayatani
        luminous colour estimation method.

    References
    ----------
    :cite:`nayatani1997`

    Examples
    --------
    >>> import colour
    >>> from colour import *
    >>> w = xy_to_Luv_uv(colour.temperature.CCT_to_xy_CIE_D(6504))
    >>> colours = XYZ_to_xy([wavelength_to_XYZ(430+i*50) for i in range(5)])
    >>> L_adapting = 65
    >>> hke_luminous_Nayatani1997(xy_to_Luv_uv(colours), white, L_adapting)
    array([ 7.44604715,  2.4767159 ,  1.47234223,  0.79386959,  4.1828629 ])
    """

    return (0.4462 * (hke_object_Nayatani1997(uv, uv_c, L_a, method) + 0.3086)
            ** 3)

def coefficient_q_Nayatani1997(theta):
    """
    Returns :math:`q(\theta)` coefficient for *Nayatani (1997)* HKE computations.

    Parameters
    ----------
    theta : numeric or array_like
        Hue angle (:math:`\theta`) in radians.

    Returns
    -------
    numeric or ndarray
        :math:`q` coefficient for *Nayatani (1997)* HKE methods.

    References
    ----------
    :cite:`nayatani1997`

    Examples
    --------
    This recreates FIG. A-1
    >>> import colour
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> angles = [(np.pi*2/100 * i) for i in range(100)]
    >>> q_values = colour.colorimetry.coefficient_q_Nayatani1997(angles)
    >>> plt.plot(np.array(angles), q_values/(np.pi*2)*180)
    >>> plt.show()
    """
    theta = as_float_array(theta)

    return (-0.01585
           - 0.03017 * np.cos(theta)     - 0.04556 * np.cos(2 * theta)
           - 0.02667 * np.cos(3 * theta) - 0.00295 * np.cos(4 * theta)
           + 0.14592 * np.sin(theta)     + 0.05084 * np.sin(2 * theta)
           - 0.01900 * np.sin(3 * theta) - 0.00764 * np.sin(4 * theta))

def coefficient_K_B_r_Nayatani1997(L_a):
    """
    Returns K_B_r coefficient for *Nayatani (1997)* HKE computations.

    Parameters
    ----------
    L_a : numeric or array_like
        Adapting luminance in :math:`cd/m^2`.

    Returns
    -------
    numeric or ndarray
        :math:`K_B_r` coefficient for *Nayatani (1997)* HKE methods.

    References
    ----------
    :cite:`nayatani1997`

    Examples
    --------
    >>> import colour
    >>> L_a_values = [10+i*20 for i in range(5)]
    >>> colour.colorimetry.coefficient_K_B_r_Nayatani1997(L_a_values)
    array([ 0.71344818,  0.87811728,  0.96062482,  1.01566892,  1.05670084])
    >>> colour.colorimetry.coefficient_K_B_r_Nayatani1997(63.66)
    1.0001284555840311
    """
    L_a_4495 = np.power(L_a, 0.4495)
    return 0.2717 * (6.469 + 6.362 * L_a_4495) / (6.469 + L_a_4495)