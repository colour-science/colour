from __future__ import division

import logging
from collections import namedtuple

import numpy as np


__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_ATD95']

logger = logging.getLogger(__name__)

ATD95_result = namedtuple('ATD95_result', ('H', 'Br', 'C', 'a_1', 't_1', 'd_1', 'a_2', 't_2', 'd_2'))


def XYZ_to_ATD95(x, y, z, x_0, y_0, z_0, y_0_abs, k_1, k_2, sigma=300):
    """
    Compute the ATD95 model color appearance correlates.

    Parameters
    ----------
    x, y, z : numeric or array_like
        CIE XYZ values of test sample in domain [0, 100].
    x_0, y_0, z_0 : numeric or array_like
        CIE XYZ values of reference white in domain [0, 100].
    y_0_abs : numeric or array_like
        Absolute adapting luminance in cd/m^2.
    k_1, k2 : numeric or array_like
    sigma : numeric or array_like

    Returns
    -------
    ATD95_result

    References
    ----------
    .. [1] Fairchild, M. D. (2013). *Color appearance models*, 3rd Ed. John Wiley & Sons.
    .. [2] Guth, S. L. (1995, April). Further applications of the ATD model for color vision. In *IS&T/SPIE's Symposium
           on Electronic Imaging: Science & Technology* (pp. 12-26). International Society for Optics and Photonics.
    """
    xyz = scale_to_luminance(np.array([x, y, z]), y_0_abs)
    xyz_0 = scale_to_luminance(np.array([x_0, y_0, z_0]), y_0_abs)
    logger.debug('Scaled XYZ: {}'.format(xyz))
    logger.debug('Scaled XYZ_0: {}'.format(xyz))

    # Adaptation Model
    lms = xyz_to_lms(xyz)
    logger.debug('LMS: {}'.format(lms))

    xyz_a = k_1 * xyz + k_2 * xyz_0
    logger.debug('XYZ_a: {}'.format(xyz_a))

    lms_a = xyz_to_lms(xyz_a)
    logger.debug('LMS_a: {}'.format(lms_a))

    l_g, m_g, s_g = lms * (sigma / (sigma + lms_a))

    # Opponent Color Dimensions
    a_1i = 3.57 * l_g + 2.64 * m_g
    t_1i = 7.18 * l_g - 6.21 * m_g
    d_1i = -0.7 * l_g + 0.085 * m_g + s_g
    a_2i = 0.09 * a_1i
    t_2i = 0.43 * t_1i + 0.76 * d_1i
    d_2i = d_1i

    a_1 = calculate_final_response(a_1i)
    t_1 = calculate_final_response(t_1i)
    d_1 = calculate_final_response(d_1i)
    a_2 = calculate_final_response(a_2i)
    t_2 = calculate_final_response(t_2i)
    d_2 = calculate_final_response(d_2i)

    # Perceptual Correlates
    brightness = (a_1 ** 2 + t_1 ** 2 + d_1 ** 2) ** 0.5
    saturation = (t_2 ** 2 + d_2 ** 2) ** 0.5 / a_2
    hue = t_2 / d_2

    return ATD95_result(hue, brightness, saturation, a_1, t_1, d_1, a_2, t_2, d_2)


def calculate_final_response(value):
    return value / (200 + abs(value))


def scale_to_luminance(xyz, absolute_adapting_luminance):
    return 18 * (absolute_adapting_luminance * xyz / 100) ** 0.8


def xyz_to_lms(xyz):
    x, y, z = xyz
    l = ((0.66 * (0.2435 * x + 0.8524 * y - 0.0516 * z)) ** 0.7) + 0.024
    m = ((-0.3954 * x + 1.1642 * y + 0.0837 * z) ** 0.7) + 0.036
    s = ((0.43 * (0.04 * y + 0.6225 * z)) ** 0.7) + 0.31
    return np.array([l, m, s])
