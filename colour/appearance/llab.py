from __future__ import division

from collections import namedtuple
import logging

import numpy as np

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ['XYZ_to_LLAB']

logger = logging.getLogger(__name__)

LLAB_result = namedtuple("LLAB_result", ("h_L", "Ch_L", "s_L", "L_L", "A_L", "B_L"))


def XYZ_to_LLAB(x, y, z, x_0, y_0, z_0, y_b, f_s, f_l, f_c, l, d=1):
    """
    Compute the LLAB model color appearance correlates.

    Parameters
    ----------
    x, y, z : numeric or array_like
        CIE XYZ values of test sample in domain [0, 100].
    x_0, y_0, z_0 : numeric or array_like
        CIE XYZ values of reference white in domain [0, 100].
    y_b : numeric or array_like
        Luminance factor of the background in cd/m^2.
    f_s : numeric or array_like
        Surround induction factor.
    f_l : numeric or array_like
        Lightness induction factor.
    f_c : numeric or array_like
        Chroma induction factor.
    l : numeric or array_like
        Absolute luminance of reference white in cd/m^2.
    d : numeric or array_like
         Discounting-the-Illuminant factor .

    Returns
    -------
    LLAB_result

    References
    ----------
    .. [1] Fairchild, M. D. (2013). *Color appearance models*, 3rd Ed. John Wiley & Sons.
    .. [2] Luo, M. R., & Morovic, J. (1996, September). Two unsolved issues in colour management-colour appearance and
           gamut mapping. In *5th International Conference on High Technology* (pp. 136-147).
    .. [3] Luo, M. R., Lo, M. C., & Kuo, W. G. (1996). The LLAB (l: c) colour model.
           *Color Research & Application*, 21(6), 412-429.

    """
    xyz = np.array([x, y, z])
    logger.debug('XYZ: {}'.format([x, y, z]))
    xyz_0 = np.array([x_0, y_0, z_0])

    r, g, b = xyz_to_rgb(xyz)
    logger.debug('RGB: {}'.format([r, g, b]))
    r_0, g_0, b_0 = xyz_to_rgb(xyz_0)
    logger.debug('RGB_0: {}'.format([r_0, g_0, b_0]))

    xyz_0r = np.array([95.05, 100, 108.88])
    r_0r, g_0r, b_0r = xyz_to_rgb(xyz_0r)
    logger.debug('RGB_0r: {}'.format([r_0r, g_0r, b_0r]))

    beta = (b_0 / b_0r) ** 0.0834
    logger.debug('beta: {}'.format(beta))
    r_r = (d * (r_0r / r_0) + 1 - d) * r
    g_r = (d * (g_0r / g_0) + 1 - d) * g
    b_r = (d * (b_0r / (b_0 ** beta)) + 1 - d) * (abs(b) ** beta)
    logger.debug('RGB_r: {}'.format([r_r, g_r, b_r]))

    rgb_r = np.array([r_r, g_r, b_r])

    # m_inv = np.linalg.inv(xyz_to_rgb_m)
    m_inv = np.array([[0.987, -0.1471, 0.16],
                      [0.4323, 0.5184, 0.0493],
                      [-0.0085, 0.04, 0.9685]])
    x_r, y_r, z_r = m_inv.dot(rgb_r * y)
    logger.debug('XYZ_r: {}'.format([x_r, y_r, z_r]))

    # Opponent Color Dimension
    def f(w):
        return np.where(w > 0.008856,
                        w ** (1 / f_s),
                        (((0.008856 ** (1 / f_s)) - (16 / 116)) / 0.008856) * w + (16 / 116))

    # lightness_contrast_exponent
    z = 1 + f_l * ((y_b / 100) ** 0.5)
    logger.debug('z: {}'.format(z))

    lightness = 116 * (f(y_r / 100) ** z) - 16
    a = 500 * (f(x_r / 95.05) - f(y_r / 100))
    b = 200 * (f(y_r / 100) - f(z_r / 108.88))
    logger.debug('A: {}'.format(a))
    logger.debug('B: {}'.format(b))

    logger.debug('f(Xr): {}'.format(f(x_r / 95.05)))
    logger.debug('f(Yr): {}'.format(f(y_r / 100)))
    logger.debug('f(Zr): {}'.format(f(z_r / 108.88)))

    # Perceptual Correlates
    c = (a ** 2 + b ** 2) ** 0.5
    chroma = 25 * np.log(1 + 0.05 * c)

    s_c = 1 + 0.47 * np.log10(l) - 0.057 * np.log10(l) ** 2
    s_m = 0.7 + 0.02 * lightness - 0.0002 * lightness ** 2
    c_l = chroma * s_m * s_c * f_c

    saturation = chroma / lightness

    hue_angle_rad = np.arctan2(b, a)
    hue_angle = hue_angle_rad * 360 / (2 * np.pi) % 360

    a_l = c_l * np.cos(hue_angle_rad)
    b_l = c_l * np.sin(hue_angle_rad)

    return LLAB_result(hue_angle, chroma, saturation, lightness, a_l, b_l)


xyz_to_rgb_m = np.array([[0.8951, 0.2664, -0.1614],
                         [-0.7502, 1.7135, 0.0367],
                         [0.0389, -0.0685, 1.0296]])


def xyz_to_rgb(xyz):
    return xyz_to_rgb_m.dot(xyz / xyz[1])

