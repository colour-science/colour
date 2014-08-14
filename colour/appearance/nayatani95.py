from __future__ import division

from collections import namedtuple
import logging

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'GPL V3.0 - http://www.gnu.org/licenses/'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_Nayatani95']

logger = logging.getLogger(__name__)

Nayatani95_result = namedtuple('Nayatani95_result', ('B_r', 'L_star_P', 'L_star_N', 'theta', 'S', 'C', 'M'))


def XYZ_to_Nayatani95(x, y, z,
                      x_n, y_n, z_n,
                      y_ob,
                      e_o, e_or,
                      n=1):
    """
    Compute the Nayatani95 model color appearance correlates.

    Parameters
    ----------
    x, y, z : numeric or array_like
        CIE XYZ values of test sample in domain [0, 100].
    x_n, y_n, z_n : numeric or array_like
        CIE XYZ values of reference white in domain [0, 100].
    y_ob : numeric or array_like
        Luminance factor of achromatic background as percentage. Required to be larger than 0.18.
    e_o : numeric or array_like
        Illuminance of the viewing field in lux.
    e_or : numeric or array_like
        Normalising illuminance in lux.
    n : numeric or array_like, optional
        Noise term.

    Returns
    -------
    Nayatani95_result

    Raises
    ------
    ValueError
        If y_ob <= 0.18

    References
    ----------
    .. [1] Fairchild, M. D. (2013). *Color appearance models*, 3rd Ed. John Wiley & Sons.
    .. [2] Nayatani, Y., Sobagaki, H., & Yano, K. H. T. (1995). Lightness dependency of chroma scales of a nonlinear
           color-appearance model and its latest formulation. *Color Research & Application*, 20(3), 156-167.

    """

    if np.any(y_ob <= 0.18):
        raise ValueError('y_ob hast be greater than 0.18.')

    l_o = y_ob * e_o / (100 * np.pi)
    l_or = y_ob * e_or / (100 * np.pi)
    logger.debug('L_o: {}'.format(l_o))
    logger.debug('L_or: {}'.format(l_or))

    x_o = x_n / (x_n + y_n + z_n)
    y_o = y_n / (x_n + y_n + z_n)
    logger.debug('x_o: {}'.format(x_o))
    logger.debug('y_o: {}'.format(y_o))

    xi = (0.48105 * x_o + 0.78841 * y_o - 0.08081) / y_o
    eta = (-0.27200 * x_o + 1.11962 * y_o + 0.04570) / y_o
    zeta = (0.91822 * (1 - x_o - y_o)) / y_o
    logger.debug('xi: {}'.format(xi))
    logger.debug('eta: {}'.format(eta))
    logger.debug('zeta: {}'.format(zeta))

    r_0, g_0, b_0 = rgb_0 = ((y_ob * e_o) / (100 * np.pi)) * np.array([xi, eta, zeta])
    logger.debug('rgb_0: {}'.format(rgb_0))

    r, g, b, = rgb = xyz_to_rgb(np.array([x, y, z]))
    logger.debug('rgb: {}'.format(rgb))

    e_r = get_scaling_coefficient(r, xi)
    logger.debug('e(R): {}'.format(e_r))
    e_g = get_scaling_coefficient(g, eta)
    logger.debug('e(G): {}'.format(e_g))

    beta_r = beta_1(r_0)
    logger.debug('beta1(rho): {}'.format(beta_r))
    beta_g = beta_1(g_0)
    logger.debug('beta1(eta): {}'.format(beta_g))
    beta_b = beta_2(b_0)
    logger.debug('beta2(zeta): {}'.format(beta_b))

    beta_l = beta_1(l_or)
    logger.debug('beta1(L_or): {}'.format(beta_l))

    # Opponent Color Dimension
    achromatic_response = (2 / 3) * beta_r * e_r * np.log10((r + n) / (20 * xi + n))
    achromatic_response += (1 / 3) * beta_g * e_g * np.log10((g + n) / (20 * eta + n))
    achromatic_response *= 41.69 / beta_l
    logger.debug('Q: {}'.format(achromatic_response))

    tritanopic_response = (1 / 1) * beta_r * np.log10((r + n) / (20 * xi + n))
    tritanopic_response += - (12 / 11) * beta_g * np.log10((g + n) / (20 * eta + n))
    tritanopic_response += (1 / 11) * beta_b * np.log10((b + n) / (20 * zeta + n))
    logger.debug('t: {}'.format(tritanopic_response))

    protanopic_response = (1 / 9) * beta_r * np.log10((r + n) / (20 * xi + n))
    protanopic_response += (1 / 9) * beta_g * np.log10((g + n) / (20 * eta + n))
    protanopic_response += - (2 / 9) * beta_b * np.log10((b + n) / (20 * zeta + n))
    logger.debug('p: {}'.format(protanopic_response))

    # Brightness
    brightness = (50 / beta_l) * ((2 / 3) * beta_r + (1 / 3) * beta_g) + achromatic_response

    brightness_ideal_white = (2 / 3) * beta_r * 1.758 * np.log10((100 * xi + n) / (20 * xi + n))
    brightness_ideal_white += (1 / 3) * beta_g * 1.758 * np.log10((100 * eta + n) / (20 * eta + n))
    brightness_ideal_white *= 41.69 / beta_l
    brightness_ideal_white += (50 / beta_l) * (2 / 3) * beta_r
    brightness_ideal_white += (50 / beta_l) * (1 / 3) * beta_g

    # Lightness
    lightness_achromatic = achromatic_response + 50
    lightness_achromatic_normalized = 100 * (brightness / brightness_ideal_white)

    # Hue
    hue_angle_rad = np.arctan2(protanopic_response, tritanopic_response)
    hue_angle = ((360 * hue_angle_rad / (2 * np.pi)) + 360) % 360
    logger.debug('theta: {}'.format(hue_angle))

    e_s_theta = get_chromatic_strength(hue_angle_rad)
    logger.debug('E_s(theta): {}'.format(e_s_theta))

    # Saturation
    saturation_rg = (488.93 / beta_l) * e_s_theta * tritanopic_response
    saturation_yb = (488.93 / beta_l) * e_s_theta * protanopic_response
    logger.debug('S_RG: {}'.format(saturation_rg))
    logger.debug('S_YB: {}'.format(saturation_yb))

    saturation = np.sqrt((saturation_rg ** 2) + (saturation_yb ** 2))
    logger.debug('S: {}'.format(saturation))

    # Chroma
    chroma_rg = ((lightness_achromatic / 50) ** 0.7) * saturation_rg
    chroma_yb = ((lightness_achromatic / 50) ** 0.7) * saturation_yb
    chroma = ((lightness_achromatic / 50) ** 0.7) * saturation
    logger.debug('C: {}'.format(chroma))

    # Colorfulness
    colorfulness_rg = chroma_rg * brightness_ideal_white / 100
    colorfulness_yb = chroma_yb * brightness_ideal_white / 100
    colorfulness = chroma * brightness_ideal_white / 100

    return Nayatani95_result(brightness,
                             lightness_achromatic, lightness_achromatic_normalized,
                             hue_angle,
                             saturation,
                             chroma,
                             colorfulness)


def get_chromatic_strength(angle):
    result = 0.9394
    result += - 0.2478 * np.sin(1 * angle)
    result += - 0.0743 * np.sin(2 * angle)
    result += + 0.0666 * np.sin(3 * angle)
    result += - 0.0186 * np.sin(4 * angle)
    result += - 0.0055 * np.cos(1 * angle)
    result += - 0.0521 * np.cos(2 * angle)
    result += - 0.0573 * np.cos(3 * angle)
    result += - 0.0061 * np.cos(4 * angle)
    return result


def get_scaling_coefficient(a, b):
    return np.where(a >= (20 * b), 1.758, 1)


def beta_1(x):
    return (6.469 + 6.362 * (x ** 0.4495)) / (6.469 + (x ** 0.4495))


def beta_2(x):
    return 0.7844 * (8.414 + 8.091 * (x ** 0.5128)) / (8.414 + (x ** 0.5128))


xyz_to_rgb_m = np.array([[0.40024, 0.70760, -0.08081],
                         [-0.22630, 1.16532, 0.04570],
                         [0, 0, 0.91822]])


def xyz_to_rgb(xyz):
    return xyz_to_rgb_m.dot(xyz)
