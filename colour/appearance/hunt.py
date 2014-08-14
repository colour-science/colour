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

__all__ = ['XYZ_to_Hunt']

logger = logging.getLogger(__name__)

Hunt_result = namedtuple('Hunt_result', ('h_S', 'C94', 's', 'Q', 'M94', 'J'))


def XYZ_to_Hunt(x, y, z,
                x_b, y_b, z_b,
                x_w, y_w, z_w,
                l_a,
                n_c,
                n_b,
                l_as=None,
                cct_w=None,
                n_cb=None,
                n_bb=None,
                x_p=None,
                y_p=None,
                z_p=None,
                p=None,
                helson_judd=False,
                discount_illuminant=True,
                s=None,
                s_w=None):
    """
    Compute the Hunt model color appearance correlates.

    Parameters
    ----------
    x, y, z : numeric or array_like
        CIE XYZ values of test sample in domain [0, 100].
    x_b, y_b, z_b : numeric or array_like
        CIE XYZ values of background in domain [0, 100].
    x_w, y_w, z_w : numeric or array_like
        CIE XYZ values of reference white in domain [0, 100].
    l_a : numeric or array_like
        Adapting luminance.
    l_a : numeric or array_like
        Adapting luminance.
    n_c : numeric or array_like
         Chromatic surround induction_factor.
    n_b : numeric or array_like
         Brightness surround induction factor.
    l_as : numeric or array_like, optional
        Scotopic luminance of the illuminant. Will be approximated if not supplied.
    cct_w : numeric or array_like, optional
        Correlated color temperature of illuminant. Only needed to approximate l_as.
    n_cb : numeric or array_like, optional
        Chromatic background induction factor. Will be approximated using y_w and y_b if not supplied.
    n_bb : numeric or array_like, optional
        Brightness background induction factor. Will be approximated using y_w and y_b if not supplied.
    x_p, y_p, z_p : numeric or array_like, optional
        CIE XYZ values of proxima field in domain [0, 100]. If not supplied, will be assumed to equal background.
    p : numeric or array_like, optional
        Simultaneous contrast/assimilation parameter.
    helson_judd : boolean or array_like, optional
        Truth value indicating whether the Heslon-Judd effect should be accounted for. Default False.
    discount_illuminant : boolean or array_like, optional
        Truth value whether discount-the-illuminant should be applied. Default True.
    s : numeric or array_like, optional
        Scotopic response to the stimulus.
    s_w : numeric or array_like, optional
        Scotopic response for th reference white.

    Returns
    -------
    Hunt_result

    Raises
    ------
    ValueError
        If illegal parameter combination is supplied.

    References
    ----------
    .. [1] Fairchild, M. D. (2013). *Color appearance models*, 3rd Ed. John Wiley & Sons.
    .. [2] Hunt, R. W. G. (2005). *The reproduction of colour*. 5th Ed., John Wiley & Sons.
    """

    if x_p is None:
        x_p = x_b
        logger.warn('Approximated x_p with x_b.')
    if y_p is None:
        y_p = y_b
        logger.warn('Approximated y_p with y_b.')
    if z_p is None:
        z_p = y_b
        logger.warn('Approximated z_p with z_b.')

    if n_cb is None:
        n_cb = 0.725 * (y_w / y_b) ** 0.2
        logger.warn('Approximated n_cb.')
    logger.debug('N_cb: {}'.format(n_cb))
    if n_bb is None:
        n_bb = 0.725 * (y_w / y_b) ** 0.2
        logger.warn('Approximated n_bb.')
    logger.debug('N_bb: {}'.format(n_cb))

    if l_as is None:
        logger.warn('Approximating scotopic luminance from supplied cct.')
        l_as = 2.26 * l_a
        l_as *= ((cct_w / 4000) - 0.4) ** (1 / 3)
    logger.debug('LA_S: {}'.format(l_as))

    if s is None != s_w is None:
        raise ValueError("Either both scotopic responses (s, s_w) need to be supplied or none.")
    elif s is None and s_w is None:
        s = y
        s_w = y_w
        logger.warn('Approximated scotopic response to stimulus and reference white.')

    if p is None:
        logger.warn('p not supplied. Model will not account for simultaneous chromatic contrast .')

    xyz = np.array([x, y, z])
    logger.debug('XYZ: {}'.format(xyz))
    xyz_w = np.array([x_w, y_w, z_w])
    logger.debug('XYZ_W: {}'.format(xyz_w))
    xyz_b = np.array([x_b, y_b, z_b])
    xyz_p = np.array([x_p, y_p, z_p])

    k = 1 / (5 * l_a + 1)
    logger.debug('k: {}'.format(k))
    # luminance adaptation factor
    f_l = 0.2 * (k ** 4) * (5 * l_a) + 0.1 * ((1 - (k ** 4)) ** 2) * ((5 * l_a) ** (1 / 3))
    logger.debug('F_L: {}'.format(f_l))

    logger.debug('--- Stimulus RGB adaptation start ----')
    rgb_a = get_adaptation(f_l, l_a, xyz, xyz_w, xyz_b, xyz_p, p, helson_judd, discount_illuminant)
    logger.debug('--- Stimulus RGB adaptation end ----')
    r_a, g_a, b_a = rgb_a
    logger.debug('RGB_A: {}'.format(rgb_a))
    logger.debug('--- White RGB adaptation start ----')
    rgb_aw = get_adaptation(f_l, l_a, xyz_w, xyz_w, xyz_b, xyz_p, p, helson_judd, discount_illuminant)
    logger.debug('--- White RGB adaptation end ----')
    r_aw, g_aw, b_aw = rgb_aw
    logger.debug('RGB_AW: {}'.format(rgb_aw))

    # ---------------------------
    # Opponent Color Dimensions
    # ---------------------------

    # achromatic_cone_signal
    a_a = 2 * r_a + g_a + (1 / 20) * b_a - 3.05 + 1
    logger.debug('A_A: {}'.format(a_a))
    a_aw = 2 * r_aw + g_aw + (1 / 20) * b_aw - 3.05 + 1
    logger.debug('A_AW: {}'.format(a_aw))

    c1 = r_a - g_a
    logger.debug('C1: {}'.format(c1))
    c2 = g_a - b_a
    logger.debug('C2: {}'.format(c2))
    c3 = b_a - r_a
    logger.debug('C3: {}'.format(c3))

    c1_w = r_aw - g_aw
    logger.debug('C1_W: {}'.format(c1_w))
    c2_w = g_aw - b_aw
    logger.debug('C2_W: {}'.format(c2_w))
    c3_w = b_aw - r_aw
    logger.debug('C3_W: {}'.format(c3_w))

    # -----
    # Hue
    # -----
    hue_angle = (180 * np.arctan2(0.5 * (c2 - c3) / 4.5, c1 - (c2 / 11)) / np.pi) % 360
    hue_angle_w = (180 * np.arctan2(0.5 * (c2_w - c3_w) / 4.5, c1_w - (c2_w / 11)) / np.pi) % 360

    # -------------
    # Saturation
    # -------------
    e_s = get_eccentricity_factor(hue_angle)
    logger.debug('es: {}'.format(e_s))
    e_s_w = get_eccentricity_factor(hue_angle_w)

    f_t = l_a / (l_a + 0.1)
    logger.debug('F_t: {}'.format(f_t))
    m_yb = 100 * (0.5 * (c2 - c3) / 4.5) * (e_s * (10 / 13) * n_c * n_cb * f_t)
    logger.debug('m_yb: {}'.format(m_yb))
    m_rg = 100 * (c1 - (c2 / 11)) * (e_s * (10 / 13) * n_c * n_cb)
    logger.debug('m_rg: {}'.format(m_rg))
    m = ((m_rg ** 2) + (m_yb ** 2)) ** 0.5
    logger.debug('m: {}'.format(m))

    saturation = 50 * m / rgb_a.sum(axis=0)

    m_yb_w = 100 * (0.5 * (c2_w - c3_w) / 4.5) * (e_s_w * (10 / 13) * n_c * n_cb * f_t)
    m_rg_w = 100 * (c1_w - (c2_w / 11)) * (e_s_w * (10 / 13) * n_c * n_cb)
    m_w = ((m_rg_w ** 2) + (m_yb_w ** 2)) ** 0.5

    # ------------
    # Brightness
    # ------------
    logger.debug('--- Stimulus achromatic signal START ----')
    a = get_achromatic_signal(l_as, s, s_w, n_bb, a_a)
    logger.debug('--- Stimulus achromatic signal END ----')
    logger.debug('A: {}'.format(a))

    logger.debug('--- White achromatic signal START ----')
    a_w = get_achromatic_signal(l_as, s_w, s_w, n_bb, a_aw)
    logger.debug('--- White achromatic signal END ----')
    logger.debug('A_w: {}'.format(a_w))

    n1 = ((7 * a_w) ** 0.5) / (5.33 * n_b ** 0.13)
    n2 = (7 * a_w * n_b ** 0.362) / 200
    logger.debug('N1: {}'.format(n1))
    logger.debug('N2: {}'.format(n2))

    brightness = ((7 * (a + (m / 100))) ** 0.6) * n1 - n2
    brightness_w = ((7 * (a_w + (m_w / 100))) ** 0.6) * n1 - n2
    logger.debug('Q: {}'.format(brightness))
    logger.debug('Q_W: {}'.format(brightness_w))

    # ----------
    # Lightness
    # ----------
    z = 1 + (y_b / y_w) ** 0.5
    logger.debug('z: {}'.format(z))
    lightness = 100 * (brightness / brightness_w) ** z

    # -------
    # Chroma
    # -------
    chroma = 2.44 * (saturation ** 0.69) * ((brightness / brightness_w) ** (y_b / y_w)) * (
        1.64 - 0.29 ** (y_b / y_w))

    # -------------
    # Colorfulness
    # -------------
    colorfulness = (f_l ** 0.15) * chroma

    return Hunt_result(hue_angle, chroma, saturation, brightness, colorfulness, lightness)


xyz_to_rgb_m = np.array([[0.38971, 0.68898, -0.07868],
                         [-0.22981, 1.18340, 0.04641],
                         [0, 0, 1]])


def xyz_to_rgb(xyz):
    return xyz_to_rgb_m.dot(xyz)


def get_adaptation(f_l, l_a, xyz, xyz_w, xyz_b, xyz_p=None, p=None, helson_judd=False, discount_illuminant=True):
    """
    :param f_l: Luminance adaptation factor
    :param l_a: Adapting luminance
    :param xyz: Stimulus color in XYZ
    :param xyz_w: Reference white color in XYZ
    :param xyz_b: Background color in XYZ
    :param xyz_p: Proxima field color in XYZ
    :param p: Simultaneous contrast/assimilation parameter.
    """
    rgb = xyz_to_rgb(xyz)
    logger.debug('RGB: {}'.format(rgb))
    rgb_w = xyz_to_rgb(xyz_w)
    logger.debug('RGB_W: {}'.format(rgb_w))
    y_w = xyz_w[1]
    y_b = xyz_b[1]

    h_rgb = 3 * rgb_w / (rgb_w.sum())
    logger.debug('H_RGB: {}'.format(h_rgb))

    # Chromatic adaptation factors
    if not discount_illuminant:
        f_rgb = (1 + (l_a ** (1 / 3)) + h_rgb) / (1 + (l_a ** (1 / 3)) + (1 / h_rgb))
    else:
        f_rgb = np.ones(np.shape(h_rgb))
    logger.debug('F_RGB: {}'.format(f_rgb))

    # Adaptation factor
    if helson_judd:
        d_rgb = f_n((y_b / y_w) * f_l * f_rgb[1]) - f_n((y_b / y_w) * f_l * f_rgb)
        assert d_rgb[1] == 0
    else:
        d_rgb = np.zeros(np.shape(f_rgb))
    logger.debug('D_RGB: {}'.format(d_rgb))

    # Cone bleaching factors
    rgb_b = (10 ** 7) / ((10 ** 7) + 5 * l_a * (rgb_w / 100))
    logger.debug('B_RGB: {}'.format(rgb_b))

    if xyz_p is not None and p is not None:
        logger.debug('Account for simultaneous chromatic contrast')
        rgb_p = xyz_to_rgb(xyz_p)
        rgb_w = adjust_white_for_scc(rgb_p, rgb_b, rgb_w, p)

    # Adapt rgb using modified
    rgb_a = 1 + rgb_b * (f_n(f_l * f_rgb * rgb / rgb_w) + d_rgb)
    logger.debug('RGB_A: {}'.format(rgb_a))

    return rgb_a


def adjust_white_for_scc(rgb_p, rgb_b, rgb_w, p):
    """
    Adjust the white point for simultaneous chromatic contrast.

    :param rgb_p: Cone signals of proxima field.
    :param rgb_b: Cone signals of background.
    :param rgb_w: Cone signals of reference white.
    :param p: Simultaneous contrast/assimilation parameter.
    :return: Adjusted cone signals for reference white.
    """
    p_rgb = rgb_p / rgb_b
    rgb_w = rgb_w * (((1 - p) * p_rgb + (1 + p) / p_rgb) ** 0.5) / (((1 + p) * p_rgb + (1 - p) / p_rgb) ** 0.5)
    return rgb_w


def calculate_scotopic_luminance(photopic_luminance, color_temperature):
    return 2.26 * photopic_luminance * ((color_temperature / 4000) - 0.4) ** (1 / 3)


def get_achromatic_signal(l_as, s, s_w, n_bb, a_a):
    j = 0.00001 / ((5 * l_as / 2.26) + 0.00001)
    logger.debug('j: {}'.format(j))

    f_ls = 3800 * (j ** 2) * (5 * l_as / 2.26)
    f_ls += 0.2 * ((1 - (j ** 2)) ** 0.4) * ((5 * l_as / 2.26) ** (1 / 6))
    logger.debug('F_LS: {}'.format(f_ls))

    b_s = 0.5 / (1 + 0.3 * ((5 * l_as / 2.26) * (s / s_w)) ** 0.3)
    b_s += 0.5 / (1 + 5 * (5 * l_as / 2.26))
    logger.debug('B_S: {}'.format(b_s))

    a_s = (f_n(f_ls * s / s_w) * 3.05 * b_s) + 0.3
    logger.debug('A_S: {}'.format(a_s))

    return n_bb * (a_a - 1 + a_s - 0.3 + np.sqrt((1 + (0.3 ** 2))))


def f_n(i):
    """
    Nonlinear response function.
    """
    return 40 * ((i ** 0.73) / (i ** 0.73 + 2))


def get_eccentricity_factor(hue_angle):
    h = np.array([20.14, 90, 164.25, 237.53])
    e = np.array([0.8, 0.7, 1.0, 1.2])

    out = np.interp(hue_angle, h, e)
    out = np.where(hue_angle < 20.14, 0.856 - (hue_angle / 20.14) * 0.056, out)
    out = np.where(hue_angle > 237.53, 0.856 + 0.344 * (360 - hue_angle) / (360 - 237.53), out)

    return out
