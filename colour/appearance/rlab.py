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

__all__ = ['XYZ_to_RLAB']

from colour.appearance import hunt

logger = logging.getLogger(__name__)

RLAB_result = namedtuple("RLAB_result", ("h", "C", "s", "L", "a", "b"))

R = np.array([[1.9569, -1.1882, 0.2313],
              [0.3612, 0.6388, 0],
              [0, 0, 1]])


def XYZ_to_RLAB(x, y, z, x_n, y_n, z_n, y_n_abs, sigma, d):
    """
    Compute the RLAB model color appearance correlates.

    Parameters
    ----------
    x, y, z : numeric or array_like
        CIE XYZ values of test sample in domain [0, 100].
    x_n, y_n, z_n : numeric or array_like
        CIE XYZ values of reference white in domain [0, 100].
    y_n_abs : numeric or array_like
        Absolute luminance of a white object in cd/m^2.
    sigma : numeric or array_like
        Relative luminance parameter. For average surround set sigma=1/2.3,
        for dim surround sigma=1/2.9 and for dark surround sigma=1/3.5.
    d : numeric or array_like
        Degree of adaptation in domain [0,1].

    Returns
    -------
    RLAB_result

    References
    ----------
    .. [1] Fairchild, M. D. (1996). Refinement of the RLAB color space. *Color Research & Application*, 21(5), 338-346.
    .. [2] Fairchild, M. D. (2013). *Color appearance models*, 3rd Ed. John Wiley & Sons.

    """

    xyz = np.array([x, y, z])
    xyz_n = np.array([x_n, y_n, z_n])

    lms = hunt.xyz_to_rgb(xyz)
    lms_n = hunt.xyz_to_rgb(xyz_n)
    logger.debug('LMS: {}'.format(lms))
    logger.debug('LMS_n: {}'.format(lms_n))

    lms_e = (3 * lms_n) / (lms_n[0] + lms_n[1] + lms_n[2])
    lms_p = (1 + (y_n_abs ** (1 / 3)) + lms_e) / (
        1 + (y_n_abs ** (1 / 3)) + (1 / lms_e))
    logger.debug('LMS_e: {}'.format(lms_e))
    logger.debug('LMS_p: {}'.format(lms_p))

    lms_a = (lms_p + d * (1 - lms_p)) / lms_n
    logger.debug('LMS_a: {}'.format(lms_a))

    # If we want to allow arrays as input we need special handling here.
    if len(np.shape(x)) == 0:
        # Okay so just a number, we can do things by the book.
        a = np.diag(lms_a)
        logger.debug('A: {}'.format(a))
        xyz_ref = R.dot(a).dot(hunt.xyz_to_rgb_m).dot(xyz)
    else:
        # So we have an array. Since constructing huge multidimensional arrays might not bee the best idea,
        # we will handle each input dimension separately.
        # First figure out how many values we have to deal with.
        input_dim = len(x)
        # No create the ouput array that we will fill layer by layer
        xyz_ref = np.zeros((3, input_dim))
        for layer in range(input_dim):
            a = np.diag(lms_a[..., layer])
            logger.debug('A layer {}: {}'.format(layer, a))
            xyz_ref[..., layer] = R.dot(a).dot(hunt.xyz_to_rgb_m).dot(xyz[..., layer])

    logger.debug('XYZ_ref: {}'.format(xyz_ref))
    x_ref, y_ref, z_ref = xyz_ref

    # Lightness
    lightness = 100 * (y_ref ** sigma)
    logger.debug('lightness: {}'.format(lightness))

    # Opponent Color Dimensions
    a = 430 * ((x_ref ** sigma) - (y_ref ** sigma))
    b = 170 * ((y_ref ** sigma) - (z_ref ** sigma))
    logger.debug('a: {}'.format(a))
    logger.debug('b: {}'.format(b))

    # Hue
    hue_angle = (360 * np.arctan2(b, a) / (2 * np.pi) + 360) % 360

    # Chroma
    chroma = np.sqrt((a ** 2) + (b ** 2))

    # Saturation
    saturation = chroma / lightness

    return RLAB_result(hue_angle, chroma, saturation, lightness, a, b, )