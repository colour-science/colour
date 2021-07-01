from colour.utilities.common import to_domain_1
from colour.models.rgb.transfer_functions import (eotf_ST2084,
                                                  eotf_inverse_ST2084)
import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_XYZ_TO_ICACB_1',
    'MATRIX_XYZ_TO_ICACB_2',
    'XYZ_to_ICaCb',
    'ICaCb_to_XYZ',
]

MATRIX_XYZ_TO_ICACB_1 = np.array([
    [0.37613, 0.70431, -0.05675],
    [-0.21649, 1.14744, 0.05356],
    [0.02567, 0.16713, 0.74235],
])

MATRIX_XYZ_TO_ICACB_2 = np.array([
    [0.4949, 0.5037, 0.0015],
    [4.2854, -4.5462, 0.2609],
    [0.3605, 1.1499, -1.5105],
])


def XYZ_to_ICaCb(XYZ):
    XYZ = to_domain_1(XYZ)
    r_g_b = np.dot(MATRIX_XYZ_TO_ICACB_1, XYZ)
    r_g_b_ = eotf_inverse_ST2084(r_g_b)
    return np.dot(MATRIX_XYZ_TO_ICACB_2, r_g_b_)


def ICaCb_to_XYZ(ICaCb):
    ICaCb = to_domain_1(ICaCb)
    r_g_b_ = np.dot(np.linalg.inv(MATRIX_XYZ_TO_ICACB_2), ICaCb)
    r_g_b = eotf_ST2084(r_g_b_)
    return np.dot(np.linalg.inv(MATRIX_XYZ_TO_ICACB_1), r_g_b)
