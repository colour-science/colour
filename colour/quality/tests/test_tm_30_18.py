# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.quality.tm_30_18` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import SDS_ILLUMINANTS
from colour.quality.tm_30_18 import (averages_area,
                                     colour_fidelity_index_TM_30_18)
from colour.utilities import as_float_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestAveragesArea']


class TestAveragesArea(unittest.TestCase):
    """
    Defines :func:`colour.quality.tm_30_18.averages_area`
    definition unit tests methods.
    """

    def test_rectangle(self):
        """
        Tests :func:`colour.quality.tm_30_18.averages_area` definition.
        """

        # A simple 3*sqrt(2) by sqrt(2) rectangle
        rectangle = as_float_array([[2, 1], [1, 2], [-2, -1], [-1, -2]])
        np.allclose(averages_area(rectangle), 6)

    def test_concave(self):
        """
        Tests :func:`colour.quality.tm_30_18.averages_area` definition using
        a concave polygon.
        """

        poly = np.array([[1., -1], [1, 1], [3, 1], [3, 3], [-1, 3], [-1, -1]])
        np.allclose(averages_area(poly), 12)


class TestColourFidelityIndexTM_30_18(unittest.TestCase):
    """
    Defines :func:`colour.quality.tm_30_18.colour_fidelity_index_TM_30_18`
    definition unit tests methods.
    """

    def test_colour_fidelity_index_TM_30_18(self):
        """
        Tests :func:`colour.quality.tm_30_18.colour_fidelity_index_TM_30_18`
        definition.

        Reference data was created using the official Excel spreadsheet,
        published by IES, available at
        http://media.ies.org/docs/errata/TM-30-18_tools_etc.zip.
        """

        spec = colour_fidelity_index_TM_30_18(
            SDS_ILLUMINANTS['FL2'], additional_data=True)

        np.testing.assert_almost_equal(spec.R_f, 70, 0)
        np.testing.assert_almost_equal(spec.R_g, 86, 0)
        np.testing.assert_almost_equal(spec.CCT, 4225, 0)
        np.testing.assert_almost_equal(spec.D_uv, 0.0019, 4)

        np.testing.assert_almost_equal(spec.Rs, [
            79, 59, 67, 66, 36, 66, 40, 35, 95, 54, 48, 45, 64, 87, 72, 49, 56,
            69, 57, 44, 47, 47, 80, 63, 48, 59, 82, 85, 62, 70, 68, 62, 74, 74,
            86, 88, 80, 76, 97, 93, 91, 89, 83, 99, 83, 81, 87, 66, 80, 81, 81,
            76, 69, 77, 77, 66, 66, 67, 79, 90, 78, 87, 77, 60, 61, 58, 56, 62,
            73, 58, 64, 84, 53, 96, 67, 57, 76, 63, 82, 85, 74, 94, 91, 86, 81,
            64, 74, 69, 66, 68, 93, 51, 70, 41, 62, 70, 80, 67, 45
        ], 0)

        np.testing.assert_almost_equal(
            spec.R_fs,
            [60, 61, 53, 68, 80, 88, 77, 73, 76, 62, 70, 77, 81, 71, 64, 65],
            0)
        np.testing.assert_almost_equal(
            spec.R_cs,
            [-25, -18, -9, 5, 11, 4, -8, -15, -17, -15, -4, 5, 11, 7, -6, -16],
            0)
        np.testing.assert_almost_equal(spec.R_hs, [
            -0.02, 0.14, 0.24, 0.20, 0.09, -0.07, -0.12, -0.08, 0.01, 0.17,
            0.19, 0.11, -0.08, -0.15, -0.26, -0.17
        ], 2)


if __name__ == '__main__':
    unittest.main()
