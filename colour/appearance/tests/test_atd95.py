# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.appearance.atd95` module.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.appearance import XYZ_to_ATD95
from colour.appearance.atd95 import XYZ_to_LMS_ATD95, final_response
from colour.appearance.tests.common import ColourAppearanceModelTest

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestATD95ColourAppearanceModel']


class TestATD95ColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.atd95` module unit tests methods for
    ATD (1995) colour vision model.
    """

    FIXTURE_BASENAME = 'atd95.csv'

    OUTPUT_ATTRIBUTES = {
        'H': 'h',
        'C': 'C',
        'Br': 'Q',
        'A_1': 'A_1',
        'T_1': 'T_1',
        'D_1': 'D_1',
        'A_2': 'A_2',
        'T_2': 'T_2',
        'D_2': 'D_2'}

    def output_specification_from_data(self, data):
        """
        Returns the ATD (1995) colour vision model output specification from
        given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        ATD95_Specification
            ATD (1995) colour vision model specification.
        """

        XYZ = np.array([data['X'], data['Y'], data['Z']])
        XYZ_0 = np.array([data['X_0'], data['Y_0'], data['Z_0']])

        specification = XYZ_to_ATD95(XYZ,
                                     XYZ_0,
                                     data['Y_02'],
                                     data['K_1'], data['K_2'],
                                     data['sigma'])
        return specification

    def test_XYZ_to_LMS_ATD95(self):
        """
        Tests :func:`colour.appearance.atd95.XYZ_to_LMS_ATD95` definition.
        """

        L, M, S = XYZ_to_LMS_ATD95(np.array([1, 1, 1]))
        np.testing.assert_almost_equal(L, 0.7946522478109985)
        np.testing.assert_almost_equal(M, 0.9303058494144267)
        np.testing.assert_almost_equal(S, 0.7252006614718631)

    def test_final_response(self):
        """
        Tests :func:`colour.appearance.atd95.final_response` definition.
        """

        np.testing.assert_almost_equal(final_response(0), 0)
        np.testing.assert_almost_equal(final_response(100), 1.0 / 3.0)
        np.testing.assert_almost_equal(final_response(200), 0.5)
        np.testing.assert_almost_equal(final_response(10000), 0.980392157)
