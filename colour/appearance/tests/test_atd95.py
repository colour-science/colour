# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.appearance.atd95` module.
"""

from __future__ import division, unicode_literals

import numpy as np
from itertools import permutations

from colour.appearance import XYZ_to_ATD95
from colour.appearance.tests.common import ColourAppearanceModelTest
from colour.utilities import domain_range_scale, ignore_numpy_errors, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestATD95ColourAppearanceModel']


class TestATD95ColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.atd95` module unit tests methods for
    *ATD (1995)* colour vision model.
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
        'D_2': 'D_2'
    }

    def output_specification_from_data(self, data):
        """
        Returns the *ATD (1995)* colour vision model output specification from
        given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        ATD95_Specification
            *ATD (1995)* colour vision model specification.
        """

        XYZ = tstack([data['X'], data['Y'], data['Z']])
        XYZ_0 = tstack([data['X_0'], data['Y_0'], data['Z_0']])

        specification = XYZ_to_ATD95(XYZ, XYZ_0, data['Y_02'], data['K_1'],
                                     data['K_2'], data['sigma'])

        return specification

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_ATD95(self):
        """
        Tests :func:`colour.appearance.atd95.XYZ_to_ATD95` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_0 = np.array([95.05, 100.00, 108.88])
        Y_0 = 318.31
        k_1 = 0.0
        k_2 = 50.0
        specification = XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)

        d_r = (
            ('reference', 1, 1),
            (1, 0.01, np.array([1 / 360, 1, 1, 1, 1, 1, 1, 1, 1])),
            (100, 1, np.array([100 / 360, 1, 1, 1, 1, 1, 1, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_ATD95(XYZ * factor_a, XYZ_0 * factor_a, Y_0, k_1,
                                 k_2),
                    specification * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_ATD95(self):
        """
        Tests :func:`colour.appearance.atd95.XYZ_to_ATD95` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_0 = np.array(case)
            Y_0 = np.array(case[0])
            k_1 = np.array(case[0])
            k_2 = np.array(case[0])
            XYZ_to_ATD95(XYZ, XYZ_0, Y_0, k_1, k_2)
