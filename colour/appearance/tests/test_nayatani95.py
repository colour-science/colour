# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.appearance.nayatani95` module.
"""

from __future__ import division, unicode_literals

import numpy as np
from itertools import permutations

from colour.appearance import XYZ_to_Nayatani95
from colour.appearance.tests.common import ColourAppearanceModelTest
from colour.utilities import domain_range_scale, ignore_numpy_errors, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestNayatani95ColourAppearanceModel']


class TestNayatani95ColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.nayatani95` module unit tests methods for
    *Nayatani (1995)* colour appearance model.
    """

    FIXTURE_BASENAME = 'nayatani95.csv'

    OUTPUT_ATTRIBUTES = {
        'L_star_P': 'L_star_P',
        'C': 'C',
        'theta': 'h',
        'S': 's',
        'B_r': 'Q',
        'M': 'M',
        'L_star_N': 'L_star_N'
    }

    def output_specification_from_data(self, data):
        """
        Returns the *Nayatani (1995)* colour appearance model output
        specification from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        Nayatani95_Specification
            *Nayatani (1995)* colour appearance model specification.
        """

        XYZ = tstack([data['X'], data['Y'], data['Z']])
        XYZ_n = tstack([data['X_n'], data['Y_n'], data['Z_n']])

        specification = XYZ_to_Nayatani95(XYZ, XYZ_n, data['Y_o'], data['E_o'],
                                          data['E_or'])

        return specification

    def test_domain_range_scale_XYZ_to_Nayatani95(self):
        """
        Tests :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
        domain and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_n = np.array([95.05, 100.00, 108.88])
        Y_o = 20.0
        E_o = 5000.0
        E_or = 1000.0
        specification = XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)[:6]

        d_r = (
            ('reference', 1, 1),
            (1, 0.01, np.array([1, 1, 1 / 360, 1, 1, 1])),
            (100, 1, np.array([1, 1, 100 / 360, 1, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Nayatani95(XYZ * factor_a, XYZ_n * factor_a, Y_o,
                                      E_o, E_or)[:6],
                    specification * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Nayatani95(self):
        """
        Tests :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_n = np.array(case)
            Y_o = case[0]
            E_o = case[0]
            E_or = case[0]
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)
