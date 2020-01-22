# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.appearance.llab` module.
"""

from __future__ import division, unicode_literals
from colour.utilities.array import tstack

import numpy as np

try:
    from unittest import mock
except ImportError:  # pragma: no cover
    import mock
from itertools import permutations

from colour.appearance import (LLAB_VIEWING_CONDITIONS, LLAB_InductionFactors,
                               XYZ_to_LLAB, llab)
from colour.appearance.tests.common import ColourAppearanceModelTest
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestLLABColourAppearanceModel']


class TestLLABColourAppearanceModel(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.llab` module unit tests methods for
    *LLAB(l:c)* colour appearance model.
    """

    FIXTURE_BASENAME = 'llab.csv'

    OUTPUT_ATTRIBUTES = {
        'L_L': 'J',
        'Ch_L': 'C',
        'h_L': 'h',
        's_L': 's',
        'C_L': 'M',
        'A_L': 'a',
        'B_L': 'b'
    }

    def output_specification_from_data(self, data):
        """
        Returns the *LLAB(l:c)* colour appearance model output specification
        from given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        LLAB_Specification
            *LLAB(l:c)* colour appearance model specification.
        """

        XYZ = tstack([data['X'], data['Y'], data['Z']])
        XYZ_0 = tstack([data['X_0'], data['Y_0'], data['Z_0']])

        specification = XYZ_to_LLAB(
            XYZ, XYZ_0, data['Y_b'], data['L'],
            LLAB_InductionFactors(1, data['F_S'], data['F_L'], data['F_C']))

        return specification

    def test_examples(self):
        """
        Tests the colour appearance model implementation.

        Returns
        -------
        tuple

        Notes
        -----
        -   Reference data was computed using a rounded
            :attr:`colour.appearance.llab.LLAB_RGB_TO_XYZ_MATRIX`, therefore a
            patched version is used for unit tests.
        """

        with mock.patch(
                'colour.appearance.llab.LLAB_RGB_TO_XYZ_MATRIX',
                np.around(
                    np.linalg.inv(llab.LLAB_XYZ_TO_RGB_MATRIX), decimals=4)):
            super(TestLLABColourAppearanceModel, self).test_examples()

    def test_n_dimensional_examples(self):
        """
        Tests the colour appearance model implementation n-dimensional arrays
        support.

        Returns
        -------
        tuple

        Notes
        -----
        -   Reference data was computed using a rounded
            :attr:`colour.appearance.llab.LLAB_RGB_TO_XYZ_MATRIX`, therefore a
            patched version is used for unit tests.
        """

        with mock.patch(
                'colour.appearance.llab.LLAB_RGB_TO_XYZ_MATRIX',
                np.around(
                    np.linalg.inv(llab.LLAB_XYZ_TO_RGB_MATRIX), decimals=4)):
            super(TestLLABColourAppearanceModel,
                  self).test_n_dimensional_examples()

    def test_colourspace_conversion_matrices_precision(self):
        """
        Tests for loss of precision in conversion between
        *LLAB(l:c)* colour appearance model *CIE XYZ* tristimulus values and
        normalised cone responses matrix.
        """

        start = np.array([1, 1, 1])
        result = np.array(start)
        for _ in range(100000):
            result = llab.LLAB_RGB_TO_XYZ_MATRIX.dot(result)
            result = llab.LLAB_XYZ_TO_RGB_MATRIX.dot(result)
        np.testing.assert_almost_equal(start, result, decimal=7)

    def test_domain_range_scale_XYZ_to_LLAB(self):
        """
        Tests :func:`colour.appearance.llab.XYZ_to_LLAB` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_0 = np.array([95.05, 100.00, 108.88])
        Y_b = 20.0
        L = 318.31
        surround = LLAB_VIEWING_CONDITIONS['ref_average_4_minus']
        specification = XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)[:5]

        d_r = (
            ('reference', 1, 1),
            (1, 0.01, np.array([1, 1, 1 / 360, 1, 1])),
            (100, 1, np.array([1, 1, 100 / 360, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_LLAB(XYZ * factor_a, XYZ_0 * factor_a, Y_b, L,
                                surround)[:5],
                    specification * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_LLAB(self):
        """
        Tests :func:`colour.appearance.llab.XYZ_to_LLAB` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_0 = np.array(case)
            Y_b = case[0]
            L = case[0]
            surround = LLAB_InductionFactors(1, case[0], case[0], case[0])
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)
