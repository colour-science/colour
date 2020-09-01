# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.appearance.hunt` module.
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
from itertools import permutations
from unittest import TestCase

from colour.appearance import (VIEWING_CONDITIONS_HUNT, InductionFactors_Hunt,
                               XYZ_to_Hunt)
from colour.appearance.tests.common import ColourAppearanceModelTest
from colour.utilities import domain_range_scale, ignore_numpy_errors, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestHuntColourAppearanceModel']


class TestHuntColourAppearanceModel(ColourAppearanceModelTest, TestCase):
    """
    Defines :mod:`colour.appearance.hunt` module unit tests methods for
    *Hunt* colour appearance model.
    """

    FIXTURE_BASENAME = 'hunt.csv'

    OUTPUT_ATTRIBUTES = {
        'J': 'J',
        'C_94': 'C',
        'h_S': 'h',
        's': 's',
        'Q': 'Q',
        'M94': 'M'
    }

    def output_specification_from_data(self, data):
        """
        Returns the *Hunt* colour appearance model output specification from
        given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        CAM_Specification_Hunt
            Hunt colour appearance model specification.
        """

        XYZ = tstack([data['X'], data['Y'], data['Z']])
        XYZ_w = tstack([data['X_w'], data['Y_w'], data['Z_w']])
        XYZ_b = tstack([data['X_w'], 0.2 * data['Y_w'], data['Z_w']])

        specification = XYZ_to_Hunt(
            XYZ,
            XYZ_w,
            XYZ_b,
            data['L_A'],
            InductionFactors_Hunt(data['N_c'], data['N_b']),
            CCT_w=data['T'])

        return specification

    def test_domain_range_scale_XYZ_to_Hunt(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT['Normal Scenes']
        CCT_w = 6504.0
        specification = XYZ_to_Hunt(
            XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)[:-2]
        print(specification, np.__name__)
        d_r = (
            ('reference', 1, 1),
            (1, 0.01, np.array([1, 1, 1 / 360, 1, 1, 1])),
            (100, 1, np.array([1, 1, 100 / 360, 1, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    XYZ_to_Hunt(
                        XYZ * factor_a,
                        XYZ_w * factor_a,
                        XYZ_b * factor_a,
                        L_A,
                        surround,
                        CCT_w=CCT_w)[:-2],
                    np.array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_raise_exception_CIECAM02_to_XYZ(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition raised
        exception.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT['Normal Scenes']
        CCT_w = 6504.0
        S = S_w = 0.5

        try:
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround)
        except ValueError:
            pass

        try:
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w, S=S)
        except ValueError:
            pass

        try:
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w, S_w=S_w)
        except ValueError:
            pass

    @ignore_numpy_errors
    def test_XYZ_p_CIECAM02_to_XYZ(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition *XYZ_p*
        argument handling.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = XYZ_p = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT['Normal Scenes']
        CCT_w = 6504.0

        np.testing.assert_array_almost_equal(
            XYZ_to_Hunt(
                XYZ,
                XYZ_w,
                XYZ_b,
                L_A,
                surround,
                XYZ_p=XYZ_p,
                CCT_w=CCT_w,
            )[:-2],
            np.array([
                30.046267861960700,
                0.121050839936350,
                269.273759446144600,
                0.019909320692942,
                22.209765491265024,
                0.123896438259997,
            ]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Hunt(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            XYZ_b = np.array(case)
            L_A = case[0]
            surround = InductionFactors_Hunt(case[0], case[0])
            CCT_w = case[0]
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)
