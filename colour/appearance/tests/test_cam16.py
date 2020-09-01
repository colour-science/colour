# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.appearance.cam16` module.
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
from itertools import permutations
from unittest import TestCase

from colour.appearance import (VIEWING_CONDITIONS_CAM16,
                               InductionFactors_CAM16, CAM_Specification_CAM16,
                               XYZ_to_CAM16, CAM16_to_XYZ)
from colour.appearance.tests.common import ColourAppearanceModelTest
from colour.utilities import (as_namedtuple, domain_range_scale,
                              ignore_numpy_errors, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestCAM16ColourAppearanceModelForward',
    'TestCAM16ColourAppearanceModelInverse'
]


class TestCAM16ColourAppearanceModelForward(ColourAppearanceModelTest,
                                            TestCase):
    """
    Defines :mod:`colour.appearance.cam16` module units tests methods for
    *CAM16* colour appearance model forward implementation.
    """

    # TODO: The current fixture data is generated from direct computations
    # using our model implementation. We have asked ground truth data to
    # Li et al. (2016) and will update the "cam16.csv" file accordingly
    # whenever we receive it.
    FIXTURE_BASENAME = 'cam16.csv'

    OUTPUT_ATTRIBUTES = {
        'J': 'J',
        'C': 'C',
        'h': 'h',
        's': 's',
        'Q': 'Q',
        'M': 'M',
        'H': 'H'
    }

    def output_specification_from_data(self, data):
        """
        Returns the *CAM16* colour appearance model output specification from
        given data.

        Parameters
        ----------
        data : list
            Fixture data.

        Returns
        -------
        CAM_Specification_CAM16
            *CAM16* colour appearance model specification.
        """

        XYZ = tstack([data['X'], data['Y'], data['Z']])
        XYZ_w = tstack([data['X_w'], data['Y_w'], data['Z_w']])

        specification = XYZ_to_CAM16(
            XYZ, XYZ_w, data['L_A'], data['Y_b'],
            InductionFactors_CAM16(data['F'], data['c'], data['N_c']))

        return specification

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_CAM16(self):
        """
        Tests :func:`colour.appearance.cam16.XYZ_to_CAM16` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20.0
        surround = VIEWING_CONDITIONS_CAM16['Average']
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)[:-1]

        d_r = (
            ('reference', 1, 1),
            (1, 0.01,
             np.array([
                 1 / 100, 1 / 100, 1 / 360, 1 / 100, 1 / 100, 1 / 100, 1 / 360
             ])),
            (100, 1, np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 360])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    XYZ_to_CAM16(XYZ * factor_a, XYZ_w * factor_a, L_A, Y_b,
                                 surround)[:-1],
                    np.array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_CAM16(self):
        """
        Tests :func:`colour.appearance.cam16.XYZ_to_CAM16` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            L_A = case[0]
            Y_b = case[0]
            surround = InductionFactors_CAM16(case[0], case[0], case[0])
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)


class TestCAM16ColourAppearanceModelInverse(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.cam16` module units tests methods for
    *CAM16* colour appearance model inverse implementation.
    """

    FIXTURE_BASENAME = 'cam16.csv'

    OUTPUT_ATTRIBUTES = {'X': 0, 'Y': 1, 'Z': 2}

    def output_specification_from_data(self, data):
        """
        Returns the colour appearance model output specification from given
        fixture data.

        Parameters
        ----------
        data : list
            Tested colour appearance model fixture data.

        Notes
        -----
        -   This method is a dummy object.
        """

        pass

    def _XYZ_from_data(self, data, correlates):
        """
        Returns the *CIE XYZ* tristimulus values from given *CAM16* colour
        appearance model input data.

        Parameters
        ----------
        data : list
            Fixture data.
        correlates : array_like
            Correlates used to build the input *CAM16* colour appearance
            model specification.

        Returns
        -------
        array_like
            *CIE XYZ* tristimulus values
        """

        XYZ_w = tstack([data['X_w'], data['Y_w'], data['Z_w']])

        i, j, k = correlates
        specification = as_namedtuple({
            i: data[i],
            j: data[j],
            k: data[k]
        }, CAM_Specification_CAM16)

        XYZ = CAM16_to_XYZ(
            specification, XYZ_w, data['L_A'], data['Y_b'],
            InductionFactors_CAM16(data['F'], data['c'], data['N_c']))

        return XYZ

    def check_specification_attribute(self, case, data, attribute, expected):
        """
        Tests *CIE XYZ* tristimulus values output from *CAM16* colour
        appearance model input data.

        Parameters
        ----------
        case : int
            Fixture case number.
        data : dict.
            Fixture case data.
        attribute : unicode.
            Tested attribute name.
        expected : float.
            Expected attribute value.

        Warning
        -------
        The method name does not reflect the underlying implementation.
        """

        for correlates in (('J', 'C', 'h'), ('J', 'M', 'h')):
            XYZ = self._XYZ_from_data(data, correlates)
            value = tsplit(XYZ)[attribute]

            error_message = ('Parameter "{0}" in test case "{1}" '
                             'does not match target value.\n'
                             'Expected: "{2}" \n'
                             'Received "{3}"').format(attribute, case,
                                                      expected, value)

            np.testing.assert_allclose(
                value,
                expected,
                err_msg=error_message,
                rtol=0.01,
                atol=0.01,
                verbose=False)

            np.testing.assert_array_almost_equal(
                value, expected, decimal=1, err_msg=error_message)

    @ignore_numpy_errors
    def test_domain_range_scale_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition domain
        and range scale support.
        """

        XYZ_i = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20.0
        surround = VIEWING_CONDITIONS_CAM16['Average']
        specification = XYZ_to_CAM16(XYZ_i, XYZ_w, L_A, Y_b, surround)
        XYZ = CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround)
        d_r = (
            ('reference', 1, 1, 1),
            (1,
             np.array([
                 1 / 100, 1 / 100, 1 / 360, 1 / 100, 1 / 100, 1 / 100, 1 / 360
             ]), 0.01, 0.01),
            (100, np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 360]), 1, 1),
        )

        for scale, factor_a, factor_b, factor_c in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    CAM16_to_XYZ(
                        np.array(specification[:-1]) * factor_a,
                        XYZ_w * factor_b, L_A, Y_b, surround),
                    XYZ * factor_c,
                    decimal=7)

    @ignore_numpy_errors
    def test_raise_exception_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition raised
        exception.
        """

        try:
            CAM16_to_XYZ(
                CAM_Specification_CAM16(
                    41.731207905126638,
                    None,
                    217.06795976739301,
                ),
                np.array([95.05, 100.00, 108.88]),
                318.31,
                20.0,
                VIEWING_CONDITIONS_CAM16['Average'],
            )
        except ValueError:
            pass

    @ignore_numpy_errors
    def test_nan_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            J = case[0]
            C = case[0]
            h = case[0]
            XYZ_w = np.array(case)
            L_A = case[0]
            Y_b = case[0]
            surround = InductionFactors_CAM16(case[0], case[0], case[0])
            CAM16_to_XYZ(
                CAM_Specification_CAM16(J, C, h), XYZ_w, L_A, Y_b, surround)
