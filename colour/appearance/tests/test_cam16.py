# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.appearance.cam16` module.
"""

from __future__ import division, unicode_literals

import numpy as np
from itertools import permutations

from colour.appearance import (CAM16_InductionFactors, CAM16_Specification,
                               XYZ_to_CAM16, CAM16_to_XYZ)
from colour.appearance.tests.common import ColourAppearanceModelTest
from colour.utilities import as_namedtuple, ignore_numpy_errors, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestCAM16ColourAppearanceModelForward',
    'TestCAM16ColourAppearanceModelReverse'
]


class TestCAM16ColourAppearanceModelForward(ColourAppearanceModelTest):
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
        CAM16_Specification
            *CAM16* colour appearance model specification.
        """

        XYZ = tstack((data['X'], data['Y'], data['Z']))
        XYZ_w = tstack((data['X_w'], data['Y_w'], data['Z_w']))

        specification = XYZ_to_CAM16(XYZ, XYZ_w, data['L_A'], data['Y_b'],
                                     CAM16_InductionFactors(
                                         data['F'], data['c'], data['N_c']))

        return specification


class TestCAM16ColourAppearanceModelReverse(ColourAppearanceModelTest):
    """
    Defines :mod:`colour.appearance.cam16` module units tests methods for
    *CAM16* colour appearance model reverse implementation.
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

        XYZ_w = tstack((data['X_w'], data['Y_w'], data['Z_w']))

        i, j, k = correlates
        CAM16_specification = as_namedtuple({
            i: data[i],
            j: data[j],
            k: data[k]
        }, CAM16_Specification)

        XYZ = CAM16_to_XYZ(
            CAM16_specification, XYZ_w, data['L_A'], data['Y_b'],
            CAM16_InductionFactors(data['F'], data['c'], data['N_c']))

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

            np.testing.assert_almost_equal(
                value, expected, decimal=1, err_msg=error_message)

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
            surround = CAM16_InductionFactors(case[0], case[0], case[0])
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)

    @ignore_numpy_errors
    def test_nan_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition
        nan support.
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
            surround = CAM16_InductionFactors(case[0], case[0], case[0])
            CAM16_to_XYZ(
                CAM16_Specification(J, C, h), XYZ_w, L_A, Y_b, surround)
