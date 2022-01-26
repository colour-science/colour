# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.recovery` module.
"""

import numpy as np
import unittest

from colour.colorimetry import (
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    SpectralShape,
    reshape_msds,
    reshape_sd,
    sd_to_XYZ_integration,
)
from colour.recovery import XYZ_to_sd
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_sd',
]


class TestXYZ_to_sd(unittest.TestCase):
    """
    Defines :func:`colour.recovery.XYZ_to_sd` definition unit tests
    methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        # pylint: disable=E1102
        self._cmfs = reshape_msds(
            MSDS_CMFS['CIE 1931 2 Degree Standard Observer'],
            SpectralShape(360, 780, 10))

        self._sd_D65 = reshape_sd(SDS_ILLUMINANTS['D65'], self._cmfs.shape)

    def test_domain_range_scale_XYZ_to_sd(self):
        """
        Tests :func:`colour.recovery.XYZ_to_sd` definition domain
        and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        m = ('Jakob 2019', 'Mallett 2019', 'Meng 2015', 'Otsu 2018',
             'Smits 1999')
        v = [
            sd_to_XYZ_integration(
                XYZ_to_sd(
                    XYZ, method, cmfs=self._cmfs, illuminant=self._sd_D65),
                self._cmfs, self._sd_D65) for method in m
        ]

        d_r = (('reference', 1, 1), ('1', 1, 0.01), ('100', 100, 1))
        for method, value in zip(m, v):
            for scale, factor_a, factor_b in d_r:
                with domain_range_scale(scale):
                    np.testing.assert_almost_equal(
                        sd_to_XYZ_integration(
                            XYZ_to_sd(
                                XYZ * factor_a,
                                method,
                                cmfs=self._cmfs,
                                illuminant=self._sd_D65), self._cmfs,
                            self._sd_D65),
                        value * factor_b,
                        decimal=7)


if __name__ == '__main__':
    unittest.main()
