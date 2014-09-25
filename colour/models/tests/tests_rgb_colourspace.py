#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb_colourspace` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import RGB_COLOURSPACES, RGB_Colourspace

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_COLOURSPACES',
           'TestRGB_Colourspace']


class TestRGB_COLOURSPACES(unittest.TestCase):
    """
    Defines :attr:`colour.models.RGB_COLOURSPACES` attribute unit tests
    methods.
    """

    def test_transformation_matrices(self):
        """
        Tests the transformations matrices from the
        :attr:`colour.models.RGB_COLOURSPACES` attribute colourspace models.
        """

        XYZ_r = np.array([0.5, 0.5, 0.5]).reshape((3, 1))
        for colourspace in RGB_COLOURSPACES.values():
            RGB = np.dot(colourspace.XYZ_to_RGB_matrix, XYZ_r)
            XYZ = np.dot(colourspace.RGB_to_XYZ_matrix, RGB)
            np.testing.assert_almost_equal(XYZ_r, XYZ, decimal=7)

    def test_opto_electronic_conversion_functions(self):
        """
        Tests the opto-electronic conversion functions from the
        :attr:`colour.models.RGB_COLOURSPACES` attribute colourspace models.
        """

        aces_proxy_colourspaces = ('ACES RGB Proxy 10', 'ACES RGB Proxy 12')

        samples = np.linspace(0, 1, 1000)
        for colourspace in RGB_COLOURSPACES.values():
            if colourspace.name in aces_proxy_colourspaces:
                continue

            samples_oecf = [colourspace.transfer_function(sample)
                            for sample in samples]
            samples_invert_oecf = [
                colourspace.inverse_transfer_function(sample)
                for sample in samples_oecf]

            np.testing.assert_almost_equal(samples,
                                           samples_invert_oecf,
                                           decimal=7)

        for colourspace in aces_proxy_colourspaces:
            colourspace = RGB_COLOURSPACES.get(colourspace)
            samples_oecf = [colourspace.transfer_function(sample)
                            for sample in samples]
            samples_invert_oecf = [
                colourspace.inverse_transfer_function(sample)
                for sample in samples_oecf]

            np.testing.assert_allclose(samples,
                                       samples_invert_oecf,
                                       rtol=0.01,
                                       atol=0.01)


class TestRGB_Colourspace(unittest.TestCase):
    """
    Defines :class:`colour.colour.models.RGB_Colourspace` class units
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('name',
                               'primaries',
                               'whitepoint',
                               'RGB_to_XYZ_matrix',
                               'XYZ_to_RGB_matrix',
                               'transfer_function',
                               'inverse_transfer_function',)

        for attribute in required_attributes:
            self.assertIn(attribute, dir(RGB_Colourspace))


if __name__ == '__main__':
    unittest.main()
