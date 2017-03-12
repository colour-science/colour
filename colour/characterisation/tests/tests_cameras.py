#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.characterisation.cameras` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.characterisation import RGB_SpectralSensitivities

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_SpectralSensitivities']


class TestRGB_SpectralSensitivities(unittest.TestCase):
    """
    Defines :class:`colour.characterisation.cameras.RGB_SpectralSensitivities`
    class units tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('name',
                               'mapping',
                               'labels',
                               'data',
                               'x',
                               'y',
                               'z',
                               'wavelengths',
                               'values',
                               'shape',
                               'red',
                               'green',
                               'blue')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(RGB_SpectralSensitivities))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__hash__',
                            '__getitem__',
                            '__setitem__',
                            '__iter__',
                            '__contains__',
                            '__len__',
                            '__eq__',
                            '__ne__',
                            '__add__',
                            '__sub__',
                            '__mul__',
                            '__div__',
                            'get',
                            'extrapolate',
                            'interpolate',
                            'align',
                            'zeros',
                            'normalise',
                            'clone')

        for method in required_methods:
            self.assertIn(method, dir(RGB_SpectralSensitivities))


if __name__ == '__main__':
    unittest.main()
