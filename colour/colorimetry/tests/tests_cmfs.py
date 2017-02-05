#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.cmfs` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.colorimetry import (
    LMS_ConeFundamentals,
    RGB_ColourMatchingFunctions,
    XYZ_ColourMatchingFunctions)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLMS_ConeFundamentals',
           'TestRGB_ColourMatchingFunctions',
           'TestXYZ_ColourMatchingFunctions']


class TestLMS_ConeFundamentals(unittest.TestCase):
    """
    Defines :class:`colour.colorimetry.cmfs.LMS_ConeFundamentals` class units
    tests methods.
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
                               'l_bar',
                               'm_bar',
                               's_bar')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(LMS_ConeFundamentals))

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
            self.assertIn(method, dir(LMS_ConeFundamentals))


class TestRGB_ColourMatchingFunctions(unittest.TestCase):
    """
    Defines :class:`colour.colorimetry.cmfs.RGB_ColourMatchingFunctions` class
    unit tests methods.
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
                               'r_bar',
                               'g_bar',
                               'b_bar')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(RGB_ColourMatchingFunctions))

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
            self.assertIn(method, dir(RGB_ColourMatchingFunctions))


class TestXYZ_ColourMatchingFunctions(unittest.TestCase):
    """
    Defines :class:`colour.colorimetry.cmfs.XYZ_ColourMatchingFunctions` class
    unit tests methods.
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
                               'x_bar',
                               'y_bar',
                               'z_bar')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(XYZ_ColourMatchingFunctions))

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
            self.assertIn(method, dir(XYZ_ColourMatchingFunctions))


if __name__ == '__main__':
    unittest.main()
