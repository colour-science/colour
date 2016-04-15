#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.utilities.metadata` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.utilities import (
    Metadata,
    UnitMetadata,
    CallableMetadata,
    FunctionMetadata)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestMetadata',
           'TestUnitMetadata',
           'TestCallableMetadata',
           'TestFunctionMetadata']


class TestMetadata(unittest.TestCase):
    """
    Defines :class:`colour.utilities.metadata.Metadata` class unit tests
    methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('family',
                               'identity',
                               'instances',
                               'name',
                               'strict_name')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Metadata))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__',
                            '__repr__')

        for method in required_methods:
            self.assertIn(method, dir(Metadata))

    def test_family(self):
        """
        Tests :attr:`colour.utilities.metadata.Metadata.family` attribute.
        """

        self.assertEqual(Metadata('Lambda', '$\Lambda$').family, 'Metadata')

    def test_instances(self):
        """
        Tests :attr:`colour.utilities.metadata.Metadata.instances` attribute
        and behaviour regarding instances reference tracking.
        """

        m1 = Metadata('Lambda', '$\Lambda$')
        self.assertEqual(len(m1.instances), 1)

        m2 = Metadata('Lambda', '$\Lambda$')
        self.assertEqual(len(m1.instances), 2)

        del m2
        self.assertEqual(len(m1.instances), 1)

    def test__str__(self):
        """
        Tests :func:`colour.utilities.metadata.Metadata.__str__` method.
        """

        self.assertEqual(
            str(Metadata('Lambda', '$\Lambda$')),
            'Metadata\n    Name        : Lambda\n    Strict name : $\Lambda$')

    def test__repr__(self):
        """
        Tests :func:`colour.utilities.metadata.Metadata.__repr__` method.
        """

        self.assertEqual(
            repr(Metadata('Lambda', '$\Lambda$')),
            "Metadata('Lambda', '$\\Lambda$')")


class TestUnitMetadata(unittest.TestCase):
    """
    Defines :class:`colour.utilities.metadata.UnitMetadata` class unit tests
    methods.
    """

    def test_family(self):
        """
        Tests :attr:`colour.utilities.metadata.UnitMetadata.family` attribute.
        """

        self.assertEqual(
            UnitMetadata('Lambda', '$\Lambda$').family,
            'Unit')


class TestCallableMetadata(unittest.TestCase):
    """
    Defines :class:`colour.utilities.metadata.CallableMetadata` class unit
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('callable',)

        for attribute in required_attributes:
            self.assertIn(attribute, dir(CallableMetadata))

    def test_family(self):
        """
        Tests :attr:`colour.utilities.metadata.CallableMetadata.family`
        attribute.
        """

        self.assertEqual(
            CallableMetadata('Lambda', '$\Lambda$').family,
            'Callable')


class TestFunctionMetadata(unittest.TestCase):
    """
    Defines :class:`colour.utilities.metadata.FunctionMetadata` class unit
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('input_unit',
                               'output_unit',
                               'method',
                               'strict_method')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(FunctionMetadata))

    def test_family(self):
        """
        Tests :attr:`colour.utilities.metadata.FunctionMetadata.family`
        attribute.
        """

        self.assertEqual(
            FunctionMetadata(
                UnitMetadata('Luminance', '$Y$'),
                UnitMetadata('Lightness', '$L^\star$'),
                'CIE 1976',
                '$CIE 1976$').family,
            'Function')

    def test__str__(self):
        """
        Tests :func:`colour.utilities.metadata.FunctionMetadata.__str__`
        method.
        """

        self.assertEqual(
            str(FunctionMetadata(
                UnitMetadata('Luminance', '$Y$'),
                UnitMetadata('Lightness', '$L^\star$'),
                'CIE 1976',
                '$CIE 1976$')),
            'Function\n    Name          : Luminance to Lightness - \
CIE 1976\n    Strict name   : $Y$ to $L^\\star$ - $CIE 1976$\n    \
Unit\n        Name        : Luminance\n        Strict name : $Y$\n    \
Unit\n        Name        : Lightness\n        Strict name : $L^\\star$\n    \
Method        : CIE 1976\n    Strict method : $CIE 1976$')

    def test__repr__(self):
        """
        Tests :func:`colour.utilities.metadata.FunctionMetadata.__repr__`
        method.
        """

        self.assertEqual(
            repr(FunctionMetadata(
                UnitMetadata('Luminance', '$Y$'),
                UnitMetadata('Lightness', '$L^\star$'),
                'CIE 1976',
                '$CIE 1976$')),
            "FunctionMetadata(UnitMetadata('Luminance', '$Y$'), \
UnitMetadata('Lightness', '$L^\star$'), 'CIE 1976', '$CIE 1976$')")


if __name__ == '__main__':
    unittest.main()
