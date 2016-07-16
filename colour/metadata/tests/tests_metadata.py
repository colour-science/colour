#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.metadata.common` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.metadata import (
    Metadata,
    EntityMetadata,
    CallableMetadata,
    FunctionMetadata,
    set_metadata)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestMetadata',
           'TestEntityMetadata',
           'TestCallableMetadata',
           'TestFunctionMetadata',
           'TestSetMetadata']


class TestMetadata(unittest.TestCase):
    """
    Defines :class:`colour.metadata.common.Metadata` class unit tests
    methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('family',
                               'instances',
                               'index',
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
        Tests :attr:`colour.metadata.common.Metadata.family` attribute.
        """

        self.assertEqual(Metadata('Lambda', '$\Lambda$').family, 'Metadata')

    def test_instances(self):
        """
        Tests :attr:`colour.metadata.common.Metadata.instances` attribute
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
        Tests :func:`colour.metadata.common.Metadata.__str__` method.
        """

        self.assertEqual(
            str(Metadata('Lambda', '$\Lambda$')),
            'Metadata\n    Name        : Lambda\n    Strict name : $\Lambda$')

    def test__repr__(self):
        """
        Tests :func:`colour.metadata.common.Metadata.__repr__` method.
        """

        self.assertEqual(
            repr(Metadata('Lambda', '$\Lambda$')),
            "Metadata('Lambda', '$\\Lambda$')")


class TestEntityMetadata(unittest.TestCase):
    """
    Defines :class:`colour.metadata.common.EntityMetadata` class unit tests
    methods.
    """

    def test_family(self):
        """
        Tests :attr:`colour.metadata.common.EntityMetadata.family` attribute.
        """

        self.assertEqual(
            EntityMetadata('Lambda', '$\Lambda$').family,
            'Entity')


class TestCallableMetadata(unittest.TestCase):
    """
    Defines :class:`colour.metadata.common.CallableMetadata` class unit
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
        Tests :attr:`colour.metadata.common.CallableMetadata.family`
        attribute.
        """

        self.assertEqual(
            CallableMetadata('Lambda', '$\Lambda$').family,
            'Callable')


class TestFunctionMetadata(unittest.TestCase):
    """
    Defines :class:`colour.metadata.common.FunctionMetadata` class unit
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('input_entity',
                               'output_entity',
                               'method',
                               'strict_method')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(FunctionMetadata))

    def test_family(self):
        """
        Tests :attr:`colour.metadata.common.FunctionMetadata.family`
        attribute.
        """

        self.assertEqual(
            FunctionMetadata(
                EntityMetadata('Luminance', '$Y$'),
                EntityMetadata('Lightness', '$L^\star$'),
                (0, 100),
                (0, 100),
                'CIE 1976',
                '$CIE 1976$').family,
            'Function')

    def test__str__(self):
        """
        Tests :func:`colour.metadata.common.FunctionMetadata.__str__`
        method.
        """

        self.assertEqual(
            str(FunctionMetadata(
                EntityMetadata('Luminance', '$Y$'),
                EntityMetadata('Lightness', '$L^\star$'),
                (0, 100),
                (0, 100),
                'CIE 1976',
                '$CIE 1976$')),
            'Function\n    Name          : Luminance [0, 100] to \
Lightness [0, 100] - CIE 1976\n    Strict name   : $Y$ [0, 100] to \
$L^\\star$ [0, 100] - $CIE 1976$\n    Entity\n        Name        : \
Luminance\n        Strict name : $Y$\n    Entity\n        Name        : \
Lightness\n        Strict name : $L^\\star$\n    Method        : \
CIE 1976\n    Strict method : $CIE 1976$')

    def test__repr__(self):
        """
        Tests :func:`colour.metadata.common.FunctionMetadata.__repr__`
        method.
        """

        self.assertEqual(
            repr(FunctionMetadata(
                EntityMetadata('Luminance', '$Y$'),
                EntityMetadata('Lightness', '$L^\star$'),
                (0, 100),
                (0, 100),
                'CIE 1976',
                '$CIE 1976$')),
            "FunctionMetadata(EntityMetadata('Luminance', '$Y$'), \
EntityMetadata('Lightness', '$L^\\star$'), \
(0, 100), (0, 100), 'CIE 1976', '$CIE 1976$')")


class TestSetMetadata(unittest.TestCase):
    """
    Defines :func:`colour.utilities.metadata.set_metadata` definition units
    tests methods.
    """

    def test_set_metadata(self):
        """
        Tests :func:`colour.utilities.metadata.set_metadata` definition.
        """

        @set_metadata(Metadata, 'Lambda', '$\Lambda$')
        def f():
            pass

        self.assertTrue(hasattr(f, '__metadata__'))

        m = Metadata('Gamma', '$\Gamma$')

        @set_metadata(m)
        def f():
            pass

        self.assertIs(f.__metadata__, m)


if __name__ == '__main__':
    unittest.main()
