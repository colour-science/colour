# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.deprecation` module.
"""

from __future__ import division, unicode_literals

import sys
import unittest

from colour.utilities.deprecation import (
    Renamed, Removed, FutureRename, FutureRemove, FutureAccessChange,
    FutureAccessRemove, ModuleAPI, get_attribute)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestRenamed', 'TestRemoved', 'TestFutureRename', 'TestFutureRemove',
    'TestFutureAccessChange', 'TestFutureAccessRemove', 'TestModuleAPI',
    'TestGetAttribute'
]


class TestRenamed(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.Renamed` class unit tests
    methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(Renamed))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.Renamed.__str__` method.
        """

        self.assertIn('name', str(Renamed('name', 'new_name')))
        self.assertIn('new_name', str(Renamed('name', 'new_name')))


class TestRemoved(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.Removed` class unit tests
    methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(Removed))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.Removed.__str__` method.
        """

        self.assertIn('name', str(Removed('name')))


class TestFutureRename(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.FutureRename` class unit tests
    methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(FutureRename))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.FutureRename.__str__` method.
        """

        self.assertIn('name', str(FutureRename('name', 'new_name')))
        self.assertIn('new_name', str(FutureRename('name', 'new_name')))


class TestFutureRemove(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.FutureRemove` class unit tests
    methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(FutureRemove))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.FutureRemove.__str__` method.
        """

        self.assertIn('name', str(FutureRemove('name', )))


class TestFutureAccessChange(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.FutureAccessChange` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(FutureAccessChange))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.FutureAccessChange.__str__`
        method.
        """

        self.assertIn('name', str(FutureAccessChange('name', 'new_access')))
        self.assertIn('new_access',
                      str(FutureAccessChange('name', 'new_access')))


class TestFutureAccessRemove(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.FutureAccessRemove` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(FutureAccessRemove))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.FutureAccessRemove.__str__`
        method.
        """

        self.assertIn('name', str(FutureAccessRemove('name', 'access')))
        self.assertIn('access', str(FutureAccessRemove('name', 'access')))


class TestModuleAPI(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ModuleAPI` class unit tests
    methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__init__', '__getattr__', '__dir__')

        for method in required_methods:
            self.assertIn(method, dir(ModuleAPI))

    def test__getattr__(self):
        """
        Tests :func:`colour.utilities.deprecation.ModuleAPI.__getattr__`
        method.
        """

        import colour.utilities.tests.test_deprecated

        self.assertIsNone(colour.utilities.tests.test_deprecated.NAME)

        # TODO: Use "assertWarns" when dropping Python 2.7.
        getattr(colour.utilities.tests.test_deprecated, 'OLD_NAME')

        del sys.modules['colour.utilities.tests.test_deprecated']

    def test_raise_exception__getattr__(self):
        """
        Tests :func:`colour.utilities.deprecation.ModuleAPI.__getattr__`
        method raised exception.
        """

        import colour.utilities.tests.test_deprecated

        self.assertRaises(AttributeError, getattr,
                          colour.utilities.tests.test_deprecated, 'REMOVED')

        del sys.modules['colour.utilities.tests.test_deprecated']


class TestGetAttribute(unittest.TestCase):
    """
    Defines :func:`colour.utilities.deprecation.get_attribute` definition unit
    tests methods.
    """

    def test_get_attribute(self):
        """
        Tests :func:`colour.utilities.deprecation.get_attribute` definition.
        """

        from colour import adaptation
        self.assertIs(get_attribute('colour.adaptation'), adaptation)

        from colour.models import eotf_reverse_sRGB
        self.assertIs(
            get_attribute('colour.models.eotf_reverse_sRGB'),
            eotf_reverse_sRGB)

        from colour.utilities.array import as_numeric
        self.assertIs(
            get_attribute('colour.utilities.array.as_numeric'), as_numeric)

        if 'colour.utilities.tests.test_deprecated' in sys.modules:
            del sys.modules['colour.utilities.tests.test_deprecated']
        attribute = get_attribute(
            'colour.utilities.tests.test_deprecated.NEW_NAME')
        import colour.utilities.tests.test_deprecated
        self.assertIs(attribute,
                      colour.utilities.tests.test_deprecated.NEW_NAME)
        del sys.modules['colour.utilities.tests.test_deprecated']


if __name__ == '__main__':
    unittest.main()
