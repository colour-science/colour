# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.deprecation` module.
"""

from __future__ import division, unicode_literals

import sys
import unittest

from colour.utilities.deprecation import (
    ObjectRenamed, ObjectRemoved, ObjectFutureRename, ObjectFutureRemove,
    ObjectFutureAccessChange, ObjectFutureAccessRemove, ModuleAPI,
    get_attribute)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestObjectRenamed', 'TestObjectRemoved', 'TestObjectFutureRename',
    'TestObjectFutureRemove', 'TestObjectFutureAccessChange',
    'TestObjectFutureAccessRemove', 'TestModuleAPI', 'TestGetAttribute',
    'TestBuildAPIChanges'
]


class TestObjectRenamed(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ObjectRenamed` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ObjectRenamed))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.ObjectRenamed.__str__`
        method.
        """

        self.assertIn('name', str(ObjectRenamed('name', 'new_name')))
        self.assertIn('new_name', str(ObjectRenamed('name', 'new_name')))


class TestObjectRemoved(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ObjectRemoved` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ObjectRemoved))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.ObjectRemoved.__str__`
        method.
        """

        self.assertIn('name', str(ObjectRemoved('name')))


class TestObjectFutureRename(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ObjectFutureRename` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ObjectFutureRename))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.ObjectFutureRename.__str__`
        method.
        """

        self.assertIn('name', str(ObjectFutureRename('name', 'new_name')))
        self.assertIn('new_name', str(ObjectFutureRename('name', 'new_name')))


class TestObjectFutureRemove(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ObjectFutureRemove` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ObjectFutureRemove))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.ObjectFutureRemove.__str__`
        method.
        """

        self.assertIn('name', str(ObjectFutureRemove('name', )))


class TestObjectFutureAccessChange(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ObjectFutureAccessChange`
    class unit tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ObjectFutureAccessChange))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.\
ObjectFutureAccessChange.__str__` method.
        """

        self.assertIn('name',
                      str(ObjectFutureAccessChange('name', 'new_access')))
        self.assertIn('new_access',
                      str(ObjectFutureAccessChange('name', 'new_access')))


class TestObjectFutureAccessRemove(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ObjectFutureAccessRemove`
    class unit tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ObjectFutureAccessRemove))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.\
ObjectFutureAccessRemove.__str__` method.
        """

        self.assertIn('name', str(ObjectFutureAccessRemove('name', 'access')))
        self.assertIn('access', str(
            ObjectFutureAccessRemove('name', 'access')))


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

        from colour.models import eotf_inverse_sRGB
        self.assertIs(
            get_attribute('colour.models.eotf_inverse_sRGB'),
            eotf_inverse_sRGB)

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
