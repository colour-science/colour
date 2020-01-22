# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.deprecation` module.
"""

from __future__ import division, unicode_literals

import sys
import unittest

from colour.utilities.deprecation import (
    ObjectRenamed, ObjectRemoved, ObjectFutureRename, ObjectFutureRemove,
    ObjectFutureAccessChange, ObjectFutureAccessRemove, ArgumentRenamed,
    ArgumentRemoved, ArgumentFutureRename, ArgumentFutureRemove, ModuleAPI,
    get_attribute, build_API_changes, handle_arguments_deprecation)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestObjectRenamed', 'TestObjectRemoved', 'TestObjectFutureRename',
    'TestObjectFutureRemove', 'TestObjectFutureAccessChange',
    'TestObjectFutureAccessRemove', 'TestArgumentRenamed',
    'TestArgumentRemoved', 'TestArgumentFutureRename',
    'TestArgumentFutureRemove', 'TestModuleAPI', 'TestGetAttribute',
    'TestBuildAPIChanges', 'TestHandleArgumentsDeprecation'
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

        self.assertIn('name', str(ObjectFutureAccessRemove('name', )))


class TestArgumentRenamed(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ArgumentRenamed` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ArgumentRenamed))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.ArgumentRenamed.__str__`
        method.
        """

        self.assertIn('name', str(ArgumentRenamed('name', 'new_name')))
        self.assertIn('new_name', str(ArgumentRenamed('name', 'new_name')))


class TestArgumentRemoved(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ArgumentRemoved` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ArgumentRemoved))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.ArgumentRemoved.__str__`
        method.
        """

        self.assertIn('name', str(ArgumentRemoved('name')))


class TestArgumentFutureRename(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ArgumentFutureRename` class
    unit tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ArgumentFutureRename))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.\
    ArgumentFutureRename.__str__` method.
        """

        self.assertIn('name', str(ArgumentFutureRename('name', 'new_name')))
        self.assertIn('new_name', str(
            ArgumentFutureRename('name', 'new_name')))


class TestArgumentFutureRemove(unittest.TestCase):
    """
    Defines :class:`colour.utilities.deprecation.ArgumentFutureRemove` class
    unit tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', )

        for method in required_methods:
            self.assertIn(method, dir(ArgumentFutureRemove))

    def test__str__(self):
        """
        Tests :meth:`colour.utilities.deprecation.\
ArgumentFutureRemove.__str__` method.
        """

        self.assertIn('name', str(ArgumentFutureRemove('name', )))


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


class TestBuildAPIChanges(unittest.TestCase):
    """
    Defines :func:`colour.utilities.deprecation.build_API_changes` definition
    unit tests methods.
    """

    def test_build_API_changes(self):
        """
        Tests :func:`colour.utilities.deprecation.build_API_changes`
        definition.
        """

        changes = build_API_changes({
            'ObjectRenamed': [[
                'module.object_1_name',
                'module.object_1_new_name',
            ]],
            'ObjectFutureRename': [[
                'module.object_2_name',
                'module.object_2_new_name',
            ]],
            'ObjectFutureAccessChange': [[
                'module.object_3_access',
                'module.sub_module.object_3_new_access',
            ]],
            'ObjectRemoved': ['module.object_4_name'],
            'ObjectFutureRemove': ['module.object_5_name'],
            'ObjectFutureAccessRemove': ['module.object_6_access'],
            'ArgumentRenamed': [[
                'argument_1_name',
                'argument_1_new_name',
            ]],
            'ArgumentFutureRename': [[
                'argument_2_name',
                'argument_2_new_name',
            ]],
            'ArgumentRemoved': ['argument_3_name'],
            'ArgumentFutureRemove': ['argument_4_name'],
        })
        for name, change_type in (
            ('object_1_name', ObjectRenamed),
            ('object_2_name', ObjectFutureRename),
            ('object_3_access', ObjectFutureAccessChange),
            ('object_4_name', ObjectRemoved),
            ('object_5_name', ObjectFutureRemove),
            ('object_6_access', ObjectFutureAccessRemove),
            ('argument_1_name', ArgumentRenamed),
            ('argument_2_name', ArgumentFutureRename),
            ('argument_3_name', ArgumentRemoved),
            ('argument_4_name', ArgumentFutureRemove),
        ):
            self.assertIsInstance(changes[name], change_type)


class TestHandleArgumentsDeprecation(unittest.TestCase):
    """
    Defines :func:`colour.utilities.deprecation.handle_arguments_deprecation`
    definition unit tests methods.
    """

    def test_handle_arguments_deprecation(self):
        """
        Tests :func:`colour.utilities.deprecation.handle_arguments_deprecation`
        definition.
        """

        changes = {
            'ArgumentRenamed': [[
                'argument_1_name',
                'argument_1_new_name',
            ]],
            'ArgumentFutureRename': [[
                'argument_2_name',
                'argument_2_new_name',
            ]],
            'ArgumentRemoved': ['argument_3_name'],
            'ArgumentFutureRemove': ['argument_4_name'],
        }

        self.assertDictEqual(
            handle_arguments_deprecation(
                changes,
                argument_1_name=True,
                argument_2_name=True,
                argument_4_name=True), {
                    'argument_1_new_name': True,
                    'argument_2_new_name': True,
                    'argument_4_name': True
                })

        self.assertRaises(
            ValueError,
            lambda: handle_arguments_deprecation(
                changes, argument_3_name=True),
            )


if __name__ == '__main__':
    unittest.main()
