#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.utilities.data_structures` module.
"""

from __future__ import division, unicode_literals

import pickle
import unittest

from colour.utilities import (
    ArbitraryPrecisionMapping,
    Structure,
    Lookup,
    CaseInsensitiveMapping)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestArbitraryPrecisionMapping',
           'TestStructure',
           'TestLookup',
           'TestCaseInsensitiveMapping']


class TestArbitraryPrecisionMapping(unittest.TestCase):
    """
    Defines :class:`colour.utilities.data_structures.\
ArbitraryPrecisionMapping` class units tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('key_decimals',)

        for attribute in required_attributes:
            self.assertIn(attribute, dir(ArbitraryPrecisionMapping))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__setitem__',
                            '__getitem__',
                            '__delitem__',
                            '__contains__',
                            '__iter__',
                            '__len__')

        for method in required_methods:
            self.assertIn(method, dir(ArbitraryPrecisionMapping))

    def test__setitem__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
ArbitraryPrecisionMapping.__setitem__` method.
        """

        data = {0.1999999998: 'Nemo', 0.2000000000: 'John'}
        mapping = ArbitraryPrecisionMapping(data, key_decimals=10)

        self.assertEqual(mapping, data)

        mapping = ArbitraryPrecisionMapping(data, key_decimals=7)

        self.assertListEqual(list(mapping.keys()), [0.2])

    def test__getitem__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
ArbitraryPrecisionMapping.__getitem__` method.
        """

        data = {0.1999999998: 'Nemo', 0.2000000000: 'John'}
        mapping = ArbitraryPrecisionMapping(data, key_decimals=10)

        self.assertEqual(mapping[0.2], 'John')

        self.assertEqual(mapping[0.1999999998], 'Nemo')

        data = {0.1999999998: 'Nemo', 0.2000000000: 'Nemo'}
        mapping = ArbitraryPrecisionMapping(data, key_decimals=7)

        self.assertEqual(mapping[0.2], 'Nemo')

    def test__delitem__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
ArbitraryPrecisionMapping.__delitem__` method.
        """

        data = {0.1999999998: 'Nemo', 0.2000000000: 'John'}
        mapping = ArbitraryPrecisionMapping(data, key_decimals=10)

        del mapping[0.2]

        self.assertNotIn(0.2, mapping)

        del mapping[0.1999999998]

        self.assertNotIn(0.1999999998, mapping)

        mapping = ArbitraryPrecisionMapping(data, key_decimals=7)

        del mapping[0.2]

        self.assertEqual(mapping, {})

    def test__contains__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
ArbitraryPrecisionMapping.__contains__` method.
        """

        data = {0.1999999998: 'Nemo', 0.2000000000: 'John'}
        mapping = ArbitraryPrecisionMapping(data, key_decimals=10)

        self.assertIn(0.2, mapping)

        self.assertIn(0.1999999998, mapping)

        mapping = ArbitraryPrecisionMapping(data, key_decimals=7)

        self.assertIn(0.2, mapping)

        self.assertNotIn(0.1999999, mapping)

    def test__iter__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
ArbitraryPrecisionMapping.__iter__` method.
        """

        data = {0.1999999998: 'Nemo', 0.2000000000: 'John'}
        mapping = ArbitraryPrecisionMapping(data, key_decimals=10)

        self.assertListEqual(sorted([item for item in mapping]),
                             [0.1999999998, 0.2])

        mapping = ArbitraryPrecisionMapping(data, key_decimals=7)

        self.assertListEqual(sorted([item for item in mapping]),
                             [0.2])

    def test__len__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
ArbitraryPrecisionMapping.__len__` method.
        """

        data = {0.1999999998: 'Nemo', 0.2000000000: 'John'}
        mapping = ArbitraryPrecisionMapping(data, key_decimals=10)

        self.assertEqual(len(mapping), 2)

        mapping = ArbitraryPrecisionMapping(data, key_decimals=7)

        self.assertEqual(len(mapping), 1)


class TestStructure(unittest.TestCase):
    """
    Defines :class:`colour.utilities.data_structures.Structure` class units
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__getattr__',
                            '__setattr__',
                            '__delattr__',
                            'update')

        for method in required_methods:
            self.assertIn(method, dir(Structure))

    def test_Structure(self):
        """
        Tests :class:`colour.utilities.data_structures.Structure` class.
        """

        structure = Structure(John='Doe', Jane='Doe')
        self.assertIn('John', structure)
        self.assertTrue(hasattr(structure, 'John'))

        setattr(structure, 'John', 'Nemo')
        self.assertEqual(structure['John'], 'Nemo')

        structure['John'] = 'Vador'
        self.assertEqual(structure['John'], 'Vador')

        del structure['John']
        self.assertNotIn('John', structure)
        self.assertFalse(hasattr(structure, 'John'))

        structure.John = 'Doe'
        self.assertIn('John', structure)
        self.assertTrue(hasattr(structure, 'John'))

        del structure.John
        self.assertNotIn('John', structure)
        self.assertFalse(hasattr(structure, 'John'))

        structure = Structure(John=None, Jane=None)
        self.assertIsNone(structure.John)
        self.assertIsNone(structure['John'])

        structure.update(**{'John': 'Doe', 'Jane': 'Doe'})
        self.assertEqual(structure.John, 'Doe')
        self.assertEqual(structure['John'], 'Doe')

    def test_Structure_pickle(self):
        """
        Tests :class:`colour.utilities.data_structures.Structure` class
        pickling.
        """

        structure = Structure(John='Doe', Jane='Doe')

        data = pickle.dumps(structure)
        data = pickle.loads(data)
        self.assertEqual(structure, data)

        data = pickle.dumps(structure, pickle.HIGHEST_PROTOCOL)
        data = pickle.loads(data)
        self.assertEqual(structure, data)


class TestLookup(unittest.TestCase):
    """
    Defines :class:`colour.utilities.data_structures.Lookup` class unit tests
    methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('first_key_from_value',
                            'keys_from_value')

        for method in required_methods:
            self.assertIn(method, dir(Lookup))

    def test_first_key_from_value(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
Lookup.first_key_from_value` method.
        """

        lookup = Lookup(first_name='Doe', last_name='John', gender='male')
        self.assertEqual('first_name', lookup.first_key_from_value('Doe'))

    def test_keys_from_value(self):
        """
        Tests :meth:`colour.utilities.data_structures.Lookup.keys_from_value`
        method.
        """

        lookup = Lookup(John='Doe', Jane='Doe', Luke='Skywalker')
        self.assertListEqual(sorted(['Jane', 'John']),
                             sorted(lookup.keys_from_value('Doe')))


class TestCaseInsensitiveMapping(unittest.TestCase):
    """
    Defines :class:`colour.utilities.data_structures.CaseInsensitiveMapping`
    class unit tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__setitem__',
                            '__getitem__',
                            '__delitem__',
                            '__contains__',
                            '__iter__',
                            '__len__',
                            '__eq__',
                            '__ne__',
                            '__repr__',
                            'copy',
                            'lower_items')

        for method in required_methods:
            self.assertIn(method, dir(CaseInsensitiveMapping))

    def test__setitem__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__setitem__` method.
        """

        mapping = CaseInsensitiveMapping()

        mapping['John'] = 'Doe'
        self.assertEqual(mapping['John'], 'Doe')
        self.assertEqual(mapping['john'], 'Doe')

    def test__getitem__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__getitem__` method.
        """

        mapping = CaseInsensitiveMapping(John='Doe', Jane='Doe')

        self.assertEqual(mapping['John'], 'Doe')

        self.assertEqual(mapping['john'], 'Doe')

        self.assertEqual(mapping['Jane'], 'Doe')

        self.assertEqual(mapping['jane'], 'Doe')

    def test__delitem__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__delitem__` method.
        """

        mapping = CaseInsensitiveMapping(John='Doe', Jane='Doe')

        del mapping['john']
        self.assertNotIn('John', mapping)

        del mapping['Jane']
        self.assertNotIn('jane', mapping)
        self.assertEqual(len(mapping), 0)

    def test__contains__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__contains__` method.
        """

        mapping = CaseInsensitiveMapping(John='Doe', Jane='Doe')

        self.assertIn('John', mapping)

        self.assertIn('john', mapping)

        self.assertIn('Jane', mapping)

        self.assertIn('jane', mapping)

    def test__iter__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__iter__` method.
        """

        mapping = CaseInsensitiveMapping(John='Doe', Jane='Doe')
        self.assertListEqual(sorted([item for item in mapping]),
                             ['Jane', 'John'])

    def test__len__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__len__` method.
        """

        self.assertEqual(len(CaseInsensitiveMapping()), 0)

        self.assertEqual(len(CaseInsensitiveMapping(John='Doe', Jane='Doe')),
                         2)

    def test__eq__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__eq__` method.
        """

        mapping1 = CaseInsensitiveMapping(John='Doe', Jane='Doe')
        mapping2 = CaseInsensitiveMapping(John='Doe', Jane='Doe')
        mapping3 = CaseInsensitiveMapping(john='Doe', jane='Doe')

        self.assertEqual(mapping1, mapping2)

        self.assertEqual(mapping2, mapping3)

    def test__ne__(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__ne__` method.
        """

        mapping1 = CaseInsensitiveMapping(John='Doe', Jane='Doe')
        mapping2 = CaseInsensitiveMapping(Gi='Doe', Jane='Doe')

        self.assertNotEqual(mapping1, mapping2)

    def test_copy(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.copy` method.
        """

        mapping1 = CaseInsensitiveMapping(John='Doe', Jane='Doe')
        mapping2 = mapping1.copy()

        self.assertEqual(mapping1, mapping2)

        self.assertNotEqual(id(mapping1), id(mapping2))

    def test_lower_items(self):
        """
        Tests :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.lower_items` method.
        """

        mapping = CaseInsensitiveMapping(John='Doe', Jane='Doe')

        self.assertListEqual(sorted([item for item in mapping.lower_items()]),
                             [('jane', 'Doe'), ('john', 'Doe')])


if __name__ == '__main__':
    unittest.main()
