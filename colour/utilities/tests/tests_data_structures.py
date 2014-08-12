#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines units tests for :mod:`colour.utilities.data_structures` module.
"""

from __future__ import unicode_literals

import pickle
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.utilities.data_structures

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2008 - 2014 - Colour Developers"
__license__ = "New BSD License - http://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["TestStructure",
           "TestLookup"]


class TestStructure(unittest.TestCase):
    """
    Defines :class:`colour.utilities.data_structures.Structure` class units
    tests methods.
    """

    def test_structure(self):
        """
        Tests :class:`colour.utilities.data_structures.Structure` class.
        """

        structure = colour.utilities.data_structures.Structure(John="Doe",
                                                               Jane="Doe")
        self.assertIn("John", structure)
        self.assertTrue(hasattr(structure, "John"))
        setattr(structure, "John", "Nemo")
        self.assertEqual(structure["John"], "Nemo")
        structure["John"] = "Vador"
        self.assertEqual(structure["John"], "Vador")
        del (structure["John"])
        self.assertNotIn("John", structure)
        self.assertFalse(hasattr(structure, "John"))
        structure.John = "Doe"
        self.assertIn("John", structure)
        self.assertTrue(hasattr(structure, "John"))
        del (structure.John)
        self.assertNotIn("John", structure)
        self.assertFalse(hasattr(structure, "John"))
        structure = colour.utilities.data_structures.Structure(John=None,
                                                               Jane=None)
        self.assertIsNone(structure.John)
        self.assertIsNone(structure["John"])
        structure.update(**{"John": "Doe", "Jane": "Doe"})
        self.assertEqual(structure.John, "Doe")
        self.assertEqual(structure["John"], "Doe")

    def test_structure_pickle(self):
        """
        Tests :class:`colour.utilities.data_structures.Structure` class
        pickling.
        """

        structure = colour.utilities.data_structures.Structure(John="Doe",
                                                               Jane="Doe")

        data = pickle.dumps(structure)
        data = pickle.loads(data)
        self.assertEqual(structure, data)

        data = pickle.dumps(structure, pickle.HIGHEST_PROTOCOL)
        data = pickle.loads(data)
        self.assertEqual(structure, data)


class TestLookup(unittest.TestCase):
    """
    Defines :class:`colour.utilities.data_structures.Lookup` class units tests
    methods.
    """

    def test_get_first_key_from_value(self):
        """
        Tests
        :meth:`colour.utilities.data_structures.Lookup.get_first_key_from_value`
        method.
        """

        lookup = colour.utilities.data_structures.Lookup(firstName="Doe",
                                                         lastName="John",
                                                         gender="male")
        self.assertEqual("firstName", lookup.get_first_key_from_value("Doe"))

    def test_get_keys_from_value(self):
        """
        Tests
        :meth:`colour.utilities.data_structures.Lookup.get_keys_from_value`
        method.
        """

        lookup = colour.utilities.data_structures.Lookup(John="Doe",
                                                         Jane="Doe",
                                                         Luke="Skywalker")
        self.assertListEqual(sorted(["Jane", "John"]),
                             sorted(lookup.get_keys_from_value("Doe")))


if __name__ == "__main__":
    unittest.main()
