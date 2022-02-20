"""Defines the unit tests for the :mod:`colour.utilities.data_structures` module."""

import numpy as np
import operator
import pickle
import unittest

from colour.utilities import (
    Structure,
    Lookup,
    CaseInsensitiveMapping,
    LazyCaseInsensitiveMapping,
    Node,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestStructure",
    "TestLookup",
    "TestCaseInsensitiveMapping",
    "TestLazyCaseInsensitiveMapping",
    "TestNode",
]


class TestStructure(unittest.TestCase):
    """
    Define :class:`colour.utilities.data_structures.Structure` class unit
    tests methods.
    """

    def test_Structure(self):
        """Test :class:`colour.utilities.data_structures.Structure` class."""

        structure = Structure(John="Doe", Jane="Doe")
        self.assertIn("John", structure)
        self.assertTrue(hasattr(structure, "John"))

        setattr(structure, "John", "Nemo")
        self.assertEqual(structure["John"], "Nemo")

        structure["John"] = "Vador"
        self.assertEqual(structure["John"], "Vador")

        del structure["John"]
        self.assertNotIn("John", structure)
        self.assertFalse(hasattr(structure, "John"))

        structure.John = "Doe"
        self.assertIn("John", structure)
        self.assertTrue(hasattr(structure, "John"))

        del structure.John
        self.assertNotIn("John", structure)
        self.assertFalse(hasattr(structure, "John"))

        structure = Structure(John=None, Jane=None)
        self.assertIsNone(structure.John)
        self.assertIsNone(structure["John"])

        structure.update(**{"John": "Doe", "Jane": "Doe"})
        self.assertEqual(structure.John, "Doe")
        self.assertEqual(structure["John"], "Doe")

    def test_Structure_pickle(self):
        """
        Test :class:`colour.utilities.data_structures.Structure` class
        pickling.
        """

        structure = Structure(John="Doe", Jane="Doe")

        data = pickle.dumps(structure)
        data = pickle.loads(data)
        self.assertEqual(structure, data)

        data = pickle.dumps(structure, pickle.HIGHEST_PROTOCOL)
        data = pickle.loads(data)
        self.assertEqual(structure, data)

        self.assertEqual(sorted(dir(data)), ["Jane", "John"])


class TestLookup(unittest.TestCase):
    """
    Define :class:`colour.utilities.data_structures.Lookup` class unit tests
    methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("keys_from_value", "first_key_from_value")

        for method in required_methods:
            self.assertIn(method, dir(Lookup))

    def test_keys_from_value(self):
        """
        Test :meth:`colour.utilities.data_structures.Lookup.keys_from_value`
        method.
        """

        lookup = Lookup(John="Doe", Jane="Doe", Luke="Skywalker")
        self.assertListEqual(
            ["Jane", "John"], sorted(lookup.keys_from_value("Doe"))
        )

        lookup = Lookup(
            A=np.array([0, 1, 2]), B=np.array([0, 1, 2]), C=np.array([1, 2, 3])
        )
        self.assertListEqual(
            ["A", "B"], sorted(lookup.keys_from_value(np.array([0, 1, 2])))
        )

    def test_first_key_from_value(self):
        """
        Test :meth:`colour.utilities.data_structures.\
Lookup.first_key_from_value` method.
        """

        lookup = Lookup(first_name="John", last_name="Doe", gender="male")
        self.assertEqual("first_name", lookup.first_key_from_value("John"))

        lookup = Lookup(
            A=np.array([0, 1, 2]), B=np.array([1, 2, 3]), C=np.array([2, 3, 4])
        )
        self.assertEqual("A", lookup.first_key_from_value(np.array([0, 1, 2])))

    def test_raise_exception_first_key_from_value(self):
        """
        Test :meth:`colour.utilities.data_structures.\
Lookup.first_key_from_value` method raised exception.
        """

        self.assertRaises(IndexError, Lookup().first_key_from_value, "John")


class TestCaseInsensitiveMapping(unittest.TestCase):
    """
    Define :class:`colour.utilities.data_structures.CaseInsensitiveMapping`
    class unit tests methods.
    """

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("data",)

        for attribute in required_attributes:
            self.assertIn(attribute, dir(CaseInsensitiveMapping))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__repr__",
            "__setitem__",
            "__getitem__",
            "__delitem__",
            "__contains__",
            "__iter__",
            "__len__",
            "__eq__",
            "__ne__",
            "copy",
            "lower_items",
        )

        for method in required_methods:
            self.assertIn(method, dir(CaseInsensitiveMapping))

    def test_data(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.data` property.
        """

        self.assertDictEqual(
            CaseInsensitiveMapping({"John": "Doe", "Jane": "Doe"}).data,
            {"jane": ("Jane", "Doe"), "john": ("John", "Doe")},
        )

    def test__repr__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__repr__` method.
        """

        mapping = CaseInsensitiveMapping()

        mapping["John"] = "Doe"
        self.assertEqual(
            repr(mapping), "CaseInsensitiveMapping({'John': 'Doe'})"
        )

    def test__setitem__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__setitem__` method.
        """

        mapping = CaseInsensitiveMapping()

        mapping["John"] = "Doe"
        self.assertEqual(mapping["John"], "Doe")
        self.assertEqual(mapping["john"], "Doe")

    def test__getitem__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__getitem__` method.
        """

        mapping = CaseInsensitiveMapping(John="Doe", Jane="Doe")

        self.assertEqual(mapping["John"], "Doe")

        self.assertEqual(mapping["john"], "Doe")

        self.assertEqual(mapping["Jane"], "Doe")

        self.assertEqual(mapping["jane"], "Doe")

        mapping = CaseInsensitiveMapping({1: "Foo", 2: "Bar"})

        self.assertEqual(mapping[1], "Foo")

    def test__delitem__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__delitem__` method.
        """

        mapping = CaseInsensitiveMapping(John="Doe", Jane="Doe")

        del mapping["john"]
        self.assertNotIn("John", mapping)

        del mapping["Jane"]
        self.assertNotIn("jane", mapping)
        self.assertEqual(len(mapping), 0)

    def test__contains__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__contains__` method.
        """

        mapping = CaseInsensitiveMapping(John="Doe", Jane="Doe")

        self.assertIn("John", mapping)

        self.assertIn("john", mapping)

        self.assertIn("Jane", mapping)

        self.assertIn("jane", mapping)

    def test__iter__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__iter__` method.
        """

        mapping = CaseInsensitiveMapping(John="Doe", Jane="Doe")
        self.assertListEqual(
            sorted(item for item in mapping), ["Jane", "John"]
        )

    def test__len__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__len__` method.
        """

        self.assertEqual(len(CaseInsensitiveMapping()), 0)

        self.assertEqual(
            len(CaseInsensitiveMapping(John="Doe", Jane="Doe")), 2
        )

    def test__eq__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__eq__` method.
        """

        mapping1 = CaseInsensitiveMapping(John="Doe", Jane="Doe")
        mapping2 = CaseInsensitiveMapping(John="Doe", Jane="Doe")
        mapping3 = CaseInsensitiveMapping(john="Doe", jane="Doe")

        self.assertEqual(mapping1, mapping2)

        self.assertEqual(mapping2, mapping3)

    def test_raise_exception__eq__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__eq__` method raised exception.
        """

        self.assertRaises(
            ValueError,
            operator.eq,
            CaseInsensitiveMapping(John="Doe", Jane="Doe"),
            ["John", "Doe", "Jane", "Doe"],
        )

    def test__ne__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__ne__` method.
        """

        mapping1 = CaseInsensitiveMapping(John="Doe", Jane="Doe")
        mapping2 = CaseInsensitiveMapping(Gi="Doe", Jane="Doe")

        self.assertNotEqual(mapping1, mapping2)

    def test_raise_exception__ne__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.__ne__` method raised exception.
        """

        self.assertRaises(
            ValueError,
            operator.ne,
            CaseInsensitiveMapping(John="Doe", Jane="Doe"),
            ["John", "Doe", "Jane", "Doe"],
        )

    def test_copy(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.copy` method.
        """

        mapping1 = CaseInsensitiveMapping(John="Doe", Jane="Doe")
        mapping2 = mapping1.copy()

        self.assertEqual(mapping1, mapping2)

        self.assertNotEqual(id(mapping1), id(mapping2))

    def test_lower_items(self):
        """
        Test :meth:`colour.utilities.data_structures.\
CaseInsensitiveMapping.lower_items` method.
        """

        mapping = CaseInsensitiveMapping(John="Doe", Jane="Doe")

        self.assertListEqual(
            sorted(item for item in mapping.lower_items()),
            [("jane", "Doe"), ("john", "Doe")],
        )


class TestLazyCaseInsensitiveMapping(unittest.TestCase):
    """
    Define :class:`colour.utilities.data_structures.\
LazyCaseInsensitiveMapping` class unit tests methods.
    """

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ()

        for attribute in required_attributes:  # pragma: no cover
            self.assertIn(attribute, dir(LazyCaseInsensitiveMapping))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__getitem__",)

        for method in required_methods:
            self.assertIn(method, dir(LazyCaseInsensitiveMapping))

    def test__getitem__(self):
        """
        Test :meth:`colour.utilities.data_structures.\
LazyCaseInsensitiveMapping.__getitem__` method.
        """

        mapping = LazyCaseInsensitiveMapping(John="Doe", Jane=lambda: "Doe")

        self.assertEqual(mapping["John"], "Doe")

        self.assertEqual(mapping["john"], "Doe")

        self.assertEqual(mapping["Jane"], "Doe")

        self.assertEqual(mapping["jane"], "Doe")


class TestNode(unittest.TestCase):
    """
    Define :class:`colour.utilities.data_structures.Node` class unit tests
    methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._data = {"John": "Doe"}

        self._node_a = Node("Node A", data=self._data)
        self._node_b = Node("Node B", self._node_a)
        self._node_c = Node("Node C", self._node_a)
        self._node_d = Node("Node D", self._node_b)
        self._node_e = Node("Node E", self._node_b)
        self._node_f = Node("Node F", self._node_d)
        self._node_g = Node("Node G", self._node_f)
        self._node_h = Node("Node H", self._node_g)

        self._tree = self._node_a

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = (
            "name",
            "parent",
            "children",
            "id",
            "root",
            "leaves",
            "siblings",
            "data",
        )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Node))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__new__",
            "__init__",
            "__str__",
            "__len__",
            "is_root",
            "is_inner",
            "is_leaf",
            "walk",
            "render",
        )

        for method in required_methods:
            self.assertIn(method, dir(Node))

    def test_name(self):
        """Test :attr:`colour.utilities.data_structures.Node.name` property."""

        self.assertEqual(self._tree.name, "Node A")
        self.assertIn("Node#", Node().name)

    def test_parent(self):
        """Test :attr:`colour.utilities.data_structures.Node.parent` property."""

        self.assertIs(self._node_b.parent, self._node_a)
        self.assertIs(self._node_h.parent, self._node_g)

    def test_children(self):
        """Test :attr:`colour.utilities.data_structures.Node.children` property."""

        self.assertListEqual(
            self._node_a.children, [self._node_b, self._node_c]
        )

    def test_id(self):
        """Test :attr:`colour.utilities.data_structures.Node.id` property."""

        self.assertIsInstance(self._node_a.id, int)

    def test_root(self):
        """Test :attr:`colour.utilities.data_structures.Node.root` property."""

        self.assertIs(self._node_a.root, self._node_a)
        self.assertIs(self._node_f.root, self._node_a)
        self.assertIs(self._node_g.root, self._node_a)
        self.assertIs(self._node_h.root, self._node_a)

    def test_leaves(self):
        """Test :attr:`colour.utilities.data_structures.Node.leaves` property."""

        self.assertListEqual(list(self._node_h.leaves), [self._node_h])

        self.assertListEqual(
            list(self._node_a.leaves),
            [self._node_h, self._node_e, self._node_c],
        )

    def test_siblings(self):
        """Test :attr:`colour.utilities.data_structures.Node.siblings` property."""

        self.assertListEqual(list(self._node_a.siblings), [])

        self.assertListEqual(list(self._node_b.siblings), [self._node_c])

    def test_data(self):
        """Test :attr:`colour.utilities.data_structures.Node.data` property."""

        self.assertIs(self._node_a.data, self._data)

    def test__str__(self):
        """Test :attr:`colour.utilities.data_structures.Node.__str__` method."""

        self.assertIn("Node#", str(self._node_a))
        self.assertIn("{'John': 'Doe'})", str(self._node_a))

    def test__len__(self):
        """Test :attr:`colour.utilities.data_structures.Node.__len__` method."""

        self.assertEqual(len(self._node_a), 7)

    def test_is_root(self):
        """Test :attr:`colour.utilities.data_structures.Node.is_root` method."""

        self.assertTrue(self._node_a.is_root())
        self.assertFalse(self._node_b.is_root())
        self.assertFalse(self._node_c.is_root())
        self.assertFalse(self._node_h.is_root())

    def test_is_inner(self):
        """Test :attr:`colour.utilities.data_structures.Node.is_inner` method."""

        self.assertFalse(self._node_a.is_inner())
        self.assertTrue(self._node_b.is_inner())
        self.assertFalse(self._node_c.is_inner())
        self.assertFalse(self._node_h.is_inner())

    def test_is_leaf(self):
        """Test :attr:`colour.utilities.data_structures.Node.is_leaf` method."""

        self.assertFalse(self._node_a.is_leaf())
        self.assertFalse(self._node_b.is_leaf())
        self.assertTrue(self._node_c.is_leaf())
        self.assertTrue(self._node_h.is_leaf())

    def test_walk(self):
        """Test :attr:`colour.utilities.data_structures.Node.walk` method."""

        self.assertListEqual(
            list(self._node_a.walk()),
            [
                self._node_b,
                self._node_d,
                self._node_f,
                self._node_g,
                self._node_h,
                self._node_e,
                self._node_c,
            ],
        )

        self.assertListEqual(
            list(self._node_h.walk(ascendants=True)),
            [
                self._node_g,
                self._node_f,
                self._node_d,
                self._node_b,
                self._node_a,
            ],
        )

    def test_render(self):
        """Test :attr:`colour.utilities.data_structures.Node.render` method."""

        self.assertIsInstance(self._node_a.render(), str)


if __name__ == "__main__":
    unittest.main()
