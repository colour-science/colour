# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.utilities.data_structures` module."""

import operator
import pickle

import numpy as np
import pytest

from colour.utilities import (
    CanonicalMapping,
    ColourUsageWarning,
    LazyCanonicalMapping,
    Lookup,
    Node,
    Structure,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestStructure",
    "TestLookup",
    "TestCanonicalMapping",
    "TestLazyCanonicalMapping",
    "TestNode",
]


class TestStructure:
    """
    Define :class:`colour.utilities.data_structures.Structure` class unit
    tests methods.
    """

    def test_Structure(self):
        """Test :class:`colour.utilities.data_structures.Structure` class."""

        structure = Structure(John="Doe", Jane="Doe")
        assert "John" in structure
        assert hasattr(structure, "John")

        structure.John = "Nemo"
        assert structure["John"] == "Nemo"

        structure["John"] = "Vador"
        assert structure["John"] == "Vador"

        del structure["John"]
        assert "John" not in structure
        assert not hasattr(structure, "John")

        structure.John = "Doe"
        assert "John" in structure
        assert hasattr(structure, "John")

        del structure.John
        assert "John" not in structure
        assert not hasattr(structure, "John")

        structure = Structure(John=None, Jane=None)
        assert structure.John is None
        assert structure["John"] is None

        structure.update(**{"John": "Doe", "Jane": "Doe"})
        assert structure.John == "Doe"
        assert structure["John"] == "Doe"

    def test_pickling(self):
        """
        Test whether :class:`colour.utilities.data_structures.Structure` class
        can be pickled.
        """

        structure = Structure(John="Doe", Jane="Doe")

        data = pickle.dumps(structure)
        data = pickle.loads(data)  # noqa: S301
        assert structure == data

        data = pickle.dumps(structure, pickle.HIGHEST_PROTOCOL)
        data = pickle.loads(data)  # noqa: S301
        assert structure == data

        assert sorted(dir(data)) == ["Jane", "John"]


class TestLookup:
    """
    Define :class:`colour.utilities.data_structures.Lookup` class unit tests
    methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("keys_from_value", "first_key_from_value")

        for method in required_methods:
            assert method in dir(Lookup)

    def test_keys_from_value(self):
        """
        Test :meth:`colour.utilities.data_structures.Lookup.keys_from_value`
        method.
        """

        lookup = Lookup(John="Doe", Jane="Doe", Luke="Skywalker")
        assert ["Jane", "John"] == sorted(lookup.keys_from_value("Doe"))

        lookup = Lookup(
            A=np.array([0, 1, 2]), B=np.array([0, 1, 2]), C=np.array([1, 2, 3])
        )
        assert ["A", "B"] == sorted(lookup.keys_from_value(np.array([0, 1, 2])))

    def test_first_key_from_value(self):
        """
        Test :meth:`colour.utilities.data_structures.\
Lookup.first_key_from_value` method.
        """

        lookup = Lookup(first_name="John", last_name="Doe", gender="male")
        assert lookup.first_key_from_value("John") == "first_name"

        lookup = Lookup(
            A=np.array([0, 1, 2]), B=np.array([1, 2, 3]), C=np.array([2, 3, 4])
        )
        assert lookup.first_key_from_value(np.array([0, 1, 2])) == "A"

    def test_raise_exception_first_key_from_value(self):
        """
        Test :meth:`colour.utilities.data_structures.\
Lookup.first_key_from_value` method raised exception.
        """

        pytest.raises(IndexError, Lookup().first_key_from_value, "John")


class TestCanonicalMapping:
    """
    Define :class:`colour.utilities.data_structures.CanonicalMapping` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("data",)

        for attribute in required_attributes:
            assert attribute in dir(CanonicalMapping)

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
            "lower_keys",
            "lower_items",
            "slugified_keys",
            "slugified_items",
            "canonical_keys",
            "canonical_items",
        )

        for method in required_methods:
            assert method in dir(CanonicalMapping)

    def test_data(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.data`
        property.
        """

        assert CanonicalMapping({"John": "Doe", "Jane": "Doe"}).data == {
            "John": "Doe",
            "Jane": "Doe",
        }

    def test__repr__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.__repr__`
        method.
        """

        mapping = CanonicalMapping()

        mapping["John"] = "Doe"
        assert repr(mapping) == "CanonicalMapping({'John': 'Doe'})"

    def test__setitem__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
__setitem__` method.
        """

        mapping = CanonicalMapping()

        mapping["John"] = "Doe"
        assert mapping["John"] == "Doe"
        assert mapping["john"] == "Doe"

    def test__getitem__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
__getitem__` method.
        """

        mapping = CanonicalMapping(John="Doe", Jane="Doe")

        assert mapping["John"] == "Doe"
        assert mapping["john"] == "Doe"
        assert mapping["JOHN"] == "Doe"
        assert mapping["Jane"] == "Doe"
        assert mapping["jane"] == "Doe"
        assert mapping["JANE"] == "Doe"

        mapping = CanonicalMapping({1: "Foo", 2: "Bar"})

        assert mapping[1] == "Foo"

        mapping = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})

        assert mapping["mccamy-1992"] == 1
        assert mapping["hernandez-1999"] == 2
        assert mapping["mccamy1992"] == 1
        assert mapping["hernandez1999"] == 2

    def test__delitem__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
__delitem__` method.
        """

        mapping = CanonicalMapping(John="Doe", Jane="Doe")

        del mapping["john"]
        assert "John" not in mapping

        del mapping["Jane"]
        assert "jane" not in mapping
        assert len(mapping) == 0

        mapping = CanonicalMapping(John="Doe", Jane="Doe")
        del mapping["JOHN"]
        assert "John" not in mapping

        del mapping["jane"]
        assert "jane" not in mapping
        assert len(mapping) == 0

        mapping = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})

        del mapping["mccamy-1992"]
        assert "McCamy 1992" not in mapping

        del mapping["hernandez-1999"]
        assert "Hernandez 1999" not in mapping

        assert len(mapping) == 0

        mapping = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})

        del mapping["mccamy1992"]
        assert "McCamy 1992" not in mapping

        del mapping["hernandez1999"]
        assert "Hernandez 1999" not in mapping

        assert len(mapping) == 0

    def test__contains__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
__contains__` method.
        """

        mapping = CanonicalMapping(John="Doe", Jane="Doe")

        assert "John" in mapping
        assert "john" in mapping
        assert "JOHN" in mapping
        assert "Jane" in mapping
        assert "jane" in mapping
        assert "JANE" in mapping

        mapping = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})

        assert "mccamy-1992" in mapping
        assert "hernandez-1999" in mapping
        assert "mccamy1992" in mapping
        assert "hernandez1999" in mapping

    def test__iter__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.__iter__`
        method.
        """

        mapping = CanonicalMapping(John="Doe", Jane="Doe")
        assert sorted(item for item in mapping) == ["Jane", "John"]

    def test__len__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.__len__`
        method.
        """

        assert len(CanonicalMapping()) == 0

        assert len(CanonicalMapping(John="Doe", Jane="Doe")) == 2

    def test__eq__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.__eq__`
        method.
        """

        mapping1 = CanonicalMapping(John="Doe", Jane="Doe")
        mapping2 = CanonicalMapping(John="Doe", Jane="Doe")
        mapping3 = CanonicalMapping(john="Doe", jane="Doe")

        assert mapping1 == mapping2

        assert mapping2 != mapping3

    def test_raise_exception__eq__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.__eq__`
        method raised exception.
        """

        pytest.raises(
            TypeError,
            operator.eq,
            CanonicalMapping(John="Doe", Jane="Doe"),
            ["John", "Doe", "Jane", "Doe"],
        )

    def test__ne__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.__ne__`
        method.
        """

        mapping1 = CanonicalMapping(John="Doe", Jane="Doe")
        mapping2 = CanonicalMapping(Gi="Doe", Jane="Doe")

        assert mapping1 != mapping2

    def test_raise_exception__ne__(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.__ne__`
        method raised exception.
        """

        pytest.raises(
            TypeError,
            operator.ne,
            CanonicalMapping(John="Doe", Jane="Doe"),
            ["John", "Doe", "Jane", "Doe"],
        )

    def test_copy(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.copy`
        method.
        """

        mapping1 = CanonicalMapping(John="Doe", Jane="Doe")
        mapping2 = mapping1.copy()

        assert mapping1 == mapping2

        assert id(mapping1) != id(mapping2)

    def test_lower_keys(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
lower_keys` method.
        """

        mapping = CanonicalMapping(John="Doe", Jane="Doe")

        assert sorted(item for item in mapping.lower_keys()) == ["jane", "john"]

        mapping = CanonicalMapping(John="Doe", john="Doe")

        pytest.warns(ColourUsageWarning, lambda: list(mapping.lower_keys()))

    def test_lower_items(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
lower_items` method.
        """

        mapping = CanonicalMapping(John="Doe", Jane="Doe")

        assert sorted(item for item in mapping.lower_items()) == [
            ("jane", "Doe"),
            ("john", "Doe"),
        ]

    def test_slugified_keys(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
slugified_keys` method.
        """

        mapping = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})

        assert sorted(item for item in mapping.slugified_keys()) == [
            "hernandez-1999",
            "mccamy-1992",
        ]

        mapping = CanonicalMapping({"McCamy 1992": 1, "McCamy-1992": 2})

        pytest.warns(ColourUsageWarning, lambda: list(mapping.slugified_keys()))

    def test_slugified_items(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
slugified_items` method.
        """

        mapping = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})
        assert sorted(item for item in mapping.slugified_items()) == [
            ("hernandez-1999", 2),
            ("mccamy-1992", 1),
        ]

    def test_canonical_keys(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
canonical_keys` method.
        """

        mapping = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})

        assert sorted(item for item in mapping.canonical_keys()) == [
            "hernandez1999",
            "mccamy1992",
        ]

        mapping = CanonicalMapping({"McCamy_1992": 1, "McCamy-1992": 2})

        pytest.warns(ColourUsageWarning, lambda: list(mapping.canonical_keys()))

    def test_canonical_items(self):
        """
        Test :meth:`colour.utilities.data_structures.CanonicalMapping.\
canonical_items` method.
        """

        mapping = CanonicalMapping({"McCamy 1992": 1, "Hernandez 1999": 2})
        assert sorted(item for item in mapping.canonical_items()) == [
            ("hernandez1999", 2),
            ("mccamy1992", 1),
        ]


class TestLazyCanonicalMapping:
    """
    Define :class:`colour.utilities.data_structures.LazyCanonicalMapping` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ()

        for attribute in required_attributes:  # pragma: no cover
            assert attribute in dir(LazyCanonicalMapping)

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__getitem__",)

        for method in required_methods:
            assert method in dir(LazyCanonicalMapping)

    def test__getitem__(self):
        """
        Test :meth:`colour.utilities.data_structures.LazyCanonicalMapping.\
__getitem__` method.
        """

        mapping = LazyCanonicalMapping(John="Doe", Jane=lambda: "Doe")

        assert mapping["John"] == "Doe"
        assert mapping["john"] == "Doe"
        assert mapping["Jane"] == "Doe"
        assert mapping["jane"] == "Doe"


class TestNode:
    """
    Define :class:`colour.utilities.data_structures.Node` class unit tests
    methods.
    """

    def setup_method(self):
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
            assert attribute in dir(Node)

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
            assert method in dir(Node)

    def test_name(self):
        """Test :attr:`colour.utilities.data_structures.Node.name` property."""

        assert self._tree.name == "Node A"
        assert "Node#" in Node().name

    def test_parent(self):
        """Test :attr:`colour.utilities.data_structures.Node.parent` property."""

        assert self._node_b.parent is self._node_a
        assert self._node_h.parent is self._node_g

    def test_children(self):
        """Test :attr:`colour.utilities.data_structures.Node.children` property."""

        assert self._node_a.children == [self._node_b, self._node_c]

    def test_id(self):
        """Test :attr:`colour.utilities.data_structures.Node.id` property."""

        assert isinstance(self._node_a.id, int)

    def test_root(self):
        """Test :attr:`colour.utilities.data_structures.Node.root` property."""

        assert self._node_a.root is self._node_a
        assert self._node_f.root is self._node_a
        assert self._node_g.root is self._node_a
        assert self._node_h.root is self._node_a

    def test_leaves(self):
        """Test :attr:`colour.utilities.data_structures.Node.leaves` property."""

        assert list(self._node_h.leaves) == [self._node_h]

        assert list(self._node_a.leaves) == [self._node_h, self._node_e, self._node_c]

    def test_siblings(self):
        """Test :attr:`colour.utilities.data_structures.Node.siblings` property."""

        assert list(self._node_a.siblings) == []

        assert list(self._node_b.siblings) == [self._node_c]

    def test_data(self):
        """Test :attr:`colour.utilities.data_structures.Node.data` property."""

        assert self._node_a.data is self._data

    def test__str__(self):
        """Test :attr:`colour.utilities.data_structures.Node.__str__` method."""

        assert "Node#" in str(self._node_a)
        assert "{'John': 'Doe'})" in str(self._node_a)

    def test__len__(self):
        """Test :attr:`colour.utilities.data_structures.Node.__len__` method."""

        assert len(self._node_a) == 7

    def test_is_root(self):
        """Test :attr:`colour.utilities.data_structures.Node.is_root` method."""

        assert self._node_a.is_root()
        assert not self._node_b.is_root()
        assert not self._node_c.is_root()
        assert not self._node_h.is_root()

    def test_is_inner(self):
        """Test :attr:`colour.utilities.data_structures.Node.is_inner` method."""

        assert not self._node_a.is_inner()
        assert self._node_b.is_inner()
        assert not self._node_c.is_inner()
        assert not self._node_h.is_inner()

    def test_is_leaf(self):
        """Test :attr:`colour.utilities.data_structures.Node.is_leaf` method."""

        assert not self._node_a.is_leaf()
        assert not self._node_b.is_leaf()
        assert self._node_c.is_leaf()
        assert self._node_h.is_leaf()

    def test_walk(self):
        """Test :attr:`colour.utilities.data_structures.Node.walk` method."""

        assert list(self._node_a.walk()) == [
            self._node_b,
            self._node_d,
            self._node_f,
            self._node_g,
            self._node_h,
            self._node_e,
            self._node_c,
        ]

        assert list(self._node_h.walk(ascendants=True)) == [
            self._node_g,
            self._node_f,
            self._node_d,
            self._node_b,
            self._node_a,
        ]

    def test_render(self):
        """Test :attr:`colour.utilities.data_structures.Node.render` method."""

        assert isinstance(self._node_a.render(), str)
