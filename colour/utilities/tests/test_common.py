"""Defines the unit tests for the :mod:`colour.utilities.common` module."""

from __future__ import annotations

import numpy as np
import unittest
from functools import partial

from colour.hints import Any, Floating, Number, Tuple
from colour.utilities import (
    CacheRegistry,
    attest,
    batch,
    multiprocessing_pool,
    is_iterable,
    is_string,
    is_numeric,
    is_integer,
    is_sibling,
    filter_kwargs,
    filter_mapping,
    first_item,
    validate_method,
    optional,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestCacheRegistry",
    "TestAttest",
    "TestBatch",
    "TestMultiprocessingPool",
    "TestIsIterable",
    "TestIsString",
    "TestIsNumeric",
    "TestIsInteger",
    "TestIsSibling",
    "TestFilterKwargs",
    "TestFilterMapping",
    "TestFirstItem",
    "TestValidateMethod",
    "TestOptional",
]


class TestCacheRegistry(unittest.TestCase):
    """
    Define :class:`colour.utilities.common.CacheRegistry` class unit
    tests methods.
    """

    @staticmethod
    def _default_test_cache_registry():
        """Create a default test cache registry."""

        cache_registry = CacheRegistry()
        cache_a = cache_registry.register_cache("Cache A")
        cache_a["Foo"] = "Bar"
        cache_b = cache_registry.register_cache("Cache B")
        cache_b["John"] = "Doe"
        cache_b["Luke"] = "Skywalker"

        return cache_registry

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("registry",)

        for attribute in required_attributes:
            self.assertIn(attribute, dir(CacheRegistry))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "register_cache",
            "unregister_cache",
            "clear_cache",
            "clear_all_caches",
        )

        for method in required_methods:
            self.assertIn(method, dir(CacheRegistry))

    def test__str__(self):
        """Test :class:`colour.utilities.common.CacheRegistry.__str__` method."""

        cache_registry = self._default_test_cache_registry()
        self.assertEqual(
            str(cache_registry),
            "{'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}",
        )

    def test_register_cache(self):
        """
        Test :class:`colour.utilities.common.CacheRegistry.register_cache`
        method.
        """

        cache_registry = CacheRegistry()
        cache_a = cache_registry.register_cache("Cache A")
        self.assertDictEqual(cache_registry.registry, {"Cache A": cache_a})
        cache_b = cache_registry.register_cache("Cache B")
        self.assertDictEqual(
            cache_registry.registry, {"Cache A": cache_a, "Cache B": cache_b}
        )

    def test_unregister_cache(self):
        """
        Test :class:`colour.utilities.common.CacheRegistry.unregister_cache`
        method.
        """

        cache_registry = self._default_test_cache_registry()
        cache_registry.unregister_cache("Cache A")
        self.assertNotIn("Cache A", cache_registry.registry)
        self.assertIn("Cache B", cache_registry.registry)

    def test_clear_cache(self):
        """
        Test :class:`colour.utilities.common.CacheRegistry.clear_cache`
        method.
        """

        cache_registry = self._default_test_cache_registry()
        cache_registry.clear_cache("Cache A")
        self.assertDictEqual(
            cache_registry.registry,
            {"Cache A": {}, "Cache B": {"John": "Doe", "Luke": "Skywalker"}},
        )

    def test_clear_all_caches(self):
        """
        Test :class:`colour.utilities.common.CacheRegistry.clear_all_caches`
        method.
        """

        cache_registry = self._default_test_cache_registry()
        cache_registry.clear_all_caches()
        self.assertDictEqual(
            cache_registry.registry, {"Cache A": {}, "Cache B": {}}
        )


class TestAttest(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.attest` definition unit
    tests methods.
    """

    def test_attest(self):
        """Test :func:`colour.utilities.common.attest` definition."""

        self.assertIsNone(attest(True, ""))

        self.assertRaises(AssertionError, attest, False)


class TestBatch(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.batch` definition unit tests
    methods.
    """

    def test_batch(self):
        """Test :func:`colour.utilities.common.batch` definition."""

        self.assertListEqual(
            list(batch(tuple(range(10)), 3)),
            [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)],
        )

        self.assertListEqual(
            list(batch(tuple(range(10)), 5)),
            [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)],
        )

        self.assertListEqual(
            list(batch(tuple(range(10)), 1)),
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)],
        )


def _add(a: Number, b: Number):
    """
    Add two numbers.

    This definition is intended to be used with a multiprocessing pool for unit
    testing.

    Parameters
    ----------
    a
        Variable :math:`a`.
    b
        Variable :math:`b`.

    Returns
    -------
    numeric
        Addition result.
    """

    # NOTE: No coverage information is available as this code is executed in
    # sub-processes.
    return a + b  # pragma: no cover


class TestMultiprocessingPool(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.multiprocessing_pool` definition
    unit tests methods.
    """

    def test_multiprocessing_pool(self):
        """Test :func:`colour.utilities.common.multiprocessing_pool` definition."""

        with multiprocessing_pool() as pool:
            self.assertListEqual(
                pool.map(partial(_add, b=2), range(10)),
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            )


class TestIsIterable(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.is_iterable` definition unit tests
    methods.
    """

    def test_is_iterable(self):
        """Test :func:`colour.utilities.common.is_iterable` definition."""

        self.assertTrue(is_iterable(""))

        self.assertTrue(is_iterable(()))

        self.assertTrue(is_iterable([]))

        self.assertTrue(is_iterable(dict()))

        self.assertTrue(is_iterable(set()))

        self.assertTrue(is_iterable(np.array([])))

        self.assertFalse(is_iterable(1))

        self.assertFalse(is_iterable(2))

        generator = (a for a in range(10))
        self.assertTrue(is_iterable(generator))
        self.assertEqual(len(list(generator)), 10)


class TestIsString(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.is_string` definition unit tests
    methods.
    """

    def test_is_string(self):
        """Test :func:`colour.utilities.common.is_string` definition."""

        self.assertTrue(is_string("Hello World!"))

        self.assertTrue(is_string("Hello World!"))

        self.assertTrue(is_string(r"Hello World!"))

        self.assertFalse(is_string(1))

        self.assertFalse(is_string([1]))

        self.assertFalse(is_string({1: None}))


class TestIsNumeric(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.is_numeric` definition unit tests
    methods.
    """

    def test_is_numeric(self):
        """Test :func:`colour.utilities.common.is_numeric` definition."""

        self.assertTrue(is_numeric(1))

        self.assertTrue(is_numeric(1))

        self.assertFalse(is_numeric((1,)))

        self.assertFalse(is_numeric([1]))

        self.assertFalse(is_numeric("1"))


class TestIsInteger(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.is_integer` definition unit
    tests methods.
    """

    def test_is_integer(self):
        """Test :func:`colour.utilities.common.is_integer` definition."""

        self.assertTrue(is_integer(1))

        self.assertTrue(is_integer(1.001))

        self.assertFalse(is_integer(1.01))


class TestIsSibling(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.is_sibling` definition unit tests
    methods.
    """

    def test_is_sibling(self):
        """Test :func:`colour.utilities.common.is_sibling` definition."""

        class Element:
            """:func:`is_sibling` unit tests :class:`Element` class."""

            def __init__(self, name):
                self.name = name

        class NotElement:
            """:func:`is_sibling` unit tests :class:`NotElement` class."""

            def __init__(self, name):
                self.name = name

        mapping = {
            "Element A": Element("A"),
            "Element B": Element("B"),
            "Element C": Element("C"),
        }

        self.assertTrue(is_sibling(Element("D"), mapping))

        self.assertFalse(is_sibling(NotElement("Not D"), mapping))


class TestFilterKwargs(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.filter_kwargs` definition unit
    tests methods.
    """

    def test_filter_kwargs(self):
        """Test :func:`colour.utilities.common.filter_kwargs` definition."""

        def fn_a(a: Any) -> Any:
            """:func:`filter_kwargs` unit tests :func:`fn_a` definition."""

            return a

        def fn_b(a: Any, b: Floating = 0) -> Tuple[Any, Floating]:
            """:func:`filter_kwargs` unit tests :func:`fn_b` definition."""

            return a, b

        def fn_c(
            a: Any, b: Floating = 0, c: Floating = 0
        ) -> Tuple[Any, Floating, Floating]:
            """:func:`filter_kwargs` unit tests :func:`fn_c` definition."""

            return a, b, c

        self.assertEqual(1, fn_a(1, **filter_kwargs(fn_a, b=2, c=3)))

        self.assertTupleEqual((1, 2), fn_b(1, **filter_kwargs(fn_b, b=2, c=3)))

        self.assertTupleEqual(
            (1, 2, 3), fn_c(1, **filter_kwargs(fn_c, b=2, c=3))
        )

        self.assertDictEqual(filter_kwargs(partial(fn_c, b=1), b=1), {"b": 1})


class TestFilterMapping(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.filter_mapping` definition unit
    tests methods.
    """

    def test_filter_mapping(self):
        """Test :func:`colour.utilities.common.filter_mapping` definition."""

        class Element:
            """:func:`filter_mapping` unit tests :class:`Element` class."""

            def __init__(self, name):
                self.name = name

        mapping = {
            "Element A": Element("A"),
            "Element B": Element("B"),
            "Element C": Element("C"),
            "Not Element C": Element("Not C"),
        }

        self.assertListEqual(
            sorted(filter_mapping(mapping, "\\w+\\s+A")), ["Element A"]
        )

        self.assertListEqual(
            sorted(filter_mapping(mapping, "Element.*")),
            [
                "Element A",
                "Element B",
                "Element C",
            ],
        )

        self.assertListEqual(
            sorted(filter_mapping(mapping, "^Element.*")),
            [
                "Element A",
                "Element B",
                "Element C",
            ],
        )

        self.assertListEqual(
            sorted(filter_mapping(mapping, "^Element.*", False)),
            [
                "Element A",
                "Element B",
                "Element C",
            ],
        )

        self.assertListEqual(
            sorted(filter_mapping(mapping, [".*A", ".*B"])),
            [
                "Element A",
                "Element B",
            ],
        )

        self.assertIsInstance(
            filter_mapping(mapping, "^Element.*", False), type(mapping)
        )


class TestFirstItem(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.first_item` definition unit
    tests methods.
    """

    def test_first_item(self):
        """Test :func:`colour.utilities.common.first_item` definition."""

        self.assertEqual(first_item(range(10)), 0)

        dictionary = {0: "a", 1: "b", 2: "c"}
        self.assertEqual(first_item(dictionary.items()), (0, "a"))

        self.assertEqual(first_item(dictionary.values()), "a")


class TestValidateMethod(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.validate_method` definition unit
    tests methods.
    """

    def test_validate_method(self):
        """Test :func:`colour.utilities.common.validate_method` definition."""

        self.assertEqual(
            validate_method("Valid", ["Valid", "Yes", "Ok"]), "valid"
        )

    def test_raise_exception_validate_method(self):
        """
        Test :func:`colour.utilities.common.validate_method` definition raised
        exception.
        """

        self.assertRaises(
            ValueError, validate_method, "Invalid", ["Valid", "Yes", "Ok"]
        )


class TestOptional(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.optional` definition unit
    tests methods.
    """

    def test_optional(self):
        """Test :func:`colour.utilities.common.optional` definition."""

        self.assertEqual(optional("Foo", "Bar"), "Foo")

        self.assertEqual(optional(None, "Bar"), "Bar")


if __name__ == "__main__":
    unittest.main()
