# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.utilities.common` module."""

from __future__ import annotations

import platform
import unicodedata
from functools import partial

import numpy as np
import pytest

from colour.hints import Any, Real, Tuple
from colour.utilities import (
    CacheRegistry,
    CanonicalMapping,
    as_bool,
    attest,
    batch,
    caching_enable,
    filter_kwargs,
    filter_mapping,
    first_item,
    int_digest,
    is_caching_enabled,
    is_integer,
    is_iterable,
    is_numeric,
    is_sibling,
    is_string,
    multiprocessing_pool,
    optional,
    set_caching_enable,
    slugify,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIsCachingEnabled",
    "TestSetCachingEnabled",
    "TestCachingEnable",
    "TestCacheRegistry",
    "TestAttest",
    "TestBatch",
    "TestMultiprocessingPool",
    "TestAsBool",
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
    "TestSlugify",
]


class TestIsCachingEnabled:
    """
    Define :func:`colour.utilities.common.is_caching_enabled` definition unit
    tests methods.
    """

    def test_is_caching_enabled(self):
        """Test :func:`colour.utilities.common.is_caching_enabled` definition."""

        with caching_enable(True):
            assert is_caching_enabled()

        with caching_enable(False):
            assert not is_caching_enabled()


class TestSetCachingEnabled:
    """
    Define :func:`colour.utilities.common.set_caching_enable` definition unit
    tests methods.
    """

    def test_set_caching_enable(self):
        """Test :func:`colour.utilities.common.set_caching_enable` definition."""

        with caching_enable(is_caching_enabled()):
            set_caching_enable(True)
            assert is_caching_enabled()

        with caching_enable(is_caching_enabled()):
            set_caching_enable(False)
            assert not is_caching_enabled()


class TestCachingEnable:
    """
    Define :func:`colour.utilities.common.caching_enable` definition unit
    tests methods.
    """

    def test_caching_enable(self):
        """Test :func:`colour.utilities.common.caching_enable` definition."""

        with caching_enable(True):
            assert is_caching_enabled()

        with caching_enable(False):
            assert not is_caching_enabled()

        @caching_enable(True)
        def fn_a():
            """:func:`caching_enable` unit tests :func:`fn_a` definition."""

            assert is_caching_enabled()

        fn_a()

        @caching_enable(False)
        def fn_b():
            """:func:`caching_enable` unit tests :func:`fn_b` definition."""

            assert not is_caching_enabled()

        fn_b()


class TestCacheRegistry:
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
            assert attribute in dir(CacheRegistry)

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
            assert method in dir(CacheRegistry)

    def test__str__(self):
        """Test :class:`colour.utilities.common.CacheRegistry.__str__` method."""

        cache_registry = self._default_test_cache_registry()
        assert str(cache_registry) == "{'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}"

    def test_register_cache(self):
        """
        Test :class:`colour.utilities.common.CacheRegistry.register_cache`
        method.
        """

        cache_registry = CacheRegistry()
        cache_a = cache_registry.register_cache("Cache A")
        assert cache_registry.registry == {"Cache A": cache_a}
        cache_b = cache_registry.register_cache("Cache B")
        assert cache_registry.registry == {"Cache A": cache_a, "Cache B": cache_b}

    def test_unregister_cache(self):
        """
        Test :class:`colour.utilities.common.CacheRegistry.unregister_cache`
        method.
        """

        cache_registry = self._default_test_cache_registry()
        cache_registry.unregister_cache("Cache A")
        assert "Cache A" not in cache_registry.registry
        assert "Cache B" in cache_registry.registry

    def test_clear_cache(self):
        """
        Test :class:`colour.utilities.common.CacheRegistry.clear_cache`
        method.
        """

        cache_registry = self._default_test_cache_registry()
        cache_registry.clear_cache("Cache A")
        assert cache_registry.registry == {
            "Cache A": {},
            "Cache B": {"John": "Doe", "Luke": "Skywalker"},
        }

    def test_clear_all_caches(self):
        """
        Test :class:`colour.utilities.common.CacheRegistry.clear_all_caches`
        method.
        """

        cache_registry = self._default_test_cache_registry()
        cache_registry.clear_all_caches()
        assert cache_registry.registry == {"Cache A": {}, "Cache B": {}}


class TestAttest:
    """
    Define :func:`colour.utilities.common.attest` definition unit
    tests methods.
    """

    def test_attest(self):
        """Test :func:`colour.utilities.common.attest` definition."""

        assert attest(True, "") is None

        pytest.raises(AssertionError, attest, False)


class TestBatch:
    """
    Define :func:`colour.utilities.common.batch` definition unit tests
    methods.
    """

    def test_batch(self):
        """Test :func:`colour.utilities.common.batch` definition."""

        assert list(batch(tuple(range(10)), 3)) == [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (9,),
        ]

        assert list(batch(tuple(range(10)), 5)) == [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)]

        assert list(batch(tuple(range(10)), 1)) == [
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
        ]


def _add(a: Real, b: Real):
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


class TestMultiprocessingPool:
    """
    Define :func:`colour.utilities.common.multiprocessing_pool` definition
    unit tests methods.
    """

    def test_multiprocessing_pool(self):
        """Test :func:`colour.utilities.common.multiprocessing_pool` definition."""

        with multiprocessing_pool() as pool:
            assert pool.map(partial(_add, b=2), range(10)) == [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            ]


class TestAsBool:
    """
    Define :func:`colour.utilities.common.as_bool` definition unit tests
    methods.
    """

    def test_as_bool(self):
        """Test :func:`colour.utilities.common.as_bool` definition."""

        assert as_bool("1")

        assert as_bool("On")

        assert as_bool("True")

        assert not as_bool("0")

        assert not as_bool("Off")

        assert not as_bool("False")

        assert not as_bool("")


class TestIsIterable:
    """
    Define :func:`colour.utilities.common.is_iterable` definition unit tests
    methods.
    """

    def test_is_iterable(self):
        """Test :func:`colour.utilities.common.is_iterable` definition."""

        assert is_iterable("")

        assert is_iterable(())

        assert is_iterable([])

        assert is_iterable({})

        assert is_iterable(set())

        assert is_iterable(np.array([]))

        assert not is_iterable(1)

        assert not is_iterable(2)

        generator = (a for a in range(10))
        assert is_iterable(generator)
        assert len(list(generator)) == 10


class TestIsString:
    """
    Define :func:`colour.utilities.common.is_string` definition unit tests
    methods.
    """

    def test_is_string(self):
        """Test :func:`colour.utilities.common.is_string` definition."""

        assert is_string("Hello World!")

        assert is_string("Hello World!")

        assert is_string(r"Hello World!")

        assert not is_string(1)

        assert not is_string([1])

        assert not is_string({1: None})


class TestIsNumeric:
    """
    Define :func:`colour.utilities.common.is_numeric` definition unit tests
    methods.
    """

    def test_is_numeric(self):
        """Test :func:`colour.utilities.common.is_numeric` definition."""

        assert is_numeric(1)

        assert is_numeric(1)

        assert not is_numeric((1,))

        assert not is_numeric([1])

        assert not is_numeric("1")


class TestIsInteger:
    """
    Define :func:`colour.utilities.common.is_integer` definition unit
    tests methods.
    """

    def test_is_integer(self):
        """Test :func:`colour.utilities.common.is_integer` definition."""

        assert is_integer(1)

        assert is_integer(1.001)

        assert not is_integer(1.01)


class TestIsSibling:
    """
    Define :func:`colour.utilities.common.is_sibling` definition unit tests
    methods.
    """

    def test_is_sibling(self):
        """Test :func:`colour.utilities.common.is_sibling` definition."""

        class Element:
            """:func:`is_sibling` unit tests :class:`Element` class."""

            def __init__(self, name: str) -> None:
                self.name = name

        class NotElement:
            """:func:`is_sibling` unit tests :class:`NotElement` class."""

            def __init__(self, name: str) -> None:
                self.name = name

        mapping = {
            "Element A": Element("A"),
            "Element B": Element("B"),
            "Element C": Element("C"),
        }

        assert is_sibling(Element("D"), mapping)

        assert not is_sibling(NotElement("Not D"), mapping)


class TestFilterKwargs:
    """
    Define :func:`colour.utilities.common.filter_kwargs` definition unit
    tests methods.
    """

    def test_filter_kwargs(self):
        """Test :func:`colour.utilities.common.filter_kwargs` definition."""

        def fn_a(a: Any) -> Any:
            """:func:`filter_kwargs` unit tests :func:`fn_a` definition."""

            return a

        def fn_b(a: Any, b: float = 0) -> Tuple[Any, float]:
            """:func:`filter_kwargs` unit tests :func:`fn_b` definition."""

            return a, b

        def fn_c(a: Any, b: float = 0, c: float = 0) -> Tuple[float, float, float]:
            """:func:`filter_kwargs` unit tests :func:`fn_c` definition."""

            return a, b, c

        assert fn_a(1, **filter_kwargs(fn_a, b=2, c=3)) == 1

        assert fn_b(1, **filter_kwargs(fn_b, b=2, c=3)) == (1, 2)

        assert fn_c(1, **filter_kwargs(fn_c, b=2, c=3)) == (1, 2, 3)

        assert filter_kwargs(partial(fn_c, b=1), b=1) == {"b": 1}


class TestFilterMapping:
    """
    Define :func:`colour.utilities.common.filter_mapping` definition unit
    tests methods.
    """

    def test_filter_mapping(self):
        """Test :func:`colour.utilities.common.filter_mapping` definition."""

        class Element:
            """:func:`filter_mapping` unit tests :class:`Element` class."""

            def __init__(self, name: str) -> None:
                self.name = name

        mapping = {
            "Element A": Element("A"),
            "Element B": Element("B"),
            "Element C": Element("C"),
            "Not Element C": Element("Not C"),
        }

        assert sorted(filter_mapping(mapping, "Element A")) == ["Element A"]

        assert filter_mapping(mapping, "Element") == {}

        mapping = CanonicalMapping(
            {
                "Element A": Element("A"),
                "Element B": Element("B"),
                "Element C": Element("C"),
                "Not Element C": Element("Not C"),
            }
        )

        assert sorted(filter_mapping(mapping, "element a")) == ["Element A"]

        assert sorted(filter_mapping(mapping, "element-a")) == ["Element A"]

        assert sorted(filter_mapping(mapping, "elementa")) == ["Element A"]


class TestFirstItem:
    """
    Define :func:`colour.utilities.common.first_item` definition unit
    tests methods.
    """

    def test_first_item(self):
        """Test :func:`colour.utilities.common.first_item` definition."""

        assert first_item(range(10)) == 0

        dictionary = {0: "a", 1: "b", 2: "c"}
        assert first_item(dictionary.items()) == (0, "a")

        assert first_item(dictionary.values()) == "a"


class TestValidateMethod:
    """
    Define :func:`colour.utilities.common.validate_method` definition unit
    tests methods.
    """

    def test_validate_method(self):
        """Test :func:`colour.utilities.common.validate_method` definition."""

        assert validate_method("Valid", ("Valid", "Yes", "Ok")) == "valid"
        assert (
            validate_method("Valid", ("Valid", "Yes", "Ok"), as_lowercase=False)
            == "Valid"
        )

    def test_raise_exception_validate_method(self):
        """
        Test :func:`colour.utilities.common.validate_method` definition raised
        exception.
        """

        pytest.raises(ValueError, validate_method, "Invalid", ("Valid", "Yes", "Ok"))


class TestOptional:
    """
    Define :func:`colour.utilities.common.optional` definition unit
    tests methods.
    """

    def test_optional(self):
        """Test :func:`colour.utilities.common.optional` definition."""

        assert optional("Foo", "Bar") == "Foo"

        assert optional(None, "Bar") == "Bar"


class TestSlugify:
    """
    Define :func:`colour.utilities.common.slugify` definition unit tests
    methods.
    """

    def test_slugify(self):
        """Test :func:`colour.utilities.common.slugify` definition."""

        assert (
            slugify(" Jack & Jill like numbers 1,2,3 and 4 and silly characters ?%.$!/")
            == "jack-jill-like-numbers-123-and-4-and-silly-characters"
        )

        assert (
            slugify("Un \xe9l\xe9phant \xe0 l'or\xe9e du bois")
            == "un-elephant-a-loree-du-bois"
        )

        # NOTE: Our "utilities/unicode_to_ascii.py" utility script normalises
        # the reference string.
        assert (
            unicodedata.normalize(
                "NFD",
                slugify(
                    "Un \xe9l\xe9phant \xe0 l'or\xe9e du bois",
                    allow_unicode=True,
                ),
            )
            == "un-éléphant-à-lorée-du-bois"
        )

        assert slugify(123) == "123"


class TestIntDigest:
    """
    Define :func:`colour.utilities.common.int_digest` definition unit tests
    methods.
    """

    def test_int_digest(self):
        """Test :func:`colour.utilities.common.int_digest` definition."""

        assert int_digest("Foo") == 7467386374397815550

        if platform.system() in ("Windows", "Microsoft"):  # pragma: no cover
            assert int_digest(np.array([1, 2, 3]).tobytes()) == 7764052002911021640
        else:
            assert int_digest(np.array([1, 2, 3]).tobytes()) == 8964613590703056768

        assert int_digest(repr((1, 2, 3))) == 5069958125469218295
