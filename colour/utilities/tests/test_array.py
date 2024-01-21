# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.utilities.array` module."""

import unittest
from copy import deepcopy
from dataclasses import dataclass, field, fields

import numpy as np

from colour.constants import (
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    TOLERANCE_ABSOLUTE_TESTS,
)
from colour.hints import NDArray, Optional, Type, Union
from colour.utilities import (
    CACHE_REGISTRY,
    MixinDataclassArithmetic,
    MixinDataclassArray,
    MixinDataclassFields,
    MixinDataclassIterable,
    as_array,
    as_float,
    as_float_array,
    as_float_scalar,
    as_int,
    as_int_array,
    as_int_scalar,
    centroid,
    closest,
    closest_indexes,
    domain_range_scale,
    fill_nan,
    format_array_as_row,
    from_range_1,
    from_range_10,
    from_range_100,
    from_range_degrees,
    from_range_int,
    full,
    get_domain_range_scale,
    has_only_nan,
    in_array,
    index_along_last_axis,
    interval,
    is_ndarray_copy_enabled,
    is_networkx_installed,
    is_uniform,
    ndarray_copy,
    ndarray_copy_enable,
    ndarray_write,
    ones,
    orient,
    row_as_diagonal,
    set_default_float_dtype,
    set_default_int_dtype,
    set_domain_range_scale,
    set_ndarray_copy_enable,
    to_domain_1,
    to_domain_10,
    to_domain_100,
    to_domain_degrees,
    to_domain_int,
    tsplit,
    tstack,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMixinDataclassFields",
    "TestMixinDataclassIterable",
    "TestMixinDataclassArray",
    "TestMixinDataclassArithmetic",
    "TestAsArray",
    "TestAsInt",
    "TestAsFloat",
    "TestAsIntArray",
    "TestAsFloatArray",
    "TestAsIntScalar",
    "TestAsFloatScalar",
    "TestSetDefaultIntegerDtype",
    "TestSetDefaultFloatDtype",
    "TestGetDomainRangeScale",
    "TestSetDomainRangeScale",
    "TestDomainRangeScale",
    "TestToDomain1",
    "TestToDomain10",
    "TestToDomain100",
    "TestToDomainDegrees",
    "TestToDomainInt",
    "TestFromRange1",
    "TestFromRange10",
    "TestFromRange100",
    "TestFromRangeDegrees",
    "TestFromRangeInt",
    "TestIsNdarrayCopyEnabled",
    "TestSetNdarrayCopyEnabled",
    "TestNdarrayCopyEnable",
    "TestNdarrayCopy",
    "TestClosestIndexes",
    "TestClosest",
    "TestInterval",
    "TestIsUniform",
    "TestInArray",
    "TestTstack",
    "TestTsplit",
    "TestRowAsDiagonal",
    "TestOrient",
    "TestCentroid",
    "TestFillNan",
    "TestHasNanOnly",
    "TestNdarrayWrite",
    "TestZeros",
    "TestOnes",
    "TestFull",
    "TestIndexAlongLastAxis",
]


class TestMixinDataclassFields(unittest.TestCase):
    """
    Define :class:`colour.utilities.array.MixinDataclassFields` class unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        @dataclass
        class Data(MixinDataclassFields):
            a: str
            b: str
            c: str

        self._data: Data = Data(a="Foo", b="Bar", c="Baz")

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("fields",)

        for method in required_attributes:
            self.assertIn(method, dir(MixinDataclassFields))

    def test_fields(self):
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable._fields`
        method.
        """

        self.assertTupleEqual(
            self._data.fields,
            fields(self._data),
        )


class TestMixinDataclassIterable(unittest.TestCase):
    """
    Define :class:`colour.utilities.array.MixinDataclassIterable` class unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        @dataclass
        class Data(MixinDataclassIterable):
            a: str
            b: str
            c: str

        self._data: Data = Data(a="Foo", b="Bar", c="Baz")

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = (
            "keys",
            "values",
            "items",
        )

        for method in required_attributes:
            self.assertIn(method, dir(MixinDataclassIterable))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__iter__",)

        for method in required_methods:
            self.assertIn(method, dir(MixinDataclassIterable))

    def test__iter__(self):
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable.__iter__`
        method.
        """

        self.assertDictEqual(
            {key: value for key, value in self._data},  # noqa: C416
            {"a": "Foo", "b": "Bar", "c": "Baz"},
        )

    def test_keys(self):
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable.keys`
        method.
        """

        self.assertTupleEqual(
            tuple(self._data.keys),
            ("a", "b", "c"),
        )

    def test_values(self):
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable.values`
        method.
        """

        self.assertTupleEqual(
            tuple(self._data.values),
            ("Foo", "Bar", "Baz"),
        )

    def test_items(self):
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable.items`
        method.
        """

        self.assertTupleEqual(
            tuple(self._data.items),
            (("a", "Foo"), ("b", "Bar"), ("c", "Baz")),
        )


class TestMixinDataclassArray(unittest.TestCase):
    """
    Define :class:`colour.utilities.array.MixinDataclassArray` class unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        @dataclass
        class Data(MixinDataclassArray):
            a: Optional[Union[float, list, tuple, np.ndarray]] = field(
                default_factory=lambda: None
            )

            b: Optional[Union[float, list, tuple, np.ndarray]] = field(
                default_factory=lambda: None
            )

            c: Optional[Union[float, list, tuple, np.ndarray]] = field(
                default_factory=lambda: None
            )

        self._data: Data = Data(
            b=np.array([0.1, 0.2, 0.3]), c=np.array([0.4, 0.5, 0.6])
        )
        self._array: NDArray = np.array(
            [
                [np.nan, 0.1, 0.4],
                [np.nan, 0.2, 0.5],
                [np.nan, 0.3, 0.6],
            ]
        )

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__array__",)

        for method in required_methods:
            self.assertIn(method, dir(MixinDataclassArray))

    def test__array__(self):
        """
        Test :meth:`colour.utilities.array.MixinDataclassArray.__array__`
        method.
        """

        np.testing.assert_array_equal(np.array(self._data), self._array)

        self.assertEqual(
            np.array(self._data, dtype=DTYPE_INT_DEFAULT).dtype,
            DTYPE_INT_DEFAULT,
        )


class TestMixinDataclassArithmetic(unittest.TestCase):
    """
    Define :class:`colour.utilities.array.MixinDataclassArithmetic` class unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        @dataclass
        class Data(MixinDataclassArithmetic):
            a: Optional[Union[float, list, tuple, np.ndarray]] = field(
                default_factory=lambda: None
            )

            b: Optional[Union[float, list, tuple, np.ndarray]] = field(
                default_factory=lambda: None
            )

            c: Optional[Union[float, list, tuple, np.ndarray]] = field(
                default_factory=lambda: None
            )

        self._factory: Type[Data] = Data
        self._data: Data = Data(
            b=np.array([0.1, 0.2, 0.3]), c=np.array([0.4, 0.5, 0.6])
        )
        self._array: NDArray = np.array(
            [
                [np.nan, 0.1, 0.4],
                [np.nan, 0.2, 0.5],
                [np.nan, 0.3, 0.6],
            ]
        )

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__iadd__",
            "__add__",
            "__isub__",
            "__sub__",
            "__imul__",
            "__mul__",
            "__idiv__",
            "__div__",
            "__ipow__",
            "__pow__",
            "arithmetical_operation",
        )

        for method in required_methods:
            self.assertIn(method, dir(MixinDataclassArithmetic))

    def test_arithmetical_operation(self):
        """
        Test :meth:`colour.utilities.array.MixinDataclassArithmetic.\
arithmetical_operation` method.
        """

        np.testing.assert_allclose(
            np.array(self._data.arithmetical_operation(10, "+", False)),
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data.arithmetical_operation(10, "-", False)),
            self._array - 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data.arithmetical_operation(10, "*", False)),
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data.arithmetical_operation(10, "/", False)),
            self._array / 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data.arithmetical_operation(10, "**", False)),
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data + 10),
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data - 10),
            self._array - 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data * 10),
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data / 10),
            self._array / 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(self._data**10),
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        data = deepcopy(self._data)

        np.testing.assert_allclose(
            np.array(data.arithmetical_operation(10, "+", True)),
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(data.arithmetical_operation(10, "-", True)),
            self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(data.arithmetical_operation(10, "*", True)),
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(data.arithmetical_operation(10, "/", True)),
            self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(data.arithmetical_operation(10, "**", True)),
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        data = deepcopy(self._data)

        np.testing.assert_allclose(
            np.array(data.arithmetical_operation(self._array, "+", False)),
            data + self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            np.array(data.arithmetical_operation(data, "+", False)),
            data + data,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        data = self._factory(1, 2, 3)

        data += 1
        self.assertEqual(data.a, 2)

        data -= 1
        self.assertEqual(data.a, 1)

        data *= 2
        self.assertEqual(data.a, 2)

        data /= 2
        self.assertEqual(data.a, 1)

        data **= 0.5
        self.assertEqual(data.a, 1)


class TestAsArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_array` definition unit tests
    methods.
    """

    def test_as_array(self):
        """Test :func:`colour.utilities.array.as_array` definition."""

        np.testing.assert_equal(as_array([1, 2, 3]), np.array([1, 2, 3]))

        self.assertEqual(
            as_array([1, 2, 3], DTYPE_FLOAT_DEFAULT).dtype, DTYPE_FLOAT_DEFAULT
        )

        self.assertEqual(
            as_array([1, 2, 3], DTYPE_INT_DEFAULT).dtype, DTYPE_INT_DEFAULT
        )

        np.testing.assert_equal(
            as_array(dict(zip("abc", [1, 2, 3])).values()), np.array([1, 2, 3])
        )


class TestAsInt(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_int` definition unit tests
    methods.
    """

    def test_as_int(self):
        """Test :func:`colour.utilities.array.as_int` definition."""

        self.assertEqual(as_int(1), 1)

        self.assertEqual(as_int(np.array([1])).ndim, 1)

        self.assertEqual(as_int(np.array([[1]])).ndim, 2)

        np.testing.assert_array_equal(
            as_int(np.array([1.0, 2.0, 3.0])), np.array([1, 2, 3])
        )

        self.assertEqual(
            as_int(np.array([1.0, 2.0, 3.0])).dtype, DTYPE_INT_DEFAULT
        )

        self.assertIsInstance(as_int(1), DTYPE_INT_DEFAULT)


class TestAsFloat(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_float` definition unit tests
    methods.
    """

    def test_as_float(self):
        """Test :func:`colour.utilities.array.as_float` definition."""

        self.assertEqual(as_float(1), 1.0)

        self.assertEqual(as_float(np.array([1])).ndim, 1)

        self.assertEqual(as_float(np.array([[1]])).ndim, 2)

        np.testing.assert_allclose(
            as_float(np.array([1, 2, 3])),
            np.array([1.0, 2.0, 3.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        self.assertEqual(
            as_float(np.array([1, 2, 3])).dtype, DTYPE_FLOAT_DEFAULT
        )

        self.assertIsInstance(as_float(1), DTYPE_FLOAT_DEFAULT)


class TestAsIntArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_int_array` definition unit tests
    methods.
    """

    def test_as_int_array(self):
        """Test :func:`colour.utilities.array.as_int_array` definition."""

        np.testing.assert_equal(
            as_int_array([1.0, 2.0, 3.0]), np.array([1, 2, 3])
        )

        self.assertEqual(as_int_array([1, 2, 3]).dtype, DTYPE_INT_DEFAULT)


class TestAsFloatArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_float_array` definition unit tests
    methods.
    """

    def test_as_float_array(self):
        """Test :func:`colour.utilities.array.as_float_array` definition."""

        np.testing.assert_equal(as_float_array([1, 2, 3]), np.array([1, 2, 3]))

        self.assertEqual(as_float_array([1, 2, 3]).dtype, DTYPE_FLOAT_DEFAULT)


class TestAsIntScalar(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_int_scalar` definition unit tests
    methods.
    """

    def test_as_int_scalar(self):
        """Test :func:`colour.utilities.array.as_int_scalar` definition."""

        self.assertEqual(as_int_scalar(1.0), 1)

        self.assertEqual(as_int_scalar(1.0).dtype, DTYPE_INT_DEFAULT)


class TestAsFloatScalar(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_float_scalar` definition unit
    tests methods.
    """

    def test_as_float_scalar(self):
        """Test :func:`colour.utilities.array.as_float_scalar` definition."""

        self.assertEqual(as_float_scalar(1), 1.0)

        self.assertEqual(as_float_scalar(1).dtype, DTYPE_FLOAT_DEFAULT)


class TestSetDefaultIntegerDtype(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.set_default_int_dtype` definition unit
    tests methods.
    """

    def test_set_default_int_dtype(self):
        """
        Test :func:`colour.utilities.array.set_default_int_dtype` definition.
        """

        self.assertEqual(as_int_array(np.ones(3)).dtype, np.int64)

        set_default_int_dtype(np.int32)

        self.assertEqual(as_int_array(np.ones(3)).dtype, np.int32)

        set_default_int_dtype(np.int64)

        self.assertEqual(as_int_array(np.ones(3)).dtype, np.int64)

    def tearDown(self):
        """After tests actions."""

        set_default_int_dtype(np.int64)


class TestSetDefaultFloatDtype(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.set_default_float_dtype` definition unit
    tests methods.
    """

    def test_set_default_float_dtype(self):
        """
        Test :func:`colour.utilities.array.set_default_float_dtype`
        definition.
        """

        try:
            self.assertEqual(as_float_array(np.ones(3)).dtype, np.float64)

            set_default_float_dtype(np.float16)

            self.assertEqual(as_float_array(np.ones(3)).dtype, np.float16)

            set_default_float_dtype(np.float64)

            self.assertEqual(as_float_array(np.ones(3)).dtype, np.float64)
        finally:
            set_default_float_dtype(np.float64)

    def test_set_default_float_dtype_enforcement(self):
        """
        Test whether :func:`colour.utilities.array.set_default_float_dtype`
        effect is applied through most of *Colour* public API.
        """

        if not is_networkx_installed():  # pragma: no cover
            return

        from colour.appearance import (
            CAM_Specification_CAM16,
            CAM_Specification_CIECAM02,
            CAM_Specification_CIECAM16,
            CAM_Specification_Hellwig2022,
            CAM_Specification_Kim2009,
            CAM_Specification_ZCAM,
        )
        from colour.graph.conversion import (
            CONVERSION_SPECIFICATIONS_DATA,
            convert,
        )

        try:
            dtype = np.float32
            set_default_float_dtype(dtype)

            for source, target, _callable in CONVERSION_SPECIFICATIONS_DATA:
                if target in ("Hexadecimal", "Munsell Colour"):
                    continue

                # Spectral distributions are instantiated with float64 data and
                # spectral up-sampling optimization fails.
                if (
                    "Spectral Distribution" in (source, target)
                    or target == "Complementary Wavelength"
                    or target == "Dominant Wavelength"
                ):
                    continue

                a = np.array([(0.25, 0.5, 0.25), (0.25, 0.5, 0.25)])

                if source == "CAM16":
                    a = CAM_Specification_CAM16(J=0.25, M=0.5, h=0.25)

                if source == "CIECAM02":
                    a = CAM_Specification_CIECAM02(J=0.25, M=0.5, h=0.25)

                if source == "CIECAM16":
                    a = CAM_Specification_CIECAM16(J=0.25, M=0.5, h=0.25)

                if source == "Hellwig 2022":
                    a = CAM_Specification_Hellwig2022(J=0.25, M=0.5, h=0.25)

                if source == "Kim 2009":
                    a = CAM_Specification_Kim2009(J=0.25, M=0.5, h=0.25)

                if source == "ZCAM":
                    a = CAM_Specification_ZCAM(J=0.25, M=0.5, h=0.25)

                if source == "CMYK":
                    a = np.array(
                        [(0.25, 0.5, 0.25, 0.5), (0.25, 0.5, 0.25, 0.5)]
                    )

                if source == "Hexadecimal":
                    a = np.array(["#FFFFFF", "#FFFFFF"])

                if source == "CSS Color 3":
                    a = "aliceblue"

                if source == "Munsell Colour":
                    a = ["4.2YR 8.1/5.3", "4.2YR 8.1/5.3"]

                if source == "Wavelength":
                    a = 555

                if (
                    source.startswith("CCT")  # noqa: PIE810
                    or source.endswith(" xy")
                    or source.endswith(" uv")
                ):
                    a = np.array([(0.25, 0.5), (0.25, 0.5)])

                def dtype_getter(x):
                    """Dtype getter callable."""

                    for specification in (
                        "ATD95",
                        "CIECAM02",
                        "CAM16",
                        "Hellwig 2022",
                        "Hunt",
                        "Kim 2009",
                        "LLAB",
                        "Nayatani95",
                        "RLAB",
                        "ZCAM",
                    ):
                        if target.endswith(specification):  # noqa: B023
                            return getattr(x, fields(x)[0].name).dtype

                    return x.dtype

                self.assertEqual(
                    dtype_getter(convert(a, source, target)), dtype
                )
        finally:
            set_default_float_dtype(np.float64)

            CACHE_REGISTRY.clear_all_caches()


class TestGetDomainRangeScale(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.get_domain_range_scale` definition
    unit tests methods.
    """

    def test_get_domain_range_scale(self):
        """
        Test :func:`colour.utilities.common.get_domain_range_scale`
        definition.
        """

        with domain_range_scale("Reference"):
            self.assertEqual(get_domain_range_scale(), "reference")

        with domain_range_scale("1"):
            self.assertEqual(get_domain_range_scale(), "1")

        with domain_range_scale("100"):
            self.assertEqual(get_domain_range_scale(), "100")


class TestSetDomainRangeScale(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.set_domain_range_scale` definition
    unit tests methods.
    """

    def test_set_domain_range_scale(self):
        """
        Test :func:`colour.utilities.common.set_domain_range_scale`
        definition.
        """

        with domain_range_scale("Reference"):
            set_domain_range_scale("1")
            self.assertEqual(get_domain_range_scale(), "1")

        with domain_range_scale("Reference"):
            set_domain_range_scale("100")
            self.assertEqual(get_domain_range_scale(), "100")

        with domain_range_scale("1"):
            set_domain_range_scale("Reference")
            self.assertEqual(get_domain_range_scale(), "reference")

        self.assertRaises(
            ValueError, lambda: set_domain_range_scale("Invalid")
        )


class TestDomainRangeScale(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.domain_range_scale` definition
    unit tests methods.
    """

    def test_domain_range_scale(self):
        """
        Test :func:`colour.utilities.common.domain_range_scale`
        definition.
        """

        self.assertEqual(get_domain_range_scale(), "reference")

        with domain_range_scale("Reference"):
            self.assertEqual(get_domain_range_scale(), "reference")

        self.assertEqual(get_domain_range_scale(), "reference")

        with domain_range_scale("1"):
            self.assertEqual(get_domain_range_scale(), "1")

        self.assertEqual(get_domain_range_scale(), "reference")

        with domain_range_scale("100"):
            self.assertEqual(get_domain_range_scale(), "100")

        self.assertEqual(get_domain_range_scale(), "reference")

        def fn_a(a):
            """Change the domain-range scale for unit testing."""

            b = to_domain_10(a)

            b *= 2

            return from_range_100(b)

        with domain_range_scale("Reference"):
            with domain_range_scale("1"):
                with domain_range_scale("100"):
                    with domain_range_scale("Ignore"):
                        self.assertEqual(get_domain_range_scale(), "ignore")
                        self.assertEqual(fn_a(4), 8)

                    self.assertEqual(get_domain_range_scale(), "100")
                    self.assertEqual(fn_a(40), 8)

                self.assertEqual(get_domain_range_scale(), "1")
                self.assertEqual(fn_a(0.4), 0.08)

            self.assertEqual(get_domain_range_scale(), "reference")
            self.assertEqual(fn_a(4), 8)

        self.assertEqual(get_domain_range_scale(), "reference")

        @domain_range_scale("1")
        def fn_b(a):
            """Change the domain-range scale for unit testing."""

            b = to_domain_10(a)

            b *= 2

            return from_range_100(b)

        self.assertEqual(fn_b(10), 2.0)


class TestToDomain1(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_1` definition unit
    tests methods.
    """

    def test_to_domain_1(self):
        """Test :func:`colour.utilities.common.to_domain_1` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(to_domain_1(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(to_domain_1(1), 1)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_1(1), 0.01)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_1(1, np.pi), 1 / np.pi)

        with domain_range_scale("100"):
            self.assertEqual(
                to_domain_1(1, dtype=np.float16).dtype, np.float16
            )


class TestToDomain10(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_10` definition unit
    tests methods.
    """

    def test_to_domain_10(self):
        """Test :func:`colour.utilities.common.to_domain_10` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(to_domain_10(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(to_domain_10(1), 10)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_10(1), 0.1)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_10(1, np.pi), 1 / np.pi)

        with domain_range_scale("100"):
            self.assertEqual(
                to_domain_10(1, dtype=np.float16).dtype, np.float16
            )


class TestToDomain100(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_100` definition unit
    tests methods.
    """

    def test_to_domain_100(self):
        """Test :func:`colour.utilities.common.to_domain_100` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(to_domain_100(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(to_domain_100(1), 100)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_100(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(to_domain_100(1, np.pi), np.pi)

        with domain_range_scale("100"):
            self.assertEqual(
                to_domain_100(1, dtype=np.float16).dtype, np.float16
            )


class TestToDomainDegrees(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_degrees` definition unit
    tests methods.
    """

    def test_to_domain_degrees(self):
        """Test :func:`colour.utilities.common.to_domain_degrees` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(to_domain_degrees(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(to_domain_degrees(1), 360)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_degrees(1), 3.6)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_degrees(1, np.pi), np.pi / 100)

        with domain_range_scale("100"):
            self.assertEqual(
                to_domain_degrees(1, dtype=np.float16).dtype, np.float16
            )


class TestToDomainInt(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_int` definition unit
    tests methods.
    """

    def test_to_domain_int(self):
        """Test :func:`colour.utilities.common.to_domain_int` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(to_domain_int(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(to_domain_int(1), 255)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_int(1), 2.55)

        with domain_range_scale("100"):
            self.assertEqual(to_domain_int(1, 10), 10.23)

        with domain_range_scale("100"):
            self.assertEqual(
                to_domain_int(1, dtype=np.float16).dtype, np.float16
            )


class TestFromRange1(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_1` definition unit
    tests methods.
    """

    def test_from_range_1(self):
        """Test :func:`colour.utilities.common.from_range_1` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(from_range_1(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(from_range_1(1), 1)

        with domain_range_scale("100"):
            self.assertEqual(from_range_1(1), 100)

        with domain_range_scale("100"):
            self.assertEqual(from_range_1(1, np.pi), 1 * np.pi)


class TestFromRange10(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_10` definition unit
    tests methods.
    """

    def test_from_range_10(self):
        """Test :func:`colour.utilities.common.from_range_10` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(from_range_10(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(from_range_10(1), 0.1)

        with domain_range_scale("100"):
            self.assertEqual(from_range_10(1), 10)

        with domain_range_scale("100"):
            self.assertEqual(from_range_10(1, np.pi), 1 * np.pi)


class TestFromRange100(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_100` definition unit
    tests methods.
    """

    def test_from_range_100(self):
        """Test :func:`colour.utilities.common.from_range_100` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(from_range_100(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(from_range_100(1), 0.01)

        with domain_range_scale("100"):
            self.assertEqual(from_range_100(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(from_range_100(1, np.pi), 1 / np.pi)


class TestFromRangeDegrees(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_degrees` definition unit
    tests methods.
    """

    def test_from_range_degrees(self):
        """Test :func:`colour.utilities.common.from_range_degrees` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(from_range_degrees(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(from_range_degrees(1), 1 / 360)

        with domain_range_scale("100"):
            self.assertEqual(from_range_degrees(1), 1 / 3.6)

        with domain_range_scale("100"):
            self.assertEqual(from_range_degrees(1, np.pi), 1 / (np.pi / 100))


class TestFromRangeInt(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_int` definition unit
    tests methods.
    """

    def test_from_range_int(self):
        """Test :func:`colour.utilities.common.from_range_int` definition."""

        with domain_range_scale("Reference"):
            self.assertEqual(from_range_int(1), 1)

        with domain_range_scale("1"):
            self.assertEqual(from_range_int(1), 1 / 255)

        with domain_range_scale("100"):
            self.assertEqual(from_range_int(1), 1 / 2.55)

        with domain_range_scale("100"):
            self.assertEqual(from_range_int(1, 10), 1 / (1023 / 100))

        with domain_range_scale("100"):
            self.assertEqual(
                from_range_int(1, dtype=np.float16).dtype, np.float16
            )


class TestIsNdarrayCopyEnabled(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.is_ndarray_copy_enabled` definition
    unit tests methods.
    """

    def test_is_ndarray_copy_enabled(self):
        """
        Test :func:`colour.utilities.array.is_ndarray_copy_enabled` definition.
        """

        with ndarray_copy_enable(True):
            self.assertTrue(is_ndarray_copy_enabled())

        with ndarray_copy_enable(False):
            self.assertFalse(is_ndarray_copy_enabled())


class TestSetNdarrayCopyEnabled(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.set_ndarray_copy_enable` definition
    unit tests methods.
    """

    def test_set_ndarray_copy_enable(self):
        """
        Test :func:`colour.utilities.array.set_ndarray_copy_enable` definition.
        """

        with ndarray_copy_enable(is_ndarray_copy_enabled()):
            set_ndarray_copy_enable(True)
            self.assertTrue(is_ndarray_copy_enabled())

        with ndarray_copy_enable(is_ndarray_copy_enabled()):
            set_ndarray_copy_enable(False)
            self.assertFalse(is_ndarray_copy_enabled())


class TestNdarrayCopyEnable(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.ndarray_copy_enable` definition unit
    tests methods.
    """

    def test_ndarray_copy_enable(self):
        """
        Test :func:`colour.utilities.array.ndarray_copy_enable` definition.
        """

        with ndarray_copy_enable(True):
            self.assertTrue(is_ndarray_copy_enabled())

        with ndarray_copy_enable(False):
            self.assertFalse(is_ndarray_copy_enabled())

        @ndarray_copy_enable(True)
        def fn_a():
            """:func:`ndarray_copy_enable` unit tests :func:`fn_a` definition."""

            self.assertTrue(is_ndarray_copy_enabled())

        fn_a()

        @ndarray_copy_enable(False)
        def fn_b():
            """:func:`ndarray_copy_enable` unit tests :func:`fn_b` definition."""

            self.assertFalse(is_ndarray_copy_enabled())

        fn_b()


class TestNdarrayCopy(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.ndarray_copy` definition unit
    tests methods.
    """

    def test_ndarray_copy(self):
        """Test :func:`colour.utilities.array.ndarray_copy` definition."""

        a = np.linspace(0, 1, 10)
        with ndarray_copy_enable(True):
            self.assertNotEqual(id(ndarray_copy(a)), id(a))

        with ndarray_copy_enable(False):
            self.assertEqual(id(ndarray_copy(a)), id(a))


class TestClosestIndexes(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.closest_indexes` definition unit
    tests methods.
    """

    def test_closest_indexes(self):
        """Test :func:`colour.utilities.array.closest_indexes` definition."""

        a = np.array(
            [
                24.31357115,
                63.62396289,
                55.71528816,
                62.70988028,
                46.84480573,
                25.40026416,
            ]
        )

        self.assertEqual(closest_indexes(a, 63.05), 3)

        self.assertEqual(closest_indexes(a, 51.15), 4)

        self.assertEqual(closest_indexes(a, 24.90), 5)

        np.testing.assert_array_equal(
            closest_indexes(a, np.array([63.05, 51.15, 24.90])),
            np.array([3, 4, 5]),
        )


class TestClosest(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.closest` definition unit tests
    methods.
    """

    def test_closest(self):
        """Test :func:`colour.utilities.array.closest` definition."""

        a = np.array(
            [
                24.31357115,
                63.62396289,
                55.71528816,
                62.70988028,
                46.84480573,
                25.40026416,
            ]
        )

        self.assertEqual(closest(a, 63.05), 62.70988028)

        self.assertEqual(closest(a, 51.15), 46.84480573)

        self.assertEqual(closest(a, 24.90), 25.40026416)

        np.testing.assert_allclose(
            closest(a, np.array([63.05, 51.15, 24.90])),
            np.array([62.70988028, 46.84480573, 25.40026416]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestInterval(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.interval` definition unit tests
    methods.
    """

    def test_interval(self):
        """Test :func:`colour.utilities.array.interval` definition."""

        np.testing.assert_array_equal(interval(range(0, 10, 2)), np.array([2]))

        np.testing.assert_array_equal(
            interval(range(0, 10, 2), False), np.array([2, 2, 2, 2])
        )

        np.testing.assert_allclose(
            interval([1, 2, 3, 4, 6, 6.5]),
            np.array([0.5, 1.0, 2.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            interval([1, 2, 3, 4, 6, 6.5], False),
            np.array([1.0, 1.0, 1.0, 2.0, 0.5]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestIsUniform(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.is_uniform` definition unit tests
    methods.
    """

    def test_is_uniform(self):
        """Test :func:`colour.utilities.array.is_uniform` definition."""

        self.assertTrue(is_uniform(range(0, 10, 2)))

        self.assertFalse(is_uniform([1, 2, 3, 4, 6]))


class TestInArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.in_array` definition unit tests
    methods.
    """

    def test_in_array(self):
        """Test :func:`colour.utilities.array.in_array` definition."""

        self.assertTrue(
            np.array_equal(
                in_array(np.array([0.50, 0.60]), np.linspace(0, 10, 101)),
                np.array([True, True]),
            )
        )

        self.assertFalse(
            np.array_equal(
                in_array(np.array([0.50, 0.61]), np.linspace(0, 10, 101)),
                np.array([True, True]),
            )
        )

        self.assertTrue(
            np.array_equal(
                in_array(np.array([[0.50], [0.60]]), np.linspace(0, 10, 101)),
                np.array([[True], [True]]),
            )
        )

    def test_n_dimensional_in_array(self):
        """
        Test :func:`colour.utilities.array.in_array` definition n-dimensional
        support.
        """

        np.testing.assert_array_equal(
            in_array(np.array([0.50, 0.60]), np.linspace(0, 10, 101)).shape,
            np.array([2]),
        )

        np.testing.assert_array_equal(
            in_array(np.array([[0.50, 0.60]]), np.linspace(0, 10, 101)).shape,
            np.array([1, 2]),
        )

        np.testing.assert_array_equal(
            in_array(
                np.array([[0.50], [0.60]]), np.linspace(0, 10, 101)
            ).shape,
            np.array([2, 1]),
        )


class TestTstack(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.tstack` definition unit tests
    methods.
    """

    def test_tstack(self):
        """Test :func:`colour.utilities.array.tstack` definition."""

        a = 0
        np.testing.assert_array_equal(tstack([a, a, a]), np.array([0, 0, 0]))

        a = np.arange(0, 6)
        np.testing.assert_array_equal(
            tstack([a, a, a]),
            np.array(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5],
                ]
            ),
        )

        a = np.reshape(a, (1, 6))
        np.testing.assert_array_equal(
            tstack([a, a, a]),
            np.array(
                [
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2],
                        [3, 3, 3],
                        [4, 4, 4],
                        [5, 5, 5],
                    ]
                ]
            ),
        )

        a = np.reshape(a, (1, 2, 3))
        np.testing.assert_array_equal(
            tstack([a, a, a]),
            np.array(
                [
                    [
                        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                        [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                    ]
                ]
            ),
        )


class TestTsplit(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.tsplit` definition unit tests
    methods.
    """

    def test_tsplit(self):
        """Test :func:`colour.utilities.array.tsplit` definition."""

        a = np.array([0, 0, 0])
        np.testing.assert_array_equal(tsplit(a), np.array([0, 0, 0]))
        a = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
            ]
        )
        np.testing.assert_array_equal(
            tsplit(a),
            np.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                ]
            ),
        )

        a = np.array(
            [
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5],
                ],
            ]
        )
        np.testing.assert_array_equal(
            tsplit(a),
            np.array(
                [
                    [[0, 1, 2, 3, 4, 5]],
                    [[0, 1, 2, 3, 4, 5]],
                    [[0, 1, 2, 3, 4, 5]],
                ]
            ),
        )

        a = np.array(
            [
                [
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                ]
            ]
        )
        np.testing.assert_array_equal(
            tsplit(a),
            np.array(
                [
                    [[[0, 1, 2], [3, 4, 5]]],
                    [[[0, 1, 2], [3, 4, 5]]],
                    [[[0, 1, 2], [3, 4, 5]]],
                ]
            ),
        )


class TestRowAsDiagonal(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.row_as_diagonal` definition unit
    tests methods.
    """

    def test_row_as_diagonal(self):
        """Test :func:`colour.utilities.array.row_as_diagonal` definition."""

        np.testing.assert_allclose(
            row_as_diagonal(
                np.array(
                    [
                        [0.25891593, 0.07299478, 0.36586996],
                        [0.30851087, 0.37131459, 0.16274825],
                        [0.71061831, 0.67718718, 0.09562581],
                        [0.71588836, 0.76772047, 0.15476079],
                        [0.92985142, 0.22263399, 0.88027331],
                    ]
                )
            ),
            np.array(
                [
                    [
                        [0.25891593, 0.00000000, 0.00000000],
                        [0.00000000, 0.07299478, 0.00000000],
                        [0.00000000, 0.00000000, 0.36586996],
                    ],
                    [
                        [0.30851087, 0.00000000, 0.00000000],
                        [0.00000000, 0.37131459, 0.00000000],
                        [0.00000000, 0.00000000, 0.16274825],
                    ],
                    [
                        [0.71061831, 0.00000000, 0.00000000],
                        [0.00000000, 0.67718718, 0.00000000],
                        [0.00000000, 0.00000000, 0.09562581],
                    ],
                    [
                        [0.71588836, 0.00000000, 0.00000000],
                        [0.00000000, 0.76772047, 0.00000000],
                        [0.00000000, 0.00000000, 0.15476079],
                    ],
                    [
                        [0.92985142, 0.00000000, 0.00000000],
                        [0.00000000, 0.22263399, 0.00000000],
                        [0.00000000, 0.00000000, 0.88027331],
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestOrient(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.orient` definition unit tests
    methods.
    """

    def test_orient(self):
        """Test :func:`colour.utilities.array.orient` definition."""

        a = np.tile(np.arange(5), (5, 1))

        np.testing.assert_array_equal(
            orient(a, "Flip"),
            np.array(
                [
                    [4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0],
                ]
            ),
        )

        np.testing.assert_array_equal(
            orient(a, "Flop"),
            np.array(
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                ]
            ),
        )

        np.testing.assert_array_equal(
            orient(a, "90 CW"),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4],
                ]
            ),
        )

        np.testing.assert_array_equal(
            orient(a, "90 CCW"),
            np.array(
                [
                    [4, 4, 4, 4, 4],
                    [3, 3, 3, 3, 3],
                    [2, 2, 2, 2, 2],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ]
            ),
        )

        np.testing.assert_array_equal(
            orient(a, "180"),
            np.array(
                [
                    [4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0],
                    [4, 3, 2, 1, 0],
                ]
            ),
        )

        np.testing.assert_array_equal(orient(a), a)


class TestCentroid(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.centroid` definition unit tests
    methods.
    """

    def test_centroid(self):
        """Test :func:`colour.utilities.array.centroid` definition."""

        a = np.arange(5)
        np.testing.assert_array_equal(centroid(a), np.array([3]))

        a = np.tile(a, (5, 1))
        np.testing.assert_array_equal(centroid(a), np.array([2, 3]))

        a = np.tile(np.linspace(0, 1, 10), (10, 1))
        np.testing.assert_array_equal(centroid(a), np.array([4, 6]))

        a = tstack([a, a, a])
        np.testing.assert_array_equal(centroid(a), np.array([4, 6, 1]))


class TestFillNan(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.fill_nan` definition unit tests
    methods.
    """

    def test_fill_nan(self):
        """Test :func:`colour.utilities.array.fill_nan` definition."""

        a = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
        np.testing.assert_allclose(
            fill_nan(a),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            fill_nan(a, method="Constant", default=8.0),
            np.array([0.1, 0.2, 8.0, 0.4, 0.5]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestHasNanOnly(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.has_only_nan` definition unit tests
    methods.
    """

    def test_has_only_nan(self):
        """Test :func:`colour.utilities.array.has_only_nan` definition."""

        self.assertTrue(has_only_nan(None))

        self.assertTrue(has_only_nan([None, None]))

        self.assertFalse(has_only_nan([True, None]))

        self.assertFalse(has_only_nan([0.1, np.nan, 0.3]))


class TestNdarrayWrite(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.ndarray_write` definition unit tests
    methods.
    """

    def test_ndarray_write(self):
        """Test :func:`colour.utilities.array.ndarray_write` definition."""

        a = np.linspace(0, 1, 10)
        a.setflags(write=False)

        with self.assertRaises(ValueError):
            a += 1

        with ndarray_write(a):
            a += 1


class TestZeros(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.zeros` definition unit tests
    methods.
    """

    def test_zeros(self):
        """Test :func:`colour.utilities.array.zeros` definition."""

        np.testing.assert_equal(zeros(3), np.zeros(3))


class TestOnes(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.ones` definition unit tests
    methods.
    """

    def test_ones(self):
        """Test :func:`colour.utilities.array.ones` definition."""

        np.testing.assert_equal(ones(3), np.ones(3))


class TestFull(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.full` definition unit tests
    methods.
    """

    def test_full(self):
        """Test :func:`colour.utilities.array.full` definition."""

        np.testing.assert_equal(full(3, 0.5), np.full(3, 0.5))


class TestIndexAlongLastAxis(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.index_along_last_axis` definition
    unit tests methods.
    """

    def test_index_along_last_axis(self):
        """Test :func:`colour.utilities.array.index_along_last_axis` definition."""
        a = np.array(
            [
                [
                    [
                        [0.51090627, 0.86191718, 0.8687926],
                        [0.82738158, 0.80587656, 0.28285687],
                    ],
                    [
                        [0.84085977, 0.03851814, 0.06057988],
                        [0.94659267, 0.79308353, 0.30870888],
                    ],
                ],
                [
                    [
                        [0.50758436, 0.24066455, 0.20199051],
                        [0.4507304, 0.84189245, 0.81160878],
                    ],
                    [
                        [0.75421871, 0.88187494, 0.01612045],
                        [0.38777511, 0.58905552, 0.32970469],
                    ],
                ],
                [
                    [
                        [0.99285824, 0.738076, 0.0716432],
                        [0.35847844, 0.0367514, 0.18586322],
                    ],
                    [
                        [0.72674561, 0.0822759, 0.9771182],
                        [0.90644279, 0.09689787, 0.93483977],
                    ],
                ],
            ]
        )

        indexes = np.array(
            [[[0, 1], [0, 1]], [[2, 1], [2, 1]], [[2, 1], [2, 0]]]
        )

        np.testing.assert_equal(
            index_along_last_axis(a, indexes),
            np.array(
                [
                    [[0.51090627, 0.80587656], [0.84085977, 0.79308353]],
                    [[0.20199051, 0.84189245], [0.01612045, 0.58905552]],
                    [[0.0716432, 0.0367514], [0.9771182, 0.90644279]],
                ]
            ),
        )

    def test_compare_with_argmin_argmax(self):
        """
        Test :func:`colour.utilities.array.index_along_last_axis` definition
        by comparison with :func:`argmin` and :func:`argmax`.
        """

        a = np.random.random((2, 3, 4, 5, 6, 7))

        np.testing.assert_equal(
            index_along_last_axis(a, np.argmin(a, axis=-1)), np.min(a, axis=-1)
        )

        np.testing.assert_equal(
            index_along_last_axis(a, np.argmax(a, axis=-1)), np.max(a, axis=-1)
        )

    def test_exceptions(self):
        """
        Test :func:`colour.utilities.array.index_along_last_axis` definition
        handling of invalid inputs.
        """

        a = as_float_array([[11, 12], [21, 22]])

        # Bad shape
        with self.assertRaises(ValueError):
            indexes = np.array([0])
            index_along_last_axis(a, indexes)

        # Indexes out of range
        with self.assertRaises(IndexError):
            indexes = np.array([123, 456])
            index_along_last_axis(a, indexes)

        # Non-int indexes
        with self.assertRaises(IndexError):
            indexes = np.array([0.0, 0.0])
            index_along_last_axis(a, indexes)


class TestFormatArrayAsRow(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.format_array_as_row` definition unit
    tests methods.
    """

    def test_format_array_as_row(self):
        """Test :func:`colour.utilities.array.format_array_as_row` definition."""

        self.assertEqual(
            format_array_as_row([1.25, 2.5, 3.75]),
            "1.2500000 2.5000000 3.7500000",
        )

        self.assertEqual(
            format_array_as_row([1.25, 2.5, 3.75], 3),
            "1.250 2.500 3.750",
        )

        self.assertEqual(
            format_array_as_row([1.25, 2.5, 3.75], 3, ", "),
            "1.250, 2.500, 3.750",
        )


if __name__ == "__main__":
    unittest.main()
