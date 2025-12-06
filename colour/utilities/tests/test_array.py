"""Define the unit tests for the :mod:`colour.utilities.array` module."""

from __future__ import annotations

import typing
import unittest
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import partial

import numpy as np
import pytest

from colour.constants import (
    DTYPE_COMPLEX_DEFAULT,
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    TOLERANCE_ABSOLUTE_TESTS,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        Annotated,
        Any,
        ArrayLike,
        Domain1,
        Domain10,
        Domain100,
        Domain100_100_360,
        Domain360,
        DType,
        NDArray,
        NDArrayFloat,
        Range1,
        Range10,
        Range100,
        Range100_100_360,
        Range360,
        Type,
    )
else:
    # Import Annotated at runtime for test helper function signatures
    # get_domain_range_scale_metadata() needs to access Annotated.__metadata__
    from colour.hints import (  # noqa: TC001
        Annotated,
        Any,
        ArrayLike,
        Domain1,
        Domain10,
        Domain100,
        Domain360,
        Domain100_100_360,
        NDArrayFloat,
        Range1,
        Range10,
        Range100,
        Range360,
        Range100_100_360,
    )

from colour.utilities import (
    MixinDataclassArithmetic,
    MixinDataclassArray,
    MixinDataclassFields,
    MixinDataclassIterable,
    as_array,
    as_complex_array,
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
    get_domain_range_scale_metadata,
    has_only_nan,
    in_array,
    index_along_last_axis,
    interval,
    is_ndarray_copy_enabled,
    is_networkx_installed,
    is_scipy_installed,
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
    "TestAsComplexArray",
    "TestAsIntScalar",
    "TestAsFloatScalar",
    "TestSetDefaultIntegerDtype",
    "TestSetDefaultFloatDtype",
    "TestGetDomainRangeScale",
    "TestSetDomainRangeScale",
    "TestDomainRangeScale",
    "TestGetDomainRangeScaleMetadata",
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

    def setUp(self) -> None:
        """Initialise the common tests attributes."""

        @dataclass
        class Data(MixinDataclassFields):
            a: str
            b: str
            c: str

        self._data: Data = Data(a="Foo", b="Bar", c="Baz")

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = ("fields",)

        for method in required_attributes:
            assert method in dir(MixinDataclassFields)

    def test_fields(self) -> None:
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable._fields`
        method.
        """

        assert self._data.fields == fields(self._data)


class TestMixinDataclassIterable(unittest.TestCase):
    """
    Define :class:`colour.utilities.array.MixinDataclassIterable` class unit
    tests methods.
    """

    def setUp(self) -> None:
        """Initialise the common tests attributes."""

        @dataclass
        class Data(MixinDataclassIterable):
            a: str
            b: str
            c: str

        self._data: Data = Data(a="Foo", b="Bar", c="Baz")

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = (
            "keys",
            "values",
            "items",
        )

        for method in required_attributes:
            assert method in dir(MixinDataclassIterable)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = ("__iter__",)

        for method in required_methods:
            assert method in dir(MixinDataclassIterable)

    def test__iter__(self) -> None:
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable.__iter__`
        method.
        """

        assert {key: value for key, value in self._data} == (
            {"a": "Foo", "b": "Bar", "c": "Baz"}
        )

    def test_keys(self) -> None:
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable.keys`
        method.
        """

        assert tuple(self._data.keys) == ("a", "b", "c")

    def test_values(self) -> None:
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable.values`
        method.
        """

        assert tuple(self._data.values) == ("Foo", "Bar", "Baz")

    def test_items(self) -> None:
        """
        Test :meth:`colour.utilities.array.MixinDataclassIterable.items`
        method.
        """

        assert tuple(self._data.items) == (("a", "Foo"), ("b", "Bar"), ("c", "Baz"))


class TestMixinDataclassArray(unittest.TestCase):
    """
    Define :class:`colour.utilities.array.MixinDataclassArray` class unit
    tests methods.
    """

    def setUp(self) -> None:
        """Initialise the common tests attributes."""

        @dataclass
        class Data(MixinDataclassArray):
            a: float | list | tuple | np.ndarray | None = field(
                default_factory=lambda: None
            )

            b: float | list | tuple | np.ndarray | None = field(
                default_factory=lambda: None
            )

            c: float | list | tuple | np.ndarray | None = field(
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

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = ("__array__",)

        for method in required_methods:
            assert method in dir(MixinDataclassArray)

    def test__array__(self) -> None:
        """
        Test :meth:`colour.utilities.array.MixinDataclassArray.__array__`
        method.
        """

        np.testing.assert_array_equal(self._data, self._array)

        assert np.array(self._data, dtype=DTYPE_INT_DEFAULT).dtype == DTYPE_INT_DEFAULT


class TestMixinDataclassArithmetic(unittest.TestCase):
    """
    Define :class:`colour.utilities.array.MixinDataclassArithmetic` class unit
    tests methods.
    """

    def setUp(self) -> None:
        """Initialise the common tests attributes."""

        @dataclass
        class Data(MixinDataclassArithmetic):
            a: float | list | tuple | np.ndarray | None = field(
                default_factory=lambda: None
            )

            b: float | list | tuple | np.ndarray | None = field(
                default_factory=lambda: None
            )

            c: float | list | tuple | np.ndarray | None = field(
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

    def test_required_methods(self) -> None:
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
            assert method in dir(MixinDataclassArithmetic)

    def test_arithmetical_operation(self) -> None:
        """
        Test :meth:`colour.utilities.array.MixinDataclassArithmetic.\
arithmetical_operation` method.
        """

        np.testing.assert_allclose(
            self._data.arithmetical_operation(10, "+", False),
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data.arithmetical_operation(10, "-", False),
            self._array - 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data.arithmetical_operation(10, "*", False),
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data.arithmetical_operation(10, "/", False),
            self._array / 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data.arithmetical_operation(10, "**", False),
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data + 10,
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data - 10,
            self._array - 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data * 10,
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data / 10,
            self._array / 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            self._data**10,
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        data = deepcopy(self._data)

        np.testing.assert_allclose(
            data.arithmetical_operation(10, "+", True),
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            data.arithmetical_operation(10, "-", True),
            self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            data.arithmetical_operation(10, "*", True),
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            data.arithmetical_operation(10, "/", True),
            self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            data.arithmetical_operation(10, "**", True),
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        data = deepcopy(self._data)

        np.testing.assert_allclose(
            data.arithmetical_operation(self._array, "+", False),
            data + self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            data.arithmetical_operation(data, "+", False),
            data + data,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        data = self._factory(1, 2, 3)

        data += 1
        assert data.a == 2

        data -= 1
        assert data.a == 1

        data *= 2
        assert data.a == 2

        data /= 2
        assert data.a == 1

        data **= 0.5
        assert data.a == 1


class TestAsArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_array` definition unit tests
    methods.
    """

    def test_as_array(self) -> None:
        """Test :func:`colour.utilities.array.as_array` definition."""

        np.testing.assert_equal(as_array([1, 2, 3]), np.array([1, 2, 3]))

        assert as_array([1, 2, 3], DTYPE_FLOAT_DEFAULT).dtype == DTYPE_FLOAT_DEFAULT

        assert as_array([1, 2, 3], DTYPE_INT_DEFAULT).dtype == DTYPE_INT_DEFAULT

        np.testing.assert_equal(
            as_array(dict(zip("abc", [1, 2, 3], strict=True)).values()),
            np.array([1, 2, 3]),
        )


class TestAsInt(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_int` definition unit tests
    methods.
    """

    def test_as_int(self) -> None:
        """Test :func:`colour.utilities.array.as_int` definition."""

        assert as_int(1) == 1

        assert as_int(np.array([1])).ndim == 1

        assert as_int(np.array([[1]])).ndim == 2

        np.testing.assert_array_equal(
            as_int(np.array([1.0, 2.0, 3.0])), np.array([1, 2, 3])
        )

        assert as_int(np.array([1.0, 2.0, 3.0])).dtype == DTYPE_INT_DEFAULT

        assert isinstance(as_int(1), DTYPE_INT_DEFAULT)


class TestAsFloat(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_float` definition unit tests
    methods.
    """

    def test_as_float(self) -> None:
        """Test :func:`colour.utilities.array.as_float` definition."""

        assert as_float(1) == 1.0

        assert as_float(np.array([1])).ndim == 1

        assert as_float(np.array([[1]])).ndim == 2

        np.testing.assert_allclose(
            as_float(np.array([1, 2, 3])),
            np.array([1.0, 2.0, 3.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert as_float(np.array([1, 2, 3])).dtype == DTYPE_FLOAT_DEFAULT

        assert isinstance(as_float(1), DTYPE_FLOAT_DEFAULT)


class TestAsIntArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_int_array` definition unit tests
    methods.
    """

    def test_as_int_array(self) -> None:
        """Test :func:`colour.utilities.array.as_int_array` definition."""

        np.testing.assert_equal(as_int_array([1.0, 2.0, 3.0]), np.array([1, 2, 3]))

        assert as_int_array([1, 2, 3]).dtype == DTYPE_INT_DEFAULT


class TestAsFloatArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_float_array` definition unit tests
    methods.
    """

    def test_as_float_array(self) -> None:
        """Test :func:`colour.utilities.array.as_float_array` definition."""

        np.testing.assert_equal(as_float_array([1, 2, 3]), np.array([1, 2, 3]))

        assert as_float_array([1, 2, 3]).dtype == DTYPE_FLOAT_DEFAULT


class TestAsComplexArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_complex_array` definition unit tests
    methods.
    """

    def test_as_complex_array(self) -> None:
        """Test :func:`colour.utilities.array.as_complex_array` definition."""

        np.testing.assert_equal(
            as_complex_array([1, 2, 3]), np.array([1 + 0j, 2 + 0j, 3 + 0j])
        )

        np.testing.assert_equal(
            as_complex_array([1 + 2j, 3 + 4j]), np.array([1 + 2j, 3 + 4j])
        )

        assert as_complex_array([1, 2, 3]).dtype == DTYPE_COMPLEX_DEFAULT

        assert as_complex_array([1, 2, 3], np.complex64).dtype == np.complex64


class TestAsIntScalar(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_int_scalar` definition unit tests
    methods.
    """

    def test_as_int_scalar(self) -> None:
        """Test :func:`colour.utilities.array.as_int_scalar` definition."""

        assert as_int_scalar(1.0) == 1

        assert as_int_scalar(1.0).dtype == DTYPE_INT_DEFAULT  # pyright: ignore


class TestAsFloatScalar(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.as_float_scalar` definition unit
    tests methods.
    """

    def test_as_float_scalar(self) -> None:
        """Test :func:`colour.utilities.array.as_float_scalar` definition."""

        assert as_float_scalar(1) == 1.0

        assert as_float_scalar(1).dtype == DTYPE_FLOAT_DEFAULT  # pyright: ignore


class TestSetDefaultIntegerDtype(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.set_default_int_dtype` definition unit
    tests methods.
    """

    def test_set_default_int_dtype(self) -> None:
        """
        Test :func:`colour.utilities.array.set_default_int_dtype` definition.
        """

        assert as_int_array(np.ones(3)).dtype == np.int64

        set_default_int_dtype(np.int32)

        assert as_int_array(np.ones(3)).dtype == np.int32

        set_default_int_dtype(np.int64)

        assert as_int_array(np.ones(3)).dtype == np.int64

    def tearDown(self) -> None:
        """After tests actions."""

        set_default_int_dtype(np.int64)


class TestSetDefaultFloatDtype(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.set_default_float_dtype` definition unit
    tests methods.
    """

    def test_set_default_float_dtype(self) -> None:
        """
        Test :func:`colour.utilities.array.set_default_float_dtype`
        definition.
        """

        try:
            assert as_float_array(np.ones(3)).dtype == np.float64

            set_default_float_dtype(np.float16)

            assert as_float_array(np.ones(3)).dtype == np.float16

            set_default_float_dtype(np.float64)

            assert as_float_array(np.ones(3)).dtype == np.float64
        finally:
            set_default_float_dtype(np.float64)

    def test_set_default_float_dtype_enforcement(self) -> None:
        """
        Test whether :func:`colour.utilities.array.set_default_float_dtype`
        effect is applied through most of *Colour* public API.
        """

        if not is_scipy_installed():  # pragma: no cover
            return

        if not is_networkx_installed():  # pragma: no cover
            return

        from colour.appearance import (  # noqa: PLC0415
            CAM_Specification_CAM16,
            CAM_Specification_CIECAM02,
            CAM_Specification_CIECAM16,
            CAM_Specification_Hellwig2022,
            CAM_Specification_Kim2009,
            CAM_Specification_sCAM,
            CAM_Specification_ZCAM,
        )
        from colour.graph.conversion import (  # noqa: PLC0415
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
                    "Spectral Distribution" in (source, target)  # noqa: PLR1714
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

                if source == "sCAM":
                    a = CAM_Specification_sCAM(J=0.25, M=0.5, h=0.25)

                if source == "ZCAM":
                    a = CAM_Specification_ZCAM(J=0.25, M=0.5, h=0.25)

                if source == "CMYK":
                    a = np.array([(0.25, 0.5, 0.25, 0.5), (0.25, 0.5, 0.25, 0.5)])

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

                def dtype_getter(x: NDArray) -> DType:
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
                        "sCAM",
                        "ZCAM",
                    ):
                        if target.endswith(specification):  # noqa: B023
                            return getattr(x, fields(x)[0].name).dtype  # pyright: ignore

                    return x.dtype  # pyright: ignore

                assert dtype_getter(convert(a, source, target)) == dtype
        finally:
            set_default_float_dtype(np.float64)


class TestGetDomainRangeScale(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.get_domain_range_scale` definition
    unit tests methods.
    """

    def test_get_domain_range_scale(self) -> None:
        """
        Test :func:`colour.utilities.common.get_domain_range_scale`
        definition.
        """

        with domain_range_scale("Reference"):
            assert get_domain_range_scale() == "reference"

        with domain_range_scale("1"):
            assert get_domain_range_scale() == "1"

        with domain_range_scale("100"):
            assert get_domain_range_scale() == "100"


class TestSetDomainRangeScale(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.set_domain_range_scale` definition
    unit tests methods.
    """

    def test_set_domain_range_scale(self) -> None:
        """
        Test :func:`colour.utilities.common.set_domain_range_scale`
        definition.
        """

        with domain_range_scale("Reference"):
            set_domain_range_scale("1")
            assert get_domain_range_scale() == "1"

        with domain_range_scale("Reference"):
            set_domain_range_scale("100")
            assert get_domain_range_scale() == "100"

        with domain_range_scale("1"):
            set_domain_range_scale("Reference")
            assert get_domain_range_scale() == "reference"

        with pytest.raises(ValueError):
            set_domain_range_scale("Invalid")


class TestDomainRangeScale(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.domain_range_scale` definition
    unit tests methods.
    """

    def test_domain_range_scale(self) -> None:
        """
        Test :func:`colour.utilities.common.domain_range_scale`
        definition.
        """

        assert get_domain_range_scale() == "reference"

        with domain_range_scale("Reference"):
            assert get_domain_range_scale() == "reference"

        assert get_domain_range_scale() == "reference"

        with domain_range_scale("1"):
            assert get_domain_range_scale() == "1"

        assert get_domain_range_scale() == "reference"

        with domain_range_scale("100"):
            assert get_domain_range_scale() == "100"

        assert get_domain_range_scale() == "reference"

        def fn_a(a: ArrayLike) -> NDArrayFloat:
            """Change the domain-range scale for unit testing."""

            b = to_domain_10(a)

            b *= 2

            return from_range_100(b)

        with domain_range_scale("Reference"):
            with domain_range_scale("1"):
                with domain_range_scale("100"):
                    with domain_range_scale("Ignore"):
                        assert get_domain_range_scale() == "ignore"
                        assert fn_a(4) == 8

                    assert get_domain_range_scale() == "100"
                    assert fn_a(40) == 8

                assert get_domain_range_scale() == "1"
                assert fn_a(0.4) == 0.08

            assert get_domain_range_scale() == "reference"
            assert fn_a(4) == 8

        assert get_domain_range_scale() == "reference"

        @domain_range_scale("1")
        def fn_b(a: ArrayLike) -> NDArrayFloat:
            """Change the domain-range scale for unit testing."""

            b = to_domain_10(a)

            b *= 2

            return from_range_100(b)

        assert fn_b(10) == 2.0


class TestGetDomainRangeScaleMetadata(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.get_domain_range_scale_metadata`
    definition unit tests methods.
    """

    def test_get_domain_range_scale_metadata(self) -> None:
        """
        Test :func:`colour.utilities.array.get_domain_range_scale_metadata`
        definition.
        """

        # Pattern 1: Uniform parameter scaling
        def function_a(
            XYZ: Annotated[ArrayLike, 1],
            illuminant: ArrayLike = None,  # type: ignore
        ) -> Annotated[NDArrayFloat, 100]:  # type: ignore
            """Test uniform parameter scaling."""

        metadata = get_domain_range_scale_metadata(function_a)
        assert metadata["domain"] == {"XYZ": 1}
        assert metadata["range"] == 100

        # Pattern 2: Per-parameter scaling (only some params scaled)
        def function_b(
            uv: ArrayLike,
            illuminant: ArrayLike = None,  # type: ignore
            L: Annotated[ArrayLike, 100] = 100,
        ) -> Annotated[NDArrayFloat, 100]:  # type: ignore
            """Test per-parameter scaling."""

        metadata = get_domain_range_scale_metadata(function_b)
        assert metadata["domain"] == {"L": 100}
        assert metadata["range"] == 100

        # Pattern 3: Per-component tuple scaling (CAM models)
        def function_c(
            XYZ: Annotated[ArrayLike, 100],
        ) -> Annotated[tuple, (100, 100, 360, 100, 100, 100, 400)]:  # type: ignore
            """Test tuple return scaling."""

        metadata = get_domain_range_scale_metadata(function_c)
        assert metadata["domain"] == {"XYZ": 100}
        assert metadata["range"] == (100, 100, 360, 100, 100, 100, 400)

        # Multiple domain parameters
        def function_d(
            XYZ: Annotated[ArrayLike, 100],
            XYZ_w: Annotated[ArrayLike, 100],
            illuminant: ArrayLike = None,  # type: ignore
        ) -> Annotated[NDArrayFloat, 100]:  # type: ignore
            """Test multiple domain parameters."""

        metadata = get_domain_range_scale_metadata(function_d)
        assert metadata["domain"] == {"XYZ": 100, "XYZ_w": 100}
        assert metadata["range"] == 100

        # No annotations (backward compatibility)
        def function_e(XYZ: Any, illuminant: Any = None) -> None:
            """Test backward compatibility."""

        metadata = get_domain_range_scale_metadata(function_e)
        assert metadata["domain"] == {}
        assert metadata["range"] is None

        # Only domain scaling, no range
        def function_f(
            XYZ: Annotated[ArrayLike, 1],
        ) -> NDArrayFloat:  # type: ignore
            """Test domain-only scaling."""

        metadata = get_domain_range_scale_metadata(function_f)
        assert metadata["domain"] == {"XYZ": 1}
        assert metadata["range"] is None

        # Only range scaling, no domain
        def function_g(
            XYZ: ArrayLike,
        ) -> Annotated[NDArrayFloat, 100]:  # type: ignore
            """Test range-only scaling."""

        metadata = get_domain_range_scale_metadata(function_g)
        assert metadata["domain"] == {}
        assert metadata["range"] == 100

        # Type aliases: Domain1/Range1
        def function_h(XYZ: Domain1, XYZ_w: Domain1 = 1) -> Range1:  # type: ignore
            """Test Domain1/Range1 type aliases."""

        metadata = get_domain_range_scale_metadata(function_h)
        assert metadata["domain"] == {"XYZ": 1, "XYZ_w": 1}

        # Union with Annotated types
        def function_i(
            value: Annotated[int, 100] | Annotated[float, 200],
        ) -> NDArrayFloat:  # type: ignore
            """Test Union with Annotated members."""

        metadata = get_domain_range_scale_metadata(function_i)
        assert metadata["domain"] == {"value": 100}
        assert metadata["range"] is None

        # Type aliases: Domain100/Range100
        def function_j(Y: Domain100, Y_n: Domain100 = 100) -> Range100:  # type: ignore
            """Test Domain100/Range100 type aliases."""

        metadata = get_domain_range_scale_metadata(function_j)
        assert metadata["domain"] == {"Y": 100, "Y_n": 100}
        assert metadata["range"] == 100

        # Type aliases: Domain10/Range10
        def function_k(L: Domain10) -> Range10:  # type: ignore
            """Test Domain10/Range10 type aliases."""

        metadata = get_domain_range_scale_metadata(function_k)
        assert metadata["domain"] == {"L": 10}
        assert metadata["range"] == 10

        # Type aliases: Domain360/Range360
        def function_l(hue: Domain360) -> Range360:  # type: ignore
            """Test Domain360/Range360 type aliases."""

        metadata = get_domain_range_scale_metadata(function_l)
        assert metadata["domain"] == {"hue": 360}
        assert metadata["range"] == 360

        # Type aliases: Domain100_100_360/Range100_100_360
        def function_m(Lab: Domain100_100_360) -> Range100_100_360:  # type: ignore
            """Test Domain100_100_360/Range100_100_360 type aliases."""

        metadata = get_domain_range_scale_metadata(function_m)
        assert metadata["domain"] == {"Lab": (100, 100, 360)}
        assert metadata["range"] == (100, 100, 360)

        # Mixed: type aliases and explicit Annotated
        def function_n(
            XYZ: Domain1, L: Domain100, custom: Annotated[ArrayLike, 50]
        ) -> Range100:  # type: ignore
            """Test mixed type aliases and Annotated."""

        metadata = get_domain_range_scale_metadata(function_n)
        assert metadata["domain"] == {"XYZ": 1, "L": 100, "custom": 50}
        assert metadata["range"] == 100

        # functools.partial with type aliases
        def function_o(
            XYZ: Domain1,
            colourspace: str,
            illuminant: ArrayLike | None = None,
        ) -> Range1:  # type: ignore
            """Test function for partial wrapping."""

        partial_func = partial(function_o, colourspace="sRGB")
        metadata = get_domain_range_scale_metadata(partial_func)
        assert metadata["domain"] == {"XYZ": 1}
        assert metadata["range"] == 1

        # functools.partial with explicit Annotated
        def function_p(
            Lab: Annotated[ArrayLike, 100],
            illuminant: ArrayLike | None = None,
            method: str = "CIE 1976",
        ) -> Annotated[NDArrayFloat, 100]:  # type: ignore
            """Test function for partial wrapping with Annotated."""

        partial_func2 = partial(function_p, method="CIE 2000")
        metadata = get_domain_range_scale_metadata(partial_func2)
        assert metadata["domain"] == {"Lab": 100}
        assert metadata["range"] == 100

        # Test string annotation with unevaluable scale (triggers exception handler)
        # This simulates what happens with `from __future__ import annotations`
        # when the annotation contains an undefined variable
        def function_q(x: Any) -> Any:
            """Test function with mock string annotation."""

        # Manually set __annotations__ to simulate string annotation with undefined var
        function_q.__annotations__ = {
            "x": "Annotated[float, undefined_variable]",
            "return": "Annotated[float, another_undefined]",
        }

        metadata = get_domain_range_scale_metadata(function_q)
        # The eval will fail, so it falls back to the string itself
        assert metadata["domain"] == {"x": "undefined_variable"}
        assert metadata["range"] == "another_undefined"


class TestToDomain1(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_1` definition unit
    tests methods.
    """

    def test_to_domain_1(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_1` definition."""

        with domain_range_scale("Reference"):
            assert to_domain_1(1) == 1

        with domain_range_scale("1"):
            assert to_domain_1(1) == 1

        with domain_range_scale("100"):
            assert to_domain_1(1) == 0.01

        with domain_range_scale("100"):
            assert to_domain_1(1, np.pi) == 1 / np.pi

        with domain_range_scale("100"):
            assert to_domain_1(1, dtype=np.float16).dtype == np.float16


class TestToDomain10(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_10` definition unit
    tests methods.
    """

    def test_to_domain_10(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_10` definition."""

        with domain_range_scale("Reference"):
            assert to_domain_10(1) == 1

        with domain_range_scale("1"):
            assert to_domain_10(1) == 10

        with domain_range_scale("100"):
            assert to_domain_10(1) == 0.1

        with domain_range_scale("100"):
            assert to_domain_10(1, np.pi) == 1 / np.pi

        with domain_range_scale("100"):
            assert to_domain_10(1, dtype=np.float16).dtype == np.float16


class TestToDomain100(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_100` definition unit
    tests methods.
    """

    def test_to_domain_100(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_100` definition."""

        with domain_range_scale("Reference"):
            assert to_domain_100(1) == 1

        with domain_range_scale("1"):
            assert to_domain_100(1) == 100

        with domain_range_scale("100"):
            assert to_domain_100(1) == 1

        with domain_range_scale("1"):
            assert to_domain_100(1, np.pi) == np.pi

        with domain_range_scale("100"):
            assert to_domain_100(1, dtype=np.float16).dtype == np.float16


class TestToDomainDegrees(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_degrees` definition unit
    tests methods.
    """

    def test_to_domain_degrees(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_degrees` definition."""

        with domain_range_scale("Reference"):
            assert to_domain_degrees(1) == 1

        with domain_range_scale("1"):
            assert to_domain_degrees(1) == 360

        with domain_range_scale("100"):
            assert to_domain_degrees(1) == 3.6

        with domain_range_scale("100"):
            assert to_domain_degrees(1, np.pi) == np.pi / 100

        with domain_range_scale("100"):
            assert to_domain_degrees(1, dtype=np.float16).dtype == np.float16


class TestToDomainInt(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.to_domain_int` definition unit
    tests methods.
    """

    def test_to_domain_int(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_int` definition."""

        with domain_range_scale("Reference"):
            assert to_domain_int(1) == 1

        with domain_range_scale("1"):
            assert to_domain_int(1) == 255

        with domain_range_scale("100"):
            assert to_domain_int(1) == 2.55

        with domain_range_scale("100"):
            assert to_domain_int(1, 10) == 10.23

        with domain_range_scale("100"):
            assert to_domain_int(1, dtype=np.float16).dtype == np.float16


class TestFromRange1(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_1` definition unit
    tests methods.
    """

    def test_from_range_1(self) -> None:
        """Test :func:`colour.utilities.common.from_range_1` definition."""

        with domain_range_scale("Reference"):
            assert from_range_1(1) == 1

        with domain_range_scale("1"):
            assert from_range_1(1) == 1

        with domain_range_scale("100"):
            assert from_range_1(1) == 100

        with domain_range_scale("100"):
            assert from_range_1(1, np.pi) == 1 * np.pi


class TestFromRange10(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_10` definition unit
    tests methods.
    """

    def test_from_range_10(self) -> None:
        """Test :func:`colour.utilities.common.from_range_10` definition."""

        with domain_range_scale("Reference"):
            assert from_range_10(1) == 1

        with domain_range_scale("1"):
            assert from_range_10(1) == 0.1

        with domain_range_scale("100"):
            assert from_range_10(1) == 10

        with domain_range_scale("100"):
            assert from_range_10(1, np.pi) == 1 * np.pi


class TestFromRange100(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_100` definition unit
    tests methods.
    """

    def test_from_range_100(self) -> None:
        """Test :func:`colour.utilities.common.from_range_100` definition."""

        with domain_range_scale("Reference"):
            assert from_range_100(1) == 1

        with domain_range_scale("1"):
            assert from_range_100(1) == 0.01

        with domain_range_scale("100"):
            assert from_range_100(1) == 1

        with domain_range_scale("1"):
            assert from_range_100(1, np.pi) == 1 / np.pi


class TestFromRangeDegrees(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_degrees` definition unit
    tests methods.
    """

    def test_from_range_degrees(self) -> None:
        """Test :func:`colour.utilities.common.from_range_degrees` definition."""

        with domain_range_scale("Reference"):
            assert from_range_degrees(1) == 1

        with domain_range_scale("1"):
            assert from_range_degrees(1) == 1 / 360

        with domain_range_scale("100"):
            assert from_range_degrees(1) == 1 / 3.6

        with domain_range_scale("100"):
            assert from_range_degrees(1, np.pi) == 1 / (np.pi / 100)


class TestFromRangeInt(unittest.TestCase):
    """
    Define :func:`colour.utilities.common.from_range_int` definition unit
    tests methods.
    """

    def test_from_range_int(self) -> None:
        """Test :func:`colour.utilities.common.from_range_int` definition."""

        with domain_range_scale("Reference"):
            assert from_range_int(1) == 1

        with domain_range_scale("1"):
            assert from_range_int(1) == 1 / 255

        with domain_range_scale("100"):
            assert from_range_int(1) == 1 / 2.55

        with domain_range_scale("100"):
            assert from_range_int(1, 10) == 1 / (1023 / 100)

        with domain_range_scale("100"):
            assert from_range_int(1, dtype=np.float16).dtype == np.float16


class TestIsNdarrayCopyEnabled(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.is_ndarray_copy_enabled` definition
    unit tests methods.
    """

    def test_is_ndarray_copy_enabled(self) -> None:
        """
        Test :func:`colour.utilities.array.is_ndarray_copy_enabled` definition.
        """

        with ndarray_copy_enable(True):
            assert is_ndarray_copy_enabled()

        with ndarray_copy_enable(False):
            assert not is_ndarray_copy_enabled()


class TestSetNdarrayCopyEnabled(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.set_ndarray_copy_enable` definition
    unit tests methods.
    """

    def test_set_ndarray_copy_enable(self) -> None:
        """
        Test :func:`colour.utilities.array.set_ndarray_copy_enable` definition.
        """

        with ndarray_copy_enable(is_ndarray_copy_enabled()):
            set_ndarray_copy_enable(True)
            assert is_ndarray_copy_enabled()

        with ndarray_copy_enable(is_ndarray_copy_enabled()):
            set_ndarray_copy_enable(False)
            assert not is_ndarray_copy_enabled()


class TestNdarrayCopyEnable(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.ndarray_copy_enable` definition unit
    tests methods.
    """

    def test_ndarray_copy_enable(self) -> None:
        """
        Test :func:`colour.utilities.array.ndarray_copy_enable` definition.
        """

        with ndarray_copy_enable(True):
            assert is_ndarray_copy_enabled()

        with ndarray_copy_enable(False):
            assert not is_ndarray_copy_enabled()

        @ndarray_copy_enable(True)
        def fn_a() -> None:
            """:func:`ndarray_copy_enable` unit tests :func:`fn_a` definition."""

            assert is_ndarray_copy_enabled()

        fn_a()

        @ndarray_copy_enable(False)
        def fn_b() -> None:
            """:func:`ndarray_copy_enable` unit tests :func:`fn_b` definition."""

            assert not is_ndarray_copy_enabled()

        fn_b()


class TestNdarrayCopy(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.ndarray_copy` definition unit
    tests methods.
    """

    def test_ndarray_copy(self) -> None:
        """Test :func:`colour.utilities.array.ndarray_copy` definition."""

        a = np.linspace(0, 1, 10)
        with ndarray_copy_enable(True):
            assert id(ndarray_copy(a)) != id(a)

        with ndarray_copy_enable(False):
            assert id(ndarray_copy(a)) == id(a)


class TestClosestIndexes(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.closest_indexes` definition unit
    tests methods.
    """

    def test_closest_indexes(self) -> None:
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

        assert closest_indexes(a, 63.05) == 3

        assert closest_indexes(a, 51.15) == 4

        assert closest_indexes(a, 24.90) == 5

        np.testing.assert_array_equal(
            closest_indexes(a, np.array([63.05, 51.15, 24.90])),
            np.array([3, 4, 5]),
        )


class TestClosest(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.closest` definition unit tests
    methods.
    """

    def test_closest(self) -> None:
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

        assert closest(a, 63.05) == 62.70988028

        assert closest(a, 51.15) == 46.84480573

        assert closest(a, 24.90) == 25.40026416

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

    def test_interval(self) -> None:
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

    def test_is_uniform(self) -> None:
        """Test :func:`colour.utilities.array.is_uniform` definition."""

        assert is_uniform(range(0, 10, 2))

        assert not is_uniform([1, 2, 3, 4, 6])


class TestInArray(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.in_array` definition unit tests
    methods.
    """

    def test_in_array(self) -> None:
        """Test :func:`colour.utilities.array.in_array` definition."""

        assert np.array_equal(
            in_array(np.array([0.50, 0.60]), np.linspace(0, 10, 101)),
            np.array([True, True]),
        )

        assert not np.array_equal(
            in_array(np.array([0.50, 0.61]), np.linspace(0, 10, 101)),
            np.array([True, True]),
        )

        assert np.array_equal(
            in_array(np.array([[0.50], [0.60]]), np.linspace(0, 10, 101)),
            np.array([[True], [True]]),
        )

    def test_n_dimensional_in_array(self) -> None:
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
            in_array(np.array([[0.50], [0.60]]), np.linspace(0, 10, 101)).shape,
            np.array([2, 1]),
        )


class TestTstack(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.tstack` definition unit tests
    methods.
    """

    def test_tstack(self) -> None:
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

        # Ensuring that contiguity is maintained.
        a = np.array([0, 1, 2], dtype=DTYPE_FLOAT_DEFAULT)
        b = tstack([a, a, a])
        assert b.flags.contiguous

        # Ensuring that independence is maintained.
        a *= 2
        np.testing.assert_array_equal(
            b,
            np.array(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                ],
            ),
        )

        a = np.array([0, 1, 2], dtype=DTYPE_FLOAT_DEFAULT)
        b = tstack([a, a, a])

        b[1] *= 2
        np.testing.assert_array_equal(
            a,
            np.array([0, 1, 2]),
        )


class TestTsplit(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.tsplit` definition unit tests
    methods.
    """

    def test_tsplit(self) -> None:
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

        # Ensuring that contiguity is maintained.
        a = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
            dtype=DTYPE_FLOAT_DEFAULT,
        )
        b = tsplit(a)
        assert b.flags.contiguous

        # Ensuring that independence is maintained.
        a *= 2
        np.testing.assert_array_equal(
            b,
            np.array(
                [
                    [0, 1, 2],
                    [0, 1, 2],
                    [0, 1, 2],
                ]
            ),
        )

        a = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
            dtype=DTYPE_FLOAT_DEFAULT,
        )
        b = tsplit(a)

        b[1] *= 2
        np.testing.assert_array_equal(
            a,
            np.array(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                ]
            ),
        )


class TestRowAsDiagonal(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.row_as_diagonal` definition unit
    tests methods.
    """

    def test_row_as_diagonal(self) -> None:
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

    def test_orient(self) -> None:
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

    def test_centroid(self) -> None:
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

    def test_fill_nan(self) -> None:
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

    def test_has_only_nan(self) -> None:
        """Test :func:`colour.utilities.array.has_only_nan` definition."""

        assert has_only_nan(None)  # pyright: ignore

        assert has_only_nan([None, None])  # pyright: ignore

        assert not has_only_nan([True, None])  # pyright: ignore

        assert not has_only_nan([0.1, np.nan, 0.3])


class TestNdarrayWrite(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.ndarray_write` definition unit tests
    methods.
    """

    def test_ndarray_write(self) -> None:
        """Test :func:`colour.utilities.array.ndarray_write` definition."""

        a = np.linspace(0, 1, 10)
        a.setflags(write=False)

        with pytest.raises(ValueError):
            a += 1

        with ndarray_write(a):
            a += 1


class TestZeros(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.zeros` definition unit tests
    methods.
    """

    def test_zeros(self) -> None:
        """Test :func:`colour.utilities.array.zeros` definition."""

        np.testing.assert_equal(zeros(3), np.zeros(3))


class TestOnes(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.ones` definition unit tests
    methods.
    """

    def test_ones(self) -> None:
        """Test :func:`colour.utilities.array.ones` definition."""

        np.testing.assert_equal(ones(3), np.ones(3))


class TestFull(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.full` definition unit tests
    methods.
    """

    def test_full(self) -> None:
        """Test :func:`colour.utilities.array.full` definition."""

        np.testing.assert_equal(full(3, 0.5), np.full(3, 0.5))


class TestIndexAlongLastAxis(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.index_along_last_axis` definition
    unit tests methods.
    """

    def test_index_along_last_axis(self) -> None:
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

        indexes = np.array([[[0, 1], [0, 1]], [[2, 1], [2, 1]], [[2, 1], [2, 0]]])

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

    def test_compare_with_argmin_argmax(self) -> None:
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

    def test_exceptions(self) -> None:
        """
        Test :func:`colour.utilities.array.index_along_last_axis` definition
        handling of invalid inputs.
        """

        a = as_float_array([[11, 12], [21, 22]])

        # Bad shape
        with pytest.raises(ValueError):
            indexes = np.array([0])
            index_along_last_axis(a, indexes)

        # Indexes out of range
        with pytest.raises(IndexError):
            indexes = np.array([123, 456])
            index_along_last_axis(a, indexes)

        # Non-int indexes
        with pytest.raises(IndexError):
            indexes = np.array([0.0, 0.0])
            index_along_last_axis(a, indexes)


class TestFormatArrayAsRow(unittest.TestCase):
    """
    Define :func:`colour.utilities.array.format_array_as_row` definition unit
    tests methods.
    """

    def test_format_array_as_row(self) -> None:
        """Test :func:`colour.utilities.array.format_array_as_row` definition."""

        assert format_array_as_row([1.25, 2.5, 3.75]) == "1.2500000 2.5000000 3.7500000"

        assert format_array_as_row([1.25, 2.5, 3.75], 3) == "1.250 2.500 3.750"

        assert format_array_as_row([1.25, 2.5, 3.75], 3, ", ") == "1.250, 2.500, 3.750"
