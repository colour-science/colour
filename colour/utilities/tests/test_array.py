"""Define the unit tests for the :mod:`colour.utilities.array` module."""

from __future__ import annotations

import types
import typing
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import partial

import numpy as np
import pytest

import colour.utilities.array as utilities_array
from colour.constants import (
    DTYPE_COMPLEX_DEFAULT,
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    TOLERANCE_ABSOLUTE_TESTS,
)

if typing.TYPE_CHECKING:
    from collections.abc import Generator

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
        ModuleType,
        NDArray,
        NDArrayFloat,
        ProtocolArrayNamespace,
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
    ColourRuntimeWarning,
    MixinDataclassArithmetic,
    MixinDataclassArray,
    MixinDataclassFields,
    MixinDataclassIterable,
    array_api_enable,
    array_namespace,
    as_array,
    as_complex_array,
    as_float,
    as_float_array,
    as_float_scalar,
    as_int,
    as_int_array,
    as_int_scalar,
    as_ndarray,
    caching_enable,
    cast_non_ndarray,
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
    is_array_api_enabled,
    is_ndarray_copy_enabled,
    is_networkx_installed,
    is_non_ndarray,
    is_numpy_namespace,
    is_uniform,
    ndarray_copy,
    ndarray_copy_enable,
    ndarray_write,
    ones,
    orient,
    row_as_diagonal,
    set_array_api_enabled,
    set_default_float_dtype,
    set_default_int_dtype,
    set_domain_range_scale,
    set_ndarray_copy_enabled,
    to_domain_1,
    to_domain_10,
    to_domain_100,
    to_domain_degrees,
    to_domain_int,
    tsplit,
    tstack,
    xp_as_array,
    xp_as_float_array,
    xp_as_int_array,
    xp_ascontiguousarray,
    xp_assert_close,
    xp_assert_equal,
    xp_astype,
    xp_atleast_1d,
    xp_atleast_2d,
    xp_average,
    xp_broadcast_to,
    xp_create_diagonal,
    xp_degrees,
    xp_eig,
    xp_eigh,
    xp_gradient,
    xp_insert,
    xp_interp,
    xp_isclose,
    xp_isin,
    xp_linspace,
    xp_lstsq,
    xp_matrix_transpose,
    xp_median,
    xp_nan_to_num,
    xp_nanmean,
    xp_pad,
    xp_radians,
    xp_reshape,
    xp_resize,
    xp_round,
    xp_select,
    xp_setxor1d,
    xp_sinc,
    xp_squeeze,
    xp_trapezoid,
    xp_unique,
    zeros,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIsArrayApiEnabled",
    "TestSetArrayApiEnabled",
    "TestArrayApiEnable",
    "TestArrayNamespace",
    "TestIsNumpyNamespace",
    "TestIsNonnumpyArray",
    "TestAsNdarray",
    "TestCastNonNdarray",
    "TestXpAsArray",
    "TestXpAsFloatArray",
    "TestXpAsIntArray",
    "TestXpAscontiguousarray",
    "TestXpAstype",
    "TestXpMatrixTranspose",
    "TestXpSelect",
    "TestRuntimeWarningXpFallback",
    "TestXpInterp",
    "TestXpTrapezoid",
    "TestXpAverage",
    "TestXpGradient",
    "TestXpResize",
    "TestXpNanmean",
    "TestXpMedian",
    "TestXpRound",
    "TestXpRadians",
    "TestXpDegrees",
    "TestXpAtleast1d",
    "TestXpAtleast2d",
    "TestXpSinc",
    "TestXpSqueeze",
    "TestXpIsclose",
    "TestXpNanToNum",
    "TestXpCreateDiagonal",
    "TestXpReshape",
    "TestXpEig",
    "TestXpEigh",
    "TestXpLstsq",
    "TestXpIsin",
    "TestXpLinspace",
    "TestXpPad",
    "TestXpUnique",
    "TestXpInsert",
    "TestXpSetxor1d",
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
    "TestFormatArrayAsRow",
    "TestAsArrayArrayApi",
    "TestTstackArrayApi",
    "TestTsplitArrayApi",
]


class TestIsArrayApiEnabled:
    """Define :func:`colour.utilities.is_array_api_enabled` unit tests."""

    def test_is_array_api_enabled(self) -> None:
        """Test :func:`colour.utilities.is_array_api_enabled` definition."""

        with array_api_enable(False):
            assert not is_array_api_enabled()

        with array_api_enable(True):
            assert is_array_api_enabled()


class TestSetArrayApiEnabled:
    """Define :func:`colour.utilities.set_array_api_enabled` unit tests."""

    def test_set_array_api_enabled(self) -> None:
        """Test :func:`colour.utilities.set_array_api_enabled` definition."""

        with array_api_enable(is_array_api_enabled()):
            set_array_api_enabled(True)
            assert is_array_api_enabled()
            set_array_api_enabled(False)
            assert not is_array_api_enabled()


class TestArrayApiEnable:
    """Define :class:`colour.utilities.array_api_enable` unit tests."""

    def test_array_api_enable(self) -> None:
        """Test :class:`colour.utilities.array_api_enable` definition."""

        with array_api_enable(True):
            assert is_array_api_enabled()

        with array_api_enable(False):
            assert not is_array_api_enabled()

        with array_api_enable(False):
            original = is_array_api_enabled()
            with array_api_enable(True):
                assert is_array_api_enabled()
            assert is_array_api_enabled() == original

        @array_api_enable(True)
        def fn_enabled() -> bool:
            return is_array_api_enabled()

        @array_api_enable(False)
        def fn_disabled() -> bool:
            return is_array_api_enabled()

        assert fn_enabled()
        assert not fn_disabled()


class TestArrayNamespace:
    """Define :func:`colour.utilities.array_namespace` unit tests."""

    def test_array_namespace(self, xp: ProtocolArrayNamespace | ModuleType) -> None:
        """Test :func:`colour.utilities.array_namespace` definition."""

        with array_api_enable(False):
            assert array_namespace(np.array([1, 2, 3])) is np

        with array_api_enable(True):
            xp = array_namespace(np.array([1, 2, 3]))

            assert is_numpy_namespace(xp)

        with array_api_enable(True):
            assert array_namespace() is np
            assert array_namespace(1.0, 2.0) is np
            assert array_namespace(None) is np


class TestIsNumpyNamespace:
    """Define :func:`colour.utilities.is_numpy_namespace` unit tests."""

    def test_is_numpy_namespace(self) -> None:
        """Test :func:`colour.utilities.is_numpy_namespace` definition."""

        assert is_numpy_namespace(np)

        mock_ns = types.ModuleType("jax.numpy")
        assert not is_numpy_namespace(mock_ns)


class TestIsNonnumpyArray:
    """Define :func:`colour.utilities.is_non_ndarray` unit tests."""

    def test_is_non_ndarray(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.is_non_ndarray` definition."""

        assert not is_non_ndarray(np.array([1, 2, 3]))
        assert not is_non_ndarray(np.float64(1.0))
        assert not is_non_ndarray([1, 2, 3])
        assert not is_non_ndarray(1.0)
        assert not is_non_ndarray(None)

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        if is_numpy_namespace(xp):
            assert not is_non_ndarray(a)
        else:
            assert is_non_ndarray(a)


class TestAsNdarray:
    """Define :func:`colour.utilities.as_ndarray` unit tests."""

    def test_as_ndarray(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.as_ndarray` definition."""

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = as_ndarray(a)
        assert isinstance(result, np.ndarray)
        xp_assert_equal(result, [1.0, 2.0, 3.0])

        result = as_ndarray(np.array([4, 5, 6]))
        assert isinstance(result, np.ndarray)
        xp_assert_equal(result, [4, 5, 6])

        # The hand-off is asked of the array's own namespace: dispatch being
        # disabled returns the *NumPy* fallback, which has no ``to_device``.
        class _Namespace:
            """Define a backend namespace offering the host hand-off."""

            def to_device(self, a: Any, _device: str) -> Any:
                """Return the array on the specified device."""

                return np.asarray(a.values)

        class _DeviceArray:
            """Define an array resident on a device *NumPy* cannot read."""

            def __init__(self, values: Any) -> None:
                self.values = values

            def __array__(self, *args: Any, **kwargs: Any) -> Any:
                """Raise as a device-resident array does."""

                raise TypeError

            def __array_namespace__(self, *, api_version: str | None = None) -> Any:
                """Return the namespace of the array."""

                return _Namespace()

        with array_api_enable(False):
            xp_assert_equal(as_ndarray(_DeviceArray([7.0, 8.0])), [7.0, 8.0])

    def test_raise_exception_as_ndarray(self) -> None:
        """Test :func:`colour.utilities.as_ndarray` definition raised exception."""

        class _Opaque:
            """Define an object that is not an array by any route."""

            def __array__(self, *args: Any, **kwargs: Any) -> Any:
                """Raise as a foreign object does."""

                raise TypeError

        pytest.raises(TypeError, as_ndarray, _Opaque())


class TestCastNonNdarray:
    """Define :func:`colour.utilities.cast_non_ndarray` unit tests."""

    def test_cast_non_ndarray(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.cast_non_ndarray` definition."""

        # A *NumPy* array is never cast.
        with array_api_enable(True):
            assert cast_non_ndarray(np.array([1.0, 2.0]), np.float32) is None

        # Disabled *Array API* dispatch returns ``None``.
        with array_api_enable(False):
            assert cast_non_ndarray(np.array([1.0, 2.0]), np.float32) is None

        if is_numpy_namespace(xp):
            return

        with array_api_enable(True):
            # A non-*NumPy* array is cast to the specified dtype.
            a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
            result = cast_non_ndarray(a, np.float32)
            assert result is not None
            assert result.dtype == getattr(xp, "float32", None)

            # The array is returned unchanged when the dtype already
            # matches. Casting to the array's *actual* dtype keeps the
            # precondition true on every backend, including *MPS* which
            # silently substitutes ``float32`` for ``float64``;
            # ``as_ndarray`` yields a genuine :class:`numpy.dtype` that
            # :func:`cast_non_ndarray` resolves back to the native dtype.
            a = xp_as_array([1.0, 2.0], xp=xp)
            assert cast_non_ndarray(a, as_ndarray(a).dtype) is a


class TestXpAsArray:
    """Define :func:`colour.utilities.xp_as_array` unit tests."""

    def test_xp_as_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_as_array` definition."""

        # Python sequence promotion.
        result = xp_as_array([1, 2, 3], xp=xp)
        xp_assert_equal(result, [1, 2, 3])

        # Dtype enforcement.
        result = xp_as_array([1, 2, 3], dtype=DTYPE_FLOAT_DEFAULT, xp=xp)
        assert result.dtype == getattr(
            xp, np.dtype(DTYPE_FLOAT_DEFAULT).name, DTYPE_FLOAT_DEFAULT
        )
        xp_assert_close(result, [1.0, 2.0, 3.0])

        result = xp_as_array([1.5, 2.5], dtype=DTYPE_INT_DEFAULT, xp=xp)
        assert result.dtype == getattr(
            xp, np.dtype(DTYPE_INT_DEFAULT).name, DTYPE_INT_DEFAULT
        )
        xp_assert_equal(result, [1, 2])

        # Empty input survives the round-trip.
        result = xp_as_array([], dtype=DTYPE_FLOAT_DEFAULT, xp=xp)
        assert result.shape == (0,)

        if is_numpy_namespace(xp):
            # *NumPy* identity when no dtype conversion is required.
            a = np.array([1.0, 2.0, 3.0])
            assert xp_as_array(a, xp=xp) is a
            assert xp_as_array(a, dtype=a.dtype, xp=xp) is a

            # *NumPy* with mismatched dtype is cast and copied.
            result = xp_as_array(a, dtype=np.float32, xp=xp)
            assert result.dtype == np.float32
            assert result is not a

            # *Array API* disabled coerces to *NumPy*.
            with array_api_enable(False):
                result = xp_as_array([1.0, 2.0], dtype=np.float32, xp=xp)
                assert isinstance(result, np.ndarray)
                assert result.dtype == np.float32

            return

        # Already on a backend, no dtype, returned identity.
        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        assert xp_as_array(a, xp=xp) is a

        # Already on a backend, dtype matches, returned identity, including
        # when a *NumPy* dtype alias is passed in.
        a = xp_as_float_array([1.0, 2.0, 3.0], xp=xp)
        assert xp_as_array(a, dtype=a.dtype, xp=xp) is a

        backend_dtype_name = getattr(a.dtype, "name", str(a.dtype)).rsplit(".", 1)[-1]
        numpy_dtype = getattr(np, backend_dtype_name, None)
        if numpy_dtype is not None:
            assert xp_as_array(a, dtype=numpy_dtype, xp=xp) is a

        # Already on a backend, dtype differs, cast through ``xp_astype``.
        a_64 = xp_astype(xp_as_array([1.0, 2.0, 3.0], xp=xp), np.float64, xp=xp)
        result = xp_as_array(a_64, dtype=np.float32, xp=xp)
        assert result.dtype == getattr(xp, "float32", np.float32)

        # Backend object without a ``dtype`` attribute is returned as-is.
        class _NoDtype:
            def __array_namespace__(self) -> ModuleType:
                return xp

        sentinel: Any = _NoDtype()
        assert xp_as_array(sentinel, dtype=np.float32, xp=xp) is sentinel

        # Scalar cache hit.
        first = xp_as_array(0.5, dtype=DTYPE_FLOAT_DEFAULT, xp=xp)
        second = xp_as_array(0.5, dtype=DTYPE_FLOAT_DEFAULT, xp=xp)
        assert first is second

        # Small constant array cache hit.
        constant = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        first = xp_as_array(constant, dtype=DTYPE_FLOAT_DEFAULT, xp=xp)
        second = xp_as_array(constant, dtype=DTYPE_FLOAT_DEFAULT, xp=xp)
        assert first is second

        # Caching disabled bypasses the scalar cache.
        with caching_enable(False):
            first = xp_as_array(0.75, dtype=DTYPE_FLOAT_DEFAULT, xp=xp)
            second = xp_as_array(0.75, dtype=DTYPE_FLOAT_DEFAULT, xp=xp)
            assert first is not second

        # ``like`` argument honoured for backends that expose a device.
        reference = xp_as_array([1.0, 2.0], xp=xp)
        if getattr(reference, "device", None) is not None:
            result = xp_as_array([3.0, 4.0], xp=xp, like=reference)
            assert getattr(result, "device", None) == reference.device

        # ``copy=True`` returns a fresh object for backend arrays
        # (short-circuit path).
        original = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        copied = xp_as_array(original, xp=xp, copy=True)
        assert copied is not original
        xp_assert_equal(copied, [1.0, 2.0, 3.0])

        # ``copy=True`` bypasses the scalar-promotion cache.
        constant = np.array([0.4, 0.5, 0.6], dtype=np.float64)
        first = xp_as_array(constant, dtype=DTYPE_FLOAT_DEFAULT, xp=xp, copy=True)
        second = xp_as_array(constant, dtype=DTYPE_FLOAT_DEFAULT, xp=xp, copy=True)
        if is_array_api_enabled() and not is_numpy_namespace(xp):
            assert first is not second

        # ``copy=None`` (default) preserves the no-copy short-circuit.
        no_copy = xp_as_array(original, xp=xp)
        assert no_copy is original

    def test_xp_as_array_copy_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test that a backend copy preserves automatic differentiation."""

        copied, (gradient,), (original,) = autodiff(
            lambda a: xp_as_array(a, xp=xp, copy=True), [1.0, 2.0, 3.0]
        )

        assert copied is not original
        xp_assert_equal(gradient, xp.ones_like(original))


class TestXpAsFloatArray:
    """Define :func:`colour.utilities.xp_as_float_array` unit tests."""

    def test_xp_as_float_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_as_float_array` definition."""

        result = xp_as_float_array([1, 2, 3], xp=xp)
        xp_assert_close(result, [1.0, 2.0, 3.0])

        # Dtype enforcement.
        result = xp_as_float_array([1, 2, 3], xp=np)
        assert result.dtype == DTYPE_FLOAT_DEFAULT


class TestXpAsIntArray:
    """Define :func:`colour.utilities.xp_as_int_array` unit tests."""

    def test_xp_as_int_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_as_int_array` definition."""

        result = xp_as_int_array([1.5, 2.7, 3.9], xp=xp)
        xp_assert_equal(result, [1, 2, 3])

        # Dtype enforcement.
        result = xp_as_int_array([1.5, 2.7], xp=np)
        assert result.dtype == DTYPE_INT_DEFAULT


class TestXpAstype:
    """Define :func:`colour.utilities.xp_astype` unit tests."""

    def test_xp_astype(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_astype` definition."""

        a = xp_as_array([1.0, 2.5, 3.7], xp=xp)

        result = xp_astype(a, np.float32)
        assert as_ndarray(result).dtype == np.float32

        result = xp_astype(a, np.int32)
        xp_assert_equal(result, [1, 2, 3])

        a_int = xp_as_array([1, 2, 3], xp=xp)
        result = xp_astype(a_int, DTYPE_FLOAT_DEFAULT)
        assert as_ndarray(result).dtype == DTYPE_FLOAT_DEFAULT


class TestXpAscontiguousarray:
    """Define :func:`colour.utilities.xp_ascontiguousarray` unit tests."""

    def test_xp_ascontiguousarray(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_ascontiguousarray` definition."""

        a = xp_as_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], xp=xp)

        # Identity-preserving fast path: an already-contiguous *NumPy*
        # array is returned with C-contig layout.
        result = xp_ascontiguousarray(a, xp=xp)
        xp_assert_equal(result, a)

        if is_numpy_namespace(xp):
            assert result.flags["C_CONTIGUOUS"]

        # Materialise a transposed view: the value-equivalent of
        # ``matrix_transpose`` but C-contiguous downstream.
        transposed = array_namespace(a).matrix_transpose(a)
        materialised = xp_ascontiguousarray(transposed, xp=xp)
        xp_assert_equal(materialised, transposed)
        if is_numpy_namespace(xp):
            assert not transposed.flags["C_CONTIGUOUS"]
            assert materialised.flags["C_CONTIGUOUS"]

        # Broadcast outputs derived from the materialised array stay
        # C-contiguous, demonstrating the cascade.
        if is_numpy_namespace(xp):
            broadcast = materialised[..., None] * xp.asarray([1.0, 2.0, 3.0])
            assert broadcast.flags["C_CONTIGUOUS"]

        # The namespace is derived from the input when none is passed.
        a = np.array([[1, 2], [3, 4]]).T
        result = xp_ascontiguousarray(a)
        assert result.flags["C_CONTIGUOUS"]
        xp_assert_equal(result, a)


class TestXpMatrixTranspose:
    """Define :func:`colour.utilities.xp_matrix_transpose` unit tests."""

    def test_xp_matrix_transpose(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_matrix_transpose` definition."""

        xpc = array_namespace(xp.asarray([0.0]))

        # 2-D case: swap the two axes.
        a = xp_as_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], xp=xp)
        result = xp_matrix_transpose(a, xp=xp)
        expected = xpc.matrix_transpose(a)
        xp_assert_equal(result, expected)
        assert result.shape == (3, 2)

        # 3-D case: swap the last two axes only.
        b = xp.reshape(
            xp_as_array(list(range(24)), dtype=DTYPE_FLOAT_DEFAULT, xp=xp), (2, 3, 4)
        )
        result_3d = xp_matrix_transpose(b, xp=xp)
        assert result_3d.shape == (2, 4, 3)
        xp_assert_equal(result_3d, xpc.matrix_transpose(b))

        # Materialised result is C-contiguous on *NumPy* even when the
        # input was a transposed view that would be F-contiguous.
        if is_numpy_namespace(xp):
            assert result.flags["C_CONTIGUOUS"]
            assert result_3d.flags["C_CONTIGUOUS"]

        # Downstream broadcast cascade: a *NumPy* broadcast with a
        # ``xp_matrix_transpose`` operand stays C-contiguous, unlike the
        # strided ``matrix_transpose`` view that propagates the F-stride
        # pattern.
        c = np.arange(8.0).reshape(2, 4)
        strided = np.matrix_transpose(c)
        broadcast_strided = strided[..., None] * np.array([1.0, 2.0, 3.0])
        assert not broadcast_strided.flags["C_CONTIGUOUS"]
        materialised = xp_matrix_transpose(c, xp=np)
        broadcast_materialised = materialised[..., None] * np.array([1.0, 2.0, 3.0])
        assert broadcast_materialised.flags["C_CONTIGUOUS"]
        xp_assert_equal(broadcast_materialised, broadcast_strided)

        # The namespace is derived from the input when none is passed.
        d = np.arange(6).reshape(2, 3)
        result = xp_matrix_transpose(d)
        assert result.shape == (3, 2)
        assert result.flags["C_CONTIGUOUS"]


class TestXpSelect:
    """Define :func:`colour.utilities.xp_select` unit tests."""

    def test_xp_select(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_select` definition."""

        x = xp.arange(10)
        condlist = [x < 3, x > 6]
        choicelist = [x * 10, x * 100]
        result = xp_select(condlist, choicelist, default=-1.0, xp=xp)
        expected = np.select(
            [as_ndarray(x < 3), as_ndarray(x > 6)],
            [as_ndarray(x * 10), as_ndarray(x * 100)],
            default=-1.0,
        )
        xp_assert_equal(result, expected)

        x = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = xp_select([x > 2], [x * 10], default=0.0, xp=xp)
        expected = np.select([as_ndarray(x > 2)], [as_ndarray(x * 10)], default=0.0)
        xp_assert_equal(result, expected)

        # All-False ``condlist``: every element must take the ``default``.
        x = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = xp_select([x > 100], [x * 10], default=-1.0, xp=xp)
        expected = np.select([as_ndarray(x > 100)], [as_ndarray(x * 10)], default=-1.0)
        xp_assert_equal(result, expected)


class TestRuntimeWarningXpFallback:
    """Define tests for the backend-to-*NumPy* fallback warning."""

    @pytest.mark.parametrize(
        ("name", "alternative"),
        [
            ("xp_interp", "LinearInterpolator"),
            ("xp_unique", "no generally gradient-preserving equivalent"),
            ("xp_lstsq", "linalg.lstsq"),
        ],
    )
    def test_runtime_warning_xp_fallback(self, name: str, alternative: str) -> None:
        """Test that fallback warnings explain graph-safe alternatives."""

        with pytest.warns(
            ColourRuntimeWarning,
            match="will not preserve automatic differentiation graphs",
        ) as warnings:
            utilities_array._runtime_warning_xp_fallback(name)  # noqa: SLF001

        assert alternative in str(warnings[0].message)


class TestXpInterp:
    """Define :func:`colour.utilities.xp_interp` unit tests."""

    def test_xp_interp(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_interp` definition."""

        xp_arr = xp_as_array([0.0, 1.0, 2.0, 3.0], xp=xp)
        fp = xp_as_array([0.0, 1.0, 4.0, 9.0], xp=xp)
        x = xp_as_array([0.5, 1.5, 2.5], xp=xp)
        result = xp_interp(x, xp_arr, fp, xp=xp)
        expected = np.interp(
            np.array([0.5, 1.5, 2.5]),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 1.0, 4.0, 9.0]),
        )
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_arr = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        fp = xp_as_array([10.0, 20.0, 30.0], xp=xp)
        x = xp_as_array([0.0, 4.0], xp=xp)
        result = xp_interp(x, xp_arr, fp, xp=xp)
        expected = np.interp(
            np.array([0.0, 4.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
        )
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpTrapezoid:
    """Define :func:`colour.utilities.xp_trapezoid` unit tests."""

    def test_xp_trapezoid(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_trapezoid` definition."""

        y = xp_as_array([1.0, 2.0, 3.0, 4.0], xp=xp)
        x = xp_as_array([0.0, 1.0, 2.0, 3.0], xp=xp)
        result = xp_trapezoid(y, x=x, xp=xp)
        expected = np.trapezoid(
            np.array([1.0, 2.0, 3.0, 4.0]), x=np.array([0.0, 1.0, 2.0, 3.0])
        )
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        y = xp_as_array([1.0, 4.0, 9.0], xp=xp)
        result = xp_trapezoid(y, dx=0.5, xp=xp)
        expected = np.trapezoid(np.array([1.0, 4.0, 9.0]), dx=0.5)
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpAverage:
    """Define :func:`colour.utilities.xp_average` unit tests."""

    def test_xp_average(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_average` definition."""

        a = xp_as_array([1.0, 2.0, 3.0, 4.0], xp=xp)
        result = xp_average(a, xp=xp)
        expected = np.average(np.array([1.0, 2.0, 3.0, 4.0]))
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        weights = xp_as_array([4.0, 3.0, 2.0, 1.0], xp=xp)
        result = xp_average(a, weights=weights, xp=xp)
        expected = np.average(
            np.array([1.0, 2.0, 3.0, 4.0]), weights=np.array([4.0, 3.0, 2.0, 1.0])
        )
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_as_array([[1.0, 2.0], [3.0, 4.0]], xp=xp)
        result = xp_average(a, axis=0, xp=xp)
        expected = np.average(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0)
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpGradient:
    """Define :func:`colour.utilities.xp_gradient` unit tests."""

    def test_xp_gradient(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_gradient` definition."""

        f = xp_as_array([1.0, 4.0, 9.0, 16.0, 25.0], xp=xp)
        result = xp_gradient(f, xp=xp)
        expected = np.gradient(np.array([1.0, 4.0, 9.0, 16.0, 25.0]))
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        f = xp_as_array([1.0, 4.0, 9.0, 16.0], xp=xp)
        result = xp_gradient(f, 0.5, xp=xp)
        expected = np.gradient(np.array([1.0, 4.0, 9.0, 16.0]), 0.5)
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpResize:
    """Define :func:`colour.utilities.xp_resize` unit tests."""

    def test_xp_resize(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_resize` definition."""

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = xp_resize(a, (6,), xp=xp)
        expected = np.resize(np.array([1.0, 2.0, 3.0]), (6,))
        xp_assert_equal(result, expected)

        a = xp_as_array([1.0, 2.0], xp=xp)
        result = xp_resize(a, (3, 2), xp=xp)
        expected = np.resize(np.array([1.0, 2.0]), (3, 2))
        xp_assert_equal(result, expected)

        # Shape contraction: target smaller than input.
        a = xp_as_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], xp=xp)
        result = xp_resize(a, (3,), xp=xp)
        expected = np.resize(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (3,))
        xp_assert_equal(result, expected)

        # Zero-element target.
        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = xp_resize(a, (0,), xp=xp)
        expected = np.resize(np.array([1.0, 2.0, 3.0]), (0,))
        xp_assert_equal(result, expected)

    def test_xp_resize_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test that resizing preserves automatic differentiation."""

        resized, (gradient,), _inputs = autodiff(
            lambda a: xp_resize(a, (2, 3), xp=xp), [1.0, 2.0, 3.0]
        )

        xp_assert_equal(resized, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        xp_assert_equal(gradient, [2.0, 2.0, 2.0])


class TestXpNanmean:
    """Define :func:`colour.utilities.xp_nanmean` unit tests."""

    def test_xp_nanmean(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_nanmean` definition."""

        a = xp_as_array([1.0, np.nan, 3.0, np.nan, 5.0], xp=xp)
        result = xp_nanmean(a, xp=xp)
        expected = np.nanmean(np.array([1.0, np.nan, 3.0, np.nan, 5.0]))
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = xp_nanmean(a, xp=xp)
        expected = np.nanmean(np.array([1.0, 2.0, 3.0]))
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_as_array([[1.0, np.nan], [3.0, 4.0]], xp=xp)
        result = xp_nanmean(a, axis=0, xp=xp)
        expected = np.nanmean(np.array([[1.0, np.nan], [3.0, 4.0]]), axis=0)
        xp_assert_close(
            result,
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpMedian:
    """Define :func:`colour.utilities.xp_median` unit tests."""

    def test_xp_median(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_median` definition."""

        a = xp_as_array([3.0, 1.0, 2.0], xp=xp)
        xp_assert_close(
            xp_median(a, xp=xp),
            np.median([3.0, 1.0, 2.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_as_array([4.0, 1.0, 3.0, 2.0], xp=xp)
        xp_assert_close(
            xp_median(a, xp=xp),
            np.median([4.0, 1.0, 3.0, 2.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_as_array([[3.0, 1.0], [2.0, 4.0]], xp=xp)
        xp_assert_close(
            xp_median(a, axis=1, xp=xp),
            np.median([[3.0, 1.0], [2.0, 4.0]], axis=1),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpRound:
    """Define :func:`colour.utilities.xp_round` unit tests."""

    def test_xp_round(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_round` definition."""

        a = xp_as_array([3.14159, 2.71828, 1.41421], xp=xp)
        a_np = np.array([3.14159, 2.71828, 1.41421])

        xp_assert_close(
            xp_round(a, decimals=0, xp=xp),
            np.round(a_np, 0),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            xp_round(a, decimals=2, xp=xp),
            np.round(a_np, 2),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            xp_round(a, decimals=4, xp=xp),
            np.round(a_np, 4),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_as_array([[1.555, 2.444], [3.666, 4.777]], xp=xp)
        xp_assert_close(
            xp_round(a, decimals=1, xp=xp),
            np.round([[1.555, 2.444], [3.666, 4.777]], 1),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpRadians:
    """Define :func:`colour.utilities.xp_radians` unit tests."""

    def test_xp_radians(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_radians` definition."""

        a = xp_as_array([0.0, 90.0, 180.0, 270.0, 360.0], xp=xp)
        xp_assert_close(
            xp_radians(a),
            np.radians([0.0, 90.0, 180.0, 270.0, 360.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xp_radians(180.0),
            np.pi,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpDegrees:
    """Define :func:`colour.utilities.xp_degrees` unit tests."""

    def test_xp_degrees(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_degrees` definition."""

        a = xp_as_array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], xp=xp)
        xp_assert_close(
            xp_degrees(a),
            np.degrees([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xp_degrees(np.pi),
            180.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpAtleast1d:
    """Define :func:`colour.utilities.xp_atleast_1d` unit tests."""

    def test_xp_atleast_1d(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_atleast_1d` definition."""

        result = xp_atleast_1d(xp_as_array(1.0, xp=xp))
        assert as_ndarray(result).ndim == 1

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = xp_atleast_1d(a)
        xp_assert_equal(result, [1.0, 2.0, 3.0])

        # Python scalar input: the canonical scalar-promotion path.
        result = xp_atleast_1d(1.0)
        assert as_ndarray(result).ndim == 1
        xp_assert_equal(result, [1.0])


class TestXpAtleast2d:
    """Define :func:`colour.utilities.xp_atleast_2d` unit tests."""

    def test_xp_atleast_2d(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_atleast_2d` definition."""

        result = xp_atleast_2d(xp_as_array([1.0, 2.0, 3.0], xp=xp))
        assert as_ndarray(result).ndim == 2
        assert as_ndarray(result).shape == (1, 3)

        a = xp_as_array([[1.0, 2.0], [3.0, 4.0]], xp=xp)
        result = xp_atleast_2d(a)
        xp_assert_equal(result, [[1.0, 2.0], [3.0, 4.0]])

        # Python scalar input: the canonical scalar-promotion path.
        result = xp_atleast_2d(1.0)
        assert as_ndarray(result).ndim == 2
        assert as_ndarray(result).shape == (1, 1)
        xp_assert_equal(result, [[1.0]])


class TestXpSqueeze:
    """Define :func:`colour.utilities.xp_squeeze` unit tests."""

    def test_xp_squeeze(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_squeeze` definition."""

        a = xp_as_array([[1.0, 2.0]], xp=xp)
        xp_assert_close(xp_squeeze(a, xp=xp), [1.0, 2.0])

        a = xp_as_array([[[1.0], [2.0]]], xp=xp)
        result = xp_squeeze(a, axis=0, xp=xp)
        xp_assert_close(result, [[1.0], [2.0]])

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        xp_assert_close(xp_squeeze(a, xp=xp), [1.0, 2.0, 3.0])


class TestXpSinc:
    """Define :func:`colour.utilities.xp_sinc` unit tests."""

    def test_xp_sinc(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_sinc` definition."""

        a = xp_as_array([0.0, 0.5, 1.0, 1.5], xp=xp)
        xp_assert_close(
            xp_sinc(a),
            np.sinc([0.0, 0.5, 1.0, 1.5]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpIsclose:
    """Define :func:`colour.utilities.xp_isclose` unit tests."""

    def test_xp_isclose(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_isclose` definition."""

        a = xp_as_array([1.0, 2.0001, 3.0], xp=xp)
        b = xp_as_array([1.0, 2.0, 3.0], xp=xp)

        xp_assert_equal(
            xp_isclose(a, b, atol=TOLERANCE_ABSOLUTE_TESTS * 10000),
            np.isclose(
                [1.0, 2.0001, 3.0],
                [1.0, 2.0, 3.0],
                atol=TOLERANCE_ABSOLUTE_TESTS * 10000,
            ),
        )
        xp_assert_equal(
            xp_isclose(a, b, atol=TOLERANCE_ABSOLUTE_TESTS * 100),
            np.isclose(
                [1.0, 2.0001, 3.0],
                [1.0, 2.0, 3.0],
                atol=TOLERANCE_ABSOLUTE_TESTS * 100,
            ),
        )


class TestXpNanToNum:
    """Define :func:`colour.utilities.xp_nan_to_num` unit tests."""

    def test_xp_nan_to_num(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_nan_to_num` definition."""

        a = xp_as_array([1.0, np.nan, np.inf, -np.inf], xp=xp)

        xp_assert_equal(
            xp_nan_to_num(a),
            np.nan_to_num(
                np.array([1.0, np.nan, np.inf, -np.inf], dtype=DTYPE_FLOAT_DEFAULT)
            ),
        )

        xp_assert_equal(
            xp_nan_to_num(a, nan=0.0, posinf=999.0, neginf=-999.0),
            np.nan_to_num(
                [1.0, np.nan, np.inf, -np.inf],
                nan=0.0,
                posinf=999.0,
                neginf=-999.0,
            ),
        )


class TestXpCreateDiagonal:
    """Define :func:`colour.utilities.xp_create_diagonal` unit tests."""

    def test_xp_create_diagonal(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_create_diagonal` definition."""

        v = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = xp_create_diagonal(v)
        xp_assert_equal(result, np.diag([1.0, 2.0, 3.0]))


class TestXpReshape:
    """Define :func:`colour.utilities.xp_reshape` unit tests."""

    def test_xp_reshape(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_reshape` definition."""

        a = xp.arange(6.0)

        result = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        expected = np.arange(6.0).reshape((2, 3))
        xp_assert_equal(result, expected)

        result = xp_reshape(xp_as_array(a, xp=xp), (-1, 2), xp=xp)
        expected = np.arange(6.0).reshape((-1, 2))
        xp_assert_equal(result, expected)

        a_int = xp_as_array([1, 2, 3, 4], xp=xp)
        result = xp_reshape(xp_as_array(a_int, xp=xp), (2, 2), xp=xp)
        expected = np.array([1, 2, 3, 4]).reshape((2, 2))
        xp_assert_equal(result, expected)


class TestXpBroadcastTo:
    """Define :func:`colour.utilities.xp_broadcast_to` unit tests."""

    def test_xp_broadcast_to(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_broadcast_to` definition."""

        # Scalar to a 2-D shape.
        result = xp_broadcast_to(xp_as_array(5.0, xp=xp), (2, 3), xp=xp)
        expected = np.broadcast_to(np.array(5.0), (2, 3))
        xp_assert_equal(result, expected)

        # 1-D row to a 2-D shape (broadcast over leading axis).
        result = xp_broadcast_to(xp_as_array([1.0, 2.0, 3.0], xp=xp), (4, 3), xp=xp)
        expected = np.broadcast_to(np.array([1.0, 2.0, 3.0]), (4, 3))
        xp_assert_equal(result, expected)

        # Identity broadcast: shape unchanged.
        a = xp_as_array([[1.0, 2.0], [3.0, 4.0]], xp=xp)
        result = xp_broadcast_to(a, (2, 2), xp=xp)
        xp_assert_equal(result, [[1.0, 2.0], [3.0, 4.0]])


class TestXpEig:
    """Define :func:`colour.utilities.xp_eig` unit tests."""

    def test_xp_eig(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_eig` definition."""

        A = xp_as_array([[1.0, 2.0], [3.0, 4.0]], xp=xp)
        w, v = xp_eig(A, xp=xp)
        w_np, v_np = np.linalg.eig(np.array([[1.0, 2.0], [3.0, 4.0]]))
        xp_assert_close(w, w_np, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xp.abs(v), np.abs(v_np), atol=TOLERANCE_ABSOLUTE_TESTS)


class TestXpEigh:
    """Define :func:`colour.utilities.xp_eigh` unit tests."""

    def test_xp_eigh(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_eigh` definition."""

        A = xp_as_array([[2.0, 1.0], [1.0, 3.0]], xp=xp)
        w, v = xp_eigh(A, xp=xp)
        w_np, v_np = np.linalg.eigh(np.array([[2.0, 1.0], [1.0, 3.0]]))
        xp_assert_close(w, w_np, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xp.abs(v), np.abs(v_np), atol=TOLERANCE_ABSOLUTE_TESTS)


class TestXpLstsq:
    """Define :func:`colour.utilities.xp_lstsq` unit tests."""

    def test_xp_lstsq(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_lstsq` definition."""

        A = xp_as_array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], xp=xp)
        b = xp_as_array([[1.0], [2.0], [3.0]], xp=xp)

        result = xp_lstsq(A, b)
        expected = np.linalg.lstsq(
            np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
            np.array([[1.0], [2.0], [3.0]]),
            rcond=None,
        )[0]
        xp_assert_close(result, expected, atol=TOLERANCE_ABSOLUTE_TESTS)


class TestXpIsin:
    """Define :func:`colour.utilities.xp_isin` unit tests."""

    def test_xp_isin(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_isin` definition."""

        a = xp_as_array([1.0, 2.0, 3.0, 4.0, 5.0], xp=xp)
        b = xp_as_array([2.0, 4.0], xp=xp)

        xp_assert_equal(
            xp_isin(a, b, xp=xp),
            np.isin([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0]),
        )

        a = xp_as_array([10.0, 20.0, 30.0], xp=xp)
        b = xp_as_array([5.0, 10.0, 15.0, 20.0], xp=xp)

        xp_assert_equal(
            xp_isin(a, b, xp=xp),
            np.isin([10.0, 20.0, 30.0], [5.0, 10.0, 15.0, 20.0]),
        )

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        b = xp_as_array([4.0, 5.0, 6.0], xp=xp)

        xp_assert_equal(
            xp_isin(a, b, xp=xp),
            np.isin([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        )

        # Empty ``test_elements``: every entry is absent.
        xp_assert_equal(
            xp_isin(a, xp_as_array([], xp=xp), xp=xp),
            np.isin([1.0, 2.0, 3.0], []),
        )

        # :class:`NaN` is not equal to itself in :func:`numpy.isin`; the
        # backend wrappers preserve that contract.
        a = xp_as_array([1.0, np.nan, 3.0], xp=xp)
        b = xp_as_array([np.nan], xp=xp)

        xp_assert_equal(
            xp_isin(a, b, xp=xp),
            np.isin([1.0, np.nan, 3.0], [np.nan]),
        )


class TestXpLinspace:
    """Define :func:`colour.utilities.xp_linspace` unit tests."""

    def test_xp_linspace(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_linspace` definition."""

        result = xp_linspace(0, 10, num=5, xp=xp)
        expected = np.linspace(0, 10, 5)
        xp_assert_equal(result, expected)  # pyright: ignore

        result, step = xp_linspace(0, 1, retstep=True, num=11, xp=xp)
        expected, expected_step = np.linspace(0, 1, 11, retstep=True)
        xp_assert_close(result, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(
            step,
            expected_step,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestXpPad:
    """Define :func:`colour.utilities.xp_pad` unit tests."""

    def test_xp_pad(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_pad` definition."""

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)

        result = xp_pad(a, (2, 3), xp=xp)
        expected = np.pad(np.array([1.0, 2.0, 3.0]), (2, 3))
        xp_assert_equal(result, expected)

        result = xp_pad(a, (1, 1), "wrap", xp=xp)
        expected = np.pad(np.array([1.0, 2.0, 3.0]), (1, 1), "wrap")
        xp_assert_equal(result, expected)


class TestXpUnique:
    """Define :func:`colour.utilities.xp_unique` unit tests."""

    def test_xp_unique(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_unique` definition."""

        a = xp_as_array([3.0, 1.0, 2.0, 1.0, 3.0], xp=xp)

        result = xp_unique(a, xp=xp)
        expected = np.unique(np.array([3.0, 1.0, 2.0, 1.0, 3.0]))
        xp_assert_equal(result, expected)

        result, indexes = xp_unique(a, return_index=True, xp=xp)
        expected, expected_indexes = np.unique(
            np.array([3.0, 1.0, 2.0, 1.0, 3.0]), return_index=True
        )
        xp_assert_equal(result, expected)
        xp_assert_equal(indexes, expected_indexes)

        a = xp_as_array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]], xp=xp)
        result, indexes = xp_unique(a, axis=0, return_index=True, xp=xp)
        expected, expected_indexes = np.unique(
            np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]),
            axis=0,
            return_index=True,
        )
        xp_assert_equal(result, expected)
        xp_assert_equal(indexes, expected_indexes)


class TestXpInsert:
    """Define :func:`colour.utilities.xp_insert` unit tests."""

    def test_xp_insert(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_insert` definition."""

        a = xp_as_array([1.0, 2.0, 3.0, 4.0, 5.0], xp=xp)
        indices = xp_as_array([1, 3], xp=xp)
        values = xp_as_array([10.0, 30.0], xp=xp)

        result = xp_insert(a, indices, values, xp=xp)
        expected = np.insert(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([1, 3]),
            np.array([10.0, 30.0]),
        )
        xp_assert_equal(result, expected)

        result = xp_insert(
            a, xp_as_array([0], xp=xp), xp_as_array([99.0], xp=xp), xp=xp
        )
        expected = np.insert(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), [0], [99.0])
        xp_assert_equal(result, expected)

        result = xp_insert(
            a, xp_as_array([5], xp=xp), xp_as_array([99.0], xp=xp), xp=xp
        )
        expected = np.insert(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), [5], [99.0])
        xp_assert_equal(result, expected)

        # 2-D row insertion: insert two new rows into a ``(4, 3)`` array.
        a = xp_as_array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            xp=xp,
        )
        indices = xp_as_array([1, 3], xp=xp)
        values = xp_as_array([[-1.0, -2.0, -3.0], [-7.0, -8.0, -9.0]], xp=xp)
        result = xp_insert(a, indices, values, axis=0, xp=xp)
        expected = np.insert(
            np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
            ),
            np.array([1, 3]),
            np.array([[-1.0, -2.0, -3.0], [-7.0, -8.0, -9.0]]),
            axis=0,
        )
        xp_assert_equal(result, expected)

        # 2-D column insertion: insert one new column at index 2.
        indices = xp_as_array([2], xp=xp)
        values = xp_as_array([[99.0], [99.0], [99.0], [99.0]], xp=xp)
        result = xp_insert(a, indices, values, axis=1, xp=xp)
        expected = np.insert(
            np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
            ),
            np.array([2]),
            np.array([[99.0], [99.0], [99.0], [99.0]]),
            axis=1,
        )
        xp_assert_equal(result, expected)

        # Unsorted indices: must normalise to the sorted equivalent, matching
        # :func:`numpy.insert` behaviour.
        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        indices = xp_as_array([2, 0], xp=xp)
        values = xp_as_array([99.0, 88.0], xp=xp)
        result = xp_insert(a, indices, values, xp=xp)
        expected = np.insert(
            np.array([1.0, 2.0, 3.0]), np.array([2, 0]), np.array([99.0, 88.0])
        )
        xp_assert_equal(result, expected)

        # Empty indices / values: identity, no-op.
        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)
        result = xp_insert(
            a,
            xp_as_array(np.array([], dtype=np.int64), xp=xp),
            xp_as_array([], xp=xp),
            xp=xp,
        )
        xp_assert_equal(result, np.array([1.0, 2.0, 3.0]))

        # Negative axis (axis=-1) on a 2-D input must equal axis=1.
        a_2d = xp_as_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], xp=xp)
        result = xp_insert(
            a_2d,
            xp_as_array([1], xp=xp),
            xp_as_array([[99.0], [88.0]], xp=xp),
            axis=-1,
            xp=xp,
        )
        expected = np.insert(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([1]),
            np.array([[99.0], [88.0]]),
            axis=-1,
        )
        xp_assert_equal(result, expected)


class TestXpSetxor1d:
    """Define :func:`colour.utilities.xp_setxor1d` unit tests."""

    def test_xp_setxor1d(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_setxor1d` definition."""

        a = xp_as_array([1.0, 2.0, 3.0, 4.0], xp=xp)
        b = xp_as_array([3.0, 4.0, 5.0, 6.0], xp=xp)

        result = xp_setxor1d(a, b, xp=xp)
        expected = np.setxor1d(
            np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0, 6.0])
        )
        xp_assert_equal(result, expected)

        result = xp_setxor1d(a, a, xp=xp)
        expected = np.setxor1d(
            np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0])
        )
        xp_assert_equal(result, expected)

        result = xp_setxor1d(a, xp_as_array([10.0, 20.0], xp=xp), xp=xp)
        expected = np.setxor1d(np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0]))
        xp_assert_equal(result, expected)

        # Empty operands: identity-on-the-other.
        result = xp_setxor1d(a, xp_as_array([], xp=xp), xp=xp)
        xp_assert_equal(result, np.array([1.0, 2.0, 3.0, 4.0]))

        result = xp_setxor1d(xp_as_array([], xp=xp), xp_as_array([], xp=xp), xp=xp)
        xp_assert_equal(result, np.array([], dtype=float))


class TestXpAssertClose:
    """Define :func:`colour.utilities.xp_assert_close` unit tests."""

    def test_xp_assert_close(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_assert_close` definition."""

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)

        xp_assert_close(a, [1.0, 2.0, 3.0])
        xp_assert_close(a, xp_as_array([1.0, 2.0, 3.0], xp=xp))

        with pytest.raises(AssertionError):
            xp_assert_close(a, [1.0, 2.0, 3.5])

        xp_assert_close(a, [1.0, 2.0, 3.5], atol=1.0)
        xp_assert_close(a, [1.0, 2.0, 3.5], rtol=1.0)

        with pytest.raises(AssertionError):
            xp_assert_close(a, [1.0, 2.0, 3.5], rtol=0.1, atol=0.1)

        # Default tolerances are resolved at call time so that fixtures
        # relaxing the module-level constant also relax defaulted calls.
        default = utilities_array.TOLERANCE_ABSOLUTE_TESTS
        try:
            utilities_array.TOLERANCE_ABSOLUTE_TESTS = 1.0
            xp_assert_close(np.array([1.0]), np.array([1.5]))
        finally:
            utilities_array.TOLERANCE_ABSOLUTE_TESTS = default


class TestXpAssertEqual:
    """Define :func:`colour.utilities.xp_assert_equal` unit tests."""

    def test_xp_assert_equal(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.xp_assert_equal` definition."""

        a = xp_as_array([1.0, 2.0, 3.0], xp=xp)

        xp_assert_equal(a, [1.0, 2.0, 3.0])
        xp_assert_equal(a, xp_as_array([1.0, 2.0, 3.0], xp=xp))

        with pytest.raises(AssertionError):
            xp_assert_equal(a, [1.0, 2.0, 4.0])


class TestMixinDataclassFields:
    """
    Define :class:`colour.utilities.array.MixinDataclassFields` class unit
    tests methods.
    """

    def setup_method(self) -> None:
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


class TestMixinDataclassIterable:
    """
    Define :class:`colour.utilities.array.MixinDataclassIterable` class unit
    tests methods.
    """

    def setup_method(self) -> None:
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


class TestMixinDataclassArray:
    """
    Define :class:`colour.utilities.array.MixinDataclassArray` class unit
    tests methods.
    """

    def setup_method(self) -> None:
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

        xp_assert_equal(self._data, self._array)

        assert np.array(self._data, dtype=DTYPE_INT_DEFAULT).dtype == DTYPE_INT_DEFAULT


class TestMixinDataclassArithmetic:
    """
    Define :class:`colour.utilities.array.MixinDataclassArithmetic` class unit
    tests methods.
    """

    def setup_method(self) -> None:
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

        xp_assert_close(
            self._data.arithmetical_operation(10, "+", False),
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data.arithmetical_operation(10, "-", False),
            self._array - 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data.arithmetical_operation(10, "*", False),
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data.arithmetical_operation(10, "/", False),
            self._array / 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data.arithmetical_operation(10, "**", False),
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data + 10,
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data - 10,
            self._array - 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data * 10,
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data / 10,
            self._array / 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            self._data**10,
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        data = deepcopy(self._data)

        xp_assert_close(
            data.arithmetical_operation(10, "+", True),
            self._array + 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            data.arithmetical_operation(10, "-", True),
            self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            data.arithmetical_operation(10, "*", True),
            self._array * 10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            data.arithmetical_operation(10, "/", True),
            self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            data.arithmetical_operation(10, "**", True),
            self._array**10,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        data = deepcopy(self._data)

        xp_assert_close(
            data.arithmetical_operation(self._array, "+", False),
            data + self._array,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
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


class TestAsArray:
    """
    Define :func:`colour.utilities.array.as_array` definition unit tests
    methods.
    """

    def test_as_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.as_array` definition."""

        xp_assert_equal(as_array(xp_as_array([1, 2, 3], xp=xp)), [1, 2, 3])

        assert as_array([1, 2, 3], DTYPE_FLOAT_DEFAULT).dtype == DTYPE_FLOAT_DEFAULT

        assert as_array([1, 2, 3], DTYPE_INT_DEFAULT).dtype == DTYPE_INT_DEFAULT

        xp_assert_equal(
            as_array(dict(zip("abc", [1, 2, 3], strict=True)).values()),
            [1, 2, 3],
        )

    def test_as_array_autodiff(self, xp: ModuleType, autodiff: typing.Callable) -> None:
        """Test that conversion preserves automatic differentiation."""

        converted, (gradient,), (original,) = autodiff(
            lambda a: as_array(a, DTYPE_FLOAT_DEFAULT), [1.0, 2.0, 3.0]
        )

        xp_assert_equal(gradient, xp.ones_like(original))

        _stacked, (gradient,), (original,) = autodiff(
            lambda a: as_array([a, a * 2]), [1.0, 2.0, 3.0]
        )

        xp_assert_equal(gradient, xp.full_like(original, 3))


class TestAsInt:
    """
    Define :func:`colour.utilities.array.as_int` definition unit tests
    methods.
    """

    def test_as_int(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.as_int` definition."""

        assert as_int(1) == 1

        assert as_int(xp_as_array([1], xp=xp)).ndim == 1

        assert as_int(xp_as_array([[1]], xp=xp)).ndim == 2

        xp_assert_equal(as_int(xp_as_array([1.0, 2.0, 3.0], xp=xp)), [1, 2, 3])

        assert as_int(np.array([1.0, 2.0, 3.0])).dtype == DTYPE_INT_DEFAULT

        assert isinstance(as_int(1), DTYPE_INT_DEFAULT)


class TestAsFloat:
    """
    Define :func:`colour.utilities.array.as_float` definition unit tests
    methods.
    """

    def test_as_float(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.as_float` definition."""

        assert as_float(1) == 1.0

        assert as_float(xp_as_array([1], xp=xp)).ndim == 1

        assert as_float(xp_as_array([[1]], xp=xp)).ndim == 2

        xp_assert_close(
            as_float(xp_as_array([1, 2, 3], xp=xp)),
            [1.0, 2.0, 3.0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert as_float(np.array([1, 2, 3])).dtype == DTYPE_FLOAT_DEFAULT

        if is_numpy_namespace(xp):
            assert isinstance(as_float(1), DTYPE_FLOAT_DEFAULT)


class TestAsIntArray:
    """
    Define :func:`colour.utilities.array.as_int_array` definition unit tests
    methods.
    """

    def test_as_int_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.as_int_array` definition."""

        xp_assert_equal(
            as_int_array(xp_as_array([1.0, 2.0, 3.0], xp=xp)),
            [1, 2, 3],
        )

        assert as_int_array([1, 2, 3]).dtype == DTYPE_INT_DEFAULT


class TestAsFloatArray:
    """
    Define :func:`colour.utilities.array.as_float_array` definition unit tests
    methods.
    """

    def test_as_float_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.as_float_array` definition."""

        xp_assert_equal(
            as_float_array(xp_as_array([1, 2, 3], xp=xp)),
            [1, 2, 3],
        )

        assert as_float_array([1, 2, 3]).dtype == DTYPE_FLOAT_DEFAULT


class TestAsComplexArray:
    """
    Define :func:`colour.utilities.array.as_complex_array` definition unit tests
    methods.
    """

    def test_as_complex_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.as_complex_array` definition."""

        if not is_numpy_namespace(xp):
            probe = xp_as_array([1], xp=xp)
            xp_compat = array_namespace(probe)
            xp_complex_dtype = getattr(
                xp_compat, np.dtype(DTYPE_COMPLEX_DEFAULT).name, None
            )
            if xp_complex_dtype is None:
                pytest.skip("Backend does not support the default complex dtype.")
            try:
                xp_compat.asarray(probe, dtype=xp_complex_dtype)
            except TypeError:
                pytest.skip("Backend does not support the default complex dtype.")

        xp_assert_equal(
            as_complex_array(xp_as_array([1, 2, 3], xp=xp)),
            [1 + 0j, 2 + 0j, 3 + 0j],
        )

        xp_assert_equal(
            as_complex_array(xp_as_array([1 + 2j, 3 + 4j], xp=xp)),
            [1 + 2j, 3 + 4j],
        )

        assert as_complex_array([1, 2, 3]).dtype == DTYPE_COMPLEX_DEFAULT

        assert as_complex_array([1, 2, 3], np.complex64).dtype == np.complex64


class TestAsIntScalar:
    """
    Define :func:`colour.utilities.array.as_int_scalar` definition unit tests
    methods.
    """

    def test_as_int_scalar(self) -> None:
        """Test :func:`colour.utilities.array.as_int_scalar` definition."""

        assert float(as_int_scalar(1.0)) == 1

        assert as_int_scalar(1.0).dtype == DTYPE_INT_DEFAULT  # pyright: ignore


class TestAsFloatScalar:
    """
    Define :func:`colour.utilities.array.as_float_scalar` definition unit
    tests methods.
    """

    def test_as_float_scalar(self) -> None:
        """Test :func:`colour.utilities.array.as_float_scalar` definition."""

        assert float(as_float_scalar(1)) == 1.0

        assert as_float_scalar(1).dtype == DTYPE_FLOAT_DEFAULT  # pyright: ignore


class TestSetDefaultIntegerDtype:
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


class TestSetDefaultFloatDtype:
    """
    Define :func:`colour.utilities.array.set_default_float_dtype` definition unit
    tests methods.
    """

    @pytest.fixture(autouse=True)
    def _restore_default_float_dtype(self) -> Generator[None, None, None]:
        """
        Restore the default float dtype after each test to avoid
        cross-test bleed under *pytest-xdist*.
        """

        yield
        set_default_float_dtype(np.float64)

    def test_set_default_float_dtype(self) -> None:
        """
        Test :func:`colour.utilities.array.set_default_float_dtype`
        definition.
        """

        assert as_float_array(np.ones(3)).dtype == np.float64

        set_default_float_dtype(np.float16)

        assert as_float_array(np.ones(3)).dtype == np.float16

        set_default_float_dtype(np.float64)

        assert as_float_array(np.ones(3)).dtype == np.float64

    def test_set_default_float_dtype_enforcement(self) -> None:
        """
        Test whether :func:`colour.utilities.array.set_default_float_dtype`
        effect is applied through most of *Colour* public API.
        """

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


class TestGetDomainRangeScale:
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


class TestSetDomainRangeScale:
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


class TestDomainRangeScale:
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


class TestGetDomainRangeScaleMetadata:
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


class TestToDomain1:
    """
    Define :func:`colour.utilities.common.to_domain_1` definition unit
    tests methods.
    """

    def test_to_domain_1(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_1` definition."""

        with domain_range_scale("Reference"):
            assert float(to_domain_1(1)) == 1

        with domain_range_scale("1"):
            assert float(to_domain_1(1)) == 1

        with domain_range_scale("100"):
            assert float(to_domain_1(1)) == 0.01

        with domain_range_scale("100"):
            assert float(to_domain_1(1, np.pi)) == 1 / np.pi

        with domain_range_scale("100"):
            assert to_domain_1(1, dtype=np.float16).dtype == np.float16


class TestToDomain10:
    """
    Define :func:`colour.utilities.common.to_domain_10` definition unit
    tests methods.
    """

    def test_to_domain_10(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_10` definition."""

        with domain_range_scale("Reference"):
            assert float(to_domain_10(1)) == 1

        with domain_range_scale("1"):
            assert float(to_domain_10(1)) == 10

        with domain_range_scale("100"):
            assert float(to_domain_10(1)) == 0.1

        with domain_range_scale("100"):
            assert float(to_domain_10(1, np.pi)) == 1 / np.pi

        with domain_range_scale("100"):
            assert to_domain_10(1, dtype=np.float16).dtype == np.float16


class TestToDomain100:
    """
    Define :func:`colour.utilities.common.to_domain_100` definition unit
    tests methods.
    """

    def test_to_domain_100(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_100` definition."""

        with domain_range_scale("Reference"):
            assert float(to_domain_100(1)) == 1

        with domain_range_scale("1"):
            assert float(to_domain_100(1)) == 100

        with domain_range_scale("100"):
            assert float(to_domain_100(1)) == 1

        with domain_range_scale("1"):
            assert float(to_domain_100(1, np.pi)) == np.pi

        with domain_range_scale("100"):
            assert to_domain_100(1, dtype=np.float16).dtype == np.float16


class TestToDomainDegrees:
    """
    Define :func:`colour.utilities.common.to_domain_degrees` definition unit
    tests methods.
    """

    def test_to_domain_degrees(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_degrees` definition."""

        with domain_range_scale("Reference"):
            assert float(to_domain_degrees(1)) == 1

        with domain_range_scale("1"):
            assert float(to_domain_degrees(1)) == 360

        with domain_range_scale("100"):
            assert float(to_domain_degrees(1)) == 3.6

        with domain_range_scale("100"):
            assert float(to_domain_degrees(1, np.pi)) == np.pi / 100

        with domain_range_scale("100"):
            assert to_domain_degrees(1, dtype=np.float16).dtype == np.float16


class TestToDomainInt:
    """
    Define :func:`colour.utilities.common.to_domain_int` definition unit
    tests methods.
    """

    def test_to_domain_int(self) -> None:
        """Test :func:`colour.utilities.common.to_domain_int` definition."""

        with domain_range_scale("Reference"):
            assert float(to_domain_int(1)) == 1

        with domain_range_scale("1"):
            assert float(to_domain_int(1)) == 255

        with domain_range_scale("100"):
            assert float(to_domain_int(1)) == 2.55

        with domain_range_scale("100"):
            assert float(to_domain_int(1, 10)) == 10.23

        with domain_range_scale("100"):
            assert to_domain_int(1, dtype=np.float16).dtype == np.float16


class TestFromRange1:
    """
    Define :func:`colour.utilities.common.from_range_1` definition unit
    tests methods.
    """

    def test_from_range_1(self) -> None:
        """Test :func:`colour.utilities.common.from_range_1` definition."""

        with domain_range_scale("Reference"):
            assert float(from_range_1(1)) == 1

        with domain_range_scale("1"):
            assert float(from_range_1(1)) == 1

        with domain_range_scale("100"):
            assert float(from_range_1(1)) == 100

        with domain_range_scale("100"):
            assert float(from_range_1(1, np.pi)) == 1 * np.pi


class TestFromRange10:
    """
    Define :func:`colour.utilities.common.from_range_10` definition unit
    tests methods.
    """

    def test_from_range_10(self) -> None:
        """Test :func:`colour.utilities.common.from_range_10` definition."""

        with domain_range_scale("Reference"):
            assert float(from_range_10(1)) == 1

        with domain_range_scale("1"):
            assert float(from_range_10(1)) == 0.1

        with domain_range_scale("100"):
            assert float(from_range_10(1)) == 10

        with domain_range_scale("100"):
            assert float(from_range_10(1, np.pi)) == 1 * np.pi


class TestFromRange100:
    """
    Define :func:`colour.utilities.common.from_range_100` definition unit
    tests methods.
    """

    def test_from_range_100(self) -> None:
        """Test :func:`colour.utilities.common.from_range_100` definition."""

        with domain_range_scale("Reference"):
            assert float(from_range_100(1)) == 1

        with domain_range_scale("1"):
            assert float(from_range_100(1)) == 0.01

        with domain_range_scale("100"):
            assert float(from_range_100(1)) == 1

        with domain_range_scale("1"):
            assert float(from_range_100(1, np.pi)) == 1 / np.pi


class TestFromRangeDegrees:
    """
    Define :func:`colour.utilities.common.from_range_degrees` definition unit
    tests methods.
    """

    def test_from_range_degrees(self) -> None:
        """Test :func:`colour.utilities.common.from_range_degrees` definition."""

        with domain_range_scale("Reference"):
            assert float(from_range_degrees(1)) == 1

        with domain_range_scale("1"):
            assert float(from_range_degrees(1)) == 1 / 360

        with domain_range_scale("100"):
            assert float(from_range_degrees(1)) == 1 / 3.6

        with domain_range_scale("100"):
            assert float(from_range_degrees(1, np.pi)) == 1 / (np.pi / 100)


class TestFromRangeInt:
    """
    Define :func:`colour.utilities.common.from_range_int` definition unit
    tests methods.
    """

    def test_from_range_int(self) -> None:
        """Test :func:`colour.utilities.common.from_range_int` definition."""

        with domain_range_scale("Reference"):
            assert float(from_range_int(1)) == 1

        with domain_range_scale("1"):
            assert float(from_range_int(1)) == 1 / 255

        with domain_range_scale("100"):
            assert float(from_range_int(1)) == 1 / 2.55

        with domain_range_scale("100"):
            assert float(from_range_int(1, 10)) == 1 / (1023 / 100)

        with domain_range_scale("100"):
            assert from_range_int(1, dtype=np.float16).dtype == np.float16


class TestIsNdarrayCopyEnabled:
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


class TestSetNdarrayCopyEnabled:
    """
    Define :func:`colour.utilities.array.set_ndarray_copy_enabled` definition
    unit tests methods.
    """

    def test_set_ndarray_copy_enabled(self) -> None:
        """
        Test :func:`colour.utilities.array.set_ndarray_copy_enabled` definition.
        """

        with ndarray_copy_enable(is_ndarray_copy_enabled()):
            set_ndarray_copy_enabled(True)
            assert is_ndarray_copy_enabled()

        with ndarray_copy_enable(is_ndarray_copy_enabled()):
            set_ndarray_copy_enabled(False)
            assert not is_ndarray_copy_enabled()


class TestNdarrayCopyEnable:
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


class TestNdarrayCopy:
    """
    Define :func:`colour.utilities.array.ndarray_copy` definition unit
    tests methods.
    """

    def test_ndarray_copy(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.ndarray_copy` definition."""

        a = xp_linspace(0, 1, num=10, xp=xp)
        with ndarray_copy_enable(True):
            assert id(ndarray_copy(a)) != id(a)  # pyright: ignore

        with ndarray_copy_enable(False):
            assert id(ndarray_copy(a)) == id(a)  # pyright: ignore

    def test_ndarray_copy_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test that copying preserves automatic differentiation."""

        copied, (gradient,), (original,) = autodiff(ndarray_copy, [1.0, 2.0, 3.0])

        assert copied is not original
        xp_assert_equal(gradient, xp.ones_like(original))


class TestClosestIndexes:
    """
    Define :func:`colour.utilities.array.closest_indexes` definition unit
    tests methods.
    """

    def test_closest_indexes(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.closest_indexes` definition."""

        a = xp_as_array(
            [
                24.31357115,
                63.62396289,
                55.71528816,
                62.70988028,
                46.84480573,
                25.40026416,
            ],
            xp=xp,
        )

        assert as_ndarray(closest_indexes(a, 63.05)).item() == 3

        assert as_ndarray(closest_indexes(a, 51.15)).item() == 4

        assert as_ndarray(closest_indexes(a, 24.90)).item() == 5

        xp_assert_equal(
            closest_indexes(a, xp_as_array([63.05, 51.15, 24.90], xp=xp)),
            [3, 4, 5],
        )


class TestClosest:
    """
    Define :func:`colour.utilities.array.closest` definition unit tests
    methods.
    """

    def test_closest(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.closest` definition."""

        a = xp_as_array(
            [
                24.31357115,
                63.62396289,
                55.71528816,
                62.70988028,
                46.84480573,
                25.40026416,
            ],
            xp=xp,
        )

        xp_assert_close(
            closest(a, xp_as_array([63.05, 51.15, 24.90], xp=xp)),
            [62.70988028, 46.84480573, 25.40026416],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestInterval:
    """
    Define :func:`colour.utilities.array.interval` definition unit tests
    methods.
    """

    def test_interval(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.interval` definition."""

        xp_assert_equal(
            interval(xp.arange(0, 10, 2)),
            [2],
        )

        xp_assert_equal(
            interval(xp.arange(0, 10, 2), False),
            [2, 2, 2, 2],
        )

        xp_assert_close(
            interval(xp_as_array([1.0, 2.0, 3.0, 4.0, 6.0, 6.5], xp=xp)),
            [0.5, 1.0, 2.0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            interval(xp_as_array([1.0, 2.0, 3.0, 4.0, 6.0, 6.5], xp=xp), False),
            [1.0, 1.0, 1.0, 2.0, 0.5],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_equal(
            interval(xp_as_array([1.0], xp=xp)),
            xp_as_array([], xp=xp),
        )

        xp_assert_equal(
            interval(xp_as_array([], xp=xp)),
            xp_as_array([], xp=xp),
        )


class TestIsUniform:
    """
    Define :func:`colour.utilities.array.is_uniform` definition unit tests
    methods.
    """

    def test_is_uniform(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.is_uniform` definition."""

        assert is_uniform(xp.arange(0, 10, 2))

        assert not is_uniform(xp_as_array([1.0, 2.0, 3.0, 4.0, 6.0], xp=xp))


class TestInArray:
    """
    Define :func:`colour.utilities.array.in_array` definition unit tests
    methods.
    """

    def test_in_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.in_array` definition."""

        b = xp_linspace(0, 10, num=101, xp=xp)

        assert np.array_equal(
            as_ndarray(in_array(xp_as_array([0.50, 0.60], xp=xp), b)),  # pyright: ignore
            np.array([True, True]),
        )

        assert not np.array_equal(
            as_ndarray(in_array(xp_as_array([0.50, 0.61], xp=xp), b)),  # pyright: ignore
            np.array([True, True]),
        )

        assert np.array_equal(
            as_ndarray(in_array(xp_as_array([[0.50], [0.60]], xp=xp), b)),  # pyright: ignore
            np.array([[True], [True]]),
        )

    def test_n_dimensional_in_array(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.utilities.array.in_array` definition n-dimensional
        support.
        """

        b = xp_linspace(0, 10, num=101, xp=xp)

        xp_assert_equal(
            in_array(xp_as_array([0.50, 0.60], xp=xp), b).shape,  # pyright: ignore
            [2],
        )

        xp_assert_equal(
            in_array(xp_as_array([[0.50, 0.60]], xp=xp), b).shape,  # pyright: ignore
            [1, 2],
        )

        xp_assert_equal(
            in_array(xp_as_array([[0.50], [0.60]], xp=xp), b).shape,  # pyright: ignore
            [2, 1],
        )


class TestTstack:
    """
    Define :func:`colour.utilities.array.tstack` definition unit tests
    methods.
    """

    def test_tstack(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.tstack` definition."""

        a = 0
        xp_assert_equal(tstack([a, a, a]), [0, 0, 0])

        a = xp.arange(0, 6)
        xp_assert_equal(
            tstack([a, a, a]),
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
            ],
        )

        a = xp_reshape(xp.arange(0, 6), (1, 6), xp=xp)
        xp_assert_equal(
            tstack([a, a, a]),
            [
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5],
                ]
            ],
        )

        a = xp_reshape(xp.arange(0, 6), (1, 2, 3), xp=xp)
        xp_assert_equal(
            tstack([a, a, a]),
            [
                [
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                ]
            ],
        )

        # Ensuring that contiguity is maintained.
        a = np.array([0, 1, 2], dtype=DTYPE_FLOAT_DEFAULT)
        b = tstack([a, a, a])
        assert b.flags.contiguous

        # Ensuring that independence is maintained.
        a *= 2
        xp_assert_equal(
            b,
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
        )

        a = np.array([0, 1, 2], dtype=DTYPE_FLOAT_DEFAULT)
        b = tstack([a, a, a])

        b[1] *= 2
        xp_assert_equal(
            a,
            [0, 1, 2],
        )


class TestTsplit:
    """
    Define :func:`colour.utilities.array.tsplit` definition unit tests
    methods.
    """

    def test_tsplit(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.tsplit` definition."""

        a = xp_as_array([0, 0, 0], xp=xp)
        xp_assert_equal(tsplit(a), [0, 0, 0])
        a = xp_as_array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
            ],
            xp=xp,
        )
        xp_assert_equal(
            tsplit(a),
            [
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ],
        )

        a = xp_as_array(
            [
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5],
                ],
            ],
            xp=xp,
        )
        xp_assert_equal(
            tsplit(a),
            [
                [[0, 1, 2, 3, 4, 5]],
                [[0, 1, 2, 3, 4, 5]],
                [[0, 1, 2, 3, 4, 5]],
            ],
        )

        a = xp_as_array(
            [
                [
                    [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                    [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                ]
            ],
            xp=xp,
        )
        xp_assert_equal(
            tsplit(a),
            [
                [[[0, 1, 2], [3, 4, 5]]],
                [[[0, 1, 2], [3, 4, 5]]],
                [[[0, 1, 2], [3, 4, 5]]],
            ],
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
        xp_assert_equal(
            b,
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ],
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
        xp_assert_equal(
            a,
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
        )


class TestRowAsDiagonal:
    """
    Define :func:`colour.utilities.array.row_as_diagonal` definition unit
    tests methods.
    """

    def test_row_as_diagonal(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.row_as_diagonal` definition."""

        xp_assert_close(
            row_as_diagonal(
                xp_as_array(
                    [
                        [0.25891593, 0.07299478, 0.36586996],
                        [0.30851087, 0.37131459, 0.16274825],
                        [0.71061831, 0.67718718, 0.09562581],
                        [0.71588836, 0.76772047, 0.15476079],
                        [0.92985142, 0.22263399, 0.88027331],
                    ],
                    xp=xp,
                )
            ),
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
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestOrient:
    """
    Define :func:`colour.utilities.array.orient` definition unit tests
    methods.
    """

    def test_orient(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.orient` definition."""

        a = xp.tile(xp.arange(5), (5, 1))

        xp_assert_equal(
            orient(a, "Flip"),
            [
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
            ],
        )

        xp_assert_equal(
            orient(a, "Flop"),
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ],
        )

        xp_assert_equal(
            orient(a, "90 CW"),
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
            ],
        )

        xp_assert_equal(
            orient(a, "90 CCW"),
            [
                [4, 4, 4, 4, 4],
                [3, 3, 3, 3, 3],
                [2, 2, 2, 2, 2],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ],
        )

        xp_assert_equal(
            orient(a, "180"),
            [
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
            ],
        )

        xp_assert_equal(orient(a), as_ndarray(a))


class TestCentroid:
    """
    Define :func:`colour.utilities.array.centroid` definition unit tests
    methods.
    """

    def test_centroid(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.centroid` definition."""

        a = xp.arange(5)
        xp_assert_equal(centroid(a), [3])

        a = xp.tile(xp.arange(5), (5, 1))
        xp_assert_equal(centroid(a), [2, 3])

        a = xp.tile(xp_linspace(0, 1, num=10, xp=xp), (10, 1))
        xp_assert_equal(centroid(a), [4, 6])

        a_np = np.tile(np.linspace(0, 1, 10), (10, 1))
        a_3d = tstack([a_np, a_np, a_np])
        xp_assert_equal(centroid(a_3d), [4, 6, 1])


class TestFillNan:
    """
    Define :func:`colour.utilities.array.fill_nan` definition unit tests
    methods.
    """

    def test_fill_nan(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.fill_nan` definition."""

        a = xp_as_array([0.1, 0.2, float("nan"), 0.4, 0.5], xp=xp)
        xp_assert_close(
            fill_nan(a),
            [0.1, 0.2, 0.3, 0.4, 0.5],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            fill_nan(a, method="Constant", default=8.0),
            [0.1, 0.2, 8.0, 0.4, 0.5],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestHasNanOnly:
    """
    Define :func:`colour.utilities.array.has_only_nan` definition unit tests
    methods.
    """

    def test_has_only_nan(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.has_only_nan` definition."""

        assert has_only_nan(None)  # pyright: ignore

        assert has_only_nan([None, None])  # pyright: ignore

        assert not has_only_nan([True, None])  # pyright: ignore

        assert not has_only_nan(xp_as_array([0.1, float("nan"), 0.3], xp=xp))


class TestNdarrayWrite:
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


class TestZeros:
    """
    Define :func:`colour.utilities.array.zeros` definition unit tests
    methods.
    """

    def test_zeros(self) -> None:
        """Test :func:`colour.utilities.array.zeros` definition."""

        xp_assert_equal(zeros(3), np.zeros(3))


class TestOnes:
    """
    Define :func:`colour.utilities.array.ones` definition unit tests
    methods.
    """

    def test_ones(self) -> None:
        """Test :func:`colour.utilities.array.ones` definition."""

        xp_assert_equal(ones(3), np.ones(3))


class TestFull:
    """
    Define :func:`colour.utilities.array.full` definition unit tests
    methods.
    """

    def test_full(self) -> None:
        """Test :func:`colour.utilities.array.full` definition."""

        xp_assert_equal(full(3, 0.5), np.full(3, 0.5))


class TestIndexAlongLastAxis:
    """
    Define :func:`colour.utilities.array.index_along_last_axis` definition
    unit tests methods.
    """

    def test_index_along_last_axis(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.array.index_along_last_axis` definition."""
        a = xp_as_array(
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
            ],
            xp=xp,
        )

        indexes = xp_as_array(
            [[[0, 1], [0, 1]], [[2, 1], [2, 1]], [[2, 1], [2, 0]]], xp=xp
        )

        xp_assert_close(
            index_along_last_axis(a, indexes),
            [
                [[0.51090627, 0.80587656], [0.84085977, 0.79308353]],
                [[0.20199051, 0.84189245], [0.01612045, 0.58905552]],
                [[0.0716432, 0.0367514], [0.9771182, 0.90644279]],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_compare_with_argmin_argmax(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.utilities.array.index_along_last_axis` definition
        by comparison with :func:`argmin` and :func:`argmax`.
        """

        a_np = np.random.random((2, 3, 4, 5, 6, 7)).astype(DTYPE_FLOAT_DEFAULT)
        a = xp_as_array(a_np, xp=xp)

        xp_assert_close(
            index_along_last_axis(a, xp.argmin(a, axis=-1)),
            np.min(as_ndarray(a), axis=-1),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            index_along_last_axis(a, xp.argmax(a, axis=-1)),
            np.max(as_ndarray(a), axis=-1),
            atol=TOLERANCE_ABSOLUTE_TESTS,
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

        # Float indexes are now converted to int by as_int_array.
        indexes = np.array([0.0, 0.0])
        index_along_last_axis(a, indexes)


class TestFormatArrayAsRow:
    """
    Define :func:`colour.utilities.array.format_array_as_row` definition unit
    tests methods.
    """

    def test_format_array_as_row(self) -> None:
        """Test :func:`colour.utilities.array.format_array_as_row` definition."""

        assert format_array_as_row([1.25, 2.5, 3.75]) == "1.2500000 2.5000000 3.7500000"

        assert format_array_as_row([1.25, 2.5, 3.75], 3) == "1.250 2.500 3.750"

        assert format_array_as_row([1.25, 2.5, 3.75], 3, ", ") == "1.250, 2.500, 3.750"


class TestAsArrayArrayApi:
    """Define :func:`colour.utilities.as_array` Array API dispatch tests."""

    def test_as_array(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.as_array` definition."""

        a = xp_as_array([1, 2, 3], xp=xp)

        with array_api_enable(False):
            result = as_array(a)
            assert isinstance(result, np.ndarray)

        with array_api_enable(True):
            result = as_array(a)
            assert array_namespace(result) is array_namespace(a)

            result = as_float_array(a)
            assert array_namespace(result) is array_namespace(a)


class TestTstackArrayApi:
    """Define :func:`colour.utilities.tstack` Array API dispatch tests."""

    def test_tstack(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.tstack` definition."""

        a = xp_as_array(np.arange(6, dtype=float), xp=xp)

        with array_api_enable(False):
            result = tstack([a, a, a])
            assert result.shape == (6, 3)

        with array_api_enable(True):
            result = tstack([a, a, a])
            assert result.shape == (6, 3)
            assert array_namespace(result) is array_namespace(a)


class TestTsplitArrayApi:
    """Define :func:`colour.utilities.tsplit` Array API dispatch tests."""

    def test_tsplit(self, xp: ModuleType) -> None:
        """Test :func:`colour.utilities.tsplit` definition."""

        a = xp_as_array(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), xp=xp)

        with array_api_enable(False):
            result = tsplit(a)
            assert result.shape == (3, 2)

        with array_api_enable(True):
            result = tsplit(a)
            assert result.shape == (3, 2)
            assert array_namespace(result) is array_namespace(a)

        a = xp_as_array(np.arange(6, dtype=float), xp=xp)
        stacked = tstack([a, a, a])
        split = tsplit(stacked)
        xp_assert_equal(split[0], a)
        xp_assert_equal(split[1], a)
        xp_assert_equal(split[2], a)
