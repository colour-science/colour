"""
Array Utilities
===============

Provide utilities for array manipulation and computational operations.

References
----------
-   :cite:`Castro2014a` : Castro, S. (2014). Numpy: Fastest way of computing
    diagonal for each row of a 2d array. Retrieved August 22, 2014, from
    http://stackoverflow.com/questions/26511401/\
numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array/\
26517247#26517247
-   :cite:`Yorke2014a` : Yorke, R. (2014). Python: Change format of np.array or
    allow tolerance in in1d function. Retrieved March 27, 2015, from
    http://stackoverflow.com/a/23521245/931625
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import os
import re
import sys
import typing
from collections.abc import KeysView, ValuesView
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from operator import add, mul, pow, sub, truediv  # noqa: A004
from typing import Union, get_args, get_origin, get_type_hints

import numpy as np

# NOTE: ``array_api_compat`` and ``array_api_extra`` are optional
# dependencies bound to *None* when unavailable; the static branch keeps
# *Pyright* seeing the real modules so no narrowing is required at the use
# sites, which are all guarded at runtime via the requirements predicates.
if typing.TYPE_CHECKING:
    import array_api_compat as xpc
    import array_api_extra as xpx
else:
    try:
        import array_api_compat as xpc
    except ImportError:
        xpc = None

    try:
        import array_api_extra as xpx
    except ImportError:
        xpx = None

from colour.constants import (
    DTYPE_COMPLEX_DEFAULT,
    DTYPE_FLOAT_DEFAULT,
    DTYPE_INT_DEFAULT,
    EPSILON,
    TOLERANCE_ABSOLUTE_TESTS,
    TOLERANCE_RELATIVE_TESTS,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        Callable,
        DType,
        DTypeBoolean,
        DTypeReal,
        Dataclass,
        Generator,
        Literal,
        ModuleType,
        NDArray,
        ProtocolArrayNamespace,
        NDArrayBoolean,
        NDArrayComplex,
        NDArrayFloat,
        NDArrayInt,
        Real,
        Self,
        Sequence,
        Type,
    )

from colour.hints import ArrayLike, DTypeComplex, DTypeFloat, DTypeInt, cast
from colour.utilities import (
    CACHE_REGISTRY,
    as_bool,
    attest,
    int_digest,
    is_array_api_compat_installed,
    is_array_api_extra_installed,
    is_caching_enabled,
    optional,
    runtime_warning,
    suppress_warnings,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_array_api_enabled",
    "set_array_api_enabled",
    "array_api_enable",
    "trace_array_namespace",
    "array_namespace",
    "is_numpy_namespace",
    "is_non_ndarray",
    "as_ndarray",
    "cast_non_ndarray",
    "xp_as_array",
    "xp_as_float_array",
    "xp_as_int_array",
    "xp_ascontiguousarray",
    "xp_astype",
    "xp_matrix_transpose",
    "xp_select",
    "xp_interp",
    "xp_trapezoid",
    "xp_average",
    "xp_gradient",
    "xp_resize",
    "xp_nanmean",
    "xp_median",
    "xp_round",
    "xp_radians",
    "xp_degrees",
    "xp_atleast_1d",
    "xp_atleast_2d",
    "xp_squeeze",
    "xp_sinc",
    "xp_isclose",
    "xp_nan_to_num",
    "xp_create_diagonal",
    "xp_reshape",
    "xp_broadcast_to",
    "xp_lstsq",
    "xp_eig",
    "xp_eigh",
    "xp_isin",
    "xp_linspace",
    "xp_pad",
    "xp_unique",
    "xp_insert",
    "xp_setxor1d",
    "xp_assert_close",
    "xp_assert_equal",
    "MixinDataclassFields",
    "MixinDataclassIterable",
    "MixinDataclassArray",
    "MixinDataclassArithmetic",
    "as_array",
    "as_int",
    "as_float",
    "as_int_array",
    "as_float_array",
    "as_int_scalar",
    "as_float_scalar",
    "as_complex_array",
    "set_default_int_dtype",
    "set_default_float_dtype",
    "set_default_complex_dtype",
    "get_domain_range_scale",
    "set_domain_range_scale",
    "domain_range_scale",
    "get_domain_range_scale_metadata",
    "to_domain_1",
    "to_domain_10",
    "to_domain_100",
    "to_domain_degrees",
    "to_domain_int",
    "from_range_1",
    "from_range_10",
    "from_range_100",
    "from_range_degrees",
    "from_range_int",
    "is_ndarray_copy_enabled",
    "set_ndarray_copy_enabled",
    "ndarray_copy_enable",
    "ndarray_copy",
    "closest_indexes",
    "closest",
    "interval",
    "is_uniform",
    "in_array",
    "tstack",
    "tsplit",
    "row_as_diagonal",
    "orient",
    "centroid",
    "fill_nan",
    "has_only_nan",
    "ndarray_write",
    "zeros",
    "ones",
    "full",
    "index_along_last_axis",
    "format_array_as_row",
]

_ARRAY_API_ENABLED_DEFAULT: bool = as_bool(
    os.environ.get("COLOUR_SCIENCE__ARRAY_API", "False")
)
"""Environment-seeded default for :attr:`_ARRAY_API_ENABLED`."""

_ARRAY_API_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_ARRAY_API_ENABLED", default=_ARRAY_API_ENABLED_DEFAULT
)
"""
:class:`contextvars.ContextVar` storing the current *Colour* Array API
dispatch enabled state. The :class:`contextvars.ContextVar` keeps nested
:class:`array_api_enable` contexts independent across concurrent threads
and async tasks. Read it via :func:`is_array_api_enabled` and toggle it
via :func:`set_array_api_enabled` or :class:`array_api_enable`. The
environment value is seeded as the :class:`contextvars.ContextVar`
``default`` so that fresh threads and async tasks observe it, rather than
via a module-level ``set`` that only applies to the importing context.
"""

_CACHE_ARRAY_NAMESPACE: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_ARRAY_NAMESPACE"
)
"""Cache for :func:`array_namespace` results, keyed by array type."""

_CACHE_SCALAR_PROMOTION: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_SCALAR_PROMOTION"
)
"""Cache for scalar-to-backend promotions in :func:`xp_as_array`."""

_CACHE_BACKEND_DTYPE: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_BACKEND_DTYPE"
)
"""Cache mapping ``(id(xp), dtype)`` pairs to the backend-native dtype."""


def _resolve_backend_dtype(xp: ProtocolArrayNamespace | ModuleType, dtype: Any) -> Any:
    """Resolve a *NumPy* dtype to the equivalent dtype in ``xp``.

    Resolution is memoised through :attr:`_CACHE_BACKEND_DTYPE` (keyed by
    ``(id(xp), dtype)``) when caching is enabled to avoid the
    ``np.dtype(...).name`` + ``getattr`` lookups on every call. Falls back
    to ``dtype`` unchanged when it is already a backend-native type.
    """

    key = (id(xp), dtype)

    if is_caching_enabled():
        resolved = _CACHE_BACKEND_DTYPE.get(key)
        if resolved is not None:
            return resolved

    try:
        resolved = getattr(xp, np.dtype(dtype).name, dtype)
    except TypeError:
        resolved = dtype

    if is_caching_enabled():
        _CACHE_BACKEND_DTYPE[key] = resolved

    return resolved


def is_array_api_enabled() -> bool:
    """
    Determine whether *Colour* Array API dispatch is enabled.

    The Array API dispatch state is controlled by the global
    *COLOUR_SCIENCE__ARRAY_API* environment variable and can be
    temporarily modified using the :func:`set_array_api_enabled` function
    or the :class:`array_api_enable` context manager.

    Returns
    -------
    :class:`bool`
        Whether *Colour* Array API dispatch is enabled.

    Examples
    --------
    >>> with array_api_enable(False):
    ...     is_array_api_enabled()
    False
    >>> with array_api_enable(True):
    ...     is_array_api_enabled()
    True
    """

    return _ARRAY_API_ENABLED.get()


def set_array_api_enabled(enable: bool) -> None:
    """
    Set the *Colour* Array API dispatch enabled state.

    Parameters
    ----------
    enable
        Whether to enable *Colour* Array API dispatch.

    Examples
    --------
    >>> with array_api_enable(True):
    ...     print(is_array_api_enabled())
    ...     set_array_api_enabled(False)
    ...     print(is_array_api_enabled())
    True
    False
    """

    _ARRAY_API_ENABLED.set(enable)


class array_api_enable:
    """
    Define a context manager and decorator to temporarily set the *Colour*
    Array API dispatch enabled state.

    Parameters
    ----------
    enable
        Whether to enable or disable *Colour* Array API dispatch.
    """

    def __init__(self, enable: bool) -> None:
        self._enable = enable
        # Token stack: nested or recursive ``__enter__`` / ``__exit__``
        # pairs against the same instance (e.g. via the decorator form on
        # a recursive function) push and pop independent reset tokens.
        self._tokens: list[contextvars.Token[bool]] = []

    def __enter__(self) -> Self:
        """Enter the context and set the Array API dispatch state."""

        self._tokens.append(_ARRAY_API_ENABLED.set(self._enable))

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore the previous Array API state."""

        _ARRAY_API_ENABLED.reset(self._tokens.pop())

    def __call__(self, function: Callable) -> Callable:
        """Decorate and call the specified function with Array API control."""

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # A fresh instance is entered per call so the token stack is never
            # shared across threads or async tasks invoking the decorated
            # definition concurrently.
            with self.__class__(self._enable):
                return function(*args, **kwargs)

        return wrapper


class trace_array_namespace:
    """
    Define a context manager to trace :func:`array_namespace` calls and
    array type flow through *Colour* functions using :func:`sys.settrace`.

    When active, every function call under ``colour/`` is logged with the
    types of all array arguments (positional and keyword). Return values
    are logged with their types. Calls where multiple array backends
    coexist in the same argument list are flagged as ``MIXED``.

    The trace output is indented to reflect the call stack depth.

    Examples
    --------
    >>> import torch  # doctest: +SKIP
    >>> with array_api_enable(True), trace_array_namespace():
    ...     pass  # doctest: +SKIP
    """

    _ARRAY_TYPES: tuple = (np.ndarray, np.generic)

    def __init__(self) -> None:
        self._depth: int = 0
        self._previous_trace: Any = None

        with contextlib.suppress(ImportError):
            import torch  # noqa: PLC0415

            self._ARRAY_TYPES = (*self._ARRAY_TYPES, torch.Tensor)

        with contextlib.suppress(ImportError):
            import jax  # noqa: PLC0415

            self._ARRAY_TYPES = (*self._ARRAY_TYPES, jax.Array)

    def _type_label(self, obj: Any) -> str:
        """Return a short type label for the specified object."""

        cls = type(obj)
        module = cls.__module__.split(".")[0]

        if isinstance(obj, np.ndarray):
            return f"ndarray{list(obj.shape)}"

        return (
            f"{module}.{cls.__name__}{list(obj.shape) if hasattr(obj, 'shape') else ''}"
        )

    def _format_args(
        self,
        code: Any,
        local_vars: dict,
    ) -> str:
        """Format function arguments with array type annotations."""

        parts = []
        param_names = list(code.co_varnames[: code.co_argcount])

        for name in param_names:
            if name == "self":
                continue

            value = local_vars.get(name)

            if value is None:
                parts.append(f"{name}: None")
            elif isinstance(value, self._ARRAY_TYPES):
                parts.append(f"{name}: {self._type_label(value)}")
            else:
                parts.append(f"{name}: {type(value).__name__}")

        return ", ".join(parts)

    def _has_mixed_backends(self, local_vars: dict) -> bool:
        """Check whether the local variables contain mixed array backends."""

        backends = set()

        for value in local_vars.values():
            if isinstance(value, self._ARRAY_TYPES):
                if isinstance(value, (np.ndarray, np.generic)):
                    backends.add("numpy")
                else:
                    backends.add(type(value).__module__.split(".")[0])

        return len(backends) > 1

    def _is_colour_frame(self, frame: Any) -> bool:
        """Check whether the specified frame belongs to *Colour*."""

        filename = frame.f_code.co_filename or ""

        return "colour/" in filename and "/site-packages/" not in filename

    def _trace(self, frame: Any, event: str, arg: Any) -> Any:
        """Trace function for :func:`sys.settrace`."""

        if not self._is_colour_frame(frame):
            return self._trace

        if event == "call":
            code = frame.f_code
            name = code.co_name

            if name.startswith("<") or (
                name.startswith("_") and not name.startswith("__")
            ):
                return self._trace

            args_str = self._format_args(code, frame.f_locals)
            mixed = self._has_mixed_backends(frame.f_locals)
            marker = " [MIXED]" if mixed else ""

            indent = "  " * self._depth
            print(f"{indent}{name}({args_str}){marker}")  # noqa: T201

            self._depth += 1

            return self._trace

        if event == "return":
            self._depth = max(0, self._depth - 1)

            if isinstance(arg, self._ARRAY_TYPES):
                indent = "  " * self._depth
                print(f"{indent}-> {self._type_label(arg)}")  # noqa: T201

            return self._trace

        return self._trace

    def __enter__(self) -> Self:
        """Enter the context and install the trace hook."""

        self._previous_trace = sys.gettrace()
        self._depth = 0
        sys.settrace(self._trace)

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore the previous trace hook."""

        sys.settrace(self._previous_trace)


def array_namespace(*arrays: Any) -> ProtocolArrayNamespace:
    """
    Return the array namespace for the specified arrays.

    When Array API dispatch is disabled (default), return :mod:`numpy`.
    When enabled, use :func:`array_api_compat.array_namespace` to detect
    the appropriate namespace from the input arrays.

    Parameters
    ----------
    *arrays
        Arrays to determine the namespace from. *NumPy* is returned as the
        explicit default fallback when no arrays are provided or all
        arrays are *None* / pure-*Python* scalars (no backend signal to
        dispatch on).

    Returns
    -------
    :class:`colour.hints.ProtocolArrayNamespace`
        Array namespace module.

    Examples
    --------
    >>> array_namespace(np.array([1, 2, 3]))  # doctest: +ELLIPSIS
    <module 'numpy'...>
    """

    if not is_array_api_enabled():
        return np

    if xpc is None:  # pragma: no cover
        is_array_api_compat_installed(raise_exception=True)

    # Fast path: cache by array type to avoid the full resolution chain
    # on every call. Only cache-hit when there is exactly one distinct
    # non-*NumPy* type; mixed backends (e.g. *JAX* + *PyTorch*) must
    # fall through to ``xpc.array_namespace`` so it can raise.
    if is_caching_enabled():
        non_numpy_types = {
            type(a)
            for a in arrays
            if a is not None and not isinstance(a, (np.ndarray, np.generic))
        }
        if len(non_numpy_types) == 1:
            cached = _CACHE_ARRAY_NAMESPACE.get(next(iter(non_numpy_types)))
            if cached is not None:
                return cached

    arrays = tuple(
        a
        for a in arrays
        if a is not None
        and (
            hasattr(a, "__array_namespace__")
            or isinstance(a, np.ndarray)
            or xpc.is_array_api_obj(a)
        )
    )

    if not arrays:
        return np

    # When inputs mix NumPy arrays (e.g., module-level constants) with a
    # non-NumPy backend, promote to the non-NumPy backend.  Only mixed
    # non-NumPy backends (e.g., JAX + CuPy) raise a ``TypeError``.
    non_numpy = tuple(a for a in arrays if not isinstance(a, (np.ndarray, np.generic)))

    if non_numpy:
        arrays = non_numpy

    # ``array_api_compat`` annotates its resolved namespaces as bare modules.
    xp = cast("ProtocolArrayNamespace", xpc.array_namespace(*arrays))

    if is_caching_enabled() and non_numpy:
        _CACHE_ARRAY_NAMESPACE[type(non_numpy[0])] = xp

    return xp


def is_numpy_namespace(xp: ProtocolArrayNamespace | ModuleType) -> bool:
    """
    Determine whether the specified namespace is :mod:`numpy`.

    Parameters
    ----------
    xp
        Namespace module to test.

    Returns
    -------
    :class:`bool`
        Whether the namespace is :mod:`numpy`.

    Examples
    --------
    >>> is_numpy_namespace(np)
    True
    """

    if xp is np:
        return True

    if xpc is not None:
        return xpc.is_numpy_namespace(xp)

    return False


def is_non_ndarray(a: Any) -> bool:
    """
    Determine whether the specified object is a non-*NumPy* array.

    Parameters
    ----------
    a
        Object to test.

    Returns
    -------
    :class:`bool`
        Whether the object is a non-*NumPy* array (e.g., *JAX*, *PyTorch*,
        *CuPy*).

    Examples
    --------
    >>> is_non_ndarray(np.array([1, 2, 3]))
    False
    >>> is_non_ndarray([1, 2, 3])
    False
    """

    if isinstance(a, (np.ndarray, np.generic)):
        return False

    if hasattr(a, "__array_namespace__"):
        return True

    if xpc is not None:
        return xpc.is_array_api_obj(a)

    return False


def as_ndarray(a: Any) -> np.ndarray:
    """
    Convert the specified array :math:`a` to a :class:`numpy.ndarray`.

    This function handles arrays from any backend (*JAX*, *PyTorch*, *CuPy*,
    etc.) by moving them to the host when direct conversion is not
    possible, e.g., for device-resident arrays.

    Parameters
    ----------
    a
        Array, scalar, or *Python* sequence to convert.

    Returns
    -------
    :class:`numpy.ndarray`
        *NumPy* array.

    Notes
    -----
    -   Unlike :func:`as_array` / :func:`as_float_array` siblings,
        :func:`as_ndarray` does **not** honour
        :attr:`_NDARRAY_COPY_ENABLED`. It is a *backend-host hand-off*
        boundary helper, not a copy toggle; the returned array shares
        storage with the input wherever the backend allows.

    Examples
    --------
    >>> import numpy as np
    >>> as_ndarray(np.array([1, 2, 3]))
    array([1, 2, 3])

    Round-trip a *PyTorch* tensor on a non-host device:

    >>> import torch  # doctest: +SKIP
    >>> as_ndarray(torch.tensor([1, 2, 3], device="mps"))  # doctest: +SKIP
    array([1, 2, 3])
    """

    # ``np.asarray`` succeeds for *NumPy*, *JAX* and host-resident *PyTorch*
    # tensors; it raises :class:`TypeError` for device-resident tensors
    # (notably *PyTorch* on *MPS*) and :class:`RuntimeError` for tensors
    # with ``requires_grad=True``, both recoverable via the rungs below.
    try:
        return np.asarray(a)
    except (TypeError, RuntimeError):
        pass

    # *PyTorch* tensors on a non-*CPU* device or with ``requires_grad=True``
    # need ``detach().cpu()`` before ``__array__`` can succeed; for other
    # backends, ``array_namespace(a).to_device(a, "cpu")`` is the *Array API*
    # standard host hand-off.
    if hasattr(a, "detach") and hasattr(a, "cpu"):
        return np.asarray(a.detach().cpu())

    # The array's own namespace is asked for the hand-off: dispatch being
    # disabled returns the *NumPy* fallback, which has no ``to_device``.
    namespace = getattr(a, "__array_namespace__", None)
    if namespace is not None:
        return cast("NDArray", np.asarray(namespace().to_device(a, "cpu")))

    error = f'"{type(a)}" cannot be converted to a "numpy.ndarray"!'

    raise TypeError(error)


def _copy_array(a: ArrayLike, xp: ProtocolArrayNamespace | ModuleType) -> NDArray:
    """Return a backend array copy while preserving its computational graph."""

    clone = getattr(xp, "clone", None)
    if callable(clone):
        return cast("NDArray", clone(a))

    return cast("NDArray", xp.asarray(a, copy=True))


def xp_as_array(
    a: ArrayLike,
    *,
    dtype: Any = None,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    like: Any = None,
    copy: bool | None = None,
) -> NDArray:
    """
    Convert the specified variable :math:`a` to the target namespace.

    When the namespace is :mod:`numpy`, the original variable :math:`a` is
    returned as a :class:`numpy.ndarray` without unnecessary copying. For
    other namespaces, the variable :math:`a` is converted via ``xp.asarray``,
    optionally matching the device of a reference array ``like``.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        Target dtype. When provided, the result is cast to this dtype.
        Accepts *NumPy* dtype objects (e.g. ``np.float64``) which are
        mapped to the backend equivalent.
    xp
        Array namespace module. If *None*, derived from ``a``.
    like
        Reference array whose device to match (for backends like *PyTorch*
        that support multiple devices).
    copy
        When *True*, always return a fresh copy of the input even when no
        dtype change is needed (the *Array API* ``xp.asarray(a, copy=True)``
        semantics). When *None* (default), copy only when necessary
        (dtype change, namespace promotion). The scalar-promotion cache
        is bypassed when ``copy=True``.

    Returns
    -------
    :class:`object`
        Variable :math:`a` in the target namespace.

    Examples
    --------
    >>> xp_as_array([1, 2, 3], xp=np)
    array([1, 2, 3])
    >>> xp_as_array([1, 2, 3], dtype=np.float64, xp=np)
    array([1., 2., 3.])
    """

    xp = array_namespace(a) if xp is None else xp

    # When the *Array API* dispatch is disabled the input is *NumPy* by
    # construction; bypass the namespace + non-ndarray probes that the
    # full path performs.
    if not is_array_api_enabled():
        result = as_array(a, dtype)
        return np.copy(result) if copy else result

    if is_numpy_namespace(xp):
        result = as_array(as_ndarray(a) if is_non_ndarray(a) else a)

        if dtype is not None and hasattr(result, "dtype") and result.dtype != dtype:
            result = result.astype(dtype)

        return np.copy(result) if copy else result

    # Non-*NumPy* namespace, input already on a backend device: short-
    # circuit when no dtype is requested or the dtype already matches.
    if is_non_ndarray(a):
        result = a

        # A ``like`` reference on another device moves the array onto it first:
        # an operand pair split across, say, CPU and *MPS* would otherwise
        # reach the arithmetic unmoved and fail with a device mismatch. The
        # move precedes the dtype cast so that the cast is evaluated against
        # the destination device's capabilities, e.g. *MPS* has no float64.
        device_like = getattr(like, "device", None)
        if device_like is not None and getattr(result, "device", None) != device_like:
            try:
                result = xp.asarray(result, device=device_like)
            except (TypeError, RuntimeError, ValueError):
                runtime_warning(
                    f'Backend "{xp.__name__}" could not move the array to '
                    f'device "{device_like}"; keeping device '
                    f'"{getattr(result, "device", None)}".'
                )

        if dtype is not None:
            a_dtype = getattr(result, "dtype", None)
            if a_dtype is not None and a_dtype != dtype:
                xp_target_dtype = _resolve_backend_dtype(xp, dtype)
                if a_dtype != xp_target_dtype:
                    try:
                        result = xp_astype(result, xp_target_dtype, xp=xp)
                    except (TypeError, RuntimeError):
                        runtime_warning(
                            f'Backend "{xp.__name__}" does not support '
                            f'dtype "{xp_target_dtype}"; keeping input '
                            f'dtype "{a_dtype}".'
                        )

        if copy and result is a:
            result = _copy_array(a, xp)
        return result  # pyright: ignore

    # Non-*NumPy* namespace: convert from *NumPy* / *Python* to the target
    # backend, caching scalar / small constant promotions to avoid repeated
    # CPU-to-GPU transfers for module-level constants. The cache is bypassed
    # when ``copy=True`` so callers asking for a fresh copy cannot
    # accidentally mutate the cached entry.
    device = getattr(like, "device", None)
    device_kwarg = device if device is not None and hasattr(device, "type") else None
    xp_target_dtype = _resolve_backend_dtype(xp, dtype) if dtype is not None else None

    cache_key = None
    if is_caching_enabled() and not copy:
        if isinstance(a, (int, float, complex)):
            # ``type(a).__name__`` disambiguates ``True`` / ``1`` / ``1.0``
            # which share a hash and compare equal; the tuple itself is the
            # cache key so distinct constants can never collide on a hash.
            cache_key = ("scalar", type(a).__name__, a, id(xp), str(device), dtype)
        elif isinstance(a, np.ndarray) and a.size <= 16:
            # ``a.dtype`` is part of the key: ``array([0])`` and ``array([0.0])``
            # share ``tobytes`` output but must not share a cache entry.
            cache_key = (
                "ndarray",
                int_digest(a.tobytes()),
                a.shape,
                str(a.dtype),
                id(xp),
                str(device),
                dtype,
            )

        if cache_key is not None:
            cached = _CACHE_SCALAR_PROMOTION.get(cache_key)
            if cached is not None:
                return cached

    # A non-contiguous *NumPy* array (e.g. a negatively-strided view from a
    # flip or transpose) is made contiguous before hand-off: backends such as
    # *PyTorch* reject negative strides in ``asarray``.
    if isinstance(a, np.ndarray) and not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)

    # Passing the target dtype to ``asarray`` avoids promoting a *Python*
    # scalar at the backend default dtype (e.g. float32 for stock *PyTorch*)
    # and only then upcasting, which would quantise the value.
    asarray_kwargs: dict[str, Any] = {}
    if device_kwarg is not None:
        asarray_kwargs["device"] = device_kwarg
    if xp_target_dtype is not None:
        asarray_kwargs["dtype"] = xp_target_dtype

    try:
        result = xp.asarray(a, **asarray_kwargs)
    except TypeError:
        # Backend does not support the input dtype (e.g., *MPS* + float64).
        a = np.asarray(a)
        original_dtype = a.dtype
        a = a.astype(np.complex64 if np.iscomplexobj(a) else np.float32)
        _runtime_warning_xp_downcast(xp, original_dtype, a.dtype)
        # The requested dtype is dropped from this retry only: it is the dtype
        # the backend just rejected. ``xp_target_dtype`` is left set so that
        # the cast below is still attempted and warns when unsupported, rather
        # than silently returning a different dtype than the caller asked for.
        asarray_kwargs.pop("dtype", None)
        result = xp.asarray(a, **asarray_kwargs)

    if (
        xp_target_dtype is not None
        and hasattr(result, "dtype")
        and result.dtype != xp_target_dtype
    ):
        try:
            result = xp_astype(result, xp_target_dtype, xp=xp)
        except (TypeError, RuntimeError):
            runtime_warning(
                f'Backend "{xp.__name__}" does not support '
                f'dtype "{xp_target_dtype}"; keeping result dtype '
                f'"{result.dtype}".'
            )

    if cache_key is not None:
        _CACHE_SCALAR_PROMOTION[cache_key] = result

    return result


def xp_as_float_array(
    a: ArrayLike,
    *,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    like: Any = None,
) -> NDArrayFloat:
    """
    Convert the specified variable :math:`a` to a float array in the target
    namespace using :attr:`colour.constants.DTYPE_FLOAT_DEFAULT`.

    Shorthand for ``xp_as_array(a, dtype=DTYPE_FLOAT_DEFAULT, xp=xp, like=like)``.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    xp
        Array namespace module. If *None*, derived from ``a``.
    like
        Reference array whose device to match.

    Returns
    -------
    :class:`object`
        Variable :math:`a` as a float array in the target namespace.

    Examples
    --------
    >>> xp_as_float_array([1, 2, 3], xp=np)
    array([1., 2., 3.])
    """

    return xp_as_array(a, dtype=DTYPE_FLOAT_DEFAULT, xp=xp, like=like)


def xp_as_int_array(
    a: ArrayLike,
    *,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    like: Any = None,
) -> NDArrayInt:
    """
    Convert the specified variable :math:`a` to an integer array in the target
    namespace using :attr:`colour.constants.DTYPE_INT_DEFAULT`.

    Shorthand for ``xp_as_array(a, dtype=DTYPE_INT_DEFAULT, xp=xp, like=like)``.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    xp
        Array namespace module. If *None*, derived from ``a``.
    like
        Reference array whose device to match.

    Returns
    -------
    :class:`object`
        Variable :math:`a` as an integer array in the target namespace.

    Examples
    --------
    >>> xp_as_int_array([1.5, 2.7, 3.9], xp=np)
    array([1, 2, 3])
    """

    return xp_as_array(a, dtype=DTYPE_INT_DEFAULT, xp=xp, like=like)


def xp_ascontiguousarray(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArray:
    """
    *Array API* compatible implementation of :func:`numpy.ascontiguousarray`.

    Materialise ``a`` into a C-contiguous array with the same shape and
    dtype. The lazy stride-permuted view returned by
    :func:`xp.matrix_transpose` (and ``.T`` / ``.mT``) poisons downstream
    broadcasts and forces *BLAS* to copy internally on every subsequent
    ``matmul``; calling this function at the transpose boundary cascades
    the contiguous layout through all downstream operations.

    Parameters
    ----------
    a
        Variable :math:`a` to materialise.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        C-contiguous copy of :math:`a` on the same backend.

    Examples
    --------
    >>> xp_ascontiguousarray(np.array([[1, 2], [3, 4]]).T, xp=np).flags["C_CONTIGUOUS"]
    True
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.ascontiguousarray(a)

    # ``PyTorch`` exposes a ``.contiguous()`` method on tensors; other
    # backends (e.g. *JAX*) manage contiguity as an implementation
    # detail and don't expose a corresponding primitive.
    contiguous = getattr(a, "contiguous", None)
    if callable(contiguous):
        return contiguous()  # pyright: ignore

    return a  # pyright: ignore


def xp_matrix_transpose(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArray:
    """
    *Array API* compatible implementation of :func:`numpy.matrix_transpose`
    materialising the result to a C-contiguous array.

    Equivalent to ``xp.matrix_transpose(a)`` followed by
    :func:`xp_ascontiguousarray`. Use whenever the transposed array will
    participate in subsequent broadcasts or ``matmul`` operations; the
    lazy stride-permuted view returned by the standard
    ``matrix_transpose`` poisons broadcast outputs (the *NumPy* broadcast
    machinery inherits the strided layout into the freshly-allocated
    output) and forces *BLAS* to copy internally on every matmul. The
    cost of materialising once is amortised by keeping all downstream
    broadcasts and matmuls on contiguous memory.

    Parameters
    ----------
    a
        Variable :math:`a`; the last two axes are swapped.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Matrix-transposed and C-contiguous array.

    Examples
    --------
    >>> a = np.arange(6).reshape(2, 3)
    >>> xp_matrix_transpose(a, xp=np)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    """

    if xp is None or not hasattr(xp, "matrix_transpose"):
        xp = array_namespace(a)

    return xp_ascontiguousarray(xp.matrix_transpose(a), xp=xp)


def xp_astype(
    a: ArrayLike, dtype: Any, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArray:
    """
    *Array API* compatible implementation of :meth:`numpy.ndarray.astype`.

    *NumPy* uses ``a.astype(dtype)`` while the *Array API* standard uses
    ``xp.astype(a, dtype)`` with backend-native dtype objects.

    Parameters
    ----------
    a
        Array to cast.
    dtype
        Target dtype (*NumPy* dtype accepted, automatically translated for
        non-*NumPy* backends).
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Cast array.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return a.astype(dtype)  # pyright: ignore

    xp_dtype = _resolve_backend_dtype(xp, dtype)

    if a.dtype == xp_dtype:  # pyright: ignore
        return a  # pyright: ignore

    # NOTE: ``array_namespace(a)`` is called again to obtain the
    # ``array-api-compat`` wrapped namespace which provides ``astype`` for
    # backends (e.g., *PyTorch*) that lack a module-level ``astype``.
    try:
        return array_namespace(a).astype(a, xp_dtype)
    except (TypeError, RuntimeError):
        # Fall back to float32 for backends that don't support float64
        # (e.g., MPS on Apple Silicon).
        xp_dtype_f32 = getattr(xp, "float32", None)
        if xp_dtype_f32 is not None and xp_dtype_f32 != xp_dtype:
            _runtime_warning_xp_downcast(xp, xp_dtype, xp_dtype_f32)
            return array_namespace(a).astype(a, xp_dtype_f32)
        raise


# NOTE: Backend capability probing follows a single canonical pattern: attempt
# the native call and catch ``AttributeError`` (the backend does not provide
# the function) and ``TypeError`` (the backend signature is incompatible),
# then warn via :func:`_runtime_warning_xp_fallback` and fall back to *NumPy*.
# ``linalg`` probes additionally catch ``NotImplementedError`` and
# ``RuntimeError`` which *PyTorch* raises at call time for operations
# unsupported on the active device (e.g. *MPS*).


def _runtime_warning_xp_fallback(name: str) -> None:
    """Emit the standard *falling back to NumPy* runtime warning."""

    runtime_warning(
        f'"{name}" is falling back to "NumPy" for non-"NumPy" '
        "arrays, this will incur a performance penalty due to array "
        "conversion."
    )


def _runtime_warning_xp_downcast(
    xp: ProtocolArrayNamespace | ModuleType, dtype: Any, dtype_target: Any
) -> None:
    """Emit the standard backend dtype downcast runtime warning."""

    runtime_warning(
        f'Backend "{xp.__name__}" does not support dtype "{dtype}"; '
        f'downcasting to "{dtype_target}".'
    )


def _xpx() -> ModuleType:
    """
    Return the :mod:`array_api_extra` module, raising when it is not
    installed: together with :mod:`array_api_compat`, it is required for
    *Array API* dispatch but both are optional dependencies; *NumPy*-only
    code paths never reach this guard.
    """

    is_array_api_extra_installed(raise_exception=True)

    return xpx


def xp_select(
    condlist: Any,
    choicelist: Any,
    *,
    default: Any = 0,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.select`.

    Parameters
    ----------
    condlist
        List of boolean arrays for conditions.
    choicelist
        List of arrays from which output elements are taken.
    default
        Value used when all conditions are ``False``.
    xp
        Array namespace module. If *None*, derived from ``condlist`` and
        ``choicelist``.

    Returns
    -------
    :class:`object`
        Array with elements from *choicelist* where *condlist* is ``True``.
    """

    xp = array_namespace(*condlist, *choicelist) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.select(condlist, choicelist, default)

    like = None
    for item in (*condlist, *choicelist):
        if hasattr(item, "device"):
            like = item
            break

    condlist = [xp_as_array(c, xp=xp, like=like) for c in condlist]
    choicelist = [xp_as_float_array(c, xp=xp, like=like) for c in choicelist]

    if hasattr(default, "shape"):
        result = xp_as_float_array(default, xp=xp, like=like)
    else:
        result = xp.full(
            condlist[0].shape,
            fill_value=default,
            dtype=choicelist[0].dtype,
            device=getattr(like, "device", None),
        )

    for condition, choice in zip(reversed(condlist), reversed(choicelist), strict=True):
        result = xp.where(xp_astype(condition, bool, xp=xp), choice, result)

    return result


def xp_interp(
    x: ArrayLike,
    x_data: ArrayLike,
    fp: ArrayLike,
    *,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.interp`.

    Parameters
    ----------
    x
        x-coordinates at which to evaluate the interpolation.
    x_data
        x-coordinates of the data points.
    fp
        y-coordinates of the data points.
    xp
        Array namespace module. If *None*, derived from ``x``, ``x_data``
        and ``fp``.

    Returns
    -------
    :class:`object`
        Interpolated values.
    """

    xp = array_namespace(x, x_data, fp) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.interp(x, x_data, fp)  # pyright: ignore

    try:
        return xp.interp(x, x_data, fp)
    except (AttributeError, TypeError):
        pass

    _runtime_warning_xp_fallback("xp_interp")

    fp_nd = as_ndarray(fp)
    result = np.interp(as_ndarray(x), as_ndarray(x_data), fp_nd)
    result = result.astype(fp_nd.dtype)

    device = getattr(x, "device", None)
    if device is not None and hasattr(device, "type"):
        like = x
        like_dtype = getattr(like, "dtype", None)
        like_is_f32 = like_dtype is not None and (
            getattr(like_dtype, "name", None) == "float32"
            or str(like_dtype) in ("torch.float32", "float32")
        )
        if like_is_f32 and result.dtype == np.float64:
            result = result.astype(np.float32)
        return xp.asarray(result, device=device)

    return xp.asarray(result)


def xp_trapezoid(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    dx: float = 1.0,
    axis: int = -1,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.trapezoid`.

    Parameters
    ----------
    y
        y-coordinates of the function values.
    x
        x-coordinates of the function values.
    dx
        Spacing between sample points when *x* is ``None``.
    axis
        Axis along which to integrate.
    xp
        Array namespace module. If *None*, derived from ``y`` and ``x``.

    Returns
    -------
    :class:`object`
        Approximation of the integral.
    """

    xp = array_namespace(y, x) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)  # pyright: ignore

    y = xp_as_float_array(y, xp=xp)

    try:
        if x is not None:
            x = xp_as_float_array(x, xp=xp, like=y)
            return xp.trapezoid(y, x=x, axis=axis)

        return xp.trapezoid(y, dx=dx, axis=axis)
    except (AttributeError, TypeError):
        pass

    _runtime_warning_xp_fallback("xp_trapezoid")

    result = np.trapezoid(
        as_ndarray(y), x=as_ndarray(x) if x is not None else None, dx=dx, axis=axis
    )

    return xp.asarray(result)


def xp_average(
    a: ArrayLike,
    *,
    axis: int | None = None,
    weights: ArrayLike | None = None,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.average`.

    Parameters
    ----------
    a
        Array to average.
    axis
        Axis along which to average.
    weights
        Weights associated with the values in *a*.
    xp
        Array namespace module. If *None*, derived from ``a`` and
        ``weights``.

    Returns
    -------
    :class:`object`
        Weighted average.
    """

    xp = array_namespace(a, weights) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.average(a, axis=axis, weights=weights)  # pyright: ignore

    a = xp_as_float_array(a, xp=xp)

    if weights is None:
        return xp.mean(a, axis=axis)

    weights = xp_as_float_array(weights, xp=xp, like=a)
    if weights.ndim == 1 and a.ndim != 1 and axis is not None:
        # Broadcast 1-D ``weights`` along ``axis`` to match ``np.average``
        # semantics for an N-D ``a``.
        broadcast_shape = [1] * a.ndim
        broadcast_shape[axis] = weights.shape[0]
        weights = xp_reshape(weights, tuple(broadcast_shape), xp=xp)

    return xp.sum(a * weights, axis=axis) / xp.sum(weights, axis=axis)


@typing.overload
def xp_gradient(
    f: ArrayLike,
    *varargs: Any,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    axis: int,
) -> NDArrayFloat: ...
@typing.overload
def xp_gradient(
    f: ArrayLike,
    *varargs: Any,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    axis: None = None,
) -> NDArrayFloat | list[NDArrayFloat]: ...
def xp_gradient(
    f: ArrayLike,
    *varargs: Any,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    axis: Any = None,
) -> NDArrayFloat | list[NDArrayFloat]:
    """
    *Array API* compatible implementation of :func:`numpy.gradient`.

    Parameters
    ----------
    f
        Array of function values.
    *varargs
        Spacing between values.
    xp
        Array namespace module. If *None*, derived from ``f``.
    axis
        Axis along which to compute the gradient.

    Returns
    -------
    :class:`object`
        Gradient of *f*.
    """

    xp = array_namespace(f) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.gradient(f, *varargs, axis=axis)

    try:
        result = xp.gradient(f, *varargs, axis=axis)
    except (AttributeError, TypeError):
        pass
    else:
        # Some backends (e.g., torch) return a tuple of tensors.
        if isinstance(result, (tuple, list)) and len(result) == 1:
            return result[0]
        return result  # pyright: ignore

    _runtime_warning_xp_fallback("xp_gradient")

    result = np.gradient(as_ndarray(f), *(as_ndarray(v) for v in varargs), axis=axis)

    if isinstance(result, list):
        return [xp.asarray(r) for r in result]

    return xp.asarray(result)


def xp_resize(
    a: ArrayLike,
    new_shape: Any,
    *,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArray:
    """
    *Array API* compatible implementation of :func:`numpy.resize`.

    Parameters
    ----------
    a
        Array to resize.
    new_shape
        Shape of the resized array.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Resized array.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.resize(a, new_shape)

    try:
        return xp.resize(a, new_shape)
    except (AttributeError, TypeError):
        pass

    # Native implementation via tile + slice for backends without resize.
    # ``numpy.resize`` accepts ``int``, ``tuple``, or ``list`` shapes;
    # normalise once at the boundary.
    shape_tuple = tuple(new_shape) if hasattr(new_shape, "__iter__") else (new_shape,)
    a = xp_as_array(a, xp=xp)
    raveled = xp.reshape(a, (-1,))
    target_size = 1
    for shape in shape_tuple:
        target_size *= shape

    if raveled.shape[0] == 0:
        return xp.zeros(
            shape_tuple,
            dtype=a.dtype,
            device=getattr(a, "device", None),
        )

    repeats = (target_size + raveled.shape[0] - 1) // raveled.shape[0]
    tiled = xp.tile(raveled, (repeats,))[:target_size]

    return xp.reshape(tiled, shape_tuple)


def xp_nanmean(
    a: ArrayLike,
    *,
    axis: int | None = None,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.nanmean`.

    Parameters
    ----------
    a
        Array containing numbers whose NaN-aware mean is desired.
    axis
        Axis along which the mean is computed.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        NaN-aware mean.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.nanmean(a, axis=axis)  # pyright: ignore

    mask = xp.isnan(a)
    zeroed = xp.where(mask, xp.asarray(0.0, dtype=a.dtype), a)  # pyright: ignore
    count = xp.sum(
        xp_astype(~mask, a.dtype, xp=xp),  # pyright: ignore
        axis=axis,
    )

    return xp.sum(zeroed, axis=axis) / count


def xp_median(
    a: ArrayLike,
    *,
    axis: int | None = None,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.median`.

    Parameters
    ----------
    a
        Array whose median is desired.
    axis
        Axis along which the median is computed.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Median value(s).
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.median(a, axis=axis)  # pyright: ignore

    _runtime_warning_xp_fallback("xp_median")

    result = np.median(as_ndarray(a), axis=axis)

    return xp.asarray(result)


def xp_round(
    a: ArrayLike,
    *,
    decimals: int = 0,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.round` with *decimals*.

    The Array API standard ``xp.round`` does not accept a *decimals*
    parameter. This helper uses the backend's native ``round`` when it
    supports *decimals* (JAX, CuPy), otherwise falls back to a
    multiply-round-divide pattern using the standard ``xp.round``.

    Parameters
    ----------
    a
        Array to round.
    decimals
        Number of decimal places.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Rounded array.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.round(a, decimals)  # pyright: ignore

    try:
        return xp.round(a, decimals)
    except (AttributeError, TypeError):
        factor = 10**decimals
        return xp.round(a * factor) / factor


def _scale_by(
    a: ArrayLike,
    factor: float,
    *,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """Multiply array :math:`a` by a scalar factor in its native namespace."""

    a = as_float_array(a)

    if not is_array_api_enabled():
        return a * factor

    xp = array_namespace(a) if xp is None else xp

    return a * xp_as_float_array(factor, xp=xp, like=a)


def xp_radians(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.radians`.

    Parameters
    ----------
    a
        Angle in degrees.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Angle in radians.
    """

    return _scale_by(a, np.pi / 180, xp=xp)


def xp_degrees(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.degrees`.

    Parameters
    ----------
    a
        Angle in radians.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Angle in degrees.
    """

    return _scale_by(a, 180 / np.pi, xp=xp)


# NOTE: The following wrappers around ``array_api_extra`` functions exist for
# typing purposes. ``array_api_extra`` returns a generic ``Array`` type that
# ``pyright`` cannot reconcile with ``NDArrayFloat`` and other *Colour* type
# aliases. These thin wrappers provide properly annotated return types,
# avoiding ``cast()`` noise at every call site. They can be removed once
# ``array-api-typing`` is released and ``array_api_extra`` adopts it.


def xp_atleast_1d(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.atleast_1d`.

    Parameters
    ----------
    a
        Array to ensure is at least 1-D.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Array with ``ndim >= 1``.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.atleast_1d(a)

    if isinstance(a, (np.ndarray, np.generic)):
        a = xp_as_array(a, xp=xp)

    return _xpx().atleast_nd(a, ndim=1, xp=xp)


def xp_atleast_2d(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.atleast_2d`.

    Parameters
    ----------
    a
        Array to ensure is at least 2-D.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Array with ``ndim >= 2``.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.atleast_2d(a)

    if isinstance(a, (np.ndarray, np.generic)):
        a = xp_as_array(a, xp=xp)

    return _xpx().atleast_nd(a, ndim=2, xp=xp)


def xp_squeeze(
    a: ArrayLike,
    *,
    axis: int | tuple[int, ...] | None = None,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArray:
    """
    Squeeze size-1 dimensions from the specified array :math:`a`.

    The *Array API* standard requires an explicit ``axis`` argument for
    :func:`squeeze`, unlike *NumPy* which allows omitting it to squeeze all
    size-1 dimensions. When ``axis`` is ``None``, this helper computes the
    axes automatically.

    Parameters
    ----------
    a
        Array :math:`a` to squeeze.
    axis
        Axis or axes to squeeze. When ``None``, all size-1 dimensions are
        squeezed.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Squeezed array.

    Examples
    --------
    >>> xp_squeeze(np.array([[1.0, 2.0]]), xp=np)
    array([1., 2.])
    >>> xp_squeeze(np.array([[[1.0], [2.0]]]), axis=-1, xp=np)
    array([[1., 2.]])
    """

    xp = array_namespace(a) if xp is None else xp

    if axis is not None:
        return xp.squeeze(a, axis=axis)

    axes = tuple(i for i in range(a.ndim) if a.shape[i] == 1)  # pyright: ignore

    if not axes:
        return a  # pyright: ignore

    return xp.squeeze(a, axis=axes)


def xp_sinc(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.sinc`.

    Parameters
    ----------
    a
        Array of values.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Sinc of *a*.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.sinc(a)  # pyright: ignore

    if isinstance(a, (np.ndarray, np.generic)):
        a = xp_as_array(a, xp=xp)

    return _xpx().sinc(a, xp=xp)


def xp_isclose(
    a: ArrayLike,
    b: ArrayLike,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayBoolean:
    """
    *Array API* compatible implementation of :func:`numpy.isclose`.

    Parameters
    ----------
    a
        First array.
    b
        Second array.
    rtol
        Relative tolerance.
    atol
        Absolute tolerance.
    xp
        Array namespace module. If *None*, derived from ``a`` and ``b``.

    Returns
    -------
    :class:`object`
        Boolean array of element-wise comparisons.
    """

    xp = array_namespace(a, b) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.isclose(a, b, rtol=rtol, atol=atol)

    if isinstance(a, (np.ndarray, np.generic)):
        a = xp_as_array(a, xp=xp)
    if isinstance(b, (np.ndarray, np.generic)):
        b = xp_as_array(b, xp=xp)

    return _xpx().isclose(a, b, rtol=rtol, atol=atol, xp=xp)


def xp_nan_to_num(
    a: ArrayLike,
    *,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.nan_to_num`.

    Parameters
    ----------
    a
        Array to process.
    nan
        Value to replace NaN entries.
    posinf
        Value to replace positive infinity entries.
    neginf
        Value to replace negative infinity entries.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Array with NaN/Inf replaced.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.nan_to_num(a, nan=nan, posinf=posinf, neginf=neginf)

    a = as_float_array(a)

    result = as_float_array(_xpx().nan_to_num(a, fill_value=nan, xp=xp))

    if posinf is not None:
        result = as_float_array(xp.where(xp.isinf(a) & (a > 0), posinf, result))

    if neginf is not None:
        result = as_float_array(xp.where(xp.isinf(a) & (a < 0), neginf, result))

    return result


def xp_create_diagonal(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.diagflat`.

    Parameters
    ----------
    a
        1-D array of diagonal values.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        2-D array with *a* on the diagonal.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.diagflat(a)

    if isinstance(a, (np.ndarray, np.generic)):
        a = xp_as_array(a, xp=xp)

    return _xpx().create_diagonal(a, xp=xp)


def xp_reshape(
    a: ArrayLike, shape: Any, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArray:
    """
    *Array API* compatible implementation of :func:`numpy.reshape`.

    This typed wrapper exists because ``xp`` is typed as :class:`Any`, so
    ``xp.reshape(...)`` returns :class:`Any`. When the result is assigned
    back to a variable that was originally a function parameter (e.g.,
    ``a: ArrayLike``), *Pyright* reverts the variable to its declared type
    instead of narrowing it. This wrapper declares ``-> NDArray`` so
    that *Pyright* can track the narrowed type through reassignments.

    Callers must promote *Python* scalars / lists to the backend
    namespace beforehand (e.g. via :func:`xp_as_float_array`), since
    strict backends like *PyTorch* reject raw scalars and the wrapper
    cannot infer the target device for a scalar input.

    Parameters
    ----------
    a
        Input array.
    shape
        New shape.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Reshaped array.
    """

    xp = array_namespace(a) if xp is None else xp

    return xp.reshape(a, shape)


def xp_broadcast_to(
    a: ArrayLike, shape: Any, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> NDArray:
    """
    *Array API* compatible implementation of :func:`numpy.broadcast_to`.

    This typed wrapper exists because ``xp`` is typed as :class:`Any`, so
    ``xp.broadcast_to(...)`` returns :class:`Any`. When the result is
    assigned back to a variable that was originally a function parameter
    (e.g., ``a: ArrayLike``), *Pyright* reverts the variable to its
    declared type instead of narrowing it. This wrapper declares ``->
    NDArray`` so that *Pyright* can track the narrowed type through
    reassignments.

    Callers must promote *Python* scalars / lists to the backend
    namespace beforehand (e.g. via :func:`xp_as_float_array`), since
    strict backends like *PyTorch* reject raw scalars and the wrapper
    cannot infer the target device for a scalar input.

    Parameters
    ----------
    a
        Input array.
    shape
        Target shape.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`object`
        Broadcast array.
    """

    xp = array_namespace(a) if xp is None else xp

    return xp.broadcast_to(a, shape)


def xp_lstsq(
    a: ArrayLike,
    b: ArrayLike,
    *,
    rcond: float | None = None,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArrayFloat:
    """
    *Array API* compatible implementation of :func:`numpy.linalg.lstsq`.

    Returns only the least-squares solution (the first element of the tuple
    returned by :func:`numpy.linalg.lstsq`). Backends that provide
    ``xp.linalg.lstsq`` (e.g., *JAX*, *PyTorch*) are used natively; others
    fall back to *NumPy* with a `ColourRuntimeWarning`.

    Parameters
    ----------
    a
        Coefficient matrix.
    b
        Ordinate values.
    rcond
        Cut-off ratio for small singular values (passed to *NumPy* only).
    xp
        Array namespace module. If *None*, derived from ``a`` and ``b``.

    Returns
    -------
    :class:`object`
        Least-squares solution.
    """

    xp = array_namespace(a, b) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.linalg.lstsq(a, b, rcond=rcond)[0]  # pyright: ignore

    try:
        return xp.linalg.lstsq(a, b)[0]
    except (AttributeError, TypeError, NotImplementedError, RuntimeError):
        pass

    _runtime_warning_xp_fallback("xp_lstsq")

    a_nd = as_ndarray(a)
    result = np.linalg.lstsq(a_nd, as_ndarray(b), rcond=rcond)[0]
    result = result.astype(a_nd.dtype)

    return xp.asarray(result)


def _xp_eig_generic(
    a: ArrayLike, xp: ProtocolArrayNamespace | ModuleType, name: str
) -> tuple[NDArray, NDArray]:
    """Shared implementation for :func:`xp_eig` and :func:`xp_eigh`."""

    try:
        return getattr(xp.linalg, name)(a)
    except (AttributeError, TypeError, NotImplementedError, RuntimeError):
        _runtime_warning_xp_fallback(f"xp_{name}")

        w, v = getattr(np.linalg, name)(as_ndarray(a))

        # ``a`` is passed as the device reference so that the host round-trip
        # returns the results on the input's device rather than the default.
        return xp_as_array(w, xp=xp, like=a), xp_as_array(v, xp=xp, like=a)


def xp_eig(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> tuple[NDArray, NDArray]:
    """
    *Array API* compatible implementation of :func:`numpy.linalg.eig`.

    Falls back to *NumPy* when the backend does not implement
    ``linalg.eig`` (e.g., *PyTorch* MPS).

    Parameters
    ----------
    a
        Input square matrix.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`tuple`
        Eigenvalues and eigenvectors.
    """

    xp = array_namespace(a) if xp is None else xp

    return _xp_eig_generic(a, xp, "eig")


def xp_eigh(
    a: ArrayLike, *, xp: ProtocolArrayNamespace | ModuleType | None = None
) -> tuple[NDArray, NDArray]:
    """
    *Array API* compatible implementation of :func:`numpy.linalg.eigh`.

    Falls back to *NumPy* when the backend does not implement
    ``linalg.eigh`` (e.g., *PyTorch* MPS).

    Parameters
    ----------
    a
        Input symmetric/Hermitian matrix.
    xp
        Array namespace module. If *None*, derived from ``a``.

    Returns
    -------
    :class:`tuple`
        Eigenvalues and eigenvectors.
    """

    xp = array_namespace(a) if xp is None else xp

    return _xp_eig_generic(a, xp, "eigh")


def xp_isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    *,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    like: Any = None,
) -> NDArrayBoolean:
    """
    *Array API* compatible implementation of :func:`numpy.isin`.

    Use the backend's native ``isin`` when available (JAX, CuPy),
    otherwise fall back to NumPy.

    Parameters
    ----------
    element
        Input array.
    test_elements
        Values against which to test each element of *element*.
    xp
        Array namespace module. If *None*, derived from ``element`` and
        ``test_elements``.
    like
        Reference array whose device ``test_elements`` should be placed
        on when promoted to ``xp``. Defaults to ``element`` when *None*.

    Returns
    -------
    :class:`object`
        Boolean array of the same shape as *element*.

    Notes
    -----
    -   :class:`NaN` is treated as not equal to itself, matching
        :func:`numpy.isin` semantics; ``xp_isin([NaN], [NaN])`` returns
        ``[False]``.
    """

    xp = array_namespace(element, test_elements) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.isin(element, test_elements)

    reference = element if like is None else like

    try:
        if isinstance(test_elements, np.ndarray) and hasattr(reference, "device"):
            test_elements = xp.asarray(test_elements, device=reference.device)  # pyright: ignore
        return xp.isin(element, test_elements)
    except (AttributeError, TypeError):
        pass

    _runtime_warning_xp_fallback("xp_isin")

    result = np.isin(np.asarray(element), np.asarray(test_elements))

    return xp.asarray(result)


def xp_linspace(
    start: ArrayLike,
    stop: ArrayLike,
    *,
    num: int = 50,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    like: Any = None,
    **kwargs: Any,
) -> NDArrayFloat | tuple[NDArrayFloat, float]:
    """
    *Array API* compatible implementation of :func:`numpy.linspace` with
    extra keyword arguments such as *retstep* and *dtype*.

    The Array API standard ``xp.linspace`` does not accept *retstep*.
    This helper tries the backend's native ``linspace`` first, falling
    back to NumPy if the keyword is unsupported.

    Parameters
    ----------
    start
        Start of the interval.
    stop
        End of the interval.
    num
        Number of samples.
    xp
        Array namespace module. If *None*, derived from ``start`` and
        ``stop``.
    like
        Reference array whose device to match (for backends like *PyTorch*
        that support multiple devices); the result is created on that device
        rather than the backend's default.
    **kwargs
        Extra keyword arguments (e.g., ``retstep``, ``dtype``).

    Returns
    -------
    :class:`object`
        Array of evenly spaced values (and step size if *retstep=True*).
    """

    xp = array_namespace(start, stop) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.linspace(start, stop, num, **kwargs)  # pyright: ignore

    device = getattr(like, "device", None)

    # The backend default dtype, e.g. float32 for stock *PyTorch*, would
    # otherwise silently determine the precision of the samples; the *Colour*
    # default float dtype is requested instead unless the caller specifies one.
    dtype = kwargs.pop("dtype", DTYPE_FLOAT_DEFAULT)

    try:
        return xp.linspace(
            start,
            stop,
            num,
            device=device,
            dtype=_resolve_backend_dtype(xp, dtype),
            **kwargs,
        )
    except (AttributeError, TypeError):
        _runtime_warning_xp_fallback("xp_linspace")

        result = np.linspace(start, stop, num, dtype=dtype, **kwargs)  # pyright: ignore

        def _to_backend(arr: NDArrayFloat) -> NDArrayFloat:
            try:
                return xp.asarray(arr, device=device)
            except TypeError:
                _runtime_warning_xp_downcast(xp, arr.dtype, "float32")
                return xp.asarray(arr.astype(np.float32), device=device)

        if isinstance(result, tuple):
            return _to_backend(np.asarray(result[0])), result[1]

        return _to_backend(np.asarray(result))


def xp_pad(
    a: ArrayLike,
    pad_width: Any,
    *args: Any,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    **kwargs: Any,
) -> NDArray:
    """
    *Array API* compatible implementation of :func:`numpy.pad`.

    Use the backend's native ``pad`` when available (JAX, CuPy),
    otherwise fall back to NumPy.

    Parameters
    ----------
    a
        Array to pad.
    pad_width
        Number of values padded to the edges of each axis.
    *args
        Positional arguments passed to the padding function.
    xp
        Array namespace module. If *None*, derived from ``a``.
    **kwargs
        Keyword arguments passed to the padding function.

    Returns
    -------
    :class:`object`
        Padded array.
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.pad(a, pad_width, *args, **kwargs)

    try:
        return xp.pad(a, pad_width, *args, **kwargs)
    except (AttributeError, TypeError):
        pass

    _runtime_warning_xp_fallback("xp_pad")

    result = np.pad(as_ndarray(a), pad_width, *args, **kwargs)

    return xp.asarray(result)


def xp_unique(
    a: ArrayLike,
    *,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
    **kwargs: Any,
) -> NDArray | tuple[NDArray, ...]:
    """
    *Array API* compatible implementation of :func:`numpy.unique` with
    extra keyword arguments such as *return_index* and *axis*.

    The Array API standard only provides ``xp.unique_values`` and
    related functions without *return_index* or *axis* support.
    This helper tries the backend's native ``unique`` first, falling
    back to NumPy if the keywords are unsupported.

    Parameters
    ----------
    a
        Input array.
    xp
        Array namespace module. If *None*, derived from ``a``.
    **kwargs
        Extra keyword arguments (e.g., ``return_index``, ``axis``).

    Returns
    -------
    :class:`object`
        Unique values (and optional indices).
    """

    xp = array_namespace(a) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.unique(a, **kwargs)

    try:
        return xp.unique(a, **kwargs)
    except (AttributeError, TypeError):
        pass

    _runtime_warning_xp_fallback("xp_unique")

    result = np.unique(as_ndarray(a), **kwargs)

    if isinstance(result, tuple):
        return tuple(xp.asarray(r) for r in result)

    return xp.asarray(result)


def xp_insert(
    a: ArrayLike,
    indices: ArrayLike,
    values: ArrayLike,
    *,
    axis: int | None = None,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArray:
    """
    *Array API* compatible implementation of :func:`numpy.insert` for sorted
    indices.

    Parameters
    ----------
    a
        Array to insert into.
    indices
        Indices before which to insert *values*.
    values
        Values to insert.
    axis
        Axis along which to insert; ``None`` flattens ``a`` first.
    xp
        Array namespace module. If *None*, derived from ``a``, ``indices``
        and ``values``.

    Returns
    -------
    :class:`object`
        Array with *values* inserted.
    """

    xp = array_namespace(a, indices, values) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.insert(a, indices, values, axis=axis)  # pyright: ignore

    a_array: NDArray = xp_as_array(a, xp=xp)
    indices_array = xp_as_array(indices, xp=xp)
    # ``numpy.insert`` treats a scalar index differently from a length-1
    # sequence: the former is a block insert whose ``values`` are moved onto
    # ``axis`` first. The distinction is captured before promoting to 1-D.
    indices_scalar = getattr(indices_array, "ndim", 0) == 0
    indices_1d: NDArray = xp_atleast_1d(indices_array, xp=xp)
    # A scalar ``values`` is promoted to at least 1-D so that it can be sliced
    # along ``axis`` below, matching ``numpy.insert`` which accepts scalars.
    values_array: NDArray = xp_atleast_1d(xp_as_array(values, xp=xp), xp=xp)

    if axis is None:
        a_array = xp_reshape(a_array, (-1,), xp=xp)
        axis = 0

    def slice_along(arr: NDArray, indexer: slice | NDArray) -> NDArray:
        """Index ``arr`` along ``axis`` with ``indexer``."""

        selector = [slice(None)] * arr.ndim
        selector[cast("int", axis)] = indexer  # pyright: ignore
        return arr[tuple(selector)]

    if indices_scalar:
        # ``numpy.insert`` pads ``values`` to the rank of ``a`` and moves its
        # leading axis onto ``axis`` before counting the insertions, so that
        # ``a[:, 0, :] = ...`` and ``a[:, [0], :] = ...`` semantics differ.
        while values_array.ndim < a_array.ndim:
            values_array = values_array[None, ...]
        if a_array.ndim > 1:
            values_array = xp.moveaxis(values_array, 0, axis)

        # ``numpy.insert`` assigns ``values`` into the insertion slot, which
        # broadcasts it against the remaining axes; the concat below requires
        # the operands to match exactly, so the broadcast is explicit here.
        shape_values = list(a_array.shape)
        shape_values[axis] = values_array.shape[axis]
        values_array = xp_broadcast_to(values_array, tuple(shape_values), xp=xp)

        # The scalar index is a block insert before one position: it is
        # broadcast to one index per inserted value for the concat below.
        indices_1d = xp_broadcast_to(indices_1d, (values_array.shape[axis],), xp=xp)

    # ``numpy.insert`` normalises negative indices against the axis length
    # before sorting; doing so here keeps the sorted concat equivalent.
    a_axis_length = a_array.shape[axis]
    indices_1d = xp.where(indices_1d < 0, indices_1d + a_axis_length, indices_1d)

    # ``numpy.insert`` accepts unsorted indices and normalises to the
    # sorted equivalent; the sequential concat below requires sorted
    # indices, so sort them and reorder ``values`` along ``axis`` to
    # match. Materialise sorted indices to *NumPy* once (single host
    # sync) to avoid per-iteration ``int(indices[i])`` syncs inside the
    # loop below.
    order = xp.argsort(indices_1d)
    values_array = slice_along(values_array, order)
    indices_sorted = as_ndarray(indices_1d[order])

    a_axis = a_array.shape[axis]
    parts = []
    prev = 0
    for i in range(indices_sorted.shape[0]):
        idx = int(indices_sorted[i])
        parts.append(slice_along(a_array, slice(prev, idx)))
        parts.append(slice_along(values_array, slice(i, i + 1)))
        prev = idx
    parts.append(slice_along(a_array, slice(prev, a_axis)))

    return xp.concat(parts, axis=axis)


def xp_setxor1d(
    a: ArrayLike,
    b: ArrayLike,
    *,
    xp: ProtocolArrayNamespace | ModuleType | None = None,
) -> NDArray:
    """
    *Array API* compatible implementation of :func:`numpy.setxor1d`.

    Return sorted unique values that are in only one of the two input arrays.

    Parameters
    ----------
    a
        First array.
    b
        Second array.
    xp
        Array namespace module. If *None*, derived from ``a`` and ``b``.

    Returns
    -------
    :class:`object`
        Sorted symmetric difference.
    """

    xp = array_namespace(a, b) if xp is None else xp

    if is_numpy_namespace(xp):
        return np.setxor1d(a, b)

    a = xp_as_array(a, xp=xp)
    b = xp_as_array(b, xp=xp, like=a)

    # NOTE: ``cast`` is used to bridge ``NDArrayFloat`` and the ``Array``
    # protocol from ``array-api-extra`` which *Pyright* cannot reconcile
    # across environments with and without type stubs.
    xpx_typed = _xpx()
    a_in_b = xpx_typed.isin(cast("Any", a), cast("Any", b))
    b_in_a = xpx_typed.isin(cast("Any", b), cast("Any", a))

    a_only = cast("Any", a)[~xp.asarray(a_in_b)]
    b_only = cast("Any", b)[~xp.asarray(b_in_a)]

    # ``numpy.setxor1d`` returns *sorted unique* values; ``a_only`` and
    # ``b_only`` are disjoint but may each carry internal duplicates, so the
    # union is deduplicated via :func:`xp_unique` (which also sorts) to match
    # the *NumPy* path rather than concatenating the raw slices.
    return cast("NDArray", xp_unique(xp.concat([a_only, b_only]), xp=xp))


def xp_assert_close(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    err_msg: str = "",
) -> None:
    """
    *Array API* compatible implementation of :func:`numpy.testing.assert_allclose`.

    Both arrays are converted to *NumPy* via :func:`as_ndarray` before
    comparison.

    Parameters
    ----------
    actual
        Array produced by the tested function.
    desired
        Expected array.
    rtol
        Relative tolerance. If *None*,
        :attr:`colour.constants.TOLERANCE_RELATIVE_TESTS`, resolved at call
        time so that test fixtures relaxing the module-level constant also
        relax calls relying on the default.
    atol
        Absolute tolerance. If *None*,
        :attr:`colour.constants.TOLERANCE_ABSOLUTE_TESTS`, resolved at call
        time so that test fixtures relaxing the module-level constant also
        relax calls relying on the default.
    err_msg
        Error message to display on failure.
    """

    rtol = TOLERANCE_RELATIVE_TESTS if rtol is None else rtol
    atol = TOLERANCE_ABSOLUTE_TESTS if atol is None else atol

    np.testing.assert_allclose(
        as_ndarray(actual),
        as_ndarray(desired),
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
    )


def xp_assert_equal(
    actual: ArrayLike,
    desired: ArrayLike,
    *,
    err_msg: str = "",
) -> None:
    """
    *Array API* compatible implementation of :func:`numpy.testing.assert_array_equal`.

    Both arrays are converted to *NumPy* via :func:`as_ndarray` before
    comparison.

    Parameters
    ----------
    actual
        Array produced by the tested function.
    desired
        Expected array.
    err_msg
        Error message to display on failure.
    """

    np.testing.assert_array_equal(
        as_ndarray(actual),
        as_ndarray(desired),
        err_msg=err_msg,
    )


class MixinDataclassFields:
    """
    Provide fields introspection for :class:`dataclass`-like classes.

    This mixin extends dataclass functionality to enable introspection
    capabilities, allowing programmatic access to field metadata and
    properties.

    Attributes
    ----------
    -   :attr:`~colour.utilities.MixinDataclassFields.fields`
    """

    @property
    def fields(self) -> tuple:
        """
        Getter for the fields of the :class:`dataclass`-like class.

        Returns
        -------
        :class:`tuple`
            :class:`dataclass`-like class fields.
        """

        return fields(self)  # pyright: ignore


class MixinDataclassIterable(MixinDataclassFields):
    """
    Provide iteration capabilities over :class:`dataclass`-like classes.

    This mixin extends dataclass functionality to enable dictionary-like
    iteration over fields, allowing access to field names, values, and
    name-value pairs through standard iteration protocols.

    Attributes
    ----------
    -   :attr:`~colour.utilities.MixinDataclassIterable.keys`
    -   :attr:`~colour.utilities.MixinDataclassIterable.values`
    -   :attr:`~colour.utilities.MixinDataclassIterable.items`

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassIterable.__iter__`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassIterable` class inherits
        the methods from the following class:

        -   :class:`colour.utilities.MixinDataclassFields`
    """

    @property
    def keys(self) -> tuple:
        """
        Getter for the :class:`dataclass`-like class keys, i.e., the field
        names.

        Returns
        -------
        :class:`tuple`
            :class:`dataclass`-like class keys.
        """

        return tuple(field for field, _value in self)

    @property
    def values(self) -> tuple:
        """
        Getter for the :class:`dataclass`-like class field values.

        Returns
        -------
        :class:`tuple`
            :class:`dataclass`-like class field values.
        """

        return tuple(value for _field, value in self)

    @property
    def items(self) -> tuple:
        """
        Getter for the :class:`dataclass`-like class items, i.e., the field
        names and values.

        Returns
        -------
        :class:`tuple`
            :class:`dataclass`-like class items.
        """

        return tuple((field, value) for field, value in self)

    def __iter__(self) -> Generator:
        """
        Yield the :class:`dataclass`-like class fields.

        Yields
        ------
        Generator
            :class:`dataclass`-like class field generator.
        """

        yield from {
            field.name: getattr(self, field.name) for field in self.fields
        }.items()


class MixinDataclassArray(MixinDataclassIterable):
    """
    Provide conversion methods for :class:`dataclass`-like classes to
    :class:`numpy.ndarray` objects.

    This mixin extends dataclass functionality to enable seamless conversion
    to NumPy arrays, facilitating numerical operations on structured data.

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassArray.__array__`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassArray` class
        inherits the methods from the following classes:

        -   :class:`colour.utilities.MixinDataclassIterable`
        -   :class:`colour.utilities.MixinDataclassFields`
    """

    def __array__(
        self, dtype: Type[DTypeReal] | None = None, copy: bool = True
    ) -> NDArray:
        """
        Implement support for :class:`dataclass`-like class conversion to
        :class:`numpy.ndarray` class.

        A field set to *None* will be filled with `np.nan` according to the
        shape of the first field not set with *None*.

        Parameters
        ----------
        dtype
            :class:`numpy.dtype` to use for conversion to `np.ndarray`,
            default to the :class:`numpy.dtype` defined by
            :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.
        copy
            Whether to return a copy of the underlying data, will always be
            `True`, irrespective of the parameter value.

        Returns
        -------
        :class:`numpy.ndarray`
            :class:`dataclass`-like class converted to
            :class:`numpy.ndarray`.
        """

        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        default = None
        for _field, value in self:
            if value is not None:
                default = full(as_float_array(value).shape, np.nan)
                break

        return tstack(
            cast(
                "ArrayLike",
                [
                    as_ndarray(value) if value is not None else default
                    for value in self.values
                ],
            ),
            dtype=dtype,
        )


class MixinDataclassArithmetic(MixinDataclassArray):
    """
    Provide mathematical operations for :class:`dataclass`-like classes.

    This mixin extends dataclass functionality to enable arithmetic
    operations, facilitating mathematical computations on dataclass instances
    containing array-like data.

    Methods
    -------
    -   :meth:`~colour.utilities.MixinDataclassArray.__iadd__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__add__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__isub__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__sub__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__imul__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__mul__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__idiv__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__div__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__ipow__`
    -   :meth:`~colour.utilities.MixinDataclassArray.__pow__`
    -   :meth:`~colour.utilities.MixinDataclassArray.arithmetical_operation`

    Notes
    -----
    -   The :class:`colour.utilities.MixinDataclassArithmetic` class inherits
        the methods from the following classes:

        -   :class:`colour.utilities.MixinDataclassArray`
        -   :class:`colour.utilities.MixinDataclassIterable`
        -   :class:`colour.utilities.MixinDataclassFields`
    """

    def __add__(self, a: Any) -> Self:
        """
        Implement support for addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add.

        Returns
        -------
        :class:`dataclass`
            Variable added :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "+")

    def __iadd__(self, a: Any) -> Self:
        """
        Implement support for in-place addition.

        Parameters
        ----------
        a
            Variable :math:`a` to add in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable added :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "+", True)

    def __sub__(self, a: Any) -> Self:
        """
        Implement support for subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract.

        Returns
        -------
        :class:`dataclass`
            Variable subtracted :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "-")

    def __isub__(self, a: Any) -> Self:
        """
        Implement support for in-place subtraction.

        Parameters
        ----------
        a
            Variable :math:`a` to subtract in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable subtracted :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "-", True)

    def __mul__(self, a: Any) -> Self:
        """
        Implement support for multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply by.

        Returns
        -------
        :class:`dataclass`
            Variable multiplied :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "*")

    def __imul__(self, a: Any) -> Self:
        """
        Implement support for in-place multiplication.

        Parameters
        ----------
        a
            Variable :math:`a` to multiply by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable multiplied :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "*", True)

    def __div__(self, a: Any) -> Self:
        """
        Implement support for division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide by.

        Returns
        -------
        :class:`dataclass`
            Variable divided :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "/")

    def __idiv__(self, a: Any) -> Self:
        """
        Implement support for in-place division.

        Parameters
        ----------
        a
            Variable :math:`a` to divide by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable divided :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "/", True)

    __itruediv__ = __idiv__
    __truediv__ = __div__

    def __pow__(self, a: Any) -> Self:
        """
        Implement support for exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to exponentiate by.

        Returns
        -------
        :class:`dataclass`
            Variable exponentiated :class:`dataclass`-like class.
        """

        return self.arithmetical_operation(a, "**")

    def __ipow__(self, a: Any) -> Self:
        """
        Implement support for in-place exponentiation.

        Parameters
        ----------
        a
            Variable :math:`a` to exponentiate by in-place.

        Returns
        -------
        :class:`dataclass`
            In-place variable exponentiated :class:`dataclass`-like
            class.
        """

        return self.arithmetical_operation(a, "**", True)

    def arithmetical_operation(
        self, a: Any, operation: str, in_place: bool = False
    ) -> Dataclass:
        """
        Perform the specified arithmetical operation with the :math:`a`
        operand on the :class:`dataclass`-like class.

        Parameters
        ----------
        a
            Operand.
        operation
            Operation to perform.
        in_place
            Operation happens in place.

        Returns
        -------
        :class:`dataclass`
            :class:`dataclass`-like class with the arithmetical operation
            performed.
        """

        callable_operation = {
            "+": add,
            "-": sub,
            "*": mul,
            "/": truediv,
            "**": pow,
        }[operation]

        if is_dataclass(a):
            a = as_float_array(a)  # pyright: ignore

        self_array = as_float_array(self)
        a = as_ndarray(a) if is_non_ndarray(a) else a

        values = tsplit(callable_operation(self_array, a))
        field_values = {field: values[i] for i, field in enumerate(self.keys)}
        field_values.update({field: None for field, value in self if value is None})

        dataclass = replace(self, **field_values)  # pyright: ignore

        if in_place:
            for field in self.keys:
                setattr(self, field, getattr(dataclass, field))

            return self

        return dataclass


# NOTE : The following messages are pre-generated for performance reasons.
_ASSERTION_MESSAGE_DTYPE_INT = (
    f'"dtype" must be one of the following types: "{DTypeInt.__args__}"'
)

_ASSERTION_MESSAGE_DTYPE_FLOAT = (
    f'"dtype" must be one of the following types: "{DTypeFloat.__args__}"'
)

_ASSERTION_MESSAGE_DTYPE_COMPLEX = (
    f'"dtype" must be one of the following types: "{DTypeComplex.__args__}"'
)


def cast_non_ndarray(a: ArrayLike, dtype: Any) -> Any | None:
    """
    Cast the specified non-:class:`numpy.ndarray` array :math:`a` to the
    specified :class:`numpy.dtype` in its native namespace.

    This is the *Array API* sibling of :func:`as_ndarray`; it preserves a
    non-NumPy array's namespace and device while applying a dtype change
    via :func:`xp_astype`, returning ``None`` when the input is a NumPy
    array, when *Array API* dispatch is disabled, when the input is not
    an array (no ``dtype`` attribute), or when the namespace does not
    expose an equivalent dtype.

    Parameters
    ----------
    a
        Array to cast. Returned as-is when its dtype already matches the
        target.
    dtype
        Target :class:`numpy.dtype`. Resolved against the input's native
        namespace via attribute lookup on the dtype name.

    Returns
    -------
    :class:`object` or :py:obj:`None`
        Cast array in its native namespace, or ``None`` when the input is
        not eligible for a non-NumPy cast.

    Examples
    --------
    >>> import numpy as np
    >>> cast_non_ndarray(np.array([1, 2, 3]), np.float32) is None
    True

    Cast a *PyTorch* tensor while preserving its device:

    >>> import torch  # doctest: +SKIP
    >>> set_array_api_enabled(True)  # doctest: +SKIP
    >>> cast_non_ndarray(torch.tensor([1, 2, 3]), np.float32).dtype  # doctest: +SKIP
    torch.float32
    >>> set_array_api_enabled(False)  # doctest: +SKIP
    """

    if not (
        is_array_api_enabled() and hasattr(a, "dtype") and not isinstance(a, np.ndarray)
    ):
        return None

    xp = array_namespace(a)

    xp_dtype = getattr(xp, np.dtype(dtype).name, None)

    if xp_dtype is None:
        return None

    if a.dtype == xp_dtype:  # pyright: ignore
        return a

    try:
        return xp_astype(a, xp_dtype, xp=xp)
    except (TypeError, AttributeError, RuntimeError):
        return None


def as_array(
    a: ArrayLike | KeysView | ValuesView,
    dtype: Type[DType] | None = None,
) -> NDArray:
    """
    Convert the specified variable :math:`a` to an array using the specified
    :class:`numpy.dtype`.

    This is a namespace-aware boundary helper. When *Array API* dispatch is
    enabled and :math:`a` is a non-*NumPy* array (e.g. *JAX*, *PyTorch*),
    the result is returned in :math:`a`'s native namespace, on its device,
    and cast to ``dtype``. Otherwise the result is a
    :class:`numpy.ndarray`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray` or backend tensor
        Variable :math:`a` converted to an array in the input's namespace.

    Examples
    --------
    >>> as_array([1, 2, 3])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    >>> as_array([1, 2, 3], dtype=DTYPE_FLOAT_DEFAULT)
    array([1., 2., 3.])
    """

    # TODO: Remove when https://github.com/numpy/numpy/issues/5718 is
    # addressed.
    if isinstance(a, (KeysView, ValuesView)):
        a = list(a)

    if is_array_api_enabled():
        # When ``a`` is a list/tuple of non-NumPy arrays, resolve the
        # namespace from the first element and use ``xp.stack`` since
        # ``xp.asarray(list)`` would fall back to NumPy.
        if isinstance(a, list) and len(a) > 0 and is_non_ndarray(a[0]):
            xp = array_namespace(a[0])

            if dtype is not None:
                dtype = getattr(xp, np.dtype(dtype).name, dtype)

            return xp.stack([xp_as_array(x, xp=xp, like=a[0]) for x in a])

        xp = array_namespace(a)

        result = (
            (a if dtype is None else cast_non_ndarray(a, dtype))
            if is_non_ndarray(a)
            else None
        )

        if result is None:
            if dtype is not None and not is_numpy_namespace(xp):
                dtype = getattr(xp, np.dtype(dtype).name, dtype)

            try:
                result = xp.asarray(a, dtype=dtype)
            except TypeError:
                # The device does not support the requested dtype, e.g. *MPS* has
                # no float64: the input dtype is kept and a warning is emitted
                # rather than failing, mirroring :func:`xp_as_array`.
                dtype_a = getattr(a, "dtype", None)
                if dtype_a is None:
                    raise

                _runtime_warning_xp_downcast(xp, dtype, dtype_a)

                result = xp.asarray(a)

        return result  # pyright: ignore

    try:
        return np.asarray(a, dtype)
    except TypeError:
        # Device-resident tensors (e.g. *PyTorch* on *MPS*) reject *NumPy*'s
        # ``__array__`` hand-off; route them through :func:`as_ndarray`. The
        # common *NumPy* / sequence path above pays nothing for this.
        if isinstance(a, (list, tuple)):
            return np.asarray([as_ndarray(x) for x in a], dtype)

        return np.asarray(as_ndarray(a), dtype)


@typing.overload
def as_int(a: float | DTypeFloat, dtype: Type[DTypeInt] | None = None) -> DTypeInt: ...
@typing.overload
def as_int(
    a: NDArray | Sequence[int], dtype: Type[DTypeInt] | None = None
) -> NDArrayInt: ...
@typing.overload
def as_int(
    a: ArrayLike, dtype: Type[DTypeInt] | None = None
) -> DTypeInt | NDArrayInt: ...
def as_int(a: ArrayLike, dtype: Type[DTypeInt] | None = None) -> DTypeInt | NDArrayInt:
    """
    Convert the specified variable :math:`a` to an integer value.

    Scalars and 0-dimensional arrays are returned as a Python / *NumPy*
    integer scalar. Higher-dimensional arrays go through
    :func:`as_int_array`, which is namespace-aware: when *Array API* dispatch
    is enabled and :math:`a` is a non-*NumPy* array, the result is returned
    in :math:`a`'s native namespace.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.integer` or :class:`numpy.ndarray` or backend tensor
        Variable :math:`a` converted to an integer scalar or array in the
        input's namespace.

    Examples
    --------
    >>> as_int(np.array(1))
    np.int64(1)
    >>> as_int(np.array([1]))  # doctest: +SKIP
    array([1])
    >>> as_int(np.arange(10))  # doctest: +SKIP
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]...)
    """

    dtype = optional(dtype, DTYPE_INT_DEFAULT)

    attest(dtype in DTypeInt.__args__, _ASSERTION_MESSAGE_DTYPE_INT)

    if is_array_api_enabled() and is_non_ndarray(a):
        return as_int_array(a, dtype)

    return dtype(a)  # pyright: ignore


@typing.overload
def as_float(
    a: float | DTypeFloat, dtype: Type[DTypeFloat] | None = None
) -> DTypeFloat: ...
@typing.overload
def as_float(
    a: NDArray | Sequence[float], dtype: Type[DTypeFloat] | None = None
) -> NDArrayFloat: ...
@typing.overload
def as_float(
    a: ArrayLike, dtype: Type[DTypeFloat] | None = None
) -> DTypeFloat | NDArrayFloat: ...
def as_float(
    a: ArrayLike, dtype: Type[DTypeFloat] | None = None
) -> DTypeFloat | NDArrayFloat:
    """
    Convert the specified variable :math:`a` to a floating-point value.

    Scalars and 0-dimensional arrays are returned as a Python / *NumPy*
    float scalar. Higher-dimensional arrays go through
    :func:`as_float_array`, which is namespace-aware: when *Array API*
    dispatch is enabled and :math:`a` is a non-*NumPy* array, the result is
    returned in :math:`a`'s native namespace.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray` or backend tensor
        Variable :math:`a` converted to a floating-point scalar or array in
        the input's namespace.

    Examples
    --------
    >>> as_float(np.array(1))
    np.float64(1.0)
    >>> as_float(np.array([1]))
    array([1.])
    >>> as_float(np.arange(10))
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    attest(dtype in DTypeFloat.__args__, _ASSERTION_MESSAGE_DTYPE_FLOAT)

    if is_array_api_enabled() and not isinstance(a, np.ndarray):
        return as_float_array(a, dtype)

    # NOTE: "np.float64" reduces dimensionality:
    # >>> np.int64(np.array([[1]]))
    # array([[1]])
    # >>> np.float64(np.array([[1]]))
    # 1.0
    # See for more information https://github.com/numpy/numpy/issues/24283
    if isinstance(a, np.ndarray) and a.size == 1 and a.ndim != 0:
        return as_float_array(a, dtype)

    return dtype(a)  # pyright: ignore


def as_int_array(a: ArrayLike, dtype: Type[DTypeInt] | None = None) -> NDArrayInt:
    """
    Convert the specified variable :math:`a` to an integer array using the
    specified :class:`numpy.dtype`.

    This is a namespace-aware boundary helper. When *Array API* dispatch is
    enabled and :math:`a` is a non-*NumPy* array (e.g. *JAX*, *PyTorch*),
    the result is returned in :math:`a`'s native namespace, on its device,
    and cast to ``dtype``. Otherwise the result is a
    :class:`numpy.ndarray`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray` or backend tensor
        Variable :math:`a` converted to an integer array in the input's
        namespace.

    Examples
    --------
    >>> as_int_array([1.0, 2.0, 3.0])  # doctest: +ELLIPSIS
    array([1, 2, 3]...)
    """

    dtype = optional(dtype, DTYPE_INT_DEFAULT)

    attest(dtype in DTypeInt.__args__, _ASSERTION_MESSAGE_DTYPE_INT)

    result = cast_non_ndarray(a, dtype)
    if result is not None:
        return result

    if is_array_api_enabled() and is_non_ndarray(a):
        a = as_ndarray(a)

    return as_array(a, dtype)


def as_float_array(a: ArrayLike, dtype: Type[DTypeFloat] | None = None) -> NDArrayFloat:
    """
    Convert the specified variable :math:`a` to a floating-point array using
    the specified :class:`numpy.dtype`.

    This is a namespace-aware boundary helper. When *Array API* dispatch is
    enabled and :math:`a` is a non-*NumPy* array (e.g. *JAX*, *PyTorch*),
    the result is returned in :math:`a`'s native namespace, on its device,
    and cast to ``dtype``. Otherwise the result is a
    :class:`numpy.ndarray`. This is the convention used at function
    boundaries: ``a = as_float_array(a)`` followed by
    ``xp = array_namespace(a)`` recovers the caller's backend.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        Floating-point :class:`numpy.dtype` to use for conversion, default
        to the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray` or backend tensor
        Variable :math:`a` converted to a floating-point array in the
        input's namespace.

    Examples
    --------
    >>> as_float_array([1, 2, 3])
    array([1., 2., 3.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    attest(dtype in DTypeFloat.__args__, _ASSERTION_MESSAGE_DTYPE_FLOAT)

    result = cast_non_ndarray(a, dtype)

    if result is not None:
        return result

    return as_array(a, dtype)


def as_int_scalar(a: ArrayLike, dtype: Type[DTypeInt] | None = None) -> int:
    """
    Convert the specified variable :math:`a` to :class:`numpy.integer` using
    the specified :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute.

    Returns
    -------
    :class:`int`
        Variable :math:`a` converted to :class:`numpy.integer`.

    Warnings
    --------
    -   The return type is effectively annotated as :class:`int` and not
        :class:`numpy.integer`.

    Examples
    --------
    >>> as_int_scalar(np.array(1))
    np.int64(1)
    """

    a = as_int_array(a, dtype)

    xp = array_namespace(a)

    a = xp_reshape(a, (), xp=xp)

    attest(a.ndim == 0, f'"{a}" cannot be converted to "int" scalar!')

    # TODO: Revisit when Numpy types are well established.
    return cast("int", as_int(a, dtype))


def as_float_scalar(a: ArrayLike, dtype: Type[DTypeFloat] | None = None) -> float:
    """
    Convert the specified variable :math:`a` to :class:`numpy.floating` using
    the specified :class:`numpy.dtype`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`float`
        Variable :math:`a` converted to :class:`numpy.floating`.

    Warnings
    --------
    -   The return type is effectively annotated as :class:`float` and not
        :class:`numpy.floating`.

    Examples
    --------
    >>> as_float_scalar(np.array(1))
    np.float64(1.0)
    """

    a = as_float_array(a, dtype)

    xp = array_namespace(a)

    a = xp_reshape(a, (), xp=xp)

    attest(a.ndim == 0, f'"{a}" cannot be converted to "float" scalar!')

    # TODO: Revisit when Numpy types are well established.
    return cast("float", as_float(a, dtype))


def as_complex_array(
    a: ArrayLike,
    dtype: Type[DTypeComplex] | None = None,
) -> NDArrayComplex:
    """
    Convert the specified variable :math:`a` to a complex array using the
    specified :class:`numpy.dtype`.

    This is a namespace-aware boundary helper. When *Array API* dispatch is
    enabled and :math:`a` is a non-*NumPy* array (e.g. *JAX*, *PyTorch*),
    the result is returned in :math:`a`'s native namespace, on its device,
    and cast to ``dtype``. Otherwise the result is a
    :class:`numpy.ndarray`.

    Parameters
    ----------
    a
        Variable :math:`a` to convert.
    dtype
        Complex :class:`numpy.dtype` to use for conversion, default
        to the :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_COMPLEX_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray` or backend tensor
        Variable :math:`a` converted to a complex array in the input's
        namespace.

    Examples
    --------
    >>> as_complex_array([1, 2, 3])
    array([1.+0.j, 2.+0.j, 3.+0.j])
    >>> as_complex_array([1 + 2j, 3 + 4j])
    array([1.+2.j, 3.+4.j])
    """

    dtype = optional(dtype, DTYPE_COMPLEX_DEFAULT)

    attest(dtype in DTypeComplex.__args__, _ASSERTION_MESSAGE_DTYPE_COMPLEX)

    result = cast_non_ndarray(a, dtype)
    if result is not None:
        return result

    # Fallback for backends that do not support the target complex dtype
    # (e.g., MPS does not support complex128): try the backend's complex64,
    # else hand the array off to host *NumPy* and cast there (mirroring the
    # :func:`as_int_array` / :func:`as_float_array` fall-through rather than
    # returning the input uncast under a complex contract).
    if is_array_api_enabled() and is_non_ndarray(a):
        result = cast_non_ndarray(a, np.complex64)
        if result is not None:
            return result
        a = as_ndarray(a)

    return as_array(a, dtype)


def set_default_int_dtype(
    dtype: Type[DTypeInt] = DTYPE_INT_DEFAULT,
) -> None:
    """
    Set the *Colour* default :class:`numpy.integer` precision by setting
    :attr:`colour.constant.DTYPE_INT_DEFAULT` attribute with the specified
    :class:`numpy.dtype` wherever the attribute is imported.

    Parameters
    ----------
    dtype
        :class:`numpy.dtype` to set
        :attr:`colour.constant.DTYPE_INT_DEFAULT` with.

    Notes
    -----
    -   It is possible to define the integer precision at import time by
        setting the *COLOUR_SCIENCE__DEFAULT_INT_DTYPE* environment
        variable, for example `set COLOUR_SCIENCE__DEFAULT_INT_DTYPE=int32`.

    Warnings
    --------
    This definition is mostly given for consistency purposes with
    :func:`colour.utilities.set_default_float_dtype` definition but contrary
    to the latter, changing *integer* precision will almost certainly
    completely break *Colour*. With great power comes great responsibility.

    Examples
    --------
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    >>> set_default_int_dtype(np.int32)  # doctest: +SKIP
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int32')
    >>> set_default_int_dtype(np.int64)
    >>> as_int_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('int64')
    """

    # TODO: Investigate behaviour on Windows.
    with suppress_warnings(colour_usage_warnings=True):
        for module in sys.modules.values():
            if not hasattr(module, "DTYPE_INT_DEFAULT"):
                continue

            module.DTYPE_INT_DEFAULT = dtype  # pyright: ignore

    CACHE_REGISTRY.clear_all_caches()


def set_default_float_dtype(
    dtype: Type[DTypeFloat] = DTYPE_FLOAT_DEFAULT,
) -> None:
    """
    Set the *Colour* default :class:`numpy.floating` precision by setting
    :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute with the
    specified :class:`numpy.dtype` wherever the attribute is imported.

    Parameters
    ----------
    dtype
        :class:`numpy.dtype` to set
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` with.

    Notes
    -----
    -   It is possible to define the *float* precision at import time by
        setting the *COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE* environment
        variable, for example
        `set COLOUR_SCIENCE__DEFAULT_FLOAT_DTYPE=float32`.
    -   Some definition returning a single-scalar ndarray might not
        honour the specified *float* precision:
        https://github.com/numpy/numpy/issues/16353

    Warnings
    --------
    Changing *float* precision might result in various *Colour*
    functionality breaking entirely:
    https://github.com/numpy/numpy/issues/6860. With great power comes
    great responsibility.

    Examples
    --------
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    >>> set_default_float_dtype(np.float16)  # doctest: +SKIP
    >>> as_float_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('float16')
    >>> set_default_float_dtype(np.float64)
    >>> as_float_array(np.ones(3)).dtype
    dtype('float64')
    """

    with suppress_warnings(colour_usage_warnings=True):
        for module in sys.modules.values():
            if not hasattr(module, "DTYPE_FLOAT_DEFAULT"):
                continue

            module.DTYPE_FLOAT_DEFAULT = dtype  # pyright: ignore

    CACHE_REGISTRY.clear_all_caches()


def set_default_complex_dtype(
    dtype: Type[DTypeComplex] = DTYPE_COMPLEX_DEFAULT,
) -> None:
    """
    Set the *Colour* default :class:`numpy.complexfloating` precision by
    setting :attr:`colour.constant.DTYPE_COMPLEX_DEFAULT` attribute with the
    specified :class:`numpy.dtype` wherever the attribute is imported.

    Parameters
    ----------
    dtype
        :class:`numpy.dtype` to set
        :attr:`colour.constant.DTYPE_COMPLEX_DEFAULT` with.

    Notes
    -----
    -   It is possible to define the *complex* precision at import time by
        setting the *COLOUR_SCIENCE__DEFAULT_COMPLEX_DTYPE* environment
        variable, for example
        `set COLOUR_SCIENCE__DEFAULT_COMPLEX_DTYPE=complex64`.

    Warnings
    --------
    Changing *complex* precision might result in various *Colour*
    functionality breaking entirely. With great power comes great
    responsibility.

    Examples
    --------
    >>> as_complex_array(np.ones(3)).dtype
    dtype('complex128')
    >>> set_default_complex_dtype(np.complex64)  # doctest: +SKIP
    >>> as_complex_array(np.ones(3)).dtype  # doctest: +SKIP
    dtype('complex64')
    >>> set_default_complex_dtype(np.complex128)
    >>> as_complex_array(np.ones(3)).dtype
    dtype('complex128')
    """

    with suppress_warnings(colour_usage_warnings=True):
        for module in sys.modules.values():
            if not hasattr(module, "DTYPE_COMPLEX_DEFAULT"):
                continue

            module.DTYPE_COMPLEX_DEFAULT = dtype  # pyright: ignore

    CACHE_REGISTRY.clear_all_caches()


_DOMAIN_RANGE_SCALE: contextvars.ContextVar[
    Literal["ignore", "reference", "1", "100"] | str
] = contextvars.ContextVar("_DOMAIN_RANGE_SCALE", default="reference")
"""
:class:`contextvars.ContextVar` storing the current *Colour* domain-range
scale. The :class:`contextvars.ContextVar` keeps nested
:class:`domain_range_scale` contexts independent across concurrent
threads and async tasks. Read it via :func:`get_domain_range_scale` and
toggle it via :func:`set_domain_range_scale` or
:class:`domain_range_scale`.
"""


def get_domain_range_scale() -> Literal["ignore", "reference", "1", "100"] | str:
    """
    Return the current *Colour* domain-range scale.

    The following scales are available:

    -   **'Reference'**, the default *Colour* domain-range scale which
        varies depending on the referenced algorithm, e.g., [0, 1],
        [0, 10], [0, 100], [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is
        important to acknowledge that this is a soft normalisation
        and it is possible to use negative out of gamut values or
        high dynamic range data exceeding 1.

    Returns
    -------
    :class:`str`
        *Colour* domain-range scale.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for
        internal usage only!
    """

    return _DOMAIN_RANGE_SCALE.get()


def set_domain_range_scale(
    scale: (
        Literal["ignore", "reference", "Ignore", "Reference", "1", "100"] | str
    ) = "reference",
) -> None:
    """
    Set the current *Colour* domain-range scale.

    The following scales are available:

    -   **'Reference'**, the default *Colour* domain-range scale which
        varies depending on the referenced algorithm, e.g., [0, 1],
        [0, 10], [0, 100], [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is
        important to acknowledge that this is a soft normalisation and it
        is possible to use negative out of gamut values or high dynamic
        range data exceeding 1.

    Parameters
    ----------
    scale
        *Colour* domain-range scale to set.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for
        internal usage only!
    """

    _DOMAIN_RANGE_SCALE.set(
        validate_method(
            str(scale),
            ("ignore", "reference", "1", "100"),
            '"{0}" scale is invalid, it must be one of {1}!',
        )
    )


class domain_range_scale:
    """
    Define a context manager and decorator to temporarily set the *Colour*
    domain-range scale.

    The following scales are available:

    -   **'Reference'**, the default *Colour* domain-range scale which
        varies depending on the referenced algorithm, e.g., [0, 1],
        [0, 10], [0, 100], [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is
        important to acknowledge that this is a soft normalisation and it
        is possible to use negative out of gamut values or high dynamic
        range data exceeding 1.

    Parameters
    ----------
    scale
        *Colour* domain-range scale to set.

    Warnings
    --------
    -   The **'Ignore'** and **'100'** domain-range scales are for
        internal usage only!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_1(1)
    array(1.)
    >>> with domain_range_scale("Reference"):
    ...     from_range_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_1(1)
    array(1.)
    >>> with domain_range_scale("1"):
    ...     from_range_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_1(1)
    array(0.01)
    >>> with domain_range_scale("100"):
    ...     from_range_1(1)
    array(100.)
    """

    def __init__(
        self,
        scale: (
            Literal["ignore", "reference", "Ignore", "Reference", "1", "100"] | str
        ),
    ) -> None:
        self._scale = scale
        # Token stack: nested or recursive ``__enter__`` / ``__exit__``
        # pairs against the same instance (e.g. via the decorator form on
        # a recursive function) push and pop independent reset tokens.
        self._tokens: list[
            contextvars.Token[Literal["ignore", "reference", "1", "100"] | str]
        ] = []

    def __enter__(self) -> Self:
        """Set the new domain-range scale upon entering the context manager."""

        self._tokens.append(
            _DOMAIN_RANGE_SCALE.set(
                validate_method(
                    str(self._scale),
                    ("ignore", "reference", "1", "100"),
                    '"{0}" scale is invalid, it must be one of {1}!',
                )
            )
        )

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Restore the previous domain-range scale upon exiting the context
        manager.
        """

        _DOMAIN_RANGE_SCALE.reset(self._tokens.pop())

    def __call__(self, function: Callable) -> Any:
        """
        Call the wrapped definition with domain-range scale management.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # A fresh instance is entered per call so the token stack is never
            # shared across threads or async tasks invoking the decorated
            # definition concurrently.
            with self.__class__(self._scale):
                return function(*args, **kwargs)

        return wrapper


_CACHE_DOMAIN_RANGE_SCALE_METADATA: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_DOMAIN_RANGE_SCALE_METADATA"
)


def get_domain_range_scale_metadata(function: Callable) -> dict[str, Any]:
    """
    Extract domain-range scale metadata from function type hints.

    Extracts scale factors from PEP 593 ``Annotated`` type hints on function
    parameters and return values. This metadata indicates which scale factors
    to use when converting between 'Reference' and '1' modes.

    Parameters
    ----------
    function
        Function to extract metadata from.

    Returns
    -------
    :class:`dict`
        Dictionary with keys:

        - ``domain``: Dict mapping parameter names to their scale factors
        - ``range``: Scale factor for return value (int, tuple, or None)

    Examples
    --------
    >>> from colour.hints import Annotated, ArrayLike, NDArrayFloat
    >>> def example_function(
    ...     XYZ: Domain1,
    ...     illuminant: ArrayLike = None,
    ... ) -> Range100:
    ...     pass
    >>> metadata = get_domain_range_scale_metadata(example_function)
    >>> metadata["domain"]
    {'XYZ': 1}
    >>> metadata["range"]
    100
    """

    # Unwrap functools.partial to get the underlying function
    if hasattr(function, "func"):
        function = function.func  # pyright: ignore

    # ``id`` alone is unsafe as a key: *CPython* reuses the address of a
    # garbage-collected object, so a short-lived definition could return the
    # metadata of an unrelated one. The qualified name and module are folded in
    # to disambiguate reused addresses.
    cache_key = (
        id(function),
        getattr(function, "__qualname__", None),
        getattr(function, "__module__", None),
    )

    if is_caching_enabled() and cache_key in _CACHE_DOMAIN_RANGE_SCALE_METADATA:
        # A deep copy is returned so that a caller mutating the nested
        # ``domain`` mapping cannot poison the cached metadata.
        return copy.deepcopy(_CACHE_DOMAIN_RANGE_SCALE_METADATA[cache_key])

    metadata: dict[str, Any] = {"domain": {}, "range": None}

    def extract_scale_from_hint(hint: Any) -> Any | None:
        """
        Extract scale metadata from a type hint, handling Union types.

        Parameters
        ----------
        hint
            Type hint to extract scale from.

        Returns
        -------
        :class:`int` | :class:`tuple` | :class:`None`
            Scale metadata if found, None otherwise.
        """

        # Direct Annotated type with __metadata__
        if hasattr(hint, "__metadata__") and hint.__metadata__:
            return next(iter(hint.__metadata__))

        # Union type: check if any arg is Annotated
        origin = get_origin(hint)
        if origin is Union:
            for arg in get_args(hint):
                if hasattr(arg, "__metadata__") and arg.__metadata__:
                    return next(iter(arg.__metadata__))

        return None

    try:
        hints = get_type_hints(function, include_extras=True)
        # Process hints from get_type_hints (actual types with __metadata__)
        for parameter_name, hint in hints.items():
            scale = extract_scale_from_hint(hint)
            if scale is not None:
                if parameter_name == "return":
                    metadata["range"] = scale
                else:
                    metadata["domain"][parameter_name] = scale
    except (AttributeError, TypeError, NameError):
        # Fallback: parse string annotations (when `from __future__ import annotations`)
        # Mapping of type alias names to their scale values
        type_alias_scales = {
            "Domain1": 1,
            "Domain10": 10,
            "Domain100": 100,
            "Domain360": 360,
            "Domain100_100_360": (100, 100, 360),
            "Range1": 1,
            "Range10": 10,
            "Range100": 100,
            "Range360": 360,
            "Range100_100_360": (100, 100, 360),
        }

        hints = getattr(function, "__annotations__", {})
        for parameter_name, hint in hints.items():
            scale = None

            # Check if hint is a type alias name
            if isinstance(hint, str) and hint in type_alias_scales:
                scale = type_alias_scales[hint]
            # Extract scale from string: "Annotated[Type, scale]" -> scale
            elif (
                isinstance(hint, str)
                and "Annotated[" in hint
                and (match := re.search(r"Annotated\[[^,]+,\s*([^\]]+)\]", hint))
            ):
                scale_string = match.group(1).strip()
                # Evaluate scale (could be int, tuple, etc.)
                try:
                    scale = eval(scale_string)  # noqa: S307
                except (SyntaxError, NameError, ValueError):
                    scale = scale_string

            if scale is not None:
                if parameter_name == "return":
                    metadata["range"] = scale
                else:
                    metadata["domain"][parameter_name] = scale

    if is_caching_enabled():
        _CACHE_DOMAIN_RANGE_SCALE_METADATA[cache_key] = copy.deepcopy(metadata)

    return metadata


def _scale_at(
    a: ArrayLike,
    target_scale: str,
    scale_factor: ArrayLike,
    dtype: Type[DTypeFloat] | None,
    *,
    divide: bool = False,
) -> NDArray:
    """
    Apply ``a * scale_factor`` or ``a / scale_factor`` when the current
    domain-range scale matches ``target_scale``.
    """

    if target_scale != _DOMAIN_RANGE_SCALE.get():
        return a  # pyright: ignore

    if not is_array_api_enabled():
        a = np.asarray(a, dtype=dtype)
        # ``np.asarray`` re-wraps the result so that ``NumPy >= 2`` does not
        # collapse a 0-d array divided / multiplied by a Python scalar to
        # a :class:`numpy.float*` instance, which would break downstream
        # callers expecting an :class:`numpy.ndarray`.
        return np.asarray(a / scale_factor if divide else a * scale_factor)

    xp = array_namespace(a)

    factor = xp_as_array(scale_factor, dtype=dtype, xp=xp, like=a)

    return xp_as_array(a / factor if divide else a * factor, xp=xp)


def to_domain_1(
    a: ArrayLike,
    scale_factor: ArrayLike = 100,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to domain **'1'**.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'1'**, the
        definition is almost entirely by-passed and will conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is divided
        by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'1'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes need different scaling to be brought to domain **'1'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'1'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_1(1)
    array(0.01)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    return _scale_at(a, "100", scale_factor, dtype, divide=True)


def to_domain_10(
    a: ArrayLike,
    scale_factor: ArrayLike = 10,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to domain **'10'**, used by the
    *Munsell Renotation System*.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is almost entirely by-passed and will conveniently convert array
        :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 10.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        divided by ``scale_factor``, typically 10.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'10'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes need different scaling to be brought to domain
        **'10'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'10'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_10(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_10(1)
    array(10.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_10(1)
    array(0.1)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    a = _scale_at(a, "1", scale_factor, dtype)

    return _scale_at(a, "100", scale_factor, dtype, divide=True)


def to_domain_100(
    a: ArrayLike,
    scale_factor: ArrayLike = 100,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to domain **'100'**.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'100'**
        (currently unsupported private value only used for unit tests), the
        definition is almost entirely by-passed and will conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to domain **'100'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes need different scaling to be brought to domain
        **'100'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to domain **'100'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_100(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_100(1)
    array(100.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_100(1)
    array(1.)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    return _scale_at(a, "1", scale_factor, dtype)


def to_domain_degrees(
    a: ArrayLike,
    scale_factor: ArrayLike = 360,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to degrees domain.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is almost entirely by-passed and will conveniently convert array
        :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 360.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor`` / 100, typically 360 / 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale to degrees domain.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes need different scaling to be brought to degrees domain.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to degrees domain.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_degrees(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_degrees(1)
    array(360.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_degrees(1)
    array(3.6)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    a = _scale_at(a, "1", scale_factor, dtype)

    return _scale_at(a, "100", scale_factor / 100, dtype)  # pyright: ignore


def to_domain_int(
    a: ArrayLike,
    bit_depth: ArrayLike = 8,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` to integer domain.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is almost entirely by-passed and will conveniently convert array
        :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by :math:`2^{bit\\_depth} - 1`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by :math:`2^{bit\\_depth} - 1`.

    Parameters
    ----------
    a
        Array :math:`a` to scale to integer domain.
    bit_depth
        Bit-depth, usually *int* but can be a :class:`numpy.ndarray` if
        some axis need different scaling to be brought to integer domain.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled to integer domain.

    Notes
    -----
    -   To avoid precision issues and rounding, the scaling is performed
        on *float* numbers.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     to_domain_int(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     to_domain_int(1)
    array(255.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     to_domain_int(1)
    array(2.55)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = ndarray_copy(as_float_array(a, dtype))

    maximum_code_value = 2**bit_depth - 1  # pyright: ignore

    a = _scale_at(a, "1", maximum_code_value, dtype)

    return _scale_at(a, "100", maximum_code_value / 100, dtype)


def from_range_1(
    a: ArrayLike,
    scale_factor: ArrayLike = 100,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from range **'1'**.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'1'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'1'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axis need different scaling to be brought from range
        **'1'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'1'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_1(1)
    array(1.)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_1(1)
    array(100.)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    return _scale_at(a, "100", scale_factor, dtype)


def from_range_10(
    a: ArrayLike,
    scale_factor: ArrayLike = 10,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from range **'10'**, used by the
    *Munsell Renotation System*.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 10.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor``, typically 10.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'10'**.
    scale_factor
        Scale factor, usually *numeric* but can be a
        :class:`numpy.ndarray` if some axis need different scaling to be
        brought from range **'10'**.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'10'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_10(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_10(1)
    array(0.1)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_10(1)
    array(10.)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    a = _scale_at(a, "1", scale_factor, dtype, divide=True)

    return _scale_at(a, "100", scale_factor, dtype)


def from_range_100(
    a: ArrayLike,
    scale_factor: ArrayLike = 100,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from range **'100'**.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'100'**
        (currently unsupported private value only used for unit tests), the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from range **'100'**.
    scale_factor
        Scale factor, usually *numeric* but can be a :class:`numpy.ndarray`
        if some axes require different scaling to be brought from range
        **'100'**.
    dtype
        Data type used for the conversion to :class:`numpy.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from range **'100'**.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_100(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_100(1)
    array(0.01)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_100(1)
    array(1.)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    return _scale_at(a, "1", scale_factor, dtype, divide=True)


def from_range_degrees(
    a: ArrayLike,
    scale_factor: ArrayLike = 360,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from degrees range.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 360.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        divided by ``scale_factor`` / 100, typically 360 / 100.

    Parameters
    ----------
    a
        Array :math:`a` to scale from degrees range.
    scale_factor
        Scale factor, usually *numeric* but can be a
        :class:`numpy.ndarray` if some axes need different scaling to be
        brought from degrees range.
    dtype
        Data type used for the conversion to :class:`numpy.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from degrees range.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_degrees(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_degrees(1)  # doctest: +ELLIPSIS
    array(0.0027777...)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_degrees(1)  # doctest: +ELLIPSIS
    array(0.2777777...)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    a = _scale_at(a, "1", scale_factor, dtype, divide=True)

    return _scale_at(a, "100", scale_factor / 100, dtype, divide=True)  # pyright: ignore


def from_range_int(
    a: ArrayLike,
    bit_depth: ArrayLike = 8,
    dtype: Type[DTypeFloat] | None = None,
) -> NDArray:
    """
    Scale the specified array :math:`a` from integer range.

    The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the definition
        is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        converted to :class:`np.ndarray` and divided by
        :math:`2^{bit\\_depth} - 1`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        converted to :class:`np.ndarray` and divided by
        :math:`2^{bit\\_depth} - 1`.

    Parameters
    ----------
    a
        Array :math:`a` to scale from integer range.
    bit_depth
        Bit-depth, usually *int* but can be a :class:`numpy.ndarray` if
        some axes need different scaling to be brought from integer range.
    dtype
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` scaled from integer range.

    Warnings
    --------
    The scale conversion of variable :math:`a` happens in-place, i.e.,
    :math:`a` will be mutated!

    Notes
    -----
    -   To avoid precision issues and rounding, the scaling is performed on
        *float* numbers.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale("Reference"):
    ...     from_range_int(1)
    array(1.)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale("1"):
    ...     from_range_int(1)  # doctest: +ELLIPSIS
    array(0.0039215...)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale("100"):
    ...     from_range_int(1)  # doctest: +ELLIPSIS
    array(0.3921568...)
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_float_array(a, dtype)

    maximum_code_value = 2**bit_depth - 1  # pyright: ignore

    a = _scale_at(a, "1", maximum_code_value, dtype, divide=True)

    return _scale_at(a, "100", maximum_code_value / 100, dtype, divide=True)


_NDARRAY_COPY_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_NDARRAY_COPY_ENABLED", default=True
)
"""
:class:`contextvars.ContextVar` storing the current *Colour* state for
:class:`numpy.ndarray` copy. The :class:`contextvars.ContextVar` keeps
nested :class:`ndarray_copy_enable` contexts independent across
concurrent threads and async tasks. Read it via
:func:`is_ndarray_copy_enabled` and toggle it via
:func:`set_ndarray_copy_enabled` or :class:`ndarray_copy_enable`.
"""


def is_ndarray_copy_enabled() -> bool:
    """
    Determine whether *Colour* :class:`numpy.ndarray` copy is enabled.

    Various API objects return a copy of their internal
    :class:`numpy.ndarray` for safety purposes, but this can be a slow
    operation impacting performance.

    Returns
    -------
    :class:`bool`
        Whether *Colour* :class:`numpy.ndarray` copy is enabled.

    Examples
    --------
    >>> with ndarray_copy_enable(False):
    ...     is_ndarray_copy_enabled()
    False
    >>> with ndarray_copy_enable(True):
    ...     is_ndarray_copy_enabled()
    True
    """

    return _NDARRAY_COPY_ENABLED.get()


def set_ndarray_copy_enabled(enable: bool) -> None:
    """
    Set the *Colour* :class:`numpy.ndarray` copy enabled state.

    Parameters
    ----------
    enable
        Whether to enable *Colour* :class:`numpy.ndarray` copy.

    Examples
    --------
    >>> with ndarray_copy_enable(is_ndarray_copy_enabled()):
    ...     print(is_ndarray_copy_enabled())
    ...     set_ndarray_copy_enabled(False)
    ...     print(is_ndarray_copy_enabled())
    True
    False
    """

    _NDARRAY_COPY_ENABLED.set(enable)


class ndarray_copy_enable:
    """
    Define a context manager and decorator to temporarily set the *Colour*
    :class:`numpy.ndarray` copy enabled state.

    Parameters
    ----------
    enable
        Whether to enable or disable *Colour* :class:`numpy.ndarray` copy.
    """

    def __init__(self, enable: bool) -> None:
        self._enable = enable
        # Token stack: nested or recursive ``__enter__`` / ``__exit__``
        # pairs against the same instance (e.g. via the decorator form on
        # a recursive function) push and pop independent reset tokens.
        self._tokens: list[contextvars.Token[bool]] = []

    def __enter__(self) -> Self:
        """
        Set the *Colour* :class:`numpy.ndarray` copy enabled state upon
        entering the context manager.
        """

        self._tokens.append(_NDARRAY_COPY_ENABLED.set(self._enable))

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Restore the *Colour* :class:`numpy.ndarray` copy enabled state upon
        exiting the context manager.
        """

        _NDARRAY_COPY_ENABLED.reset(self._tokens.pop())

    def __call__(self, function: Callable) -> Callable:
        """
        Decorate and call the specified function with array copy control.

        Parameters
        ----------
        function
            Function to be decorated with array copy state management.

        Returns
        -------
        :class:`Callable`
            Decorated function that executes within the configured array copy
            state context.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # A fresh instance is entered per call so the token stack is never
            # shared across threads or async tasks invoking the decorated
            # definition concurrently.
            with self.__class__(self._enable):
                return function(*args, **kwargs)

        return wrapper


def ndarray_copy(a: NDArray) -> NDArray:
    """
    Return a :class:`numpy.ndarray` copy if the relevant *Colour* state is
    enabled.

    Various API objects return a copy of their internal
    :class:`numpy.ndarray` for safety purposes, but this can be a slow
    operation impacting performance.

    Parameters
    ----------
    a
        Array :math:`a` to return a copy of.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` copy according to *Colour* state.

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> id(a) == id(ndarray_copy(a))
    False
    >>> with ndarray_copy_enable(False):
    ...     id(a) == id(ndarray_copy(a))
    True
    """

    if _NDARRAY_COPY_ENABLED.get():
        xp = array_namespace(a)

        if is_numpy_namespace(xp):
            return np.copy(a)
        return _copy_array(a, xp)
    return a


def closest_indexes(a: ArrayLike, b: ArrayLike) -> NDArray:
    """
    Return the closest element indexes from array :math:`a` to reference array
    :math:`b` elements.

    Parameters
    ----------
    a
        Array :math:`a` to search for the closest elements.
    b
        Reference array :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        Closest array :math:`a` element indexes.

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         24.31357115,
    ...         63.62396289,
    ...         55.71528816,
    ...         62.70988028,
    ...         46.84480573,
    ...         25.40026416,
    ...     ]
    ... )
    >>> print(closest_indexes(a, 63))
    [3]
    >>> print(closest_indexes(a, [63, 25]))
    [3 5]
    """

    xp = array_namespace(a, b)

    a = xp_as_float_array(a, xp=xp, like=b)
    b = xp_as_float_array(b, xp=xp, like=a)

    a = xp_reshape(a, (-1,), xp=xp)[:, None]
    b = xp_reshape(b, (-1,), xp=xp)[None, :]

    return xp.abs(a - b).argmin(axis=0)


def closest(a: ArrayLike, b: ArrayLike) -> NDArray:
    """
    Return the closest array :math:`a` elements to reference array
    :math:`b` elements.

    Parameters
    ----------
    a
        Array :math:`a` to search for the closest elements.
    b
        Reference array :math:`b`.

    Returns
    -------
    :class:`numpy.ndarray`
        Closest array :math:`a` elements.

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         24.31357115,
    ...         63.62396289,
    ...         55.71528816,
    ...         62.70988028,
    ...         46.84480573,
    ...         25.40026416,
    ...     ]
    ... )
    >>> closest(a, 63)
    array([62.70988028])
    >>> closest(a, [63, 25])
    array([62.70988028, 25.40026416])
    """

    b = as_float_array(b)

    xp = array_namespace(a, b)

    # The dtype is preserved rather than forced to float, matching the
    # historical ``numpy.array(a)`` conversion: ``closest`` returns elements
    # OF ``a`` and must not silently upcast an integer or float32 table.
    a = xp_as_array(a, xp=xp, like=b)

    return a[closest_indexes(a, b)]


_CACHE_DISTRIBUTION_INTERVAL: dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_DISTRIBUTION_INTERVAL"
)


def interval(distribution: ArrayLike, unique: bool = True) -> NDArray:
    """
    Return the interval size of the specified distribution.

    Parameters
    ----------
    distribution
        Distribution to retrieve the interval from.
    unique
        Whether to return unique intervals if the distribution is
        non-uniformly spaced or the complete intervals.

    Returns
    -------
    :class:`numpy.ndarray`
        Distribution interval.

    Examples
    --------
    Uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 5])
    >>> interval(y)
    array([1.])
    >>> interval(y, False)
    array([1., 1., 1., 1.])

    Non-uniformly spaced variable:

    >>> y = np.array([1, 2, 3, 4, 8])
    >>> interval(y)
    array([1., 4.])
    >>> interval(y, False)
    array([1., 1., 1., 4.])

    Distribution with a single element or empty distribution, i.e. without any
    defined interval:

    >>> interval(np.array([1]))
    array([], dtype=float64)
    >>> interval(np.array([]))
    array([], dtype=float64)
    """

    distribution = as_float_array(distribution)

    xp = array_namespace(distribution)

    hash_key = hash(
        (
            int_digest(as_ndarray(distribution).tobytes()),
            distribution.shape,
            unique,
        )
    )

    if is_caching_enabled() and hash_key in _CACHE_DISTRIBUTION_INTERVAL:
        return xp_as_array(
            _CACHE_DISTRIBUTION_INTERVAL[hash_key],
            xp=xp,
            like=distribution,
            copy=True,
        )

    differences = xp.abs(distribution[1:] - distribution[:-1])

    if differences.shape[0] == 0:
        interval_ = differences
    elif unique and xp.all(differences == differences[0]):
        interval_ = differences[0:1]
    elif unique:
        interval_ = xp.unique_values(differences)
    else:
        interval_ = differences

    if is_caching_enabled():
        _CACHE_DISTRIBUTION_INTERVAL[hash_key] = xp_as_array(
            interval_, xp=xp, copy=True
        )

    return interval_


def is_uniform(distribution: ArrayLike) -> bool:
    """
    Determine whether the specified distribution is uniform.

    Parameters
    ----------
    distribution
        Distribution to check for uniformity.

    Returns
    -------
    :class:`bool`
        Whether the distribution is uniform.

    Examples
    --------
    Uniformly spaced variable:

    >>> a = np.array([1, 2, 3, 4, 5])
    >>> is_uniform(a)
    True

    Non-uniformly spaced variable:

    >>> a = np.array([1, 2, 3.1415, 4, 5])
    >>> is_uniform(a)
    False
    """

    return len(interval(distribution)) == 1


def in_array(a: ArrayLike, b: ArrayLike, tolerance: Real = EPSILON) -> NDArray:
    """
    Determine whether each element of array :math:`a` is present in array
    :math:`b` within the specified tolerance.

    Parameters
    ----------
    a
        Array :math:`a` to test the elements from.
    b
        Array :math:`b` against which to test the elements of array
        :math:`a`.
    tolerance
        Tolerance value.

    Returns
    -------
    :class:`numpy.ndarray`
        Boolean array with array :math:`a` shape indicating whether each
        element of array :math:`a` is present in array :math:`b` within the
        specified tolerance.

    References
    ----------
    :cite:`Yorke2014a`

    Examples
    --------
    >>> a = np.array([0.50, 0.60])
    >>> b = np.linspace(0, 10, 101)
    >>> np.isin(a, b)
    array([ True, False])
    >>> in_array(a, b)
    array([ True,  True])
    """

    a = as_float_array(a)
    b = as_float_array(b)

    xp = array_namespace(a, b)

    d = xp.abs(xp_reshape(a, (-1,), xp=xp) - b[..., None])

    return xp_reshape(xp.any(d <= tolerance, axis=0), a.shape, xp=xp)


def tstack(
    a: ArrayLike,
    dtype: Type[DTypeBoolean] | Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Stack the specified array of arrays :math:`a` along the last axis (tail)
    to produce a stacked array.

    Used to stack an array of arrays produced by the
    :func:`colour.utilities.tsplit` definition.

    Parameters
    ----------
    a
        Array of arrays :math:`a` to stack along the last axis.
    dtype
        :class:`numpy.dtype` to use for initial conversion to
        :class:`numpy.ndarray`, default to the :class:`numpy.dtype` defined
        by :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Stacked array.

    Notes
    -----
    -   The returned array is always a freshly-allocated, contiguous stack of
        the components along the last axis and never aliases the inputs. It is
        the inverse of the :func:`colour.utilities.tsplit` definition, whose
        *NumPy* path likewise returns an independent, contiguous copy.

    Examples
    --------
    >>> a = 0
    >>> tstack([a, a, a])
    array([0., 0., 0.])
    >>> a = np.arange(0, 6)
    >>> tstack([a, a, a])
    array([[0., 0., 0.],
           [1., 1., 1.],
           [2., 2., 2.],
           [3., 3., 3.],
           [4., 4., 4.],
           [5., 5., 5.]])
    >>> a = np.reshape(a, (1, 6))
    >>> tstack([a, a, a])
    array([[[0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
            [4., 4., 4.],
            [5., 5., 5.]]])
    >>> a = np.reshape(a, (1, 1, 6))
    >>> tstack([a, a, a])
    array([[[[0., 0., 0.],
             [1., 1., 1.],
             [2., 2., 2.],
             [3., 3., 3.],
             [4., 4., 4.],
             [5., 5., 5.]]]])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    if (
        is_array_api_enabled()
        and isinstance(a, (list, tuple))
        and a
        and is_non_ndarray(a[0])
    ):
        xp = array_namespace(a[0])

        return xp.stack([xp_as_array(x, xp=xp, like=a[0]) for x in a], axis=-1)

    if isinstance(a, (list, tuple)) and a:
        # Stack the components directly, avoiding the ``as_array(list)``
        # round-trip that materialises an intermediate ``(n, ...)`` array only
        # to re-split and re-stack it on the tail axis.
        components = [as_array(x, dtype) for x in a]

        xp = array_namespace(components[0])

        return xp.stack(components, axis=-1)

    a = as_array(a, dtype)

    xp = array_namespace(a)

    return xp.stack(list(a), axis=-1)


def tsplit(
    a: ArrayLike,
    dtype: Type[DTypeBoolean] | Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Split the specified stacked array :math:`a` along the last axis (tail)
    to produce an array of arrays.

    Used to split a stacked array produced by the :func:`colour.utilities.tstack`
    definition.

    Parameters
    ----------
    a
        Stacked array :math:`a` to split.
    dtype
        :class:`numpy.dtype` to use for initial conversion to
        :class:`numpy.ndarray`, default to the :class:`numpy.dtype` defined
        by :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of arrays.

    Notes
    -----
    -   On the *NumPy* path, the returned array is an **independent,
        contiguous** copy: the leading-axis sub-arrays do not alias the input
        :math:`a` and can be safely written to in-place. This copy is
        deliberate and not an avoidable overhead, splitting with a
        :func:`numpy.moveaxis` view instead would alias :math:`a` (breaking
        that guarantee), and the contiguity also keeps downstream operations
        such as matrix multiplications fast.
    -   On the *Array API* path, the split is a zero-copy
        :func:`numpy.moveaxis` view whose contiguity and aliasing semantics
        are managed by the backend.

    Examples
    --------
    >>> a = np.array([0, 0, 0])
    >>> tsplit(a)
    array([0., 0., 0.])
    >>> a = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    >>> tsplit(a)
    array([[0., 1., 2., 3., 4., 5.],
           [0., 1., 2., 3., 4., 5.],
           [0., 1., 2., 3., 4., 5.]])
    >>> a = np.array(
    ...     [
    ...         [
    ...             [0, 0, 0],
    ...             [1, 1, 1],
    ...             [2, 2, 2],
    ...             [3, 3, 3],
    ...             [4, 4, 4],
    ...             [5, 5, 5],
    ...         ]
    ...     ]
    ... )
    >>> tsplit(a)
    array([[[0., 1., 2., 3., 4., 5.]],
    <BLANKLINE>
           [[0., 1., 2., 3., 4., 5.]],
    <BLANKLINE>
           [[0., 1., 2., 3., 4., 5.]]])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    a = as_array(a, dtype)

    xp = array_namespace(a)

    if a.shape[-1] == 0:
        # A zero-length last axis yields no components to stack; the empty
        # result is built directly as ``xp.stack`` rejects an empty sequence.
        # The 1-D shape matches the historical ``numpy.array([])`` behaviour.
        return xp_reshape(a, (0,), xp=xp)

    if is_numpy_namespace(xp):
        return xp.stack([a[..., x] for x in range(a.shape[-1])])

    return xp.moveaxis(a, -1, 0)


def row_as_diagonal(a: ArrayLike) -> NDArray:
    """
    Return the rows of the specified array :math:`a` as diagonal matrices.

    Parameters
    ----------
    a
        Array :math:`a` to return the rows of as diagonal matrices.

    Returns
    -------
    :class:`numpy.ndarray`
        Array :math:`a` rows as diagonal matrices.

    References
    ----------
    :cite:`Castro2014a`

    Examples
    --------
    >>> a = np.array(
    ...     [
    ...         [0.25891593, 0.07299478, 0.36586996],
    ...         [0.30851087, 0.37131459, 0.16274825],
    ...         [0.71061831, 0.67718718, 0.09562581],
    ...         [0.71588836, 0.76772047, 0.15476079],
    ...         [0.92985142, 0.22263399, 0.88027331],
    ...     ]
    ... )
    >>> row_as_diagonal(a)
    array([[[0.25891593, 0.        , 0.        ],
            [0.        , 0.07299478, 0.        ],
            [0.        , 0.        , 0.36586996]],
    <BLANKLINE>
           [[0.30851087, 0.        , 0.        ],
            [0.        , 0.37131459, 0.        ],
            [0.        , 0.        , 0.16274825]],
    <BLANKLINE>
           [[0.71061831, 0.        , 0.        ],
            [0.        , 0.67718718, 0.        ],
            [0.        , 0.        , 0.09562581]],
    <BLANKLINE>
           [[0.71588836, 0.        , 0.        ],
            [0.        , 0.76772047, 0.        ],
            [0.        , 0.        , 0.15476079]],
    <BLANKLINE>
           [[0.92985142, 0.        , 0.        ],
            [0.        , 0.22263399, 0.        ],
            [0.        , 0.        , 0.88027331]]])
    """

    d = as_array(a)

    xp = array_namespace(d)

    d = xp.expand_dims(d, axis=-2)

    eye = xp.eye(d.shape[-1], device=getattr(d, "device", None))

    return eye * d


def orient(
    a: ArrayLike,
    orientation: (
        Literal["Ignore", "Flip", "Flop", "90 CW", "90 CCW", "180"] | str
    ) = "Ignore",
) -> NDArray:
    """
    Orient the specified array :math:`a` using the specified orientation.

    Parameters
    ----------
    a
        Array :math:`a` to orient.
    orientation
        Orientation to perform.

    Returns
    -------
    :class:`numpy.ndarray`
        Oriented array.

    Examples
    --------
    >>> a = np.tile(np.arange(5), (5, 1))
    >>> a
    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])
    >>> orient(a, "90 CW")
    array([[0., 0., 0., 0., 0.],
           [1., 1., 1., 1., 1.],
           [2., 2., 2., 2., 2.],
           [3., 3., 3., 3., 3.],
           [4., 4., 4., 4., 4.]])
    >>> orient(a, "Flip")
    array([[4., 3., 2., 1., 0.],
           [4., 3., 2., 1., 0.],
           [4., 3., 2., 1., 0.],
           [4., 3., 2., 1., 0.],
           [4., 3., 2., 1., 0.]])
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    orientation = validate_method(
        orientation, ("Ignore", "Flip", "Flop", "90 CW", "90 CCW", "180")
    )

    oriented = a
    if orientation == "ignore":
        oriented = a
    elif orientation == "flip":
        oriented = xp.flip(a, axis=1)
    elif orientation == "flop":
        oriented = xp.flip(a, axis=0)
    elif orientation == "90 cw":
        oriented = xp_matrix_transpose(xp.flip(a, axis=0), xp=xp)
    elif orientation == "90 ccw":
        oriented = xp_matrix_transpose(xp.flip(a, axis=1), xp=xp)
    elif orientation == "180":
        oriented = xp.flip(xp.flip(a, axis=0), axis=1)

    return oriented


def centroid(a: ArrayLike) -> NDArrayInt:
    """
    Return the centroid indexes of the specified array :math:`a`.

    Parameters
    ----------
    a
        Array :math:`a` to return the centroid indexes of.

    Returns
    -------
    :class:`numpy.ndarray`
        Centroid indexes of array :math:`a`.

    Examples
    --------
    >>> a = np.tile(np.arange(0, 5), (5, 1))
    >>> centroid(a)  # doctest: +ELLIPSIS
    array([2, 3]...)
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    a_s = xp.sum(a)

    device = getattr(a, "device", None)
    ranges = [xp.arange(0, a.shape[i], device=device) for i in range(a.ndim)]
    coordinates = xp.meshgrid(*ranges)

    a_ci = []
    for axis in coordinates:
        axis = xp.permute_dims(axis, tuple(reversed(range(axis.ndim))))  # noqa: PLW2901
        # Aligning axis for N-D arrays where N is normalised to
        # range [3, :math:`\\\infty`]
        for i in range(axis.ndim - 2, 0, -1):
            axis = xp.moveaxis(axis, i - 1, -1)  # noqa: PLW2901

        a_ci.append(xp.sum(axis * a) // a_s)

    # NOTE: Cannot use ``as_int_array`` as presence of NaN will raise a
    # ``ValueError`` exception.
    return xp_astype(xp.stack(a_ci), DTYPE_INT_DEFAULT, xp=xp)


def fill_nan(
    a: ArrayLike,
    method: Literal["Interpolation", "Constant"] | str = "Interpolation",
    default: Real = 0,
) -> NDArray:
    """
    Fill the NaN values in the specified array :math:`a` using the specified
    method.

    Parameters
    ----------
    a
        Array :math:`a` to fill the NaNs of.
    method
        *Interpolation* method linearly interpolates through the NaN values,
        *Constant* method replaces NaN values with ``default``.
    default
        Value to use with the *Constant* method.

    Returns
    -------
    :class:`numpy.ndarray`
        NaN-filled array :math:`a`.

    Examples
    --------
    >>> a = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    >>> fill_nan(a)
    array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> fill_nan(a, method="Constant")
    array([0.1, 0.2, 0. , 0.4, 0.5])
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    a = xp_as_array(a, xp=xp, copy=True)

    method = validate_method(method, ("Interpolation", "Constant"))

    mask = xp.isnan(a)

    if not xp.any(mask):
        return a

    if method == "interpolation":
        indices = xp.arange(len(a), device=getattr(a, "device", None))
        valid = ~mask
        a = xp.where(
            mask,
            xp_interp(indices, indices[valid], a[valid], xp=xp),
            a,
        )
    elif method == "constant":
        a = xp.where(mask, default, a)

    return a


def has_only_nan(a: ArrayLike) -> bool:
    """
    Return whether the specified array :math:`a` contains only *NaN* values.

    Parameters
    ----------
    a
        Array :math:`a` to check whether it contains only *NaN* values.

    Returns
    -------
    :class:`bool`
        Whether array :math:`a` contains only *NaN* values.

    Examples
    --------
    >>> has_only_nan(None)
    True
    >>> has_only_nan([None, None])
    True
    >>> has_only_nan([True, None])
    False
    >>> has_only_nan([0.1, np.nan, 0.3])
    False
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    return bool(xp.all(xp.isnan(a)))


@contextmanager
def ndarray_write(a: ArrayLike) -> Generator:
    """
    Define a context manager that temporarily sets the specified array
    :math:`a` to writeable for operations, then restores it to read-only.

    Parameters
    ----------
    a
        Array :math:`a` to operate on.

    Yields
    ------
    Generator
        Array :math:`a` made temporarily writeable.

    Examples
    --------
    >>> a = np.linspace(0, 1, 10)
    >>> a.setflags(write=False)
    >>> try:
    ...     a += 1
    ... except ValueError:
    ...     pass
    >>> with ndarray_write(a):
    ...     a += 1
    """

    a = as_float_array(a)

    a.setflags(write=True)

    try:
        yield a
    finally:
        a.setflags(write=False)


def zeros(
    shape: int | Sequence[int],
    dtype: Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Create an array of zeros with the active dtype.

    Wrap :func:`np.zeros` definition to create an array with the active
    :class:`numpy.dtype` defined by the
    :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of the specified shape and :class:`numpy.dtype`, filled
        with zeros.

    Examples
    --------
    >>> zeros(3)
    array([0., 0., 0.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    xp = array_namespace()

    return xp.zeros(shape, dtype=dtype)


def ones(
    shape: int | Sequence[int],
    dtype: Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Create an array of ones with the active dtype.

    Wrap :func:`np.ones` definition to create an array with the active
    :class:`numpy.dtype` defined by the
    :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of the specified shape and :class:`numpy.dtype`, filled with ones.

    Examples
    --------
    >>> ones(3)
    array([1., 1., 1.])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    xp = array_namespace()

    return xp.ones(shape, dtype=dtype)


def full(
    shape: int | Sequence[int],
    fill_value: Real,
    dtype: Type[DTypeReal] | None = None,
) -> NDArray:
    """
    Create an array of the specified value with the active dtype.

    Wrap :func:`np.full` definition to create an array with the active
    :class:`numpy.dtype` defined by the
    :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Parameters
    ----------
    shape
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value
        Fill value.
    dtype
        :class:`numpy.dtype` to use for conversion, default to the
        :class:`numpy.dtype` defined by the
        :attr:`colour.constant.DTYPE_FLOAT_DEFAULT` attribute.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of the specified shape and :class:`numpy.dtype`, filled with
        the specified value.

    Examples
    --------
    >>> full(3, 2.5)
    array([2.5, 2.5, 2.5])
    """

    dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

    xp = array_namespace()

    return xp.full(shape, fill_value, dtype=dtype)


def index_along_last_axis(a: ArrayLike, indexes: ArrayLike) -> NDArray:
    """
    Reduce the dimension of array :math:`a` by one, using an array of
    indexes to select elements from the last axis.

    Parameters
    ----------
    a
        Array :math:`a` to be indexed.
    indexes
        *Integer* array with the same shape as :math:`a` but with one
        dimension fewer, containing indices to the last dimension of
        :math:`a`. All elements must be numbers between 0 and
        :math:`m - 1`.

    Returns
    -------
    :class:`numpy.ndarray`
        Indexed array :math:`a`.

    Raises
    ------
    :class:`ValueError`
        If the array :math:`a` and ``indexes`` have incompatible shapes.
    :class:`IndexError`
        If ``indexes`` has elements outside of the allowed range of 0 to
        :math:`m - 1` or if it is not an *integer* array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(
    ...     [
    ...         [
    ...             [0.3, 0.5, 6.9],
    ...             [3.3, 4.4, 1.6],
    ...             [4.4, 7.5, 2.3],
    ...             [2.3, 1.6, 7.4],
    ...         ],
    ...         [
    ...             [2.0, 5.9, 2.8],
    ...             [6.2, 4.9, 8.6],
    ...             [3.7, 9.7, 7.3],
    ...             [6.3, 4.3, 3.2],
    ...         ],
    ...         [
    ...             [0.8, 1.9, 0.7],
    ...             [5.6, 4.0, 1.7],
    ...             [6.7, 8.2, 1.7],
    ...             [1.2, 7.1, 1.4],
    ...         ],
    ...         [
    ...             [4.0, 4.8, 8.9],
    ...             [4.0, 0.3, 6.9],
    ...             [3.5, 7.1, 4.5],
    ...             [1.4, 1.9, 1.6],
    ...         ],
    ...     ]
    ... )
    >>> indexes = np.array([[2, 0, 1, 1], [2, 1, 1, 0], [0, 0, 1, 2], [0, 0, 1, 2]])
    >>> index_along_last_axis(a, indexes)
    array([[6.9, 3.3, 7.5, 1.6],
           [2.8, 4.9, 9.7, 6.3],
           [0.8, 5.6, 8.2, 1.4],
           [4. , 4. , 7.1, 1.6]])

    This function can be used to compute the result of :func:`np.min` along
    the last axis given the corresponding :func:`np.argmin` indexes.

    >>> indexes = np.argmin(a, axis=-1)
    >>> np.array_equal(index_along_last_axis(a, indexes), np.min(a, axis=-1))
    True

    In particular, this can be used to manipulate the indexes specified by
    functions like :func:`np.min` before indexing the array. For example, to
    get elements directly following the smallest elements:

    >>> index_along_last_axis(a, (indexes + 1) % 3)
    array([[0.5, 3.3, 4.4, 7.4],
           [5.9, 8.6, 9.7, 6.3],
           [0.8, 5.6, 6.7, 7.1],
           [4.8, 6.9, 7.1, 1.9]])
    """

    a = as_float_array(a)
    indexes = as_int_array(indexes)

    xp = array_namespace(a, indexes)
    a = xp_as_float_array(a, xp=xp, like=indexes)
    indexes = xp_as_int_array(indexes, xp=xp, like=a)

    if a.shape[:-1] != indexes.shape:
        error = (
            f"Array and indexes have incompatible shapes: {a.shape} and {indexes.shape}"
        )

        raise ValueError(error)

    return xp.take_along_axis(a, indexes[..., None], axis=-1).squeeze(axis=-1)


def format_array_as_row(a: ArrayLike, decimals: int = 7, separator: str = " ") -> str:
    """
    Format the specified array :math:`a` as a row.

    Parameters
    ----------
    a
        Array to format.
    decimals
        Decimal count to use when formatting as a row.
    separator
        Separator used to join the array :math:`a` items.

    Returns
    -------
    :class:`str`
        Array formatted as a row.

    Examples
    --------
    >>> format_array_as_row([1.25, 2.5, 3.75])
    '1.2500000 2.5000000 3.7500000'
    >>> format_array_as_row([1.25, 2.5, 3.75], 3)
    '1.250 2.500 3.750'
    >>> format_array_as_row([1.25, 2.5, 3.75], 3, ", ")
    '1.250, 2.500, 3.750'
    """

    a = as_float_array(a)

    xp = array_namespace(a)

    a = xp_reshape(a, (-1,), xp=xp)

    return separator.join(
        "{1:0.{0}f}".format(decimals, x)
        for x in a  # noqa: PLE1300, RUF100
    )
