"""
Common Utilities
================

Provide common utility objects that don't fall in any specific category.

References
----------
-   :cite:`DjangoSoftwareFoundation2022` : Django Software Foundation. (2022).
    slugify. Retrieved June 1, 2022, from https://github.com/django/django/\
blob/0dd29209091280ccf34e07c9468746c396b7778e/django/utils/text.py#L400
-   :cite:`Kienzle2011a` : Kienzle, P., Patel, N., & Krycka, J. (2011).
    refl1d.numpyerrors - Refl1D v0.6.19 documentation. Retrieved January 30,
    2015, from
    http://www.reflectometry.org/danse/docs/refl1d/_modules/refl1d/\
numpyerrors.html
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import types
import typing
import unicodedata
import warnings
from contextlib import contextmanager
from copy import copy
from pprint import pformat

import numpy as np

from colour.constants import THRESHOLD_INTEGER
from colour.utilities import as_bool

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        Callable,
        DTypeBoolean,
        Generator,
        Iterable,
        Literal,
        Mapping,
        Self,
        Sequence,
    )

from colour.hints import TypeVar
from colour.utilities import CanonicalMapping, Lookup, is_xxhash_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_caching_enabled",
    "set_caching_enable",
    "caching_enable",
    "CacheRegistry",
    "CACHE_REGISTRY",
    "handle_numpy_errors",
    "ignore_numpy_errors",
    "raise_numpy_errors",
    "print_numpy_errors",
    "warn_numpy_errors",
    "ignore_python_warnings",
    "attest",
    "batch",
    "disable_multiprocessing",
    "multiprocessing_pool",
    "is_iterable",
    "is_numeric",
    "is_integer",
    "is_sibling",
    "filter_kwargs",
    "filter_mapping",
    "first_item",
    "copy_definition",
    "validate_method",
    "optional",
    "slugify",
    "int_digest",
]

_CACHING_ENABLED: bool = not as_bool(
    os.environ.get("COLOUR_SCIENCE__DISABLE_CACHING", "False")
)
"""
Global variable storing the current *Colour* caching enabled state.
"""


def is_caching_enabled() -> bool:
    """
    Determine whether *Colour* caching is enabled.

    The caching state is controlled by the global
    *COLOUR_SCIENCE__DISABLE_CACHING* environment variable and can be
    temporarily modified using the :func:`set_caching_enable` function or the
    :class:`caching_enable` context manager.

    Returns
    -------
    :class:`bool`
        Whether *Colour* caching is enabled.

    Examples
    --------
    >>> with caching_enable(False):
    ...     is_caching_enabled()
    False
    >>> with caching_enable(True):
    ...     is_caching_enabled()
    True
    """

    return _CACHING_ENABLED


def set_caching_enable(enable: bool) -> None:
    """
    Set the *Colour* caching enabled state.

    Parameters
    ----------
    enable
        Whether to enable *Colour* caching.

    Examples
    --------
    >>> with caching_enable(True):
    ...     print(is_caching_enabled())
    ...     set_caching_enable(False)
    ...     print(is_caching_enabled())
    True
    False
    """

    global _CACHING_ENABLED  # noqa: PLW0603

    _CACHING_ENABLED = enable


class caching_enable:
    """
    Define a context manager and decorator to temporarily set the *Colour*
    caching enabled state.

    Parameters
    ----------
    enable
        Whether to enable or disable *Colour* caching.
    """

    def __init__(self, enable: bool) -> None:
        self._enable = enable
        self._previous_state = is_caching_enabled()

    def __enter__(self) -> Self:
        """
        Enter the caching context and set the *Colour* caching state.
        """

        set_caching_enable(self._enable)

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Exit the caching context manager and restore the previous *Colour*
        caching state.
        """

        set_caching_enable(self._previous_state)

    def __call__(self, function: Callable) -> Callable:
        """
        Decorate and call the specified function with caching control.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return function(*args, **kwargs)

        return wrapper


class CacheRegistry:
    """
    Provide a registry for managing mapping-based caches.

    The registry maintains a collection of named caches that can be
    registered, cleared, and unregistered. Each cache operates as a
    dictionary-like mapping for storing key-value pairs.

    Attributes
    ----------
    -   :attr:`~colour.utilities.CacheRegistry.registry`

    Methods
    -------
    -   :meth:`~colour.SpectralShape.__init__`
    -   :meth:`~colour.SpectralShape.__str__`
    -   :meth:`~colour.SpectralShape.register_cache`
    -   :meth:`~colour.SpectralShape.unregister_cache`
    -   :meth:`~colour.SpectralShape.clear_cache`
    -   :meth:`~colour.SpectralShape.clear_all_caches`

    Examples
    --------
    >>> cache_registry = CacheRegistry()
    >>> cache_a = cache_registry.register_cache("Cache A")
    >>> cache_a["Foo"] = "Bar"
    >>> cache_b = cache_registry.register_cache("Cache B")
    >>> cache_b["John"] = "Doe"
    >>> cache_b["Luke"] = "Skywalker"
    >>> print(cache_registry)
    {'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}
    >>> cache_registry.clear_cache("Cache A")
    >>> print(cache_registry)
    {'Cache A': '0 item(s)', 'Cache B': '2 item(s)'}
    >>> cache_registry.unregister_cache("Cache B")
    >>> print(cache_registry)
    {'Cache A': '0 item(s)'}
    >>> print(cache_b)
    {}
    """

    def __init__(self) -> None:
        self._registry: dict = {}

    @property
    def registry(self) -> dict:
        """
        Getter for the cache registry.

        Returns
        -------
        :class:`dict`
            Cache registry containing cached computation results.
        """

        return self._registry

    def __str__(self) -> str:
        """
        Return a formatted string representation of the cache registry.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return pformat(
            {
                name: f"{len(self._registry[name])} item(s)"
                for name in sorted(self._registry)
            }
        )

    def register_cache(self, name: str) -> dict:
        """
        Register a new cache with the specified name in the registry.

        Parameters
        ----------
        name
            Cache name for the registry.

        Returns
        -------
        :class:`dict`
            Registered cache.

        Examples
        --------
        >>> cache_registry = CacheRegistry()
        >>> cache_a = cache_registry.register_cache("Cache A")
        >>> cache_a["Foo"] = "Bar"
        >>> cache_b = cache_registry.register_cache("Cache B")
        >>> cache_b["John"] = "Doe"
        >>> cache_b["Luke"] = "Skywalker"
        >>> print(cache_registry)
        {'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}
        """

        self._registry[name] = {}

        return self._registry[name]

    def unregister_cache(self, name: str) -> None:
        """
        Unregister the cache with the specified name from the registry.

        Parameters
        ----------
        name
            Cache name in the registry.

        Notes
        -----
        -   The cache is cleared before being unregistered.

        Examples
        --------
        >>> cache_registry = CacheRegistry()
        >>> cache_a = cache_registry.register_cache("Cache A")
        >>> cache_a["Foo"] = "Bar"
        >>> cache_b = cache_registry.register_cache("Cache B")
        >>> cache_b["John"] = "Doe"
        >>> cache_b["Luke"] = "Skywalker"
        >>> print(cache_registry)
        {'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}
        >>> cache_registry.unregister_cache("Cache B")
        >>> print(cache_registry)
        {'Cache A': '1 item(s)'}
        >>> print(cache_b)
        {}
        """

        self.clear_cache(name)

        del self._registry[name]

    def clear_cache(self, name: str) -> None:
        """
        Clear the cache with the specified name.

        Parameters
        ----------
        name
            Cache name in the registry.

        Examples
        --------
        >>> cache_registry = CacheRegistry()
        >>> cache_a = cache_registry.register_cache("Cache A")
        >>> cache_a["Foo"] = "Bar"
        >>> print(cache_registry)
        {'Cache A': '1 item(s)'}
        >>> cache_registry.clear_cache("Cache A")
        >>> print(cache_registry)
        {'Cache A': '0 item(s)'}
        """

        self._registry[name].clear()

    def clear_all_caches(self) -> None:
        """
        Clear all caches in the registry.

        Examples
        --------
        >>> cache_registry = CacheRegistry()
        >>> cache_a = cache_registry.register_cache("Cache A")
        >>> cache_a["Foo"] = "Bar"
        >>> cache_b = cache_registry.register_cache("Cache B")
        >>> cache_b["John"] = "Doe"
        >>> cache_b["Luke"] = "Skywalker"
        >>> print(cache_registry)
        {'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}
        >>> cache_registry.clear_all_caches()
        >>> print(cache_registry)
        {'Cache A': '0 item(s)', 'Cache B': '0 item(s)'}
        """

        for key in self._registry:
            self.clear_cache(key)


CACHE_REGISTRY: CacheRegistry = CacheRegistry()
"""
*Colour* cache registry referencing all the caches used for repetitive or long
processes.
"""


def handle_numpy_errors(**kwargs: Any) -> Callable:
    """
    Handle *Numpy* errors through function decoration.

    Other Parameters
    ----------------
    kwargs
        Keyword arguments passed to :func:`numpy.seterr` to control
        error handling behaviour.

    Returns
    -------
    Callable
        Decorated function with specified *Numpy* error handling.

    References
    ----------
    :cite:`Kienzle2011a`

    Examples
    --------
    >>> import numpy
    >>> @handle_numpy_errors(all="ignore")
    ... def f():
    ...     1 / numpy.zeros(3)
    >>> f()
    """

    keyword_arguments = kwargs

    def wrapper(function: Callable) -> Callable:
        """Wrap specified function wrapper."""

        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Wrap specified function."""

            with np.errstate(**keyword_arguments):
                return function(*args, **kwargs)

        return wrapped

    return wrapper


ignore_numpy_errors = handle_numpy_errors(all="ignore")
raise_numpy_errors = handle_numpy_errors(all="raise")
print_numpy_errors = handle_numpy_errors(all="print")
warn_numpy_errors = handle_numpy_errors(all="warn")


def ignore_python_warnings(function: Callable) -> Callable:
    """
    Decorate a function to ignore *Python* warnings.

    Parameters
    ----------
    function
        Function to decorate.

    Returns
    -------
    Callable
        Decorated function that suppresses *Python* warnings during
        execution.

    Examples
    --------
    >>> @ignore_python_warnings
    ... def f():
    ...     warnings.warn("This is an ignored warning!")
    >>> f()
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrap specified function."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return function(*args, **kwargs)

    return wrapper


def attest(condition: bool | DTypeBoolean, message: str = "") -> None:
    """
    Provide the ``assert`` statement functionality without being disabled by
    optimised Python execution.

    Parameters
    ----------
    condition
        Condition to attest/assert.
    message
        Message to display when the assertion fails.
    """

    if not condition:
        raise AssertionError(message)


def batch(sequence: Sequence, k: int | Literal[3] = 3) -> Generator:
    """
    Generate batches from the specified sequence.

    Parameters
    ----------
    sequence
        Sequence to create batches from.
    k
        Batch size.

    Yields
    ------
    Generator
        Batch generator.

    Examples
    --------
    >>> batch(tuple(range(10)), 3)  # doctest: +ELLIPSIS
    <generator object batch at 0x...>
    """

    for i in range(0, len(sequence), k):
        yield sequence[i : i + k]


_MULTIPROCESSING_ENABLED: bool = True
"""*Colour* multiprocessing state."""


class disable_multiprocessing:
    """
    Define a context manager and decorator to temporarily disable *Colour*
    multiprocessing state.
    """

    def __enter__(self) -> Self:
        """
        Disable *Colour* multiprocessing state upon entering the context
        manager.
        """

        global _MULTIPROCESSING_ENABLED  # noqa: PLW0603

        _MULTIPROCESSING_ENABLED = False

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Enable *Colour* multiprocessing state upon exiting the context
        manager.
        """

        global _MULTIPROCESSING_ENABLED  # noqa: PLW0603

        _MULTIPROCESSING_ENABLED = True

    def __call__(self, function: Callable) -> Callable:
        """
        Execute the decorated function with optional multiprocessing support.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap specified function."""

            with self:
                return function(*args, **kwargs)

        return wrapper


def _initializer(kwargs: Any) -> None:
    """
    Initialize a multiprocessing pool worker process.

    Ensure that worker processes on *Windows* correctly inherit the current
    domain-range scale configuration from the parent process.

    Parameters
    ----------
    kwargs
        Initialization arguments for configuring the worker process state.
    """

    # NOTE: No coverage information is available as this code is executed in
    # sub-processes.

    import colour.utilities.array  # pragma: no cover  # noqa: PLC0415

    colour.utilities.array._DOMAIN_RANGE_SCALE = kwargs.get(  # noqa: SLF001
        "scale", "reference"
    )  # pragma: no cover

    import colour.algebra.common  # pragma: no cover  # noqa: PLC0415

    colour.algebra.common._SDIV_MODE = kwargs.get(  # noqa: SLF001
        "sdiv_mode", "Ignore Zero Conversion"
    )  # pragma: no cover
    colour.algebra.common._SPOW_ENABLED = kwargs.get(  # noqa: SLF001
        "spow_enabled", True
    )  # pragma: no cover


@contextmanager
def multiprocessing_pool(*args: Any, **kwargs: Any) -> Generator:
    """
    Provide a context manager for a multiprocessing pool.

    Other Parameters
    ----------------
    args
        Arguments passed to the multiprocessing pool constructor.
    kwargs
        Keyword arguments passed to the multiprocessing pool
        constructor.

    Yields
    ------
    Generator
        Multiprocessing pool context manager.

    Examples
    --------
    >>> from functools import partial
    >>> def _add(a, b):
    ...     return a + b
    >>> with multiprocessing_pool() as pool:
    ...     pool.map(partial(_add, b=2), range(10))
    ... # doctest: +SKIP
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """

    from colour.algebra import get_sdiv_mode, is_spow_enabled  # noqa: PLC0415
    from colour.utilities import get_domain_range_scale  # noqa: PLC0415

    class _DummyPool:
        """
        A dummy multiprocessing pool that does not perform multiprocessing.

        Other Parameters
        ----------------
        args
            Arguments.
        kwargs
            Keywords arguments.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def map(
            self,
            func: Callable,
            iterable: Sequence,
            chunksize: int | None = None,  # noqa: ARG002
        ) -> list[Any]:
            """Apply specified function to each element of the specified iterable."""

            return [func(a) for a in iterable]

        def terminate(self) -> None:
            """Terminate the process."""

    kwargs["initializer"] = _initializer
    kwargs["initargs"] = (
        {
            "scale": get_domain_range_scale(),
            "sdiv_mode": get_sdiv_mode(),
            "spow_enabled": is_spow_enabled(),
        },
    )

    pool_factory: Callable
    if _MULTIPROCESSING_ENABLED:
        import multiprocessing  # noqa: PLC0415

        pool_factory = multiprocessing.Pool
    else:
        pool_factory = _DummyPool

    pool = pool_factory(*args, **kwargs)

    try:
        yield pool
    finally:
        pool.terminate()


def is_iterable(a: Any) -> bool:
    """
    Determine whether the specified variable :math:`a` is iterable.

    Parameters
    ----------
    a
        Variable :math:`a` to check for iterability.

    Returns
    -------
    :class:`bool`
        Whether the variable :math:`a` is iterable.

    Examples
    --------
    >>> is_iterable([1, 2, 3])
    True
    >>> is_iterable(1)
    False
    """

    return isinstance(a, str) or (bool(getattr(a, "__iter__", False)))


def is_numeric(a: Any) -> bool:
    """
    Determine whether the specified variable :math:`a` is a
    :class:`Real`-like variable.

    Parameters
    ----------
    a
        Variable :math:`a` to test.

    Returns
    -------
    :class:`bool`
        Whether variable :math:`a` is a :class:`Real`-like variable.

    Examples
    --------
    >>> is_numeric(1)
    True
    >>> is_numeric((1,))
    False
    """

    return isinstance(
        a,
        (
            int,
            float,
            complex,
            np.integer,
            np.int8,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.floating,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ),  # pyright: ignore
    )


def is_integer(a: Any) -> bool:
    """
    Determine whether the specified variable :math:`a` is an
    :class:`numpy.integer`-like variable under the specified threshold.

    Parameters
    ----------
    a
        Variable :math:`a` to test.

    Returns
    -------
    :class:`bool`
        Whether variable :math:`a` is an :class:`numpy.integer`-like
        variable.

    Notes
    -----
    -   The determination threshold is defined by the
        :attr:`colour.algebra.common.THRESHOLD_INTEGER` attribute.

    Examples
    --------
    >>> is_integer(1)
    True
    >>> is_integer(1.01)
    False
    """

    return abs(a - np.around(a)) <= THRESHOLD_INTEGER


def is_sibling(element: Any, mapping: Mapping) -> bool:
    """
    Determine whether the type of the specified element is present in the
    specified mapping types.

    Parameters
    ----------
    element
        Element to check whether its type is present in the mapping
        types.
    mapping
        Mapping types to check against.

    Returns
    -------
    :class:`bool`
        Whether the type of the specified element is present in the
        specified mapping types.
    """

    return isinstance(element, tuple({type(element) for element in mapping.values()}))


def filter_kwargs(function: Callable, **kwargs: Any) -> dict:
    """
    Filter keyword arguments incompatible with the specified function
    signature.

    Parameters
    ----------
    function
        Callable to filter the incompatible keyword arguments against.

    Other Parameters
    ----------------
    kwargs
        Keyword arguments to be filtered.

    Returns
    -------
    dict
        Filtered keyword arguments compatible with the function signature.

    Examples
    --------
    >>> def fn_a(a):
    ...     return a
    >>> def fn_b(a, b=0):
    ...     return a, b
    >>> def fn_c(a, b=0, c=0):
    ...     return a, b, c
    >>> fn_a(1, **filter_kwargs(fn_a, b=2, c=3))
    1
    >>> fn_b(1, **filter_kwargs(fn_b, b=2, c=3))
    (1, 2)
    >>> fn_c(1, **filter_kwargs(fn_c, b=2, c=3))
    (1, 2, 3)
    """

    kwargs = copy(kwargs)

    try:
        args = list(inspect.signature(function).parameters.keys())
    except ValueError:  # pragma: no cover
        return {}

    for key in set(kwargs.keys()) - set(args):
        kwargs.pop(key)

    return kwargs


def filter_mapping(mapping: Mapping, names: str | Sequence[str]) -> dict:
    """
    Filter the specified mapping with specified names.

    Parameters
    ----------
    mapping
        Mapping to filter.
    names
        Name for the mapping elements to filter or a sequence of names.

    Returns
    -------
    dict
        Filtered mapping containing only the specified elements.

    Notes
    -----
    -   If the mapping is a :class:`colour.utilities.CanonicalMapping`
        instance, then the lower, slugified and canonical keys are also
        used for matching.
    -   To honour the filterers ordering, the return value is a
        :class:`dict` instance.

    Examples
    --------
    >>> class Element:
    ...     pass
    >>> mapping = {
    ...     "Element A": Element(),
    ...     "Element B": Element(),
    ...     "Element C": Element(),
    ...     "Not Element C": Element(),
    ... }
    >>> filter_mapping(mapping, "Element A")  # doctest: +ELLIPSIS
    {'Element A': <colour.utilities.common.Element object at 0x...>}
    """

    def filter_mapping_with_name(mapping: Mapping, name: str) -> dict:
        """
        Filter specified mapping with the specified name.

        Parameters
        ----------
        mapping
            Mapping to filter.
        name
            Name for the specified mapping elements.

        Returns
        -------
        dict
            Filtered mapping elements.
        """

        keys = list(mapping.keys())

        if isinstance(mapping, CanonicalMapping):
            keys += list(mapping.lower_keys())
            keys += list(mapping.slugified_keys())
            keys += list(mapping.canonical_keys())

        elements = [mapping[key] for key in keys if name == key]

        lookup = Lookup(mapping)

        return {lookup.first_key_from_value(element): element for element in elements}

    names = [str(names)] if isinstance(names, str) else names

    filtered_mapping = {}

    for filterer in names:
        filtered_mapping.update(filter_mapping_with_name(mapping, filterer))

    return filtered_mapping


def first_item(a: Iterable) -> Any:
    """
    Return the first item from the specified iterable.

    Parameters
    ----------
    a
        Iterable to retrieve the first item from.

    Returns
    -------
    :class:`object`
        First item from the iterable.

    Raises
    ------
    :class:`StopIteration`
        If the iterable is empty.

    Examples
    --------
    >>> a = range(10)
    >>> first_item(a)
    0
    """

    return next(iter(a))


def copy_definition(definition: Callable, name: str | None = None) -> Callable:
    """
    Copy a definition using the same code, globals, defaults, closure, and
    name.

    Parameters
    ----------
    definition
        Definition to be copied.
    name
        Optional name for the definition copy.

    Returns
    -------
    Callable
        Copy of the specified definition.
    """

    copy = types.FunctionType(
        definition.__code__,
        definition.__globals__,
        str(name or definition.__name__),
        definition.__defaults__,
        definition.__closure__,
    )
    copy.__dict__.update(definition.__dict__)
    copy.__annotations__ = definition.__annotations__.copy()

    return copy


@functools.cache
def validate_method(
    method: str,
    valid_methods: tuple,
    message: str = '"{0}" method is invalid, it must be one of {1}!',
    as_lowercase: bool = True,
) -> str:
    """
    Validate whether the specified method exists in the specified valid
    methods and optionally return the method lower cased.

    Parameters
    ----------
    method
        Method to validate.
    valid_methods
        Valid methods.
    message
        Message for the exception.
    as_lowercase
        Whether to convert the specified method to lower case or not.

    Returns
    -------
    :class:`str`
        Method optionally lower cased.

    Raises
    ------
    :class:`ValueError`
         If the method does not exist.

    Examples
    --------
    >>> validate_method("Valid", ("Valid", "Yes", "Ok"))
    'valid'
    >>> validate_method("Valid", ("Valid", "Yes", "Ok"), as_lowercase=False)
    'Valid'
    """

    valid_methods = tuple([str(valid_method) for valid_method in valid_methods])

    method_lower = method.lower()
    if method_lower not in [valid_method.lower() for valid_method in valid_methods]:
        raise ValueError(message.format(method, valid_methods))

    return method_lower if as_lowercase else method


T = TypeVar("T")


def optional(value: T | None, default: T) -> T:
    """
    Return the specified value or a default if the value is *None*.

    Parameters
    ----------
    value
        Optional argument value.
    default
        Default argument value if ``value`` is *None*.

    Returns
    -------
    T
        Argument value.

    Examples
    --------
    >>> optional("Foo", "Bar")
    'Foo'
    >>> optional(None, "Bar")
    'Bar'
    """

    if value is None:
        return default

    return value


def slugify(object_: Any, allow_unicode: bool = False) -> str:
    """
    Generate a *SEO* friendly and human-readable slug from the specified
    object.

    Convert to ASCII if ``allow_unicode`` is *False*. Convert spaces or
    repeated dashes to single dashes. Remove characters that are not
    alphanumerics, underscores, or hyphens. Convert to lowercase. Strip
    leading and trailing whitespace, dashes, and underscores.

    Parameters
    ----------
    object_
        Object to convert to a slug.
    allow_unicode
        Whether to allow unicode characters in the generated slug.

    Returns
    -------
    :class:`str`
        Generated slug.

    References
    ----------
    :cite:`DjangoSoftwareFoundation2022`

    Examples
    --------
    >>> slugify(" Jack & Jill like numbers 1,2,3 and 4 and silly characters ?%.$!/")
    'jack-jill-like-numbers-123-and-4-and-silly-characters'
    """

    value = str(object_)

    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    value = re.sub(r"[^\w\s-]", "", value.lower())

    return re.sub(r"[-\s]+", "-", value).strip("-_")


if is_xxhash_installed():
    import xxhash

    int_digest = xxhash.xxh3_64_intdigest
else:
    int_digest = hash  # pragma: no cover
