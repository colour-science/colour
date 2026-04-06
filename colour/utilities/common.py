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

import contextvars
import functools
import hashlib
import inspect
import os
import re
import tempfile
import types
import typing
import unicodedata
import urllib.error
import urllib.request
import warnings
from copy import copy
from pprint import pformat
from urllib.parse import urlparse

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
    "set_caching_enabled",
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
    "hash_sha256",
    "download_url",
]

_CACHING_ENABLED_DEFAULT: bool = not as_bool(
    os.environ.get("COLOUR_SCIENCE__DISABLE_CACHING", "False")
)
"""Environment-seeded default for :attr:`_CACHING_ENABLED`."""

_CACHING_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_CACHING_ENABLED", default=_CACHING_ENABLED_DEFAULT
)
"""
:class:`contextvars.ContextVar` storing the current *Colour* caching
enabled state. The :class:`contextvars.ContextVar` keeps nested
:class:`caching_enable` contexts independent across concurrent threads
and async tasks. Read it via :func:`is_caching_enabled` and toggle it
via :func:`set_caching_enabled` or :class:`caching_enable`. The
environment value is seeded as the :class:`contextvars.ContextVar`
``default`` so that fresh threads and async tasks observe it, rather than
via a module-level ``set`` that only applies to the importing context.
"""


def is_caching_enabled() -> bool:
    """
    Determine whether *Colour* caching is enabled.

    The caching state is controlled by the global
    *COLOUR_SCIENCE__DISABLE_CACHING* environment variable and can be
    temporarily modified using the :func:`set_caching_enabled` function or the
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

    return _CACHING_ENABLED.get()


def set_caching_enabled(enable: bool) -> None:
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
    ...     set_caching_enabled(False)
    ...     print(is_caching_enabled())
    True
    False
    """

    _CACHING_ENABLED.set(enable)


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
        # Token stack: nested or recursive ``__enter__`` / ``__exit__``
        # pairs against the same instance (e.g. via the decorator form on
        # a recursive function) push and pop independent reset tokens.
        self._tokens: list[contextvars.Token[bool]] = []

    def __enter__(self) -> Self:
        """
        Enter the caching context and set the *Colour* caching state.
        """

        self._tokens.append(_CACHING_ENABLED.set(self._enable))

        return self

    def __exit__(self, *args: Any) -> None:
        """
        Exit the caching context manager and restore the previous *Colour*
        caching state.
        """

        _CACHING_ENABLED.reset(self._tokens.pop())

    def __call__(self, function: Callable) -> Callable:
        """
        Decorate and call the specified function with caching control.
        """

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # A fresh instance is entered per call so the token stack is never
            # shared across threads or async tasks invoking the decorated
            # definition concurrently.
            with self.__class__(self._enable):
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

    try:
        a_float = float(a)
        return abs(a_float - round(a_float)) <= THRESHOLD_INTEGER
    except (OverflowError, ValueError, TypeError):
        return False


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


def hash_sha256(filename: str, chunk_size: int = 2**16) -> str:
    """
    Compute the *SHA-256* hash of given file.

    Parameters
    ----------
    filename
        File to compute the hash of.
    chunk_size
        Chunk size to read from the file.

    Returns
    -------
    :class:`str`
        *SHA-256* hash of the file.
    """

    sha256 = hashlib.sha256()

    with open(filename, "rb") as file_object:
        while True:
            chunk = file_object.read(chunk_size)
            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def download_url(
    url: str,
    filename: str | None = None,
    sha256: str | None = None,
    retries: int = 6,
) -> str:
    """
    Download a file from *url* and cache it locally.

    Parameters
    ----------
    url
        URL to download.
    filename
        Explicit file path to save to. If provided, the URL-derived
        cache path is ignored.
    sha256
        Expected *SHA-256* hash of the file. If provided, the downloaded
        file will be verified and re-downloaded on mismatch.
    retries
        Number of retries in case of network errors or hash mismatches.

    Returns
    -------
    :class:`str`
        Absolute path to the cached file.
    """

    if filename is not None:
        local_path = filename
    else:
        import colour  # noqa: PLC0415

        root = colour.ROOT_COLOUR_SCIENCE

        parsed = urlparse(url)
        relative = parsed.path.lstrip("/")

        # Strip the HuggingFace URL prefix to get a clean local path,
        # e.g., ``colour-science/learning-munsell/resolve/main/models/...``
        # becomes ``learning-munsell/models/...``.
        prefix = "colour-science/"
        relative = relative.removeprefix(prefix)

        resolve_main = "/resolve/main/"
        if resolve_main in relative:
            parts = relative.split(resolve_main, 1)
            relative = f"{parts[0]}/{parts[1]}"

        local_path = os.path.join(root, relative)

    if os.path.isfile(local_path):
        if sha256 is not None and hash_sha256(local_path) != sha256.lower():
            os.remove(local_path)
        else:
            return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    attempt = 0
    while attempt < retries:
        # Download to a unique temporary file in the destination directory,
        # then atomically rename it into place. Concurrent downloaders (e.g.
        # parallel test workers sharing the cache) thus never observe a
        # partially-written file.
        descriptor, temporary_path = tempfile.mkstemp(
            dir=os.path.dirname(local_path), suffix=".tmp"
        )
        try:
            with (
                os.fdopen(descriptor, "wb") as out_file,
                urllib.request.urlopen(url) as response,  # noqa: S310
            ):
                while True:
                    chunk = response.read(2**16)
                    if not chunk:
                        break
                    out_file.write(chunk)

            if sha256 is not None:
                actual_hash = hash_sha256(temporary_path)
                if actual_hash != sha256.lower():
                    file_size = os.path.getsize(temporary_path)

                    message = (
                        f'"SHA-256" hash of "{local_path}" file '
                        f"({file_size} bytes) does not match the "
                        f"expected hash: "
                        f"{actual_hash} != {sha256.lower()}"
                    )
                    raise ValueError(message)  # noqa: TRY301

            os.replace(temporary_path, local_path)
        except (urllib.error.URLError, OSError, ValueError):
            attempt += 1
            if attempt == retries:
                raise

            import time  # noqa: PLC0415

            time.sleep(min(2**attempt, 2**8))
        else:
            return local_path
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    return local_path
