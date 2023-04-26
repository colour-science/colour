"""
Common Utilities
================

Defines the common utilities objects that don't fall in any specific category.

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
from collections import OrderedDict

import inspect
import functools
import numpy as np
import re
import subprocess  # nosec
import unicodedata
import types
import warnings
from contextlib import contextmanager
from copy import copy
from pprint import pformat

from colour.constants import INTEGER_THRESHOLD
from colour.hints import (
    Any,
    Callable,
    DTypeBoolean,
    Generator,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
)
from colour.utilities import CanonicalMapping, Lookup

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CacheRegistry",
    "CACHE_REGISTRY",
    "COLOUR_DEFAULT_LRU_CACHE_SIZE",
    "ColourLRUCache",
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
    "is_ctlrender_installed",
    "is_graphviz_installed",
    "is_matplotlib_installed",
    "is_networkx_installed",
    "is_opencolorio_installed",
    "is_openimageio_installed",
    "is_pandas_installed",
    "is_tqdm_installed",
    "is_trimesh_installed",
    "required",
    "is_iterable",
    "is_string",
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
]

COLOUR_DEFAULT_LRU_CACHE_SIZE: int | None = None


class ColourLRUCache(OrderedDict):
    """A dict-like object for implementing least recently used (LRU) caching
    with a maximum number of entries

    Parameters
    ----------
    OrderedDict : _type_
        _description_
    """

    def __init__(self, max_size: int | None):
        """Create a new LRU size limited dict

        Parameters
        ----------
        max_size : int | None
            Maximum number of entries in the cache
        """
        self.max_size = max_size
        super().__init__()

    @property
    def max_size(self) -> int | None:
        """Max entries in LRU cache"""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int | None):
        """Set max entries in LRU cache

        Parameters
        ----------
        value : int | None
            new max value size. None means the cache is unbounded
        """
        self._max_size = value
        self._check_size_limit()

    def _check_size_limit(self):
        """Check the size limit of the cache and evict the oldest updated /
        checked item
        """
        if self.max_size is not None:
            while len(self) > self.max_size:
                self.popitem(False)

    def __contains__(self, key: Any) -> bool:
        """Check if this cache contains an entry for the given `key`

        Parameters
        ----------
        key : Any
            Dictionary key

        Returns
        -------
        bool
        """
        in_status = super().__contains__(key)
        if in_status:
            self.move_to_end(key=key)
        return in_status

    def __setitem__(self, key: Any, value: Any):
        """Insert / update an item in the cache

        Parameters
        ----------
        key : Any
            Dictionary key
        value : Any
            New value
        """
        super().__setitem__(key, value)
        self.move_to_end(key)

        self._check_size_limit()


class CacheRegistry:
    """
    A registry for mapping-based caches.

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
        Getter property for the cache registry.

        Returns
        -------
        :class:`dict`
            Cache registry.
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

    def register_cache(
        self, name: str, max_size: int | None = COLOUR_DEFAULT_LRU_CACHE_SIZE
    ) -> dict:
        """
        Register a new cache with given name in the registry.

        Parameters
        ----------
        name
            Cache name for the registry.
        max_size
            Maximum number of entries in the cache. Internally controlls an LRU
            cache. default: None

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

        self._registry[name] = (
            {} if max_size is None else ColourLRUCache(max_size=max_size)
        )

        return self._registry[name]

    def unregister_cache(self, name: str):
        """
        Unregister cache with given name in the registry.

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

    def clear_cache(self, name: str):
        """
        Clear the cache with given name.

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

    def clear_all_caches(self):
        """
        Clear all the caches in the registry.

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

CACHE_REGISTRY
"""


def handle_numpy_errors(**kwargs: Any) -> Callable:
    """
    Decorate a function to handle *Numpy* errors.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments.

    Returns
    -------
    Callable

    References
    ----------
    :cite:`Kienzle2011a`

    Examples
    --------
    >>> import numpy
    >>> @handle_numpy_errors(all="ignore")
    ... def f():
    ...     1 / numpy.zeros(3)
    ...
    >>> f()
    """

    context = np.errstate(**kwargs)

    def wrapper(function: Callable) -> Callable:
        """Wrap given function wrapper."""

        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Wrap given function."""

            with context:
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

    Examples
    --------
    >>> @ignore_python_warnings
    ... def f():
    ...     warnings.warn("This is an ignored warning!")
    ...
    >>> f()
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrap given function."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return function(*args, **kwargs)

    return wrapper


def attest(condition: bool | DTypeBoolean, message: str = ""):
    """
    Provide the `assert` statement functionality without being disabled by
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
    Return a batch generator from given sequence.

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
    Define a context manager and decorator to temporarily disabling *Colour*
    multiprocessing state.
    """

    def __enter__(self) -> disable_multiprocessing:
        """
        Disable *Colour* multiprocessing state upon entering the context
        manager.
        """

        global _MULTIPROCESSING_ENABLED

        _MULTIPROCESSING_ENABLED = False

        return self

    def __exit__(self, *args: Any):
        """
        Enable *Colour* multiprocessing state upon exiting the context
        manager.
        """

        global _MULTIPROCESSING_ENABLED

        _MULTIPROCESSING_ENABLED = True

    def __call__(self, function: Callable) -> Callable:
        """Call the wrapped definition."""

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap given function."""

            with self:
                return function(*args, **kwargs)

        return wrapper


def _initializer(kwargs: Any):
    """
    Initialize a multiprocessing pool.

    It is used to ensure that processes on *Windows* inherit correctly from the
    current domain-range scale.

    Parameters
    ----------
    kwargs
        Initialisation arguments.
    """

    # NOTE: No coverage information is available as this code is executed in
    # sub-processes.

    import colour.utilities.array  # pragma: no cover

    colour.utilities.array._DOMAIN_RANGE_SCALE = kwargs.get(
        "scale", "reference"
    )  # pragma: no cover

    import colour.algebra.common  # pragma: no cover

    colour.algebra.common._SDIV_MODE = kwargs.get(
        "sdiv_mode", "Ignore Zero Conversion"
    )  # pragma: no cover
    colour.algebra.common._SPOW_ENABLED = kwargs.get(
        "spow_enabled", True
    )  # pragma: no cover


@contextmanager
def multiprocessing_pool(*args: Any, **kwargs: Any) -> Generator:
    """
    Define a context manager providing a multiprocessing pool.

    Other Parameters
    ----------------
    args
        Arguments.
    kwargs
        Keywords arguments.

    Yields
    ------
    Generator
        Multiprocessing pool.

    Examples
    --------
    >>> from functools import partial
    >>> def _add(a, b):
    ...     return a + b
    ...
    >>> with multiprocessing_pool() as pool:
    ...     pool.map(partial(_add, b=2), range(10))
    ... # doctest: +SKIP
    ...
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """

    from colour.algebra import get_sdiv_mode, is_spow_enabled
    from colour.utilities import get_domain_range_scale

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

        def map(self, func, iterable, chunksize=None):  # noqa: A003, ARG002
            """Apply given function to each element of given iterable."""

            return [func(a) for a in iterable]

        def terminate(self):
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
        import multiprocessing

        pool_factory = multiprocessing.Pool
    else:
        pool_factory = _DummyPool

    pool = pool_factory(*args, **kwargs)

    try:
        yield pool
    finally:
        pool.terminate()


def is_ctlrender_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *ctlrender* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *ctlrender* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *ctlrender* is installed.

    Raises
    ------
    :class:`ImportError`
        If *ctlrender* is not installed.
    """

    try:  # pragma: no cover
        stdout = subprocess.run(
            ["ctlrender", "-help"], capture_output=True
        ).stdout.decode(
            "utf-8"
        )  # nosec

        if "transforms an image using one or more CTL scripts" not in stdout:
            raise FileNotFoundError  # noqa: TRY301

        return True
    except FileNotFoundError as error:  # pragma: no cover
        if raise_exception:
            raise FileNotFoundError(
                '"ctlrender" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


def is_graphviz_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *Graphviz* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *Graphviz* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *Graphviz* is installed.

    Raises
    ------
    :class:`ImportError`
        If *Graphviz* is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0611
        import pygraphviz  # noqa: F401

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"Graphviz" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


def is_matplotlib_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *Matplotlib* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *Matplotlib* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *Matplotlib* is installed.

    Raises
    ------
    :class:`ImportError`
        If *Matplotlib* is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0611
        import matplotlib as mpl  # noqa: F401

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"Matplotlib" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


def is_networkx_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *NetworkX* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *NetworkX* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *NetworkX* is installed.

    Raises
    ------
    :class:`ImportError`
        If *NetworkX* is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0611
        import networkx  # noqa: F401

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"NetworkX" related API features, e.g. the automatic colour '
                f'conversion graph, are not available: "{error}".\nPlease refer '
                "to the installation guide for more information: "
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


def is_opencolorio_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *OpenColorIO* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *OpenColorIO* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *OpenColorIO* is installed.

    Raises
    ------
    :class:`ImportError`
        If *OpenColorIO* is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0611
        import PyOpenColorIO  # noqa: F401

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"OpenColorIO" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


def is_openimageio_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *OpenImageIO* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *OpenImageIO* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *OpenImageIO* is installed.

    Raises
    ------
    :class:`ImportError`
        If *OpenImageIO* is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0611
        import OpenImageIO  # noqa: F401

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"OpenImageIO" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


def is_pandas_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *Pandas* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *Pandas* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *Pandas* is installed.

    Raises
    ------
    :class:`ImportError`
        If *Pandas* is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0611
        import pandas  # noqa: F401, ICN001

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                f'"Pandas" related API features are not available: "{error}".\n'
                "See the installation guide for more information: "
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


def is_tqdm_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *tqdm* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *tqdm* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *tqdm* is installed.

    Raises
    ------
    :class:`ImportError`
        If *tqdm* is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0611
        import tqdm  # noqa: F401

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                f'"tqdm" related API features are not available: "{error}".\n'
                "See the installation guide for more information: "
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


def is_trimesh_installed(raise_exception: bool = False) -> bool:
    """
    Return whether *Trimesh* is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *Trimesh* is unavailable.

    Returns
    -------
    :class:`bool`
        Whether *Trimesh* is installed.

    Raises
    ------
    :class:`ImportError`
        If *Trimesh* is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0611
        import trimesh  # noqa: F401

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"Trimesh" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            ) from error

        return False


_REQUIREMENTS_TO_CALLABLE: CanonicalMapping = CanonicalMapping(
    {
        "ctlrender": is_ctlrender_installed,
        "Graphviz": is_graphviz_installed,
        "Matplotlib": is_matplotlib_installed,
        "NetworkX": is_networkx_installed,
        "OpenColorIO": is_opencolorio_installed,
        "OpenImageIO": is_openimageio_installed,
        "Pandas": is_pandas_installed,
        "tqdm": is_tqdm_installed,
        "trimesh": is_trimesh_installed,
    }
)
"""
Mapping of requirements to their respective callables.
"""


def required(
    *requirements: Literal[
        "ctlrender",
        "Graphviz",
        "Matplotlib",
        "NetworkX",
        "OpenColorIO",
        "OpenImageIO",
        "Pandas",
        "tqdm",
        "trimesh",
    ]
) -> Callable:
    """
    Decorate a function to check whether various ancillary package requirements
    are satisfied.

    Other Parameters
    ----------------
    requirements
        Requirements to check whether they are satisfied.

    Returns
    -------
    Callable
    """

    def wrapper(function: Callable) -> Callable:
        """Wrap given function wrapper."""

        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Wrap given function."""

            for requirement in requirements:
                _REQUIREMENTS_TO_CALLABLE[requirement](raise_exception=True)

            return function(*args, **kwargs)

        return wrapped

    return wrapper


def is_iterable(a: Any) -> bool:
    """
    Return whether given variable :math:`a` is iterable.

    Parameters
    ----------
    a
        Variable :math:`a` to check the iterability.

    Returns
    -------
    :class:`bool`
        Whether variable :math:`a` is iterable.

    Examples
    --------
    >>> is_iterable([1, 2, 3])
    True
    >>> is_iterable(1)
    False
    """

    return is_string(a) or (bool(getattr(a, "__iter__", False)))


def is_string(a: Any) -> bool:
    """
    Return whether given variable :math:`a` is a :class:`str`-like variable.

    Parameters
    ----------
    a
        Variable :math:`a` to test.

    Returns
    -------
    :class:`bool`
        Whether variable :math:`a` is a :class:`str`-like variable.

    Examples
    --------
    >>> is_string("I'm a string!")
    True
    >>> is_string(["I'm a string!"])
    False
    """

    return bool(isinstance(a, str))


def is_numeric(a: Any) -> bool:
    """
    Return whether given variable :math:`a` is a :class:`Real`-like
    variable.

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
            np.csingle,
            np.cdouble,
        ),  # pyright: ignore
    )


def is_integer(a: Any) -> bool:
    """
    Return whether given variable :math:`a` is an :class:`numpy.integer`-like
    variable under given threshold.

    Parameters
    ----------
    a
        Variable :math:`a` to test.

    Returns
    -------
    :class:`bool`
        Whether variable :math:`a` is an :class:`numpy.integer`-like variable.

    Notes
    -----
    -   The determination threshold is defined by the
        :attr:`colour.algebra.common.INTEGER_THRESHOLD` attribute.

    Examples
    --------
    >>> is_integer(1)
    True
    >>> is_integer(1.01)
    False
    """

    return abs(a - np.around(a)) <= INTEGER_THRESHOLD


def is_sibling(element: Any, mapping: Mapping) -> bool:
    """
    Return whether given element type is present in given mapping types.

    Parameters
    ----------
    element
        Element to check whether its type is present in the mapping types.
    mapping
        Mapping types.

    Returns
    -------
    :class:`bool`
        Whether given element type is present in given mapping types.
    """

    return isinstance(
        element, tuple({type(element) for element in mapping.values()})
    )


def filter_kwargs(function: Callable, **kwargs: Any) -> dict:
    """
    Filter keyword arguments incompatible with the given function signature.

    Parameters
    ----------
    function
        Callable to filter the incompatible keyword arguments.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments.

    Returns
    -------
    dict
        Filtered keyword arguments.

    Examples
    --------
    >>> def fn_a(a):
    ...     return a
    ...
    >>> def fn_b(a, b=0):
    ...     return a, b
    ...
    >>> def fn_c(a, b=0, c=0):
    ...     return a, b, c
    ...
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
    Filter given mapping with given names.

    Parameters
    ----------
    mapping
        Mapping to filter.
    names
        Name for given mapping elements or a list of names.

    Returns
    -------
    dict
        Filtered mapping elements.

    Notes
    -----
    -   If the mapping passed is a :class:`colour.utilities.CanonicalMapping`
        class instance, then the lower, slugified and canonical keys are also
        used for matching.
    -   To honour the filterers ordering, the return value is a :class:`dict`
        class instance.

    Examples
    --------
    >>> class Element:
    ...     pass
    ...
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
        Filter given mapping with given name.

        Parameters
        ----------
        mapping
            Mapping to filter.
        name
            Name for given mapping elements.

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

        return {
            lookup.first_key_from_value(element): element
            for element in elements
        }

    names = [str(names)] if is_string(names) else names

    filtered_mapping = {}

    for filterer in names:
        filtered_mapping.update(filter_mapping_with_name(mapping, filterer))

    return filtered_mapping


def first_item(a: Iterable) -> Any:
    """
    Return the first item of given iterable.

    Parameters
    ----------
    a
        Iterable to get the first item from.

    Returns
    -------
    :class:`object`

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
        Optional definition copy name.

    Returns
    -------
    Callable
        Definition copy.
    """

    copy = types.FunctionType(
        definition.__code__,
        definition.__globals__,
        str(name or definition.__name__),
        definition.__defaults__,
        definition.__closure__,
    )
    copy.__dict__.update(definition.__dict__)

    return copy


def validate_method(
    method: str,
    valid_methods: Sequence | Mapping,
    message: str = '"{0}" method is invalid, it must be one of {1}!',
) -> str:
    """
    Validate whether given method exists in the given valid methods and
    returns the method lower cased.

    Parameters
    ----------
    method
        Method to validate.
    valid_methods
        Valid methods.
    message
        Message for the exception.

    Returns
    -------
    :class:`str`
        Method lower cased.

    Raises
    ------
    :class:`ValueError`
         If the method does not exist.

    Examples
    --------
    >>> validate_method("Valid", ["Valid", "Yes", "Ok"])
    'valid'
    """

    valid_methods = [str(valid_method) for valid_method in valid_methods]

    method_lower = method.lower()
    if method_lower not in [
        valid_method.lower() for valid_method in valid_methods
    ]:
        raise ValueError(message.format(method, valid_methods))

    return method_lower


T = TypeVar("T")


def optional(value: T | None, default: T) -> T:
    """
    Handle optional argument value by providing a default value.

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
    else:
        return value


def slugify(object_: Any, allow_unicode: bool = False) -> str:
    """
    Generate a *SEO* friendly and human-readable slug from given object.

    Convert to ASCII if ``allow_unicode`` is *False*. Convert spaces or
    repeated dashes to single dashes. Remove characters that aren't
    alphanumerics, underscores, or hyphens. Convert to lowercase. Also strip
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
    >>> slugify(
    ...     " Jack & Jill like numbers 1,2,3 and 4 and silly characters ?%.$!/"
    ... )
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
