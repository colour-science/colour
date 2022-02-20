"""
Common Utilities
================

Defines the common utilities objects that don't fall in any specific category.

References
----------
-   :cite:`Kienzle2011a` : Kienzle, P., Patel, N., & Krycka, J. (2011).
    refl1d.numpyerrors - Refl1D v0.6.19 documentation. Retrieved January 30,
    2015, from
    http://www.reflectometry.org/danse/docs/refl1d/_modules/refl1d/\
numpyerrors.html
"""

from __future__ import annotations

import inspect
import multiprocessing
import multiprocessing.pool
import functools
import numpy as np
import re
import types
import warnings
from contextlib import contextmanager
from copy import copy
from pprint import pformat

from colour.constants import INTEGER_THRESHOLD
from colour.hints import (
    Any,
    Boolean,
    Callable,
    Dict,
    Generator,
    Integer,
    Iterable,
    Literal,
    Mapping,
    Optional,
    RegexFlag,
    Sequence,
    TypeVar,
    Union,
)
from colour.utilities import CaseInsensitiveMapping, Lookup

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
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
    "is_matplotlib_installed",
    "is_networkx_installed",
    "is_opencolorio_installed",
    "is_openimageio_installed",
    "is_pandas_installed",
    "is_sklearn_installed",
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
]


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
    >>> cache_a = cache_registry.register_cache('Cache A')
    >>> cache_a['Foo'] = 'Bar'
    >>> cache_b = cache_registry.register_cache('Cache B')
    >>> cache_b['John'] = 'Doe'
    >>> cache_b['Luke'] = 'Skywalker'
    >>> print(cache_registry)
    {'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}
    >>> cache_registry.clear_cache('Cache A')
    >>> print(cache_registry)
    {'Cache A': '0 item(s)', 'Cache B': '2 item(s)'}
    >>> cache_registry.unregister_cache('Cache B')
    >>> print(cache_registry)
    {'Cache A': '0 item(s)'}
    >>> print(cache_b)
    {}
    """

    def __init__(self):
        self._registry = {}

    @property
    def registry(self) -> Dict:
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

    def register_cache(self, name: str) -> Dict:
        """
        Register a new cache with given name in the registry.

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
        >>> cache_a = cache_registry.register_cache('Cache A')
        >>> cache_a['Foo'] = 'Bar'
        >>> cache_b = cache_registry.register_cache('Cache B')
        >>> cache_b['John'] = 'Doe'
        >>> cache_b['Luke'] = 'Skywalker'
        >>> print(cache_registry)
        {'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}
        """

        self._registry[name] = {}

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
        >>> cache_a = cache_registry.register_cache('Cache A')
        >>> cache_a['Foo'] = 'Bar'
        >>> cache_b = cache_registry.register_cache('Cache B')
        >>> cache_b['John'] = 'Doe'
        >>> cache_b['Luke'] = 'Skywalker'
        >>> print(cache_registry)
        {'Cache A': '1 item(s)', 'Cache B': '2 item(s)'}
        >>> cache_registry.unregister_cache('Cache B')
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
        >>> cache_a = cache_registry.register_cache('Cache A')
        >>> cache_a['Foo'] = 'Bar'
        >>> print(cache_registry)
        {'Cache A': '1 item(s)'}
        >>> cache_registry.clear_cache('Cache A')
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
        >>> cache_a = cache_registry.register_cache('Cache A')
        >>> cache_a['Foo'] = 'Bar'
        >>> cache_b = cache_registry.register_cache('Cache B')
        >>> cache_b['John'] = 'Doe'
        >>> cache_b['Luke'] = 'Skywalker'
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
    >>> @handle_numpy_errors(all='ignore')
    ... def f():
    ...     1 / numpy.zeros(3)
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
    ...     warnings.warn('This is an ignored warning!')
    >>> f()
    """

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrap given function."""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return function(*args, **kwargs)

    return wrapper


def attest(condition: Boolean, message: str = ""):
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


def batch(sequence: Sequence, k: Union[Integer, Literal[3]] = 3) -> Generator:
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


_MULTIPROCESSING_ENABLED: Boolean = True
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
    >>> with multiprocessing_pool() as pool:
    ...     pool.map(partial(_add, b=2), range(10))
    ... # doctest: +SKIP
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """

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

        def __init__(self, *args: Any, **kwargs: Any):
            pass

        def map(self, func, iterable, chunksize=None):
            """Apply given function to each element of given iterable."""

            return [func(a) for a in iterable]

        def terminate(self):
            """Terminate the process."""

            pass

    kwargs["initializer"] = _initializer
    kwargs["initargs"] = ({"scale": get_domain_range_scale()},)

    pool_factory: Callable
    if _MULTIPROCESSING_ENABLED:
        pool_factory = multiprocessing.Pool
    else:
        pool_factory = _DummyPool

    pool = pool_factory(*args, **kwargs)

    try:
        yield pool
    finally:
        pool.terminate()


def is_matplotlib_installed(raise_exception: Boolean = False) -> Boolean:
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
        # pylint: disable=W0612
        import matplotlib  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"Matplotlib" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            )

        return False


def is_networkx_installed(raise_exception: Boolean = False) -> Boolean:
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
        # pylint: disable=W0612
        import networkx  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"NetworkX" related API features, e.g. the automatic colour '
                f'conversion graph, are not available: "{error}".\nPlease refer '
                "to the installation guide for more information: "
                "https://www.colour-science.org/installation-guide/"
            )

        return False


def is_opencolorio_installed(raise_exception: Boolean = False) -> Boolean:
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
        # pylint: disable=W0612
        import PyOpenColorIO  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"OpenColorIO" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            )

        return False


def is_openimageio_installed(raise_exception: Boolean = False) -> Boolean:
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
        # pylint: disable=W0612
        import OpenImageIO  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"OpenImageIO" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            )

        return False


def is_pandas_installed(raise_exception: Boolean = False) -> Boolean:
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
        # pylint: disable=W0612
        import pandas  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                f'"Pandas" related API features are not available: "{error}".\n'
                "See the installation guide for more information: "
                "https://www.colour-science.org/installation-guide/"
            )

        return False


def is_sklearn_installed(raise_exception: Boolean = False) -> Boolean:
    """
    Return whether *Scikit-Learn* (sklearn) is installed and available.

    Parameters
    ----------
    raise_exception
        Whether to raise an exception if *Scikit-Learn* (sklearn) is
        unavailable.

    Returns
    -------
    :class:`bool`
        Whether *Scikit- isLearn* (sklearn) installed.

    Raises
    ------
    :class:`ImportError`
        If *Scikit-Learn* (sklearn) is not installed.
    """

    try:  # pragma: no cover
        # pylint: disable=W0612
        import sklearn  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"Scikit-Learn" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            )

        return False


def is_tqdm_installed(raise_exception: Boolean = False) -> Boolean:
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
        # pylint: disable=W0612
        import tqdm  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                f'"tqdm" related API features are not available: "{error}".\n'
                "See the installation guide for more information: "
                "https://www.colour-science.org/installation-guide/"
            )

        return False


def is_trimesh_installed(raise_exception: Boolean = False) -> Boolean:
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
        # pylint: disable=W0612
        import trimesh  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(
                '"Trimesh" related API features are not available: '
                f'"{error}".\nSee the installation guide for more information: '
                "https://www.colour-science.org/installation-guide/"
            )

        return False


_REQUIREMENTS_TO_CALLABLE: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "Matplotlib": is_matplotlib_installed,
        "NetworkX": is_networkx_installed,
        "OpenColorIO": is_opencolorio_installed,
        "OpenImageIO": is_openimageio_installed,
        "Pandas": is_pandas_installed,
        "Scikit-Learn": is_sklearn_installed,
        "tqdm": is_tqdm_installed,
        "trimesh": is_trimesh_installed,
    }
)
"""
Mapping of requirements to their respective callables.

_REQUIREMENTS_TO_CALLABLE
    **{'Matplotlib', 'NetworkX', 'OpenColorIO', 'OpenImageIO', 'Pandas',
    'Scikit-Learn', 'tqdm', 'trimesh'}**
"""


def required(
    *requirements: Literal[
        "Matplotlib",
        "NetworkX",
        "OpenColorIO",
        "OpenImageIO",
        "Pandas",
        "Scikit-Learn",
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


def is_iterable(a: Any) -> Boolean:
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

    return is_string(a) or (True if getattr(a, "__iter__", False) else False)


def is_string(a: Any) -> Boolean:
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

    return True if isinstance(a, str) else False


def is_numeric(a: Any) -> Boolean:
    """
    Return whether given variable :math:`a` is a :class:`Number`-like
    variable.

    Parameters
    ----------
    a
        Variable :math:`a` to test.

    Returns
    -------
    :class:`bool`
        Whether variable :math:`a` is a :class:`Number`-like variable.

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
            np.floating,
        ),
    )


def is_integer(a: Any) -> Boolean:
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


def is_sibling(element: Any, mapping: Mapping) -> Boolean:
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


def filter_kwargs(function: Callable, **kwargs: Any) -> Dict:
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


def filter_mapping(
    mapping: Mapping,
    filterers: Union[str, Sequence[str]],
    anchors: Boolean = True,
    flags: Union[Integer, RegexFlag] = re.IGNORECASE,
) -> Dict:
    """
    Filter given mapping with given filterers.

    Parameters
    ----------
    mapping
        Mapping to filter.
    filterers
        Filterer pattern for given mapping elements or a list of filterers.
    anchors
        Whether to use Regex line anchors, i.e. *^* and *$* are added,
        surrounding the filterer pattern.
    flags
        Regex flags.

    Returns
    -------
    dict
        Filtered mapping elements.

    Notes
    -----
    -   To honour the filterers ordering, the return value is an
        :class:`dict` class instance.

    Examples
    --------
    >>> class Element:
    ...     pass
    >>> mapping = {
    ...     'Element A': Element(),
    ...     'Element B': Element(),
    ...     'Element C': Element(),
    ...     'Not Element C': Element(),
    ... }
    >>> filter_mapping(mapping, '\\w+\\s+A')  # doctest: +ELLIPSIS
    {'Element A': <colour.utilities.common.Element object at 0x...>}
    >>> sorted(filter_mapping(mapping, 'Element.*'))
    ['Element A', 'Element B', 'Element C']
    """

    def filter_mapping_with_filter(
        mapping: Mapping,
        filterer: str,
        anchors: Boolean = True,
        flags: Union[Integer, RegexFlag] = re.IGNORECASE,
    ) -> Dict:
        """
        Filter given mapping with given filterer.

        Parameters
        ----------
        mapping
            Mapping to filter.
        filterer
            Filterer pattern for given mapping elements.
        anchors
            Whether to use Regex line anchors, i.e. *^* and *$* are added,
            surrounding the filterer pattern.
        flags
            Regex flags.

        Returns
        -------
        dict
            Filtered mapping elements.
        """

        if anchors:
            filterer = f"^{filterer}$"
            filterer = filterer.replace("^^", "^").replace("$$", "$")

        elements = [
            mapping[element]
            for element in mapping
            if re.match(filterer, element, flags)
        ]

        lookup = Lookup(mapping)

        return {
            lookup.first_key_from_value(element): element
            for element in elements
        }

    filterers = [str(filterers)] if is_string(filterers) else filterers

    filtered_mapping = {}

    for filterer in filterers:
        filtered_mapping.update(
            filter_mapping_with_filter(mapping, filterer, anchors, flags)
        )

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


def copy_definition(
    definition: Callable, name: Optional[str] = None
) -> Callable:
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
        definition.__globals__,  # type: ignore[attr-defined]
        str(name or definition.__name__),
        definition.__defaults__,  # type: ignore[attr-defined]
        definition.__closure__,  # type: ignore[attr-defined]
    )
    copy.__dict__.update(definition.__dict__)

    return copy


def validate_method(
    method: str,
    valid_methods: Union[Sequence, Mapping],
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
    >>> validate_method('Valid', ['Valid', 'Yes', 'Ok'])
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


def optional(value: Optional[T], default: T) -> T:
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
    >>> optional('Foo', 'Bar')
    'Foo'
    >>> optional(None, 'Bar')
    'Bar'
    """

    if value is None:
        return default
    else:
        return value
