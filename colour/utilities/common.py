# -*- coding: utf-8 -*-
"""
Common Utilities
================

Defines common utilities objects that don't fall in any specific category.

References
----------
-   :cite:`Kienzle2011a` : Kienzle, P., Patel, N., & Krycka, J. (2011).
    refl1d.numpyerrors - Refl1D v0.6.19 documentation. Retrieved January 30,
    2015, from http://www.reflectometry.org/danse/docs/refl1d/_modules/\
refl1d/numpyerrors.html
"""

from __future__ import division, unicode_literals

import inspect
import multiprocessing
import multiprocessing.pool
import functools
import numpy as np
import re
import six
import warnings
from contextlib import contextmanager
from collections import OrderedDict
from copy import copy
from six import integer_types, string_types

from colour.constants import INTEGER_THRESHOLD, DEFAULT_FLOAT_DTYPE
from colour.utilities import Lookup

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'handle_numpy_errors', 'ignore_numpy_errors', 'raise_numpy_errors',
    'print_numpy_errors', 'warn_numpy_errors', 'ignore_python_warnings',
    'batch', 'disable_multiprocessing', 'multiprocessing_pool',
    'is_matplotlib_installed', 'is_networkx_installed',
    'is_openimageio_installed', 'is_pandas_installed', 'is_iterable',
    'is_string', 'is_numeric', 'is_integer', 'is_sibling', 'filter_kwargs',
    'filter_mapping', 'first_item', 'get_domain_range_scale',
    'set_domain_range_scale', 'domain_range_scale', 'to_domain_1',
    'to_domain_10', 'to_domain_100', 'to_domain_degrees', 'to_domain_int',
    'from_range_1', 'from_range_10', 'from_range_100', 'from_range_degrees',
    'from_range_int'
]


def handle_numpy_errors(**kwargs):
    """
    Decorator for handling *Numpy* errors.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    object

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

    def wrapper(function):
        """
        Wrapper for given function.
        """

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            """
            Wrapped function.
            """

            with context:
                return function(*args, **kwargs)

        return wrapped

    return wrapper


ignore_numpy_errors = handle_numpy_errors(all='ignore')
raise_numpy_errors = handle_numpy_errors(all='raise')
print_numpy_errors = handle_numpy_errors(all='print')
warn_numpy_errors = handle_numpy_errors(all='warn')


def ignore_python_warnings(function):
    """
    Decorator for ignoring *Python* warnings.

    Parameters
    ----------
    function : object
        Function to decorate.

    Returns
    -------
    object

    Examples
    --------
    >>> @ignore_python_warnings
    ... def f():
    ...     warnings.warn('This is an ignored warning!')
    >>> f()
    """

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        """
        Wrapped function.
        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            return function(*args, **kwargs)

    return wrapped


def batch(iterable, k=3):
    """
    Returns a batch generator from given iterable.

    Parameters
    ----------
    iterable : iterable
        Iterable to create batches from.
    k : integer
        Batches size.

    Returns
    -------
    bool
        Is *string_like* variable.

    Examples
    --------
    >>> batch(tuple(range(10)))  # doctest: +ELLIPSIS
    <generator object batch at 0x...>
    """

    for i in range(0, len(iterable), k):
        yield iterable[i:i + k]


_MULTIPROCESSING_ENABLED = True
"""
Whether *Colour* multiprocessing is enabled.

_MULTIPROCESSING_ENABLED : bool
"""


class disable_multiprocessing(object):
    """
    A context manager and decorator temporarily disabling *Colour*
    multiprocessing.
    """

    def __enter__(self):
        """
        Called upon entering the context manager and decorator.
        """

        global _MULTIPROCESSING_ENABLED

        _MULTIPROCESSING_ENABLED = False

        return self

    def __exit__(self, *args):
        """
        Called upon exiting the context manager and decorator.
        """

        global _MULTIPROCESSING_ENABLED

        _MULTIPROCESSING_ENABLED = True

    def __call__(self, function):
        """
        Calls the wrapped definition.
        """

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with self:
                return function(*args, **kwargs)

        return wrapper


def _initializer(kwargs):
    """
    Initializer for the multiprocessing pool. It is mainly use to ensure that
    processes on *Windows* correctly inherit from the current domain-range
    scale.

    Parameters
    ----------
    kwargs : dict
        Initialisation arguments.
    """

    global _DOMAIN_RANGE_SCALE

    # NOTE: No coverage information is available as this code is executed in
    # sub-processes.
    _DOMAIN_RANGE_SCALE = kwargs.get('scale', 'reference')  # pragma: no cover


@contextmanager
def multiprocessing_pool(*args, **kwargs):
    """
    A context manager providing a multiprocessing pool.

    Other Parameters
    ----------------
    \\*args : list, optional
        Arguments.
    \\**kwargs : dict, optional
        Keywords arguments.

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

    class _DummyPool(object):
        """
        A dummy multiprocessing pool that does not perform multiprocessing.

        Other Parameters
        ----------------
        \\*args : list, optional
            Arguments.
        \\**kwargs : dict, optional
            Keywords arguments.
        """

        def __init__(self, *args, **kwargs):
            pass

        def map(self, func, iterable, chunksize=None):
            """
            Applies given function to each element of given iterable.
            """

            return [func(a) for a in iterable]

        def terminate(self):
            """
            Terminate the process.
            """

            pass

    kwargs['initializer'] = _initializer
    kwargs['initargs'] = ({'scale': get_domain_range_scale()}, )

    if _MULTIPROCESSING_ENABLED:
        pool_factory = multiprocessing.Pool
    else:
        pool_factory = _DummyPool

    pool = pool_factory(*args, **kwargs)

    try:
        yield pool
    finally:
        pool.terminate()


def is_matplotlib_installed(raise_exception=False):
    """
    Returns if *Matplotlib* is installed and available.

    Parameters
    ----------
    raise_exception : bool
        Raise exception if *Matplotlib* is unavailable.

    Returns
    -------
    bool
        Is *Matplotlib* installed.

    Raises
    ------
    ImportError
        If *Matplotlib* is not installed.
    """

    try:  # pragma: no cover
        import matplotlib  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(('"Matplotlib" related API features '
                               'are not available: "{0}".').format(error))
        return False


def is_networkx_installed(raise_exception=False):
    """
    Returns if *NetworkX* is installed and available.

    Parameters
    ----------
    raise_exception : bool
        Raise exception if *NetworkX* is unavailable.

    Returns
    -------
    bool
        Is *NetworkX* installed.

    Raises
    ------
    ImportError
        If *NetworkX* is not installed.
    """

    try:  # pragma: no cover
        import networkx  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(('"NetworkX" related API features, e.g. '
                               'the automatic colour conversion graph, '
                               'are not available: "{0}".').format(error))
        return False


def is_openimageio_installed(raise_exception=False):
    """
    Returns if *OpenImageIO* is installed and available.

    Parameters
    ----------
    raise_exception : bool
        Raise exception if *OpenImageIO* is unavailable.

    Returns
    -------
    bool
        Is *OpenImageIO* installed.

    Raises
    ------
    ImportError
        If *OpenImageIO* is not installed.
    """

    try:  # pragma: no cover
        import OpenImageIO  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(('"OpenImageIO" related API features '
                               'are not available: "{0}".').format(error))
        return False


def is_pandas_installed(raise_exception=False):
    """
    Returns if *Pandas* is installed and available.

    Parameters
    ----------
    raise_exception : bool
        Raise exception if *Pandas* is unavailable.

    Returns
    -------
    bool
        Is *Pandas* installed.

    Raises
    ------
    ImportError
        If *Pandas* is not installed.
    """

    try:  # pragma: no cover
        import pandas  # noqa

        return True
    except ImportError as error:  # pragma: no cover
        if raise_exception:
            raise ImportError(('"Pandas" related API features '
                               'are not available: "{0}".').format(error))
        return False


def is_iterable(a):
    """
    Returns if given :math:`a` variable is iterable.

    Parameters
    ----------
    a : object
        Variable to check the iterability.

    Returns
    -------
    bool
        :math:`a` variable iterability.

    Examples
    --------
    >>> is_iterable([1, 2, 3])
    True
    >>> is_iterable(1)
    False
    """

    return is_string(a) or (True if getattr(a, '__iter__', False) else False)


def is_string(a):
    """
    Returns if given :math:`a` variable is a *string* like variable.

    Parameters
    ----------
    a : object
        Data to test.

    Returns
    -------
    bool
        Is :math:`a` variable a *string* like variable.

    Examples
    --------
    >>> is_string("I'm a string!")
    True
    >>> is_string(["I'm a string!"])
    False
    """

    return True if isinstance(a, string_types) else False


def is_numeric(a):
    """
    Returns if given :math:`a` variable is a number.

    Parameters
    ----------
    a : object
        Variable to check.

    Returns
    -------
    bool
        Is :math:`a` variable a number.

    Examples
    --------
    >>> is_numeric(1)
    True
    >>> is_numeric((1,))
    False
    """

    return isinstance(
        a,
        tuple(
            list(integer_types) +
            [float, complex, np.integer, np.floating, np.complex]))


def is_integer(a):
    """
    Returns if given :math:`a` variable is an integer under given threshold.

    Parameters
    ----------
    a : object
        Variable to check.

    Returns
    -------
    bool
        Is :math:`a` variable an integer.

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

    return abs(a - round(a)) <= INTEGER_THRESHOLD


def is_sibling(element, mapping):
    """
    Returns whether given element type is present in given mapping types.

    Parameters
    ----------
    element : object
        Element to check if its type is present in the mapping types.
    mapping : dict
        Mapping.

    Returns
    -------
    bool
        Whether given element type is present in given mapping types.
    """

    return isinstance(
        element, tuple(set(type(element) for element in mapping.values())))


def filter_kwargs(function, **kwargs):
    """
    Filters keyword arguments incompatible with the given function signature.

    Parameters
    ----------
    function : callable
        Callable to filter the incompatible keyword arguments.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    dict
        Filtered keyword arguments.

    Warnings
    --------
    Python 2.7 does not support inspecting the signature of *partial*
    functions, this could cause unexpected behaviour when using this
    definition.

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

    # TODO: Remove when dropping Python 2.7.
    if six.PY2:  # pragma: no cover
        try:
            args, _varargs, _keywords, _defaults = inspect.getargspec(function)
        except (TypeError, ValueError):
            return {}
    else:  # pragma: no cover
        try:
            args = list(inspect.signature(function).parameters.keys())
        except ValueError:
            return {}

    args = set(kwargs.keys()) - set(args)
    for key in args:
        kwargs.pop(key)

    return kwargs


def filter_mapping(mapping, filterers, anchors=True, flags=re.IGNORECASE):
    """
    Filters given mapping with given filterers.

    Parameters
    ----------
    mapping : dict_like
        Mapping to filter.
    filterers : unicode or object or array_like
        Filterer pattern for given mapping elements or a list of filterers.
    anchors : bool, optional
        Whether to use Regex line anchors, i.e. *^* and *$* are added,
        surrounding the filterer pattern.
    flags : int, optional
        Regex flags.

    Returns
    -------
    OrderedDict
        Filtered mapping elements.

    Notes
    -----
    -   To honour the filterers ordering, the return value is an
        :class:`OrderedDict` class instance.

    Examples
    --------
    >>> class Element(object):
    ...     pass
    >>> mapping = {
    ...     'Element A': Element(),
    ...     'Element B': Element(),
    ...     'Element C': Element(),
    ...     'Not Element C': Element(),
    ... }
    >>> # Doctests skip for Python 2.x compatibility.
    >>> filter_mapping(mapping, '\\w+\\s+A')  # doctest: +SKIP
    {u'Element A': <colour.utilities.common.Element object at 0x...>}
    >>> # Doctests skip for Python 2.x compatibility.
    >>> sorted(filter_mapping(mapping, 'Element.*'))  # doctest: +SKIP
    [u'Element A', u'Element B', u'Element C']
    """

    def filter_mapping_with_filter(mapping, filterer, anchors, flags):
        """
        Filters given mapping with given filterer.

        Parameters
        ----------
        mapping : dict_like
            Mapping to filter.
        filterer : unicode or object
            Filterer pattern for given mapping elements.
        anchors : bool, optional
            Whether to use Regex line anchors, i.e. *^* and *$* are added,
            surrounding the filterer pattern.
        flags : int, optional
            Regex flags.

        Returns
        -------
        OrderedDict
            Filtered mapping elements.
        """

        if anchors:
            filterer = '^{0}$'.format(filterer)
            filterer = filterer.replace('^^', '^').replace('$$', '$')

        elements = [
            mapping[element] for element in mapping
            if re.match(filterer, element, flags)
        ]

        lookup = Lookup(mapping)

        return OrderedDict((lookup.first_key_from_value(element), element)
                           for element in elements)

    if is_string(filterers):
        filterers = [filterers]

    filtered_mapping = OrderedDict()

    for filterer in filterers:
        filtered_mapping.update(
            filter_mapping_with_filter(mapping, filterer, anchors, flags))

    return filtered_mapping


def first_item(a):
    """
    Return the first item of an iterable.

    Parameters
    ----------
    a : object
        Iterable to get the first item from.

    Returns
    -------
    object

    Raises
    ------
    StopIteration
        If the iterable is empty.

    Examples
    --------
    >>> a = range(10)
    >>> first_item(a)
    0
    """

    return next(iter(a))


_DOMAIN_RANGE_SCALE = 'reference'
"""
Global variable storing the current *Colour* domain-range scale.

_DOMAIN_RANGE_SCALE : unicode
"""


def get_domain_range_scale():
    """
    Returns the current *Colour* domain-range scale. The following scales are
    available:

    -   **'Reference'**, the default *Colour* domain-range scale which varies
        depending on the referenced algorithm, e.g. [0, 1], [0, 10], [0, 100],
        [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is important to
        acknowledge that this is a soft normalisation and it is possible to
        use negative out of gamut values or high dynamic range data exceeding
        1.

    Returns
    -------
    unicode
        *Colour* domain-range scale.
    """

    return _DOMAIN_RANGE_SCALE


def set_domain_range_scale(scale='Reference'):
    """
    Sets the current *Colour* domain-range scale. The following scales are
    available:

    -   **'Reference'**, the default *Colour* domain-range scale which varies
        depending on the referenced algorithm, e.g. [0, 1], [0, 10], [0, 100],
        [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is important to
        acknowledge that this is a soft normalisation and it is possible to
        use negative out of gamut values or high dynamic range data exceeding
        1.

    Parameters
    ----------
    scale : unicode or int
        **{'Reference', '1'}**,
        *Colour* domain-range scale to set.
    """

    global _DOMAIN_RANGE_SCALE

    scale = str(scale).lower()
    valid = ('1', '100', 'reference', 'ignore')
    assert scale in valid, 'Scale must be one of "{0}".'.format(valid)

    _DOMAIN_RANGE_SCALE = scale


class domain_range_scale(object):
    """
    A context manager and decorator temporarily setting *Colour* domain-range
    scale. The following scales are available:

    -   **'Reference'**, the default *Colour* domain-range scale which varies
        depending on the referenced algorithm, e.g. [0, 1], [0, 10], [0, 100],
        [0, 255], etc...
    -   **'1'**, a domain-range scale normalised to [0, 1], it is important to
        acknowledge that this is a soft normalisation and it is possible to
        use negative out of gamut values or high dynamic range data exceeding
        1.

    Parameters
    ----------
    scale : unicode
        **{'Reference', '1'}**,
        *Colour* domain-range scale to set.
    """

    def __init__(self, scale):
        self._scale = scale
        self._previous_scale = get_domain_range_scale()

    def __enter__(self):
        """
        Called upon entering the context manager and decorator.
        """

        set_domain_range_scale(self._scale)

        return self

    def __exit__(self, *args):
        """
        Called upon exiting the context manager and decorator.
        """

        set_domain_range_scale(self._previous_scale)

    def __call__(self, function):
        """
        Calls the wrapped definition.
        """

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with self:
                return function(*args, **kwargs)

        return wrapper


def to_domain_1(a, scale_factor=100, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Scales given array :math:`a` to domain **'1'**. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'1'**, the
        definition is almost entirely by-passed and will just conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is divided by
        ``scale_factor``, typically 100.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale to domain **'1'**.
    scale_factor : numeric or array_like, optional
        Scale factor, usually *numeric* but can be an *array_like* if some
        axis need different scaling to be brought to domain **'1'**.
    dtype : object, optional
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    ndarray
        :math:`a` scaled to domain **'1'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_1(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_1(1)
    array(1.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_1(1)
    array(0.01)
    """

    a = np.asarray(a, dtype).copy()

    if _DOMAIN_RANGE_SCALE == '100':
        a /= scale_factor

    return a


def to_domain_10(a, scale_factor=10, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Scales given array :math:`a` to domain **'10'**, used by
    *Munsell Renotation System*. The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is almost entirely by-passed and will just conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 10.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        divided by ``scale_factor``, typically 10.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale to domain **'10'**.
    scale_factor : numeric or array_like, optional
        Scale factor, usually *numeric* but can be an *array_like* if some
        axis need different scaling to be brought to domain **'10'**.
    dtype : object, optional
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    ndarray
        :math:`a` scaled to domain **'10'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_10(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_10(1)
    array(10.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_10(1)
    array(0.1)
    """

    a = np.asarray(a, dtype).copy()

    if _DOMAIN_RANGE_SCALE == '1':
        a *= scale_factor

    if _DOMAIN_RANGE_SCALE == '100':
        a /= scale_factor

    return a


def to_domain_100(a, scale_factor=100, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Scales given array :math:`a` to domain **'100'**. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'100'**
        (currently unsupported private value only used for unit tests), the
        definition is almost entirely by-passed and will just conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 100.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale to domain **'100'**.
    scale_factor : numeric or array_like, optional
        Scale factor, usually *numeric* but can be an *array_like* if some
        axis need different scaling to be brought to domain **'100'**.
    dtype : object, optional
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    ndarray
        :math:`a` scaled to domain **'100'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_100(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_100(1)
    array(100.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_100(1)
    array(1.0)
    """

    a = np.asarray(a, dtype).copy()

    if _DOMAIN_RANGE_SCALE == '1':
        a *= scale_factor

    return a


def to_domain_degrees(a, scale_factor=360, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Scales given array :math:`a` to degrees domain. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is almost entirely by-passed and will just conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by ``scale_factor``, typically 360.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor`` / 100, typically 360 / 100.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale to degrees domain.
    scale_factor : numeric or array_like, optional
        Scale factor, usually *numeric* but can be an *array_like* if some
        axis need different scaling to be brought to degrees domain.
    dtype : object, optional
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    ndarray
        :math:`a` scaled to degrees domain.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_degrees(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_degrees(1)
    array(360.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_degrees(1)
    array(3.6)
    """

    a = np.asarray(a, dtype).copy()

    if _DOMAIN_RANGE_SCALE == '1':
        a *= scale_factor

    if _DOMAIN_RANGE_SCALE == '100':
        a *= scale_factor / 100

    return a


def to_domain_int(a, bit_depth=8, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Scales given array :math:`a` to int domain. The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is almost entirely by-passed and will just conveniently
        convert array :math:`a` to :class:`np.ndarray`.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        multiplied by :math:`2^{bit\\_depth} - 1`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by :math:`2^{bit\\_depth} - 1`.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale to int domain.
    bit_depth : numeric or array_like, optional
        Bit depth, usually *int* but can be an *array_like* if some axis need
        different scaling to be brought to int domain.
    dtype : object, optional
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    ndarray
        :math:`a` scaled to int domain.

    Notes
    -----
    -   To avoid precision issues and rounding, the scaling is performed on
        floating-point numbers.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     to_domain_int(1)
    array(1.0)

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     to_domain_int(1)
    array(255.0)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     to_domain_int(1)
    array(2.55)
    """

    a = np.asarray(a, dtype).copy()

    maximum_code_value = 2 ** bit_depth - 1
    if _DOMAIN_RANGE_SCALE == '1':
        a *= maximum_code_value

    if _DOMAIN_RANGE_SCALE == '100':
        a *= maximum_code_value / 100

    return a


def from_range_1(a, scale_factor=100):
    """
    Scales given array :math:`a` from range **'1'**. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'1'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is multiplied
        by ``scale_factor``, typically 100.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale from range **'1'**.
    scale_factor : numeric or array_like, optional
        Scale factor, usually *numeric* but can be an *array_like* if some
        axis need different scaling to be brought from range **'1'**.

    Returns
    -------
    ndarray
        :math:`a` scaled from range **'1'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_1(1)
    1

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_1(1)
    1

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_1(1)
    100
    """

    if _DOMAIN_RANGE_SCALE == '100':
        a *= scale_factor

    return a


def from_range_10(a, scale_factor=10):
    """
    Scales given array :math:`a` from range **'10'**, used by
    *Munsell Renotation System*. The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 10.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        multiplied by ``scale_factor``, typically 10.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale from range **'10'**.
    scale_factor : numeric or array_like, optional
        Scale factor, usually *numeric* but can be an *array_like* if some
        axis need different scaling to be brought from range **'10'**.

    Returns
    -------
    ndarray
        :math:`a` scaled from range **'10'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_10(1)
    1

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_10(1)
    0.1

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_10(1)
    10
    """

    if _DOMAIN_RANGE_SCALE == '1':
        a /= scale_factor

    if _DOMAIN_RANGE_SCALE == '100':
        a *= scale_factor

    return a


def from_range_100(a, scale_factor=100):
    """
    Scales given array :math:`a` from range **'100'**. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'** or **'100'**
        (currently unsupported private value only used for unit tests), the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 100.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale from range **'100'**.
    scale_factor : numeric or array_like, optional
        Scale factor, usually *numeric* but can be an *array_like* if some
        axis need different scaling to be brought from range **'100'**.

    Returns
    -------
    ndarray
        :math:`a` scaled from range **'100'**.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_100(1)
    1

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_100(1)
    0.01

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_100(1)
    1
    """

    if _DOMAIN_RANGE_SCALE == '1':
        a /= scale_factor

    return a


def from_range_degrees(a, scale_factor=360):
    """
    Scales given array :math:`a` from degrees range. The behaviour is as
    follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is
        divided by ``scale_factor``, typically 360.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is
        divided by ``scale_factor`` / 100, typically 360 / 100.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale from degrees range.
    scale_factor : numeric or array_like, optional
        Scale factor, usually *numeric* but can be an *array_like* if some
        axis need different scaling to be brought from degrees range.

    Returns
    -------
    ndarray
        :math:`a` scaled from degrees range.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_degrees(1)
    1

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_degrees(1)  # doctest: +ELLIPSIS
    0.0027777...

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_degrees(1)  # doctest: +ELLIPSIS
    0.2777777...
    """

    if _DOMAIN_RANGE_SCALE == '1':
        a /= scale_factor

    if _DOMAIN_RANGE_SCALE == '100':
        a /= scale_factor / 100

    return a


def from_range_int(a, bit_depth=8, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Scales given array :math:`a` from int range. The behaviour is as follows:

    -   If *Colour* domain-range scale is **'Reference'**, the
        definition is entirely by-passed.
    -   If *Colour* domain-range scale is **'1'**, array :math:`a` is converted
        to :class:`np.ndarray` and divided by :math:`2^{bit\\_depth} - 1`.
    -   If *Colour* domain-range scale is **'100'** (currently unsupported
        private value only used for unit tests), array :math:`a` is converted
        to :class:`np.ndarray` and divided by :math:`2^{bit\\_depth} - 1`.

    Parameters
    ----------
    a : array_like
        :math:`a` to scale from int range.
    bit_depth : numeric or array_like, optional
        Bit depth, usually *int* but can be an *array_like* if some axis need
        different scaling to be brought from int range.
    dtype : object, optional
        Data type used for the conversion to :class:`np.ndarray`.

    Returns
    -------
    ndarray
        :math:`a` scaled from int range.

    Notes
    -----
    -   To avoid precision issues and rounding, the scaling is performed on
        floating-point numbers.

    Examples
    --------
    With *Colour* domain-range scale set to **'Reference'**:

    >>> with domain_range_scale('Reference'):
    ...     from_range_int(1)
    1

    With *Colour* domain-range scale set to **'1'**:

    >>> with domain_range_scale('1'):
    ...     from_range_int(1)  # doctest: +ELLIPSIS
    array(0.0039215...)

    With *Colour* domain-range scale set to **'100'** (unsupported):

    >>> with domain_range_scale('100'):
    ...     from_range_int(1)  # doctest: +ELLIPSIS
    array(0.3921568...)
    """

    maximum_code_value = 2 ** bit_depth - 1
    if _DOMAIN_RANGE_SCALE == '1':
        a = np.asarray(a, dtype)
        a /= maximum_code_value

    if _DOMAIN_RANGE_SCALE == '100':
        a = np.asarray(a, dtype)
        a /= maximum_code_value / 100

    return a
