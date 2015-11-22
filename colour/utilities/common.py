#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Utilities
================

Defines common utilities objects that don't fall in any specific category.
"""

from __future__ import division, unicode_literals

import functools
import numpy as np
import warnings

from colour.constants import INTEGER_THRESHOLD

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['handle_numpy_errors',
           'ignore_numpy_errors',
           'raise_numpy_errors',
           'print_numpy_errors',
           'warn_numpy_errors',
           'ignore_python_warnings',
           'batch',
           'is_openimageio_installed',
           'is_scipy_installed',
           'is_iterable',
           'is_string',
           'is_numeric',
           'is_integer']


def handle_numpy_errors(**kwargs):
    """
    Decorator for handling *Numpy* errors.

    Parameters
    ----------
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    object

    References
    ----------
    .. [1]  Kienzle, P., Patel, N., & Krycka, J. (2011). refl1d.numpyerrors -
            Refl1D v0.6.19 documentation. Retrieved January 30, 2015, from
            http://www.reflectometry.org/danse/docs/refl1d/_modules/\
refl1d/numpyerrors.html

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

    try:
        import OpenImageIO  # noqa

        return True
    except ImportError as error:
        if raise_exception:
            raise ImportError(('"OpenImageIO" related Api features '
                               'are not available: "{0}".').format(error))
        return False


def is_scipy_installed(raise_exception=False):
    """
    Returns if *scipy* is installed and available.

    Parameters
    ----------
    raise_exception : bool
        Raise exception if *scipy* is unavailable.

    Returns
    -------
    bool
        Is *scipy* installed.

    Raises
    ------
    ImportError
        If *scipy* is not installed.
    """

    try:
        # Importing *scipy* Api features used in *Colour*.
        import scipy.interpolate  # noqa
        import scipy.ndimage  # noqa
        import scipy.spatial  # noqa

        return True
    except ImportError as error:
        if raise_exception:
            raise ImportError(('"scipy" or specific "scipy" Api features '
                               'are not available: "{0}".').format(error))
        return False


def is_iterable(x):
    """
    Returns if given :math:`x` variable is iterable.

    Parameters
    ----------
    x : object
        Variable to check the iterability.

    Returns
    -------
    bool
        :math:`x` variable iterability.

    Examples
    --------
    >>> is_iterable([1, 2, 3])
    True
    >>> is_iterable(1)
    False
    """

    try:
        for _ in x:
            break
        return True
    except TypeError:
        return False


def is_string(data):
    """
    Returns if given data is a *string_like* variable.

    Parameters
    ----------
    data : object
        Data to test.

    Returns
    -------
    bool
        Is *string_like* variable.

    Examples
    --------
    >>> is_string('I`m a string!')
    True
    >>> is_string(['I`m a string!'])
    False
    """

    return True if isinstance(data, basestring) else False  # noqa


def is_numeric(x):
    """
    Returns if given :math:`x` variable is a number.

    Parameters
    ----------
    x : object
        Variable to check.

    Returns
    -------
    bool
        Is :math:`x` variable a number.

    See Also
    --------
    is_integer

    Examples
    --------
    >>> is_numeric(1)
    True
    >>> is_numeric((1,))
    False
    """

    return isinstance(x, (int, float, complex,
                          np.integer, np.floating, np.complex))


def is_integer(x):
    """
    Returns if given :math:`x` variable is an integer under given threshold.

    Parameters
    ----------
    x : object
        Variable to check.

    Returns
    -------
    bool
        Is :math:`x` variable an integer.

    Notes
    -----
    -   The determination threshold is defined by the
        :attr:`colour.algebra.common.INTEGER_THRESHOLD` attribute.

    See Also
    --------
    is_numeric

    Examples
    --------
    >>> is_integer(1)
    True
    >>> is_integer(1.01)
    False
    """

    return abs(x - round(x)) <= INTEGER_THRESHOLD
