# -*- coding: utf-8 -*-
"""
Common Utilities
================

Defines common algebra utilities objects that don't fall in any specific
category:

-   :func:`colour.algebra.spow`: Safe (symmetrical) power.
-   :func:`colour.algebra.smoothstep_function`: *Smoothstep* sigmoid-like
    function.
"""

from __future__ import division, unicode_literals

import functools
import colour.ndarray as np

from colour.utilities import as_float_array, as_float

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'is_spow_enabled', 'set_spow_enable', 'spow_enable', 'spow',
    'smoothstep_function'
]

_SPOW_ENABLED = True
"""
Global variable storing the current *Colour* safe / symmetrical power function
enabled state.

_SPOW_ENABLED : bool
"""


def is_spow_enabled():
    """
    Returns whether *Colour* safe / symmetrical power function is enabled.

    Returns
    -------
    bool
        Whether *Colour* safe / symmetrical power function is enabled.

    Examples
    --------
    >>> with spow_enable(False):
    ...     is_spow_enabled()
    False
    >>> with spow_enable(True):
    ...     is_spow_enabled()
    True
    """

    return _SPOW_ENABLED


def set_spow_enable(enable):
    """
    Sets *Colour* safe / symmetrical power function enabled state.

    Parameters
    ----------
    enable : bool
        Whether to enable *Colour* safe / symmetrical power function.

    Examples
    --------
    >>> with spow_enable(is_spow_enabled()):
    ...     print(is_spow_enabled())
    ...     set_spow_enable(False)
    ...     print(is_spow_enabled())
    True
    False
    """

    global _SPOW_ENABLED

    _SPOW_ENABLED = enable


class spow_enable(object):
    """
    A context manager and decorator temporarily setting *Colour* safe /
    symmetrical power function enabled state.

    Parameters
    ----------
    enable : bool
        Whether to enable or disable *Colour* safe / symmetrical power
        function.
    """

    def __init__(self, enable):
        self._enable = enable
        self._previous_state = is_spow_enabled()

    def __enter__(self):
        """
        Called upon entering the context manager and decorator.
        """

        set_spow_enable(self._enable)

        return self

    def __exit__(self, *args):
        """
        Called upon exiting the context manager and decorator.
        """

        set_spow_enable(self._previous_state)

    def __call__(self, function):
        """
        Calls the wrapped definition.
        """

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with self:
                return function(*args, **kwargs)

        return wrapper


def spow(a, p):
    """
    Raises given array :math:`a` to the power :math:`p` as follows:
    :math:`sign(a) * |a|^p`.

    This definition avoids NaNs generation when array :math:`a` is negative and
    the power :math:`p` is fractional. This behaviour can be enabled or
    disabled with the :func:`colour.algebra.set_spow_enable` definition or with
    the :func:`spow_enable` context manager.

    Parameters
    ----------------
    a : numeric or array_like
        Array :math:`a`.
    p : numeric or array_like
        Power :math:`p`.

    Returns
    -------
    numeric or ndarray
        Array :math:`a` safely raised to the power :math:`p`.

    Examples
    --------
    >>> np.power(-2, 0.15)
    nan
    >>> spow(-2, 0.15)  # doctest: +ELLIPSIS
    -1.1095694...
    >>> spow(0, 0)
    0.0
    """

    if not _SPOW_ENABLED:
        return np.power(a, p)

    a = np.atleast_1d(a)
    p = as_float_array(p)

    a_p = np.sign(a) * np.abs(a) ** p

    a_p[np.isnan(a_p)] = 0

    return as_float(a_p)


def smoothstep_function(x, a=0, b=1, clip=False):
    """
    Evaluates the *smoothstep* sigmoid-like function on array :math:`x`.

    Parameters
    ----------
    x : numeric or array_like
        Array :math:`x`.
    a : numeric, optional
        Low input domain limit, i.e. the left edge.
    b : numeric, optional
        High input domain limit, i.e. the right edge.
    clip : bool, optional
        Whether to scale, bias and clip input values to domain [0, 1].

    Returns
    -------
    array_like
        Array :math:`x` after *smoothstep* sigmoid-like function evaluation.

    Examples
    --------
    >>> x = np.linspace(-2, 2, 5)
    >>> smoothstep_function(x, -2, 2, clip=True)
    array([ 0.     ,  0.15625,  0.5    ,  0.84375,  1.     ])
    """

    x = as_float_array(x)

    i = np.clip((x - a) / (b - a), 0, 1) if clip else x

    return (i ** 2) * (3 - 2 * i)
