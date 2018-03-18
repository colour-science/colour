# -*- coding: utf-8 -*-
"""
Verbose
=======

Defines verbose related objects.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys
import traceback
import warnings
from contextlib import contextmanager
from itertools import chain
from operator import lt, le, gt, ge
from textwrap import TextWrapper
from warnings import filterwarnings, formatwarning, warn

from colour.utilities import is_numeric
from colour.utilities.documentation import DocstringInt

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ColourWarning', 'message_box', 'show_warning', 'warning',
    'filter_warnings', 'suppress_warnings', 'numpy_print_options',
    'inspect_domain', 'inspect_domain_1', 'inspect_domain_10',
    'inspect_domain_100', 'inspect_domain_int'
]


class ColourWarning(Warning):
    """
    This is the base class of *Colour* warnings. It is a subclass of
    :class:`Warning`.
    """

    DOMAIN_INSPECTION = DocstringInt(1)
    DOMAIN_INSPECTION.__doc__ = """
    Enables or disables domain inspection warnings.

    DOMAIN_INSPECTION : bool
    """


def message_box(message, width=79, padding=3):
    """
    Prints a message inside a box.

    Parameters
    ----------
    message : unicode
        Message to print.
    width : int, optional
        Message box width.
    padding : unicode
        Padding on each sides of the message.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> message = ('Lorem ipsum dolor sit amet, consectetur adipiscing elit, '
    ...     'sed do eiusmod tempor incididunt ut labore et dolore magna '
    ...     'aliqua.')
    >>> message_box(message, width=75)
    ===========================================================================
    *                                                                         *
    *   Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do       *
    *   eiusmod tempor incididunt ut labore et dolore magna aliqua.           *
    *                                                                         *
    ===========================================================================
    True
    >>> message_box(message, width=60)
    ============================================================
    *                                                          *
    *   Lorem ipsum dolor sit amet, consectetur adipiscing     *
    *   elit, sed do eiusmod tempor incididunt ut labore et    *
    *   dolore magna aliqua.                                   *
    *                                                          *
    ============================================================
    True
    >>> message_box(message, width=75, padding=16)
    ===========================================================================
    *                                                                         *
    *                Lorem ipsum dolor sit amet, consectetur                  *
    *                adipiscing elit, sed do eiusmod tempor                   *
    *                incididunt ut labore et dolore magna                     *
    *                aliqua.                                                  *
    *                                                                         *
    ===========================================================================
    True
    """

    ideal_width = width - padding * 2 - 2

    def inner(text):
        """
        Formats and pads inner text for the message box.
        """

        return '*{0}{1}{2}{0}*'.format(
            ' ' * padding, text, (' ' * (width - len(text) - padding * 2 - 2)))

    print('=' * width)
    print(inner(''))

    wrapper = TextWrapper(
        width=ideal_width, break_long_words=False, replace_whitespace=False)

    lines = [wrapper.wrap(line) for line in message.split("\n")]
    lines = [' ' if len(line) == 0 else line for line in lines]
    for line in chain(*lines):
        print(inner(line.expandtabs()))

    print(inner(''))
    print('=' * width)
    return True


def show_warning(message,
                 category,
                 path,
                 line,
                 file_=None,
                 code=None,
                 frame_range=(1, 2)):
    """
    Replaces :func:`warnings.showwarning` definition to allow traceback
    printing.

    Parameters
    ----------
    message : unicode
        Warning message.
    category : Warning
        :class:`Warning` sub-class.
    path : unicode
        File path to read the line at ``lineno`` from if ``line`` is None.
    line : int
        Line number to read the line at in ``filename`` if ``line`` is None.
    file_ : file, optional
        :class:`file` object to write the warning to, defaults to
        :attr:`sys.stderr` attribute.
    code : unicode, optional
        Source code to be included in the warning message.
    frame_range : array_like, optional
        Traceback frame range, i.e first frame and numbers of frame above it.
    """

    if file_ is None:
        file_ = sys.stderr
        if file_ is None:
            return

    try:
        # Generating a traceback to print useful warning origin.
        frame_in, frame_out = frame_range

        try:
            raise ZeroDivisionError
        except ZeroDivisionError:
            frame = sys.exc_info()[2].tb_frame.f_back
            while frame_in:
                frame = frame.f_back
                frame_in -= 1

        traceback.print_stack(frame, frame_out, file_)

        file_.write(formatwarning(message, category, path, line, code))
    except (IOError, UnicodeError):
        pass


warnings.showwarning = show_warning


def warning(*args, **kwargs):
    """
    Issues a warning.

    Other Parameters
    ----------------
    \*args : list, optional
        Arguments.
    \**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> warning('This is a warning!')  # doctest: +SKIP
    /Users/.../colour/utilities/verbose.py:132: UserWarning: This is a warning!
    """

    kwargs['category'] = ColourWarning
    warn(*args, **kwargs)

    return True


def filter_warnings(state=True, colour_warnings_only=True):
    """
    Filters *Colour* and also optionally overall Python warnings.

    Parameters
    ----------
    state : bool, optional
        Warnings filter state.
    colour_warnings_only : bool, optional
        Whether to only filter *Colour* warnings or also overall Python
        warnings.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    # Filtering *Colour* only warnings:
    >>> filter_warnings()
    True

    # Filtering *Colour* and also Python warnings:
    >>> filter_warnings(colour_warnings_only=False)
    True
    """

    if not hasattr(ColourWarning, '_DOMAIN_INSPECTION'):
        ColourWarning._DOMAIN_INSPECTION = ColourWarning.DOMAIN_INSPECTION

    if state:
        ColourWarning._DOMAIN_INSPECTION = ColourWarning.DOMAIN_INSPECTION
        ColourWarning.DOMAIN_INSPECTION = False
    else:
        ColourWarning.DOMAIN_INSPECTION = ColourWarning._DOMAIN_INSPECTION

    filterwarnings(
        'ignore' if state else 'default',
        category=ColourWarning if colour_warnings_only else Warning)

    return True


@contextmanager
def suppress_warnings(colour_warnings_only=True):
    """
    A context manager filtering *Colour* and also optionally overall Python
    warnings.

    Parameters
    ----------
    colour_warnings_only : bool, optional
        Whether to only filter *Colour* warnings or also overall Python
        warnings.
    """

    filters = warnings.filters
    show_warnings = warnings.showwarning

    filter_warnings(colour_warnings_only=colour_warnings_only)
    try:
        yield
    finally:
        warnings.filters = filters
        warnings.showwarning = show_warnings


@contextmanager
def numpy_print_options(*args, **kwargs):
    """
    A context manager implementing context changes to *Numpy* print behaviour.

    Other Parameters
    ----------------
    \*args : list, optional
        Arguments.
    \**kwargs : dict, optional
        Keywords arguments.

    Examples
    -------
    >>> np.array([np.pi])  # doctest: +ELLIPSIS
    array([ 3.1415926...])
    >>> with numpy_print_options(formatter={'float': '{:0.1f}'.format}):
    ...      np.array([np.pi])
    array([3.1])
    """

    options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**options)


def inspect_domain(a=None,
                   value=1,
                   ratio=0.9,
                   comparer=lt,
                   message=None,
                   frame_range=(3, 3)):
    """
    Inspects the domain of given numeric or array :math:`a` to check whether
    its value or a given ``ratio`` of its values compare to given ``value`` and
    emits a warning accordingly.

    The definition can be skipped by setting
    :attr:`colour.utilities.ColourWarning.DOMAIN_INSPECTION` attribute to
    `False`.

    Parameters
    ----------
    a : numeric or array_like, optional
        Numeric or array :math:`a` to inspect the domain of. For convenience,
        ``a`` can be set to `None` so that the definition returns directly.
    value : numeric, optional
        Value against which the comparison is performed.
    ratio : numeric, optional
        Ratio above which the warning is emitted.
    comparer : callable, optional
        Callable performing the comparison.
    message : unicode, optional
        Warning message to use instead of the default message.
    frame_range : array_like, optional
        Traceback frame range, i.e. first frame and numbers of frame above it.

    Returns
    -------
    numeric or array_like
        Inspected numeric or array :math:`a`.

    Examples
    --------
    >>> def domain_100(a):
    ...     inspect_domain(a)
    >>> domain_100(np.linspace(0, 1, 10))  # doctest: +SKIP
      File ".../docrunner.py", line 140, in __run
        compileflags, 1), test.globs)
      File "<doctest inspect_domain[1]>", line 1, in <module>
        domain_100(np.linspace(0, 1, 10))
      File "<doctest inspect_domain[0]>", line 2, in domain_100
        inspect_domain(a)
    /.../colour/utilities/verbose.py:206: ColourWarning:
    Unusual values encountered when inspecting domain: \
100% of values are less than value of 1.
      warn(*args, **kwargs)
    """

    if not ColourWarning.DOMAIN_INSPECTION:
        return a

    if a is None:
        return

    default_operator_mapping = {
        lt: 'less than',
        le: 'less or equal to',
        gt: 'greater than',
        ge: 'greater or equal to',
    }

    if message is None:
        message = ('\nUnusual values encountered when inspecting domain: '
                   '{unusual}% of values are {comparison} value of {value}.')

    def _show_warning(message, category, path, line, file_=None, code=None):
        """
        Temporary :func:`colour.utilities.show_warning` definition with
        different frame range.
        """

        return show_warning(message, category, path, line, file_, code,
                            frame_range)

    numeric = is_numeric(a)
    if isinstance(a, np.ndarray):
        if a.ndim == 0:
            numeric = True

    unusual = False
    if numeric:
        if comparer(a, value):
            unusual = True
    else:
        a = np.asarray(a)
        len_a, len_s = a.size, a[comparer(a, value)].size
        unusual = len_s / len_a >= ratio

    if unusual:
        formatter = {
            'comparison': default_operator_mapping[comparer],
            'ratio': ratio,
            'unusual': unusual * 100,
            'value': value
        }
        try:
            warnings.showwarning = _show_warning
            warning(message.format(**formatter))
        finally:
            warnings.showwarning = show_warning

    return a


def inspect_domain_1(a):
    """
    Checks if given numeric or array :math:`a` is normalised to domain [0, 1].

    Parameters
    ----------
    a : numeric or array_like, optional
        Numeric or array :math:`a` to inspect the domain of. For convenience,
        ``a`` can be set to `None` so that the definition returns directly.

    Returns
    -------
    numeric or array_like
        Inspected numeric or array :math:`a`.
    """

    return inspect_domain(
        a,
        comparer=gt,
        message=('\nUnusual values encountered when inspecting domain: '
                 '{unusual}% of values are {comparison} {value} however '
                 'they are usually normalised to domain [0, 1].\n'),
        frame_range=(4, 3))


def inspect_domain_10(a):
    """
    Checks if given numeric or array :math:`a` is normalised to domain [0, 10].

    Parameters
    ----------
    a : numeric or array_like, optional
        Numeric or array :math:`a` to inspect the domain of. For convenience,
        ``a`` can be set to `None` so that the definition returns directly.

    Returns
    -------
    numeric or array_like
        Inspected numeric or array :math:`a`.
    """

    return inspect_domain(
        a,
        message=('\nUnusual values encountered when inspecting domain: '
                 '{unusual}% of values are {comparison} {value} however '
                 'they are usually normalised to domain [0, 10].\n'),
        frame_range=(4, 3))


def inspect_domain_100(a):
    """
    Checks if given numeric or array :math:`a` is normalised to domain
    [0, 100].

    Parameters
    ----------
    a : numeric or array_like, optional
        Numeric or array :math:`a` to inspect the domain of. For convenience,
        ``a`` can be set to `None` so that the definition returns directly.

    Returns
    -------
    numeric or array_like
        Inspected numeric or array :math:`a`.
    """

    return inspect_domain(
        a,
        message=('\nUnusual values encountered when inspecting domain: '
                 '{unusual}% of values are {comparison} {value} however '
                 'they are usually normalised to domain [0, 100].\n'),
        frame_range=(4, 3))


def inspect_domain_int(a):
    """
    Checks if given numeric or array :math:`a` is normalised to domain
    [0, 2**n - 1].

    Parameters
    ----------
    a : numeric or array_like, optional
        Numeric or array :math:`a` to inspect the domain of. For convenience,
        ``a`` can be set to `None` so that the definition returns directly.

    Returns
    -------
    numeric or array_like
        Inspected numeric or array :math:`a`.
    """

    return inspect_domain(
        a,
        message=('\nUnusual values encountered when inspecting domain: '
                 '{unusual}% of values are {comparison} {value} however '
                 'they are usually integer like values normalised '
                 'to domain [0, 2**n - 1].\n'),
        frame_range=(4, 3))
