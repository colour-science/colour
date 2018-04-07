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
from textwrap import TextWrapper
from warnings import filterwarnings, formatwarning, warn

from colour.utilities.documentation import DocstringInt

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ColourWarning', 'message_box', 'show_warning', 'warning',
    'filter_warnings', 'suppress_warnings', 'numpy_print_options'
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
