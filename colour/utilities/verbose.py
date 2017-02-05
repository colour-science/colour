#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verbose
=======

Defines verbose related objects.
"""

from __future__ import division, unicode_literals

from itertools import chain
from textwrap import TextWrapper
from warnings import filterwarnings, warn

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ColourWarning',
           'message_box',
           'warning',
           'filter_warnings']


class ColourWarning(Warning):
    """
    This is the base class of *Colour* warnings. It is a subclass of `Warning`.
    """

    pass


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

    filterwarnings(
        'ignore' if state else 'default',
        category=ColourWarning if colour_warnings_only else Warning)

    return True
