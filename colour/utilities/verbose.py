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
from warnings import warn

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['message_box',
           'warning']


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
    inner = lambda text: '*{0}{1}{2}{0}*'.format(
        ' ' * padding,
        text,
        (' ' * (width - len(text) - padding * 2 - 2)))

    print('=' * width)
    print(inner(''))

    wrapper = TextWrapper(width=ideal_width,
                          break_long_words=False,
                          replace_whitespace=False)

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

    Parameters
    ----------
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
    >>> colour.utilities.warning('This is a warning!')  # doctest: +SKIP
    /Users/.../colour/utilities/verbose.py:42: UserWarning: This is a warning!
    """

    warn(*args, **kwargs)
    return True
