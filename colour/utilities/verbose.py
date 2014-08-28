#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verbose
=======

Defines verbose related objects.
"""

from __future__ import division, unicode_literals

from textwrap import wrap
from warnings import warn

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
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
    >>> message_box(message)
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │   Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod   │
    │   tempor incididunt ut labore et dolore magna aliqua.                       │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    True
    >>> message_box(message, width=60)
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   Lorem ipsum dolor sit amet, consectetur adipiscing     │
    │   elit, sed do eiusmod tempor incididunt ut labore et    │
    │   dolore magna aliqua.                                   │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    True
    >>> message_box(message, padding=16)
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │                Lorem ipsum dolor sit amet, consectetur                      │
    │                adipiscing elit, sed do eiusmod tempor                       │
    │                incididunt ut labore et dolore magna aliqua.                 │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    True
    """

    ideal_width = width - padding * 2
    inner = lambda text: '│{0}{1}{2}{0}│'.format(
        ' ' * padding,
        text,
        (' ' * (width - len(text) - padding * 2 - 2)))

    print('┌{0}┐'.format('─' * (width - 2)))
    print(inner(''))

    for line in wrap(message, width=ideal_width):
        print(inner(line.expandtabs()))

    print(inner(''))
    print('└{0}┘'.format('─' * (width - 2)))
    return True


def warning(*args, **kwargs):
    """
    Issues a warning.

    Parameters
    ----------
    \*args : \*
        Arguments.
    \*\*kwargs : \*\*
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

