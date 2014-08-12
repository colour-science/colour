#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verbose
=======

Defines verbose related objects.
"""

from __future__ import unicode_literals

import warnings

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "New BSD License - http://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["warning"]


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
    >>> colour.utilities.warning("This is a warning!")
    /Users/.../colour/utilities/verbose.py:42: UserWarning: This is a warning!
    """

    warnings.warn(*args, **kwargs)
    return True