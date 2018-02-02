#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Documentation
=============

Defines documentation related objects.
"""

from __future__ import division, unicode_literals

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['DocstringDict', 'DocstringTuple', 'DocstringFloat']


class DocstringDict(dict):
    """
    A `dict` sub-class that allows settings a docstring to `dict` instances.
    """

    pass


class DocstringTuple(tuple):
    """
    A `tuple` sub-class that allows settings a docstring to `tuple` instances.
    """

    pass


class DocstringFloat(float):
    """
    A `float` sub-class that allows settings a docstring to `float` instances.
    """

    pass
