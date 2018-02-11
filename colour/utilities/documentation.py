#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Documentation
=============

Defines documentation related objects.
"""

from __future__ import division, unicode_literals

from six import text_type

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'DocstringText', 'DocstringDict', 'DocstringTuple', 'DocstringFloat'
]


class DocstringText(text_type):
    """
    A :class:`unicode` sub-class that allows settings a docstring to
    :class:`unicode` instances.
    """

    pass


class DocstringDict(dict):
    """
    A :class:`dict` sub-class that allows settings a docstring to :class:`dict`
    instances.
    """

    pass


class DocstringTuple(tuple):
    """
    A :class:`tuple` sub-class that allows settings a docstring to
    :class:`tuple` instances.
    """

    pass


class DocstringFloat(float):
    """
    A :class:`float` sub-class that allows settings a docstring to
    :class:`float` instances.
    """

    pass
