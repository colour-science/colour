# -*- coding: utf-8 -*-
"""
Documentation
=============

Defines documentation related objects.
"""

from __future__ import division, unicode_literals

import os
from six import text_type

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'DocstringText', 'DocstringDict', 'DocstringTuple', 'DocstringFloat',
    'is_documentation_building'
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


def is_documentation_building():
    """
    Returns whether the documentation is being built by checking whether the
    *READTHEDOCS* or *COLOUR_SCIENCE_DOCUMENTATION_BUILD* environment variables
    are defined, their value is not accounted for.

    Returns
    -------
    bool
        Whether the documentation is being built.

    Examples
    --------
    >>> is_documentation_building()
    False
    >>> os.environ['READTHEDOCS'] = 'True'
    >>> is_documentation_building()
    True
    >>> os.environ['READTHEDOCS'] = 'False'
    >>> is_documentation_building()
    True
    >>> del os.environ['READTHEDOCS']
    >>> is_documentation_building()
    False
    >>> os.environ['COLOUR_SCIENCE_DOCUMENTATION_BUILD'] = 'Yes'
    >>> is_documentation_building()
    True
    >>> del os.environ['COLOUR_SCIENCE_DOCUMENTATION_BUILD']
    """

    return bool(
        os.environ.get('READTHEDOCS') or
        os.environ.get('COLOUR_SCIENCE_DOCUMENTATION_BUILD'))
