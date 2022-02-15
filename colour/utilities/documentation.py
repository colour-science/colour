"""
Documentation
=============

Defines the documentation related objects.
"""

from __future__ import annotations

from colour.hints import Boolean

import os

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DocstringDict",
    "DocstringFloat",
    "DocstringInt",
    "DocstringText",
    "DocstringTuple",
    "is_documentation_building",
]


class DocstringDict(dict):
    """
    A :class:`dict` sub-class that allows settings a docstring to :class:`dict`
    instances.
    """

    pass


class DocstringFloat(float):
    """
    A :class:`float` sub-class that allows settings a docstring to
    :class:`float` instances.
    """

    pass


class DocstringInt(int):
    """
    A :class:`numpy.integer` sub-class that allows settings a docstring to
    :class:`numpy.integer` instances.
    """

    pass


class DocstringText(str):
    """
    A :class:`str` sub-class that allows settings a docstring to
    :class:`str` instances.
    """

    pass


class DocstringTuple(tuple):
    """
    A :class:`tuple` sub-class that allows settings a docstring to
    :class:`tuple` instances.
    """

    pass


def is_documentation_building() -> Boolean:
    """
    Return whether the documentation is being built by checking whether the
    *READTHEDOCS* or *COLOUR_SCIENCE__DOCUMENTATION_BUILD* environment
    variables are defined, their value is not accounted for.

    Returns
    -------
    :class:`bool`
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
    >>> os.environ['COLOUR_SCIENCE__DOCUMENTATION_BUILD'] = 'Yes'
    >>> is_documentation_building()
    True
    >>> del os.environ['COLOUR_SCIENCE__DOCUMENTATION_BUILD']
    """

    return bool(
        os.environ.get("READTHEDOCS")
        or os.environ.get("COLOUR_SCIENCE__DOCUMENTATION_BUILD")
    )
