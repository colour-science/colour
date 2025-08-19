"""
Documentation
=============

Define objects and utilities for documentation generation and processing.
"""

from __future__ import annotations

import os

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
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
    Define a :class:`dict` sub-class that allows docstring attachment to
    :class:`dict` instances.
    """


class DocstringFloat(float):
    """
    Define a :class:`float` sub-class that allows docstring attachment to
    :class:`float` instances.
    """


class DocstringInt(int):
    """
    Define an :class:`int` sub-class that allows docstring attachment to
    :class:`int` instances.
    """


class DocstringText(str):  # noqa: SLOT000
    """
    Define a :class:`str` sub-class that allows docstring attachment to
    :class:`str` instances.
    """


class DocstringTuple(tuple):  # noqa: SLOT001
    """
    Define a :class:`tuple` sub-class that allows docstring attachment to
    :class:`tuple` instances.
    """


def is_documentation_building() -> bool:
    """
    Determine whether the documentation is being built by checking for the
    *READTHEDOCS* or *COLOUR_SCIENCE__DOCUMENTATION_BUILD* environment
    variables.

    Returns
    -------
    :class:`bool`
        Whether the documentation is being built.

    Examples
    --------
    >>> is_documentation_building()
    False
    >>> os.environ["READTHEDOCS"] = "True"
    >>> is_documentation_building()
    True
    >>> os.environ["READTHEDOCS"] = "False"
    >>> is_documentation_building()
    True
    >>> del os.environ["READTHEDOCS"]
    >>> is_documentation_building()
    False
    >>> os.environ["COLOUR_SCIENCE__DOCUMENTATION_BUILD"] = "Yes"
    >>> is_documentation_building()
    True
    >>> del os.environ["COLOUR_SCIENCE__DOCUMENTATION_BUILD"]
    """

    return bool(
        os.environ.get("READTHEDOCS")
        or os.environ.get("COLOUR_SCIENCE__DOCUMENTATION_BUILD")
    )
