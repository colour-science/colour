"""
LUT Processing Common Utilities
===============================

Define *LUT* processing common utilities objects that do not fall within any
specific category.
"""

from __future__ import annotations

import os
import re
import typing

if typing.TYPE_CHECKING:
    from colour.hints import PathLike

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "path_to_title",
]


def path_to_title(path: str | PathLike) -> str:
    """
    Convert the specified file path to a human-readable title.

    Extract the base filename from the specified path, remove the file
    extension, and replace underscores, hyphens, and dots with spaces to
    create a readable title format.

    Parameters
    ----------
    path
        File path to convert to title.

    Returns
    -------
    :class:`str`
        File path converted to title.

    Examples
    --------
    >>> path_to_title("colour/io/luts/tests/resources/sony_spi3d/Colour_Correct.spi3d")
    'Colour Correct'
    """

    path = str(path)

    return re.sub("_|-|\\.", " ", os.path.splitext(os.path.basename(path))[0])
