# -*- coding: utf-8 -*-
"""
LUT Processing Common Utilities
===============================

Defines the *LUT* processing common utilities objects that don't fall in any
specific category.
"""

from __future__ import annotations

import os
import re

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'path_to_title',
]


def path_to_title(path: str) -> str:
    """
    Converts given file path to title.

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
    >>> path_to_title(
    ...     'colour/io/luts/tests/resources/sony_spi3d/Colour_Correct.spi3d'
    ... )
    'Colour Correct'
    """

    return re.sub('_|-|\\.', ' ', os.path.splitext(os.path.basename(path))[0])
