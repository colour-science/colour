"""
Typing
================================

Define return types of delta E computation when additional
data should be returned.
"""

from __future__ import annotations

import typing
from typing import TypedDict

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DeltaEData",
    "DeltaELabData",
    "DeltaELCHData",
    "DeltaEJabData",
    "DeltaEJCHData",
]


class DeltaEData(TypedDict):
    """
    Colour difference data containing the colour difference
    :math:`\\Delta E`.
    """

    dE: NDArrayFloat


class DeltaELabData(DeltaEData):
    """
    Colour difference data in the *CIE L\\*a\\*b\\** colourspace.
    """

    dL: NDArrayFloat
    da: NDArrayFloat
    db: NDArrayFloat


class DeltaELCHData(DeltaEData):
    """
    Colour difference data in the *CIE L\\*C\\*h°* colourspace.
    """

    dL: NDArrayFloat
    dC: NDArrayFloat
    dH: NDArrayFloat


class DeltaEJabData(DeltaEData):
    """
    Colour difference data in the :math:`J'a'b'` colourspace.
    """

    dJ: NDArrayFloat
    da: NDArrayFloat
    db: NDArrayFloat


class DeltaEJCHData(DeltaEData):
    """
    Colour difference data in the :math:`J'C'H'` colourspace.
    """

    dJ: NDArrayFloat
    dC: NDArrayFloat
    dH: NDArrayFloat


class DeltaEITPData(DeltaEData):
    """
    Colour difference data in the :math:`I_C T_C P_C` colourspace.
    """

    dI: NDArrayFloat
    dT: NDArrayFloat
    dP: NDArrayFloat
