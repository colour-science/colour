"""
Typing
================================

Define return types of delta E computation when additional
data should be returned.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat

from dataclasses import dataclass, field

from colour.utilities import MixinDataclassArithmetic

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DeltaEData",
    "DeltaEITPData",
    "DeltaEJabData",
    "DeltaELabData",
    "DeltaELCHData",
]


@dataclass
class DeltaEData(MixinDataclassArithmetic):
    """
    Colour difference data containing the colour difference :math:`\\Delta E`.

    This class is the base container for colour difference computation results
    returned when ``additional_data=True`` in :mod:`colour.difference` functions.
    """

    dE: NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class DeltaELabData(DeltaEData):
    """
    Colour difference data expressed in a Lab-like colourspace.

    This data structure is returned by the following functions when
    ``additional_data=True``:

    - :func:`colour.difference.delta_E_CIE1976`
    - :func:`colour.difference.delta_E_HyAB`
    - :func:`colour.difference.delta_E_DIN99`

    Notes
    -----
    The meaning of the components depends on the originating function:

    - *CIE 1976* (:func:`colour.difference.delta_E_CIE1976`):
      raw differences in *CIE L\\*a\\*b\\** coordinates.
    - *HyAB* (:func:`colour.difference.delta_E_HyAB`):
      raw differences in *CIE L\\*a\\*b\\** coordinates.
    - *DIN99* (:func:`colour.difference.delta_E_DIN99`):
      raw differences in *DIN99* colourspace.
    """

    dL: NDArrayFloat | None = field(default_factory=lambda: None)
    da: NDArrayFloat | None = field(default_factory=lambda: None)
    db: NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class DeltaELCHData(DeltaEData):
    """
    Colour difference data expressed in an LCH-like colourspace.

    This data structure is returned by the following functions when
    ``additional_data=True``:

    - :func:`colour.difference.delta_E_CIE1994`
    - :func:`colour.difference.delta_E_CIE2000`
    - :func:`colour.difference.delta_E_CMC`
    - :func:`colour.difference.delta_E_HyCH`

    Notes
    -----
    The components ``dL``, ``dC`` and ``dH`` are generally *weighted*
    differences and **not raw coordinate differences**:

    - *CIE 1994* (:func:`colour.difference.delta_E_CIE1994`):
      differences divided by :math:`k_L S_L`, :math:`k_C S_C`, :math:`k_H S_H`.
    - *CIE 2000* (:func:`colour.difference.delta_E_CIE2000`):
      differences divided by parametric weighting functions
      :math:`k_L S_L`, :math:`k_C S_C`, :math:`k_H S_H`.
    - *CMC* (:func:`colour.difference.delta_E_CMC`):
      differences divided by :math:`l S_L`, :math:`c S_C` and hue weighting.
    - *HyCH* (:func:`colour.difference.delta_E_HyCH`):
      weighted differences based on *CIE 2000* intermediate attributes.

    For *CIE 1994*, *CIE 2000* and *HyCH*, enabling the ``textiles`` parameter
    modifies the parametric weighting factors and therefore directly affects
    the returned component values ``dL``, ``dC`` and ``dH``.
    """

    dL: NDArrayFloat | None = field(default_factory=lambda: None)
    dC: NDArrayFloat | None = field(default_factory=lambda: None)
    dH: NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class DeltaEJabData(DeltaEData):
    """
    Colour difference data expressed in a :math:`J'a'b'` colourspace.

    This data structure is returned by the following functions when
    ``additional_data=True``:

    - :func:`colour.difference.delta_E_Luo2006`
    - :func:`colour.difference.delta_E_CAM02LCD`
    - :func:`colour.difference.delta_E_CAM02SCD`
    - :func:`colour.difference.delta_E_CAM02UCS`

    Notes
    -----
    The interpretation of the components is as follows:

    - ``dJ`` is weighted by the lightness coefficient :math:`K_L`
      defined by the selected *Luo et al. (2006)* uniform colourspace.
    - ``da`` and ``db`` are raw differences in the corresponding
      *CAM02-LCD*, *CAM02-SCD* or *CAM02-UCS* colourspace.
    """

    dJ: NDArrayFloat | None = field(default_factory=lambda: None)
    da: NDArrayFloat | None = field(default_factory=lambda: None)
    db: NDArrayFloat | None = field(default_factory=lambda: None)


@dataclass
class DeltaEITPData(DeltaEData):
    """
    Colour difference data expressed in the :math:`I_C T_C P_C` colourspace.

    This data structure is returned by
    :func:`colour.difference.delta_E_ITP` when ``additional_data=True``.

    Notes
    -----
    - ``dT`` is **half-scaled prior to differencing** as specified by
      *Recommendation ITU-R BT.2124*.
    - The returned ``dT`` value is **not** a raw :math:`T` difference.
    """

    dI: NDArrayFloat | None = field(default_factory=lambda: None)
    dT: NDArrayFloat | None = field(default_factory=lambda: None)
    dP: NDArrayFloat | None = field(default_factory=lambda: None)
