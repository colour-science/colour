"""
:math:`\\Delta E_{99}` DIN99 - Colour Difference Formula
========================================================

Define the :math:`\\Delta E_{99}` *DIN99* colour difference formula.

-   :func:`colour.difference.delta_E_DIN99`

References
----------
-   :cite:`ASTMInternational2007` : ASTM International. (2007). ASTM D2244-07 -
    Standard Practice for Calculation of Color Tolerances and Color Differences
    from Instrumentally Measured Color Coordinates: Vol. i (pp. 1-10).
    doi:10.1520/D2244-16
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import Domain100, Literal, NDArrayFloat

from colour.algebra import euclidean_distance
from colour.difference.typing import DeltaELabData
from colour.models import Lab_to_DIN99
from colour.utilities import as_float_array, get_domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "delta_E_DIN99",
]


@typing.overload
def delta_E_DIN99(
    Lab_1: Domain100,
    Lab_2: Domain100,
    textiles: bool = ...,
    *,
    additional_data: Literal[False] = False,
) -> NDArrayFloat: ...


@typing.overload
def delta_E_DIN99(
    Lab_1: Domain100,
    Lab_2: Domain100,
    textiles: bool = ...,
    *,
    additional_data: Literal[True],
) -> DeltaELabData: ...


def delta_E_DIN99(
    Lab_1: Domain100,
    Lab_2: Domain100,
    textiles: bool = False,
    additional_data: bool = False,
) -> NDArrayFloat | DeltaELabData:
    """
    Compute the colour difference :math:`\\Delta E_{DIN99}` between two
    specified *CIE L\\*a\\*b\\** colourspace arrays using the *DIN99* formula.

    Parameters
    ----------
    Lab_1
        *CIE L\\*a\\*b\\** colourspace array 1.
    Lab_2
        *CIE L\\*a\\*b\\** colourspace array 2.
    textiles
        Textiles application specific parametric factors,
        :math:`k_E=2,\\ k_{CH}=0.5` weights are used instead of
        :math:`k_E=1,\\ k_{CH}=1`.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`dict`
        Colour difference :math:`\\Delta E_{DIN99}`.

    Notes
    -----
    +------------+-----------------------+-------------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**     |
    +============+=======================+===================+
    | ``Lab_1``  | 100                   | 1                 |
    +------------+-----------------------+-------------------+
    | ``Lab_2``  | 100                   | 1                 |
    +------------+-----------------------+-------------------+

    References
    ----------
    :cite:`ASTMInternational2007`

    Examples
    --------
    >>> import numpy as np
    >>> Lab_1 = np.array([60.2574, -34.0099, 36.2677])
    >>> Lab_2 = np.array([60.4626, -34.1751, 39.4387])
    >>> delta_E_DIN99(Lab_1, Lab_2)  # doctest: +ELLIPSIS
    np.float64(1.1772166...)
    >>> delta_E_DIN99(
    ...     Lab_1,
    ...     Lab_2,
    ...     additional_data=True,
    ... )  # doctest: +ELLIPSIS
    DeltaELabData(dE=np.float64(1.1772166...), \
dL=array(-0.1750930...), da=array(-0.5804045...), \
db=array(-1.0091144...))
    """

    k_E = 2 if textiles else 1
    k_CH = 0.5 if textiles else 1

    factor = 100 if get_domain_range_scale() == "1" else 1

    Lab_99_1 = Lab_to_DIN99(Lab_1, k_E, k_CH) * factor
    Lab_99_2 = Lab_to_DIN99(Lab_2, k_E, k_CH) * factor

    dE = euclidean_distance(Lab_99_1, Lab_99_2)

    if not additional_data:
        return dE

    dLab = as_float_array(Lab_99_1) - as_float_array(Lab_99_2)

    return DeltaELabData(
        dE,
        dLab[..., 0],
        dLab[..., 1],
        dLab[..., 2],
    )
