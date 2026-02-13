"""
:math:`\\Delta E'` - Delta E Colour Difference - Luo, Cui and Li (2006)
=======================================================================

Define the :math:`\\Delta E'` colour difference computation objects based on
*Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, and *CAM02-UCS* colourspaces:

-   :func:`colour.difference.delta_E_CAM02LCD`
-   :func:`colour.difference.delta_E_CAM02SCD`
-   :func:`colour.difference.delta_E_CAM02UCS`

References
----------
-   :cite:`Luo2006b` : Luo, M. Ronnier, Cui, G., & Li, C. (2006). Uniform
    colour spaces based on CIECAM02 colour appearance model. Color Research &
    Application, 31(4), 320-330. doi:10.1002/col.20227
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat, Literal

from dataclasses import dataclass, field

from colour.hints import Domain100  # noqa: TC001
from colour.models.cam02_ucs import COEFFICIENTS_UCS_LUO2006, Coefficients_UCS_Luo2006
from colour.utilities import MixinDataclassArithmetic, as_float, tsplit

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DeltaE_Specification_Luo2006",
    "delta_E_Luo2006",
    "delta_E_CAM02LCD",
    "delta_E_CAM02SCD",
    "delta_E_CAM02UCS",
]


@dataclass
class DeltaE_Specification_Luo2006(MixinDataclassArithmetic):
    """
    Define the *Luo et al. (2006)* colour difference specification.

    This data structure is returned by
    :func:`colour.difference.delta_E_Luo2006` (and by extension
    :func:`colour.difference.delta_E_CAM02LCD`,
    :func:`colour.difference.delta_E_CAM02SCD`, and
    :func:`colour.difference.delta_E_CAM02UCS`) when
    ``additional_data=True``.

    Parameters
    ----------
    dE
        Colour difference :math:`\\Delta E'`.
    dJ
        Weighted *lightness* difference :math:`\\Delta J' / K_L`.
    da
        Raw :math:`\\Delta a'` difference.
    db
        Raw :math:`\\Delta b'` difference.
    """

    dE: NDArrayFloat | None = field(default_factory=lambda: None)
    dJ: NDArrayFloat | None = field(default_factory=lambda: None)
    da: NDArrayFloat | None = field(default_factory=lambda: None)
    db: NDArrayFloat | None = field(default_factory=lambda: None)


@typing.overload
def delta_E_Luo2006(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    coefficients: Coefficients_UCS_Luo2006,
    *,
    additional_data: Literal[False] = False,
) -> NDArrayFloat: ...


@typing.overload
def delta_E_Luo2006(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    coefficients: Coefficients_UCS_Luo2006,
    *,
    additional_data: Literal[True],
) -> DeltaE_Specification_Luo2006: ...


def delta_E_Luo2006(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    coefficients: Coefficients_UCS_Luo2006,
    additional_data: bool = False,
) -> NDArrayFloat | DeltaE_Specification_Luo2006:
    """
    Compute the colour difference :math:`\\Delta E'` between two specified
    *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
    colourspaces :math:`J'a'b'` arrays.

    Parameters
    ----------
    Jpapbp_1
        Standard / reference *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*,
        or *CAM02-UCS* colourspaces :math:`J'a'b'` array.
    Jpapbp_2
        Sample / test *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or
        *CAM02-UCS* colourspaces :math:`J'a'b'` array.
    coefficients
        Coefficients of one of the *Luo et al. (2006)* *CAM02-LCD*,
        *CAM02-SCD*, or *CAM02-UCS* colourspaces.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`DeltaE_Specification_Luo2006`
        Colour difference :math:`\\Delta E'`.

    Warnings
    --------
    The :math:`J'a'b'` array should have been computed with a
    *Luo et al. (2006)* *CAM02-LCD*, *CAM02-SCD*, or *CAM02-UCS*
    colourspace and not with the *CIE L\\*a\\*b\\** colourspace.

    Notes
    -----
    +--------------+------------------------+--------------------+
    | **Domain**   |  **Scale - Reference** | **Scale - 1**      |
    +==============+========================+====================+
    | ``Jpapbp_1`` | 100                    | 1                  |
    +--------------+------------------------+--------------------+
    | ``Jpapbp_2`` | 100                    | 1                  |
    +--------------+------------------------+--------------------+

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
    >>> delta_E_Luo2006(Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])
    ... # doctest: +ELLIPSIS
    np.float64(14.0555464...)
    >>> delta_E_Luo2006(
    ...     Jpapbp_1,
    ...     Jpapbp_2,
    ...     COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
    ...     additional_data=True,
    ... )  # doctest: +ELLIPSIS
    DeltaE_Specification_Luo2006(dE=np.float64(14.0555464...), \
dJ=np.float64(0.1309140...), da=np.float64(3.8848968...), \
db=np.float64(13.5073618...))
    """

    J_p_1, a_p_1, b_p_1 = tsplit(Jpapbp_1)
    J_p_2, a_p_2, b_p_2 = tsplit(Jpapbp_2)
    K_L, _c_1, _c_2 = coefficients.values

    J = (J_p_1 - J_p_2) / K_L
    a = a_p_1 - a_p_2
    b = b_p_1 - b_p_2

    d_E = as_float(np.sqrt(J**2 + a**2 + b**2))

    if not additional_data:
        return d_E

    return DeltaE_Specification_Luo2006(
        d_E,
        J,
        a,
        b,
    )


@typing.overload
def delta_E_CAM02LCD(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    *,
    additional_data: Literal[False] = False,
) -> NDArrayFloat: ...


@typing.overload
def delta_E_CAM02LCD(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    *,
    additional_data: Literal[True],
) -> DeltaE_Specification_Luo2006: ...


def delta_E_CAM02LCD(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    additional_data: bool = False,
) -> NDArrayFloat | DeltaE_Specification_Luo2006:
    """
    Compute the colour difference :math:`\\Delta E'` between two specified
    *CAM02-LCD* colourspace :math:`J'a'b'` arrays using the
    *Luo et al. (2006)* formula.

    Parameters
    ----------
    Jpapbp_1
        Standard / reference *CAM02-LCD* colourspace :math:`J'a'b'` array as
        computed by the *Luo et al. (2006)* uniform colour space model.
    Jpapbp_2
        Sample / test *CAM02-LCD* colourspace :math:`J'a'b'` array as computed
        by the *Luo et al. (2006)* uniform colour space model.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`DeltaE_Specification_Luo2006`
        Colour difference :math:`\\Delta E'`.

    Warnings
    --------
    The :math:`J'a'b'` arrays should have been computed with the
    *Luo et al. (2006)* *CAM02-LCD* colourspace and not with the
    *CIE L\\*a\\*b\\** colourspace.

    Notes
    -----
    +--------------+------------------------+--------------------+
    | **Domain**   |  **Scale - Reference** | **Scale - 1**      |
    +==============+========================+====================+
    | ``Jpapbp_1`` | 100                    | 1                  |
    +--------------+------------------------+--------------------+
    | ``Jpapbp_2`` | 100                    | 1                  |
    +--------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
    >>> delta_E_CAM02LCD(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    np.float64(14.0555464...)
    >>> delta_E_CAM02LCD(
    ...     Jpapbp_1,
    ...     Jpapbp_2,
    ...     additional_data=True,
    ... )  # doctest: +ELLIPSIS
    DeltaE_Specification_Luo2006(dE=np.float64(14.0555464...), \
dJ=np.float64(0.1309140...), da=np.float64(3.8848968...), \
db=np.float64(13.5073618...))
    """

    return delta_E_Luo2006(
        Jpapbp_1,
        Jpapbp_2,
        COEFFICIENTS_UCS_LUO2006["CAM02-LCD"],
        additional_data=additional_data,
    )


@typing.overload
def delta_E_CAM02SCD(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    *,
    additional_data: Literal[False] = False,
) -> NDArrayFloat: ...


@typing.overload
def delta_E_CAM02SCD(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    *,
    additional_data: Literal[True],
) -> DeltaE_Specification_Luo2006: ...


def delta_E_CAM02SCD(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    additional_data: bool = False,
) -> NDArrayFloat | DeltaE_Specification_Luo2006:
    """
    Compute the colour difference :math:`\\Delta E'` between two specified
    *CAM02-SCD* colourspace :math:`J'a'b'` arrays using the
    *Luo et al. (2006)* formula.

    Parameters
    ----------
    Jpapbp_1
        Standard / reference *CAM02-SCD* colourspace :math:`J'a'b'` array as
        computed by the *Luo et al. (2006)* uniform colour space model.
    Jpapbp_2
        Sample / test *CAM02-SCD* colourspace :math:`J'a'b'` array as computed
        by the *Luo et al. (2006)* uniform colour space model.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`DeltaE_Specification_Luo2006`
        Colour difference :math:`\\Delta E'`.

    Warnings
    --------
    The :math:`J'a'b'` arrays should have been computed with the
    *Luo et al. (2006)* *CAM02-SCD* colourspace and not with the
    *CIE L\\*a\\*b\\** colourspace.

    Notes
    -----
    +--------------+------------------------+--------------------+
    | **Domain**   |  **Scale - Reference** | **Scale - 1**      |
    +==============+========================+====================+
    | ``Jpapbp_1`` | 100                    | 1                  |
    +--------------+------------------------+--------------------+
    | ``Jpapbp_2`` | 100                    | 1                  |
    +--------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
    >>> delta_E_CAM02SCD(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    np.float64(14.0551718...)
    >>> delta_E_CAM02SCD(
    ...     Jpapbp_1,
    ...     Jpapbp_2,
    ...     additional_data=True,
    ... )  # doctest: +ELLIPSIS
    DeltaE_Specification_Luo2006(dE=np.float64(14.0551718...), \
dJ=np.float64(0.0812933...), da=np.float64(3.8848968...), \
db=np.float64(13.5073618...))
    """

    return delta_E_Luo2006(
        Jpapbp_1,
        Jpapbp_2,
        COEFFICIENTS_UCS_LUO2006["CAM02-SCD"],
        additional_data=additional_data,
    )


@typing.overload
def delta_E_CAM02UCS(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    *,
    additional_data: Literal[False] = False,
) -> NDArrayFloat: ...


@typing.overload
def delta_E_CAM02UCS(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    *,
    additional_data: Literal[True],
) -> DeltaE_Specification_Luo2006: ...


def delta_E_CAM02UCS(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    additional_data: bool = False,
) -> NDArrayFloat | DeltaE_Specification_Luo2006:
    """
    Compute the colour difference :math:`\\Delta E'` between two specified
    *CAM02-UCS* colourspace :math:`J'a'b'` arrays using the
    *Luo et al. (2006)* formula.

    Parameters
    ----------
    Jpapbp_1
        Standard / reference *CAM02-UCS* colourspace :math:`J'a'b'` array as
        computed by the *Luo et al. (2006)* uniform colour space model.
    Jpapbp_2
        Sample / test *CAM02-UCS* colourspace :math:`J'a'b'` array as computed
        by the *Luo et al. (2006)* uniform colour space model.
    additional_data
        Whether to output additional data.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`DeltaE_Specification_Luo2006`
        Colour difference :math:`\\Delta E'`.

    Warnings
    --------
    The :math:`J'a'b'` arrays should have been computed with the
    *Luo et al. (2006)* *CAM02-UCS* colourspace and not with the
    *CIE L\\*a\\*b\\** colourspace.

    Notes
    -----
    +--------------+------------------------+--------------------+
    | **Domain**   |  **Scale - Reference** | **Scale - 1**      |
    +==============+========================+====================+
    | ``Jpapbp_1`` | 100                    | 1                  |
    +--------------+------------------------+--------------------+
    | ``Jpapbp_2`` | 100                    | 1                  |
    +--------------+------------------------+--------------------+

    References
    ----------
    :cite:`Luo2006b`

    Examples
    --------
    >>> Jpapbp_1 = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> Jpapbp_2 = np.array([54.80352754, -3.96940084, -13.57591013])
    >>> delta_E_CAM02UCS(Jpapbp_1, Jpapbp_2)  # doctest: +ELLIPSIS
    np.float64(14.0552982...)
    >>> delta_E_CAM02UCS(
    ...     Jpapbp_1,
    ...     Jpapbp_2,
    ...     additional_data=True,
    ... )  # doctest: +ELLIPSIS
    DeltaE_Specification_Luo2006(dE=np.float64(14.0552982...), \
dJ=np.float64(0.1008038...), da=np.float64(3.8848968...), \
db=np.float64(13.5073618...))
    """

    return delta_E_Luo2006(
        Jpapbp_1,
        Jpapbp_2,
        COEFFICIENTS_UCS_LUO2006["CAM02-UCS"],
        additional_data=additional_data,
    )
