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
    from colour.hints import NDArrayFloat

from colour.hints import Domain100  # noqa: TC001
from colour.models.cam02_ucs import COEFFICIENTS_UCS_LUO2006, Coefficients_UCS_Luo2006
from colour.utilities import as_float, tsplit

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "delta_E_Luo2006",
    "delta_E_CAM02LCD",
    "delta_E_CAM02SCD",
    "delta_E_CAM02UCS",
]


def delta_E_Luo2006(
    Jpapbp_1: Domain100,
    Jpapbp_2: Domain100,
    coefficients: Coefficients_UCS_Luo2006,
) -> NDArrayFloat:
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

    Returns
    -------
    :class:`numpy.ndarray`
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
    14.0555464...
    """

    J_p_1, a_p_1, b_p_1 = tsplit(Jpapbp_1)
    J_p_2, a_p_2, b_p_2 = tsplit(Jpapbp_2)
    K_L, _c_1, _c_2 = coefficients.values

    d_E = np.sqrt(
        ((J_p_1 - J_p_2) / K_L) ** 2 + (a_p_1 - a_p_2) ** 2 + (b_p_1 - b_p_2) ** 2
    )

    return as_float(d_E)


def delta_E_CAM02LCD(Jpapbp_1: Domain100, Jpapbp_2: Domain100) -> NDArrayFloat:
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

    Returns
    -------
    :class:`numpy.ndarray`
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
    14.0555464...
    """

    return delta_E_Luo2006(Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-LCD"])


def delta_E_CAM02SCD(Jpapbp_1: Domain100, Jpapbp_2: Domain100) -> NDArrayFloat:
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

    Returns
    -------
    :class:`numpy.ndarray`
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
    14.0551718...
    """

    return delta_E_Luo2006(Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-SCD"])


def delta_E_CAM02UCS(Jpapbp_1: Domain100, Jpapbp_2: Domain100) -> NDArrayFloat:
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

    Returns
    -------
    :class:`numpy.ndarray`
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
    14.0552982...
    """

    return delta_E_Luo2006(Jpapbp_1, Jpapbp_2, COEFFICIENTS_UCS_LUO2006["CAM02-UCS"])
