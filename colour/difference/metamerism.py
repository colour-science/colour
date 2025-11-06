"""
:math:`M_{t}` - Metamerism Index
====================================================

Define the :math:`M_{t}` metamerism computation objects.

-   :func:`colour.difference.metamerism_index_from_Lab`
-   :func:`colour.difference.metamerism_index_from_XYZ`

References
----------
-   :cite:ISO18314-4_2024 : International Organization for Standardization. (2024).
    ISO 18314-4:2024. Analytical colorimetry, Part 4: Metamerism index for pairs
    of samples for change of illuminant (2nd ed., 24 pp.).
    ISO/TC 256. https://www.iso.org/standard/85116.html
"""

from __future__ import annotations

import colour
from colour.hints import (  # noqa: TC001
    Any,
    Domain1,
    Domain100,
    Literal,
    LiteralDeltaEMethod,
    NDArrayFloat,
)
from colour.models import XYZ_to_Lab
from colour.utilities import (
    as_array,
    filter_kwargs,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "metamerism_index_from_Lab",
    "metamerism_index_from_XYZ",
]


def metamerism_index_from_Lab(
    Lab_spl_t: Domain100,
    Lab_std_t: Domain100,
    Lab_spl_r: Domain100,
    Lab_std_r: Domain100,
    correction: Literal["additive", "multiplicative"] | str = "additive",
    method: LiteralDeltaEMethod | str = "CIE 2000",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute the metamerism index :math:`M_{t}` between four specified
    *CIE L\\*a\\*b\\** colourspace arrays. Before computing the metamerism index,
    we apply either an additive or multiplicative correction. The correction is
    based on the difference between the colour sample and colour standard under
    reference illuminant and applied to the colour sample under test illuminant.
    The correction is applied in *CIE L\\*a\\*b\\** colourspace.

    :cite:ISO18314-4_2024 recommends to use additive correction in *CIE L\\*a\\*b\\**.

    Parameters
    ----------
    Lab_spl_t
        *CIE L\\*a\\*b\\** colourspace array of colour sample under
        test illuminant.
    Lab_std_t
        *CIE L\\*a\\*b\\** colourspace array of colour standard under
        test illuminant.
    Lab_spl_r
        *CIE L\\*a\\*b\\** colourspace array of colour sample under
        reference illuminant.
    Lab_std_r
        *CIE L\\*a\\*b\\** colourspace array of colour standard under
        reference illuminant.
    correction
        Correction method to apply, either 'additive' or 'multiplicative'.
    metric
        Colour difference metric to use.

    Other Parameters
    ----------------
    c
        {:func:`colour.difference.delta_E_CMC`},
        *Chroma* weighting factor.
    l
        {:func:`colour.difference.delta_E_CMC`},
        *Lightness* weighting factor.
    textiles
        {:func:`colour.difference.delta_E_CIE1994`,
        :func:`colour.difference.delta_E_CIE2000`,
        :func:`colour.difference.delta_E_DIN99`},
        Textiles application specific parametric factors
        :math:`k_L=2,\\ k_C=k_H=1,\\ k_1=0.048,\\ k_2=0.014,\\ k_E=2,\\ k_{CH}=0.5`
        weights are used instead of
        :math:`k_L=k_C=k_H=1,\\ k_1=0.045,\\ k_2=0.015,\\ k_E=k_{CH}=1.0`.

    Returns
    -------
    :class:`numpy.ndarray`
        Metamerism index :math:`M_{t}`.

    Notes
    -----
    +----------------+-----------------------+-------------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**     |
    +================+=======================+===================+
    | ``Lab_spl_t``  | 100                   | 1                 |
    +----------------+-----------------------+-------------------+
    | ``Lab_std_t``  | 100                   | 1                 |
    +----------------+-----------------------+-------------------+
    | ``Lab_spl_r``  | 100                   | 1                 |
    +----------------+-----------------------+-------------------+
    | ``Lab_std_r``  | 100                   | 1                 |
    +----------------+-----------------------+-------------------+

    References
    ----------
    :cite:ISO18314-4_2024

    Examples
    --------
    >>> import numpy as np
    >>> Lab_std_r = np.array([39.0908, -21.3269, 22.6657])
    >>> Lab_std_t = np.array([38.17781, -17.4939, 21.0618])
    >>> Lab_spl_r = np.array([38.83253, -19.8787, 20.0453])
    >>> Lab_spl_t = np.array([37.9013, -19.56327, 16.9346])
    >>> metamerism_index_from_Lab(
    ...     Lab_spl_t,
    ...     Lab_std_t,
    ...     Lab_spl_r,
    ...     Lab_std_r,
    ...     correction="additive",
    ...     method="CIE 1976",
    ... )  # doctest: +ELLIPSIS
    3.8267581...
    >>> metamerism_index_from_Lab(
    ...     Lab_spl_t,
    ...     Lab_std_t,
    ...     Lab_spl_r,
    ...     Lab_std_r,
    ...     correction="multiplicative",
    ...     method="CIE 1976",
    ... )  # doctest: +ELLIPSIS
    3.9842216...
    """

    if correction == "additive":
        Lab_corr_t = as_array(Lab_spl_t) - (as_array(Lab_spl_r) - as_array(Lab_std_r))

    elif correction == "multiplicative":
        Lab_corr_t = as_array(Lab_spl_t) * (as_array(Lab_std_r) / as_array(Lab_spl_r))

    return colour.difference.delta_E(
        Lab_std_t,
        Lab_corr_t,
        method=method,
        **kwargs,
    )


def metamerism_index_from_XYZ(
    XYZ_spl_t: Domain1,
    XYZ_std_t: Domain1,
    XYZ_spl_r: Domain1,
    XYZ_std_r: Domain1,
    correction: Literal["additive", "multiplicative"] | str = "multiplicative",
    method: LiteralDeltaEMethod | str = "CIE 2000",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute the metamerism index :math:`M_{t}` between four specified
    *CIE XYZ* colourspace arrays. Before computing the metamerism index,
    we apply either an additive or multiplicative correction. The correction is
    based on the difference between the colour sample and colour standard under
    reference illuminant and applied to the colour sample under test illuminant.
    The correction is applied in *CIE XYZ* colourspace. Afterwards, we convert to
    *CIE L\\*a\\*b\\** colourspace to compute the metamerism index.

    :cite:ISO18314-4_2024 recommends to use multiplicative correction in
    *CIE L\\*a\\*b\\**.

    Parameters
    ----------
    XYZ_spl_t
        *CIE XYZ* tristimulus array of the sample under test
        illuminant.
    XYZ_std_t
        *CIE XYZ* tristimulus array of the standard under test
        illuminant.
    XYZ_spl_r
        *CIE XYZ* tristimulus array of the sample under reference
        illuminant.
    XYZ_std_r
        *CIE XYZ* tristimulus array of the standard under reference
        illuminant.
    correction
        Correction method to apply, either 'additive' or
        'multiplicative'.
    method
        Colour-difference method.

    Other Parameters
    ----------------
    illuminant
        {:func:`colour.models.XYZ_to_Lab`},
        Reference *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
        colourspace array for conversion from *CIE XYZ* to *CIE L\\*a\\*b\\**.
    c
        {:func:`colour.difference.delta_E_CMC`},
        *Chroma* weighting factor.
    l
        {:func:`colour.difference.delta_E_CMC`},
        *Lightness* weighting factor.
    textiles
        {:func:`colour.difference.delta_E_CIE1994`,
        :func:`colour.difference.delta_E_CIE2000`,
        :func:`colour.difference.delta_E_DIN99`},
        Textiles application specific parametric factors
        :math:`k_L=2,\\ k_C=k_H=1,\\ k_1=0.048,\\ k_2=0.014,\\ k_E=2,\\ k_{CH}=0.5`
        weights are used instead of
        :math:`k_L=k_C=k_H=1,\\ k_1=0.045,\\ k_2=0.015,\\ k_E=k_{CH}=1.0`.

    Returns
    -------
    :class:`numpy.ndarray`
        Metamerism index :math:`M_{t}`.

    Notes
    -----
    +----------------+-----------------------+-------------------+
    | **Domain**     | **Scale - Reference** | **Scale - 1**     |
    +================+=======================+===================+
    | ``XYZ_spl_t``  | 1                     | 1                 |
    +----------------+-----------------------+-------------------+
    | ``XYZ_std_t``  | 1                     | 1                 |
    +----------------+-----------------------+-------------------+
    | ``XYZ_spl_r``  | 1                     | 1                 |
    +----------------+-----------------------+-------------------+
    | ``XYZ_std_r``  | 1                     | 1                 |
    +----------------+-----------------------+-------------------+

    References
    ----------
    :cite:ISO18314-4_2024

    Examples
    --------
    >>> import numpy as np
    >>> from colour import CCS_ILLUMINANTS
    >>> XYZ_std_r = np.array([7.6576, 10.7116, 5.0731]) / 100
    >>> XYZ_std_t = np.array([8.96442, 10.1878, 1.6663]) / 100
    >>> XYZ_spl_r = np.array([7.6933, 10.5616, 5.54474]) / 100
    >>> XYZ_spl_t = np.array([8.56438, 10.0324, 1.9315]) / 100
    >>> metamerism_index_from_XYZ(
    ...     XYZ_spl_t,
    ...     XYZ_std_t,
    ...     XYZ_spl_r,
    ...     XYZ_std_r,
    ...     correction="multiplicative",
    ...     method="CIE 1976",
    ...     illuminant=CCS_ILLUMINANTS["CIE 1964 10 Degree Standard Observer"]["A"],
    ... )  # doctest: +ELLIPSIS
    3.7906989...
    >>> metamerism_index_from_XYZ(
    ...     XYZ_spl_t,
    ...     XYZ_std_t,
    ...     XYZ_spl_r,
    ...     XYZ_std_r,
    ...     correction="additive",
    ...     method="CIE 1976",
    ...     illuminant=CCS_ILLUMINANTS["CIE 1964 10 Degree Standard Observer"]["A"],
    ... )  # doctest: +ELLIPSIS
    4.6910648...
    """

    if correction == "additive":
        XYZ_corr_t = as_array(XYZ_spl_t) - (as_array(XYZ_spl_r) - as_array(XYZ_std_r))

    elif correction == "multiplicative":
        XYZ_corr_t = as_array(XYZ_spl_t) * (as_array(XYZ_std_r) / as_array(XYZ_spl_r))

    Lab_std_t = XYZ_to_Lab(XYZ_std_t, **filter_kwargs(XYZ_to_Lab, **kwargs))
    Lab_corr_t = XYZ_to_Lab(XYZ_corr_t, **filter_kwargs(XYZ_to_Lab, **kwargs))

    return colour.difference.delta_E(
        Lab_std_t,
        Lab_corr_t,
        method=method,
        **kwargs,
    )
