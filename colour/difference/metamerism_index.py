"""
:math:`M_{t}` - Metamerism Index
================================

Define the :math:`M_{t}` *metamerism index* computation objects:

-   :func:`colour.difference.Lab_to_metamerism_index`
-   :func:`colour.difference.XYZ_to_metamerism_index`

References
----------
-   :cite:`InternationalOrganizationforStandardization2024` : International
    Organization for Standardization. (2024). INTERNATIONAL STANDARD ISO
    18314-4 - Analytical colorimetry Part 4: Metamerism index for pairs of
    samples for change of illuminant. https://www.iso.org/standard/85116.html
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        Domain1,
        Domain100,
        Literal,
        LiteralDeltaEMethod,
        NDArrayFloat,
    )
    from colour.difference import DeltaE_Specification

import colour
from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    tristimulus_weighting_factors_integration,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import XYZ_to_Lab
from colour.utilities import (
    as_array,
    attest,
    domain_range_scale,
    filter_kwargs,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Lab_to_metamerism_index",
    "XYZ_to_metamerism_index",
    "sd_to_metamerism_index",
]


@typing.overload
def Lab_to_metamerism_index(
    Lab_spl_t: Domain100,
    Lab_std_t: Domain100,
    Lab_spl_r: Domain100,
    Lab_std_r: Domain100,
    correction: str = ...,
    method: LiteralDeltaEMethod | str = ...,
    *,
    additional_data: Literal[False] = False,
    **kwargs: Any,
) -> NDArrayFloat: ...


@typing.overload
def Lab_to_metamerism_index(
    Lab_spl_t: Domain100,
    Lab_std_t: Domain100,
    Lab_spl_r: Domain100,
    Lab_std_r: Domain100,
    correction: str = ...,
    method: LiteralDeltaEMethod | str = ...,
    *,
    additional_data: Literal[True],
    **kwargs: Any,
) -> DeltaE_Specification: ...


def Lab_to_metamerism_index(
    Lab_spl_t: Domain100,
    Lab_std_t: Domain100,
    Lab_spl_r: Domain100,
    Lab_std_r: Domain100,
    correction: Literal["Additive", "Multiplicative"] | str = "Additive",
    method: LiteralDeltaEMethod | str = "CIE 2000",
    additional_data: bool = False,
    **kwargs: Any,
) -> NDArrayFloat | DeltaE_Specification:
    """
    Compute the *metamerism index* :math:`M_{t}` between four specified
    *CIE L\\*a\\*b\\** colourspace arrays.

    Before computing the *metamerism index*, apply either an additive or
    multiplicative correction. The correction is based on the difference
    between the colour sample and colour standard under the reference
    illuminant and is applied to the colour sample under the test illuminant.
    The correction is applied in *CIE L\\*a\\*b\\** colourspace, which is then
    used to compute the *metamerism index*.

    :cite:`InternationalOrganizationforStandardization2024` recommends using
    additive correction in *CIE L\\*a\\*b\\**.

    Parameters
    ----------
    Lab_spl_t
        *CIE L\\*a\\*b\\** colourspace array of the colour sample under the test
        illuminant.
    Lab_std_t
        *CIE L\\*a\\*b\\** colourspace array of the colour standard under the
        test illuminant.
    Lab_spl_r
        *CIE L\\*a\\*b\\** colourspace array of the colour sample under the
        reference illuminant.
    Lab_std_r
        *CIE L\\*a\\*b\\** colourspace array of the colour standard under the
        reference illuminant.
    correction
        Correction method to apply, either ``'Additive'`` or
        ``'Multiplicative'``.
    method
        Colour-difference method.
    additional_data
        Whether to output additional data.

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
    :class:`numpy.ndarray` or :class:`DeltaE_Specification`
        *Metamerism index* :math:`M_{t}`.

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
    :cite:`InternationalOrganizationforStandardization2024`

    Examples
    --------
    >>> import numpy as np
    >>> Lab_std_r = np.array([39.0908, -21.3269, 22.6657])
    >>> Lab_std_t = np.array([38.17781, -17.4939, 21.0618])
    >>> Lab_spl_r = np.array([38.83253, -19.8787, 20.0453])
    >>> Lab_spl_t = np.array([37.9013, -19.56327, 16.9346])
    >>> Lab_to_metamerism_index(
    ...     Lab_spl_t,
    ...     Lab_std_t,
    ...     Lab_spl_r,
    ...     Lab_std_r,
    ...     correction="Additive",
    ...     method="CIE 1976",
    ... )  # doctest: +ELLIPSIS
    np.float64(3.8267581...)
    >>> Lab_to_metamerism_index(
    ...     Lab_spl_t,
    ...     Lab_std_t,
    ...     Lab_spl_r,
    ...     Lab_std_r,
    ...     correction="Multiplicative",
    ...     method="CIE 1976",
    ... )  # doctest: +ELLIPSIS
    np.float64(3.9842216...)
    """

    correction = validate_method(correction, ("Additive", "Multiplicative"))

    if correction == "additive":
        Lab_corr_t = as_array(Lab_spl_t) - (as_array(Lab_spl_r) - as_array(Lab_std_r))

    elif correction == "multiplicative":
        Lab_corr_t = as_array(Lab_spl_t) * (as_array(Lab_std_r) / as_array(Lab_spl_r))

    return colour.difference.delta_E(
        Lab_std_t,
        Lab_corr_t,
        method=method,
        additional_data=additional_data,
        **kwargs,
    )


@typing.overload
def XYZ_to_metamerism_index(
    XYZ_spl_t: Domain1,
    XYZ_std_t: Domain1,
    XYZ_spl_r: Domain1,
    XYZ_std_r: Domain1,
    correction: str = ...,
    method: LiteralDeltaEMethod | str = ...,
    *,
    additional_data: Literal[False] = False,
    **kwargs: Any,
) -> NDArrayFloat: ...


@typing.overload
def XYZ_to_metamerism_index(
    XYZ_spl_t: Domain1,
    XYZ_std_t: Domain1,
    XYZ_spl_r: Domain1,
    XYZ_std_r: Domain1,
    correction: str = ...,
    method: LiteralDeltaEMethod | str = ...,
    *,
    additional_data: Literal[True],
    **kwargs: Any,
) -> DeltaE_Specification: ...


def XYZ_to_metamerism_index(
    XYZ_spl_t: Domain1,
    XYZ_std_t: Domain1,
    XYZ_spl_r: Domain1,
    XYZ_std_r: Domain1,
    correction: Literal["Additive", "Multiplicative"] | str = "Multiplicative",
    method: LiteralDeltaEMethod | str = "CIE 2000",
    additional_data: bool = False,
    **kwargs: Any,
) -> NDArrayFloat | DeltaE_Specification:
    """
    Compute the *metamerism index* :math:`M_{t}` from four specified
    *CIE XYZ* colourspace arrays.

    Before computing the *metamerism index*, apply either an additive or
    multiplicative correction. The correction is based on the difference
    between the colour sample and colour standard under the reference
    illuminant and is applied to the colour sample under the test illuminant.
    The correction is applied in *CIE XYZ* colourspace. Afterwards, convert
    to *CIE L\\*a\\*b\\** colourspace to compute the *metamerism index*.

    :cite:`InternationalOrganizationforStandardization2024` recommends using
    multiplicative correction in *CIE L\\*a\\*b\\**.

    Parameters
    ----------
    XYZ_spl_t
        *CIE XYZ* tristimulus array of the colour sample under the test
        illuminant.
    XYZ_std_t
        *CIE XYZ* tristimulus array of the colour standard under the test
        illuminant.
    XYZ_spl_r
        *CIE XYZ* tristimulus array of the colour sample under the reference
        illuminant.
    XYZ_std_r
        *CIE XYZ* tristimulus array of the colour standard under the reference
        illuminant.
    correction
        Correction method to apply, either ``'Additive'`` or
        ``'Multiplicative'``.
    method
        Colour-difference method.
    additional_data
        Whether to output additional data.

    Other Parameters
    ----------------
    illuminant
        {:func:`colour.models.XYZ_to_Lab`},
        Test *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
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
    :class:`numpy.ndarray` or :class:`DeltaE_Specification`
        *Metamerism index* :math:`M_{t}`.

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
    :cite:`InternationalOrganizationforStandardization2024`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import CCS_ILLUMINANTS
    >>> XYZ_std_r = np.array([7.6576, 10.7116, 5.0731]) / 100
    >>> XYZ_std_t = np.array([8.96442, 10.1878, 1.6663]) / 100
    >>> XYZ_spl_r = np.array([7.6933, 10.5616, 5.54474]) / 100
    >>> XYZ_spl_t = np.array([8.56438, 10.0324, 1.9315]) / 100
    >>> XYZ_to_metamerism_index(
    ...     XYZ_spl_t,
    ...     XYZ_std_t,
    ...     XYZ_spl_r,
    ...     XYZ_std_r,
    ...     correction="multiplicative",
    ...     method="CIE 1976",
    ...     illuminant=CCS_ILLUMINANTS["CIE 1964 10 Degree Standard Observer"]["A"],
    ... )  # doctest: +ELLIPSIS
    np.float64(3.7906989...)
    >>> XYZ_to_metamerism_index(
    ...     XYZ_spl_t,
    ...     XYZ_std_t,
    ...     XYZ_spl_r,
    ...     XYZ_std_r,
    ...     correction="additive",
    ...     method="CIE 1976",
    ...     illuminant=CCS_ILLUMINANTS["CIE 1964 10 Degree Standard Observer"]["A"],
    ... )  # doctest: +ELLIPSIS
    np.float64(4.6910648...)
    """

    correction = validate_method(correction, ("Additive", "Multiplicative"))

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
        additional_data=additional_data,
        **kwargs,
    )


def sd_to_metamerism_index(
    sd_spl: SpectralDistribution | MultiSpectralDistributions,
    sd_std: SpectralDistribution | MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions,
    illuminant_r: SpectralDistribution,
    illuminant_t: SpectralDistribution,
    method: LiteralDeltaEMethod | str = "CIE 2000",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute the *metamerism index* :math:`M_t` of a sample pair for change of
    illuminant using the spectral correction method as defined in
    *ISO 18314-4:2024*.

    The spectral correction method (*Cohen-Kappauf* matrix decomposition)
    computes a corrected spectral distribution :math:`\\bar{N}_{spl,corr}` such
    that the sample and standard have identical tristimulus values under the
    reference illuminant. The *metamerism index* is then calculated as the
    colour difference between the standard and corrected sample under the test
    illuminant.

    The projection matrix :math:`R` is computed from the weighting matrix
    :math:`A` as:

    :math:`R = A \\cdot (A^T \\cdot A)^{-1} \\cdot A^T`

    The corrected spectral distribution is:

    :math:`\\bar{N}_{spl,corr} = R \\cdot \\bar{N}_{std} + (I - R) \\cdot \
\\bar{N}_{spl}`

    Parameters
    ----------
    sd_spl
        Spectral distribution of the sample :math:`\\bar{N}_{spl}`.
    sd_std
        Spectral distribution of the standard :math:`\\bar{N}_{std}`.
    cmfs
        Standard observer colour matching functions.
    illuminant_r
        Spectral distribution of the reference illuminant.
    illuminant_t
        Spectral distribution of the test illuminant.
    method
        Colour difference formula.

    Other Parameters
    ----------------
    illuminant
        {:func:`colour.models.XYZ_to_Lab`},
        Test *illuminant* *CIE xy* chromaticity coordinates or *CIE xyY*
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
        *Metamerism index* :math:`M_t`.

    Notes
    -----
    -   This method implements *ISO 18314-4:2024*, Section 8.3.3,
        *Spectral correction (Cohen-Kappauf matrix R)*.
    -   The spectral correction ensures that
        :math:`\\bar{W}_{std} = \\bar{W}_{spl,corr}` under the reference
        illuminant, i.e., the colour difference is zero.

    References
    ----------
    :cite:`InternationalOrganizationforStandardization2024`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, CCS_ILLUMINANTS
    >>> from colour.colorimetry import SpectralShape
    >>> shape = SpectralShape(400, 700, 10)
    >>> N_spl = np.array(
    ...     [
    ...         0.0379,
    ...         0.0403,
    ...         0.0415,
    ...         0.0427,
    ...         0.045,
    ...         0.0483,
    ...         0.0521,
    ...         0.0572,
    ...         0.0624,
    ...         0.0673,
    ...         0.0777,
    ...         0.1026,
    ...         0.1307,
    ...         0.145,
    ...         0.1484,
    ...         0.1455,
    ...         0.1375,
    ...         0.1254,
    ...         0.1099,
    ...         0.0908,
    ...         0.0698,
    ...         0.0526,
    ...         0.0423,
    ...         0.0368,
    ...         0.0331,
    ...         0.0306,
    ...         0.0297,
    ...         0.0311,
    ...         0.034,
    ...         0.038,
    ...         0.0421,
    ...     ]
    ... )
    >>> N_std = np.array(
    ...     [
    ...         0.099,
    ...         0.1244,
    ...         0.0933,
    ...         0.0596,
    ...         0.0405,
    ...         0.0322,
    ...         0.0299,
    ...         0.0316,
    ...         0.0377,
    ...         0.0507,
    ...         0.0681,
    ...         0.0968,
    ...         0.1522,
    ...         0.2014,
    ...         0.1991,
    ...         0.159,
    ...         0.1162,
    ...         0.0843,
    ...         0.0655,
    ...         0.057,
    ...         0.0553,
    ...         0.0582,
    ...         0.0638,
    ...         0.0716,
    ...         0.0818,
    ...         0.0959,
    ...         0.1131,
    ...         0.1317,
    ...         0.149,
    ...         0.1656,
    ...         0.1832,
    ...     ]
    ... )
    >>> N_spl = SpectralDistribution(N_spl, shape)
    >>> N_std = SpectralDistribution(N_std, shape)
    >>> cmfs = MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    >>> r = SDS_ILLUMINANTS["D65"]
    >>> t = SDS_ILLUMINANTS["A"]
    >>> sd_to_metamerism_index(
    ...     N_spl,
    ...     N_std,
    ...     cmfs,
    ...     r,
    ...     t,
    ...     method="CIE 1976",
    ...     illuminant=colour.CCS_ILLUMINANTS["CIE 1964 10 Degree Standard Observer"][
    ...         "A"
    ...     ],
    ... )  # doctest: +ELLIPSIS
    np.float64(3.4766679...)
    """

    attest(
        sd_spl.shape == sd_std.shape,
        "`sd_spl` and `sd_std` spectral distributions must have the same shape!",
    )

    shape = sd_spl.shape

    A_r = tristimulus_weighting_factors_integration(cmfs, illuminant_r, shape=shape)
    A_t = tristimulus_weighting_factors_integration(cmfs, illuminant_t, shape=shape)

    R = np.dot(
        np.dot(A_r, np.linalg.inv(np.dot(np.transpose(A_r), A_r))), np.transpose(A_r)
    )

    sd_spl_corr = np.dot(R, sd_std.values) + np.dot(
        np.identity(R.shape[0]) - R, sd_spl.values
    )
    sd_spl_corr = SpectralDistribution(sd_spl_corr, shape)

    XYZ_spl_corr_t = np.dot(sd_spl_corr.values, A_t) / 100
    XYZ_std_t = np.dot(sd_std.values, A_t) / 100
    XYZ_spl_corr = np.dot(sd_spl_corr.values, A_r) / 100
    XYZ_std = np.dot(sd_std.values, A_r) / 100

    attest(
        np.allclose(XYZ_std, XYZ_spl_corr, atol=TOLERANCE_ABSOLUTE_TESTS),
        "The corrected sample under reference illuminant must be equal "
        "to the standard under reference illuminant!",
    )

    with domain_range_scale("ignore"):
        Lab_std_t = XYZ_to_Lab(XYZ_std_t, **filter_kwargs(XYZ_to_Lab, **kwargs))
        Lab_spl_corr_t = XYZ_to_Lab(
            XYZ_spl_corr_t, **filter_kwargs(XYZ_to_Lab, **kwargs)
        )

        return colour.difference.delta_E(
            Lab_std_t,
            Lab_spl_corr_t,
            method=method,
            **kwargs,
        )
