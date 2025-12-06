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

from numpy import allclose, identity, linalg

import colour
from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    get_tristimulus_weighting_factors_integration,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
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
    attest,
    filter_kwargs,
    from_range_1,
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


def Lab_to_metamerism_index(
    Lab_spl_t: Domain100,
    Lab_std_t: Domain100,
    Lab_spl_r: Domain100,
    Lab_std_r: Domain100,
    correction: Literal["Additive", "Multiplicative"] | str = "Additive",
    method: LiteralDeltaEMethod | str = "CIE 2000",
    **kwargs: Any,
) -> NDArrayFloat:
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
    3.8267581...
    >>> Lab_to_metamerism_index(
    ...     Lab_spl_t,
    ...     Lab_std_t,
    ...     Lab_spl_r,
    ...     Lab_std_r,
    ...     correction="Multiplicative",
    ...     method="CIE 1976",
    ... )  # doctest: +ELLIPSIS
    3.9842216...
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
        **kwargs,
    )


def XYZ_to_metamerism_index(
    XYZ_spl_t: Domain1,
    XYZ_std_t: Domain1,
    XYZ_spl_r: Domain1,
    XYZ_std_r: Domain1,
    correction: Literal["Additive", "Multiplicative"] | str = "Multiplicative",
    method: LiteralDeltaEMethod | str = "CIE 2000",
    **kwargs: Any,
) -> NDArrayFloat:
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
    3.7906989...
    >>> XYZ_to_metamerism_index(
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
    Compute the *metamerism index* :math:`M_{t}` from the specified
    spectral distributions.

    Before computing the *metamerism index*, we apply a spectral correction.
    The correction aligns the sample spectral distribution to the standard
    spectral distribution so that under reference illumination there exists
    no colour different between the two.
    Afterwards, we compute the corresponding *CIE XYZ* colourspace coordinates
    under both reference and test illuminants and convert them to
    *CIE L\\*a\\*b\\** colourspace to compute the *metamerism index*.

    Parameters
    ----------
    sd_spl
        Spectral distribution of the colour sample.
        If an `ArrayLike` the wavelengths are expected to be in the last axis,
        e.g., for a spectral array with 77 bins, ``sd`` shape could be (77, )
        or (1, 77).
    sd_std
        Spectral distribution of the colour standard.
        If an `ArrayLike` the wavelengths are expected to be in the last axis,
        e.g., for a spectral array with 77 bins, ``sd`` shape could be (77, )
        or (1, 77).
    cmfs
        Standard observer colour matching functions.
    illuminant_r
        Illuminant spectral distribution of the reference illuminant.
    illuminant_t
        Illuminant spectral distribution of the test illuminant.
    method
        Colour-difference method.

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
    3.4766679...
    """

    attest(
        sd_spl.shape == sd_std.shape,
        "`sd_spl` and `sd_std` spectral distributions must have the same shape!",
    )

    shape = sd_spl.shape

    A = get_tristimulus_weighting_factors_integration(cmfs, illuminant_r, shape=shape)
    A_t = get_tristimulus_weighting_factors_integration(cmfs, illuminant_t, shape=shape)

    R = A @ linalg.inv(A.T @ A) @ A.T

    sd_corr = R @ sd_std.values + (identity(R.shape[0]) - R) @ sd_spl.values
    sd_corr = SpectralDistribution(sd_corr, shape)

    XYZ_corr_t = from_range_1((sd_corr.values @ A_t) / 100)
    XYZ_std_t = from_range_1((sd_std.values @ A_t) / 100)
    XYZ_corr_r = from_range_1((sd_corr.values @ A) / 100)
    XYZ_std_r = from_range_1((sd_std.values @ A) / 100)

    # must be equal ! otherwise correction failed !
    attest(
        allclose(XYZ_std_r, XYZ_corr_r, atol=TOLERANCE_ABSOLUTE_TESTS),
        "The corrected sample under reference illuminant must be equal "
        "to the standard under reference illuminant! Otherwise the correction"
        "has failed.",
    )

    Lab_std_t = XYZ_to_Lab(XYZ_std_t, **filter_kwargs(XYZ_to_Lab, **kwargs))
    Lab_corr_t = XYZ_to_Lab(XYZ_corr_t, **filter_kwargs(XYZ_to_Lab, **kwargs))

    return colour.difference.delta_E(
        Lab_std_t,
        Lab_corr_t,
        method=method,
        **kwargs,
    )
