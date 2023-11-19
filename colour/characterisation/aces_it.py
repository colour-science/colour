"""
Academy Color Encoding System - Input Transform
===============================================

Defines the *Academy Color Encoding System* (ACES) *Input Transform* utilities:

-   :func:`colour.sd_to_aces_relative_exposure_values`
-   :func:`colour.sd_to_ACES2065_1`
-   :func:`colour.characterisation.read_training_data_rawtoaces_v1`
-   :func:`colour.characterisation.generate_illuminants_rawtoaces_v1`
-   :func:`colour.characterisation.white_balance_multipliers`
-   :func:`colour.characterisation.best_illuminant`
-   :func:`colour.characterisation.normalise_illuminant`
-   :func:`colour.characterisation.training_data_sds_to_RGB`
-   :func:`colour.characterisation.training_data_sds_to_XYZ`
-   :func:`colour.characterisation.optimisation_factory_rawtoaces_v1`
-   :func:`colour.characterisation.optimisation_factory_Jzazbz`
-   :func:`colour.characterisation.optimisation_factory_Oklab_15`
-   :func:`colour.matrix_idt`
-   :func:`colour.camera_RGB_to_ACES2065_1`

References
----------
-   :cite:`Dyer2017` : Dyer, S., Forsythe, A., Irons, J., Mansencal, T., & Zhu,
    M. (2017). RAW to ACES (Version 1.0) [Computer software].
-   :cite:`Forsythe2018` : Borer, T. (2017). Private Discussion with Mansencal,
    T. and Shaw, N.
-   :cite:`Finlayson2015` : Finlayson, G. D., MacKiewicz, M., & Hurlbert, A.
    (2015). Color Correction Using Root-Polynomial Regression. IEEE
    Transactions on Image Processing, 24(5), 1460-1470.
    doi:10.1109/TIP.2015.2405336
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014q` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-004 - Informative Notes on SMPTE ST 2065-1 - Academy Color
    Encoding Specification (ACES) (pp. 1-40). Retrieved December 19, 2014, from
    http://j.mp/TB-2014-004
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014r` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-012 - Academy Color Encoding System Version 1.0 Component
    Names (pp. 1-8). Retrieved December 19, 2014, from http://j.mp/TB-2014-012
-   :cite:`TheAcademyofMotionPictureArtsandSciences2015c` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2015). Procedure
    P-2013-001 - Recommended Procedures for the Creation and Use of Digital
    Camera System Input Device Transforms (IDTs) (pp. 1-29). Retrieved April
    24, 2015, from http://j.mp/P-2013-001
-   :cite:`TheAcademyofMotionPictureArtsandSciencese` : The Academy of Motion
    Picture Arts and Sciences, Science and Technology Council, & Academy Color
    Encoding System (ACES) Project Subcommittee. (n.d.). Academy Color Encoding
    System. Retrieved February 24, 2014, from
    http://www.oscars.org/science-technology/council/projects/aces.html
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import minimize

from colour.adaptation import matrix_chromatic_adaptation_VonKries
from colour.algebra import (
    euclidean_distance,
    vector_dot,
)
from colour.characterisation import (
    MSDS_ACES_RICD,
    RGB_CameraSensitivities,
    polynomial_expansion_Finlayson2015,
)
from colour.colorimetry import (
    SDS_ILLUMINANTS,
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    handle_spectral_arguments,
    reshape_msds,
    reshape_sd,
    sd_blackbody,
    sd_CIE_illuminant_D_series,
    sd_to_XYZ,
    sds_and_msds_to_msds,
)
from colour.hints import (
    ArrayLike,
    Callable,
    DTypeFloat,
    LiteralChromaticAdaptationTransform,
    Mapping,
    NDArrayFloat,
    Tuple,
    cast,
)
from colour.io import read_sds_from_csv_file
from colour.models import (
    XYZ_to_Jzazbz,
    XYZ_to_Lab,
    XYZ_to_Oklab,
    XYZ_to_xy,
    xy_to_XYZ,
)
from colour.models.rgb import (
    RGB_COLOURSPACE_ACES2065_1,
    RGB_Colourspace,
    RGB_to_XYZ,
    XYZ_to_RGB,
)
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import (
    CanonicalMapping,
    as_float,
    as_float_array,
    from_range_1,
    ones,
    optional,
    runtime_warning,
    suppress_warnings,
    tsplit,
    zeros,
)
from colour.utilities.deprecation import handle_arguments_deprecation

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "FLARE_PERCENTAGE",
    "S_FLARE_FACTOR",
    "sd_to_aces_relative_exposure_values",
    "sd_to_ACES2065_1",
    "SPECTRAL_SHAPE_RAWTOACES",
    "ROOT_RESOURCES_RAWTOACES",
    "read_training_data_rawtoaces_v1",
    "generate_illuminants_rawtoaces_v1",
    "white_balance_multipliers",
    "best_illuminant",
    "normalise_illuminant",
    "training_data_sds_to_RGB",
    "training_data_sds_to_XYZ",
    "whitepoint_preserving_matrix",
    "optimisation_factory_rawtoaces_v1",
    "optimisation_factory_Jzazbz",
    "optimisation_factory_Oklab_15",
    "matrix_idt",
    "camera_RGB_to_ACES2065_1",
]

FLARE_PERCENTAGE: float = 0.00500
"""Flare percentage in the *ACES* system."""

S_FLARE_FACTOR: float = 0.18000 / (0.18000 + FLARE_PERCENTAGE)
"""Flare modulation factor in the *ACES* system."""


def sd_to_aces_relative_exposure_values(
    sd: SpectralDistribution,
    illuminant: SpectralDistribution | None = None,
    chromatic_adaptation_transform: LiteralChromaticAdaptationTransform
    | str
    | None = "CAT02",
    **kwargs,
) -> NDArrayFloat:
    """
    Convert given spectral distribution to *ACES2065-1* colourspace relative
    exposure values.

    Parameters
    ----------
    sd
        Spectral distribution.
    illuminant
        *Illuminant* spectral distribution, default to
        *CIE Standard Illuminant D65*.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform.

    Returns
    -------
    :class:`numpy.ndarray`
        *ACES2065-1* colourspace relative exposure values array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The chromatic adaptation method implemented here is a bit unusual
        as it involves building a new colourspace based on *ACES2065-1*
        colourspace primaries but using the whitepoint of the illuminant that
        the spectral distribution was measured under.

    References
    ----------
    :cite:`Forsythe2018`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> from colour import SDS_COLOURCHECKERS
    >>> sd = SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"]
    >>> sd_to_aces_relative_exposure_values(
    ...     sd, chromatic_adaptation_transform=None
    ... )  # doctest: +ELLIPSIS
    array([ 0.1171814...,  0.0866360...,  0.0589726...])
    >>> sd_to_aces_relative_exposure_values(
    ...     sd, apply_chromatic_adaptation=True
    ... )
    ... # doctest: +ELLIPSIS
    array([ 0.1180779...,  0.0869031...,  0.0589125...])
    """

    if isinstance(chromatic_adaptation_transform, bool):  # pragma: no cover
        if chromatic_adaptation_transform is True:
            chromatic_adaptation_transform = "CAT02"
        elif chromatic_adaptation_transform is False:
            chromatic_adaptation_transform = None

        kwargs = {"apply_chromatic_adaptation": True}

    handle_arguments_deprecation(
        {
            "ArgumentRemoved": ["apply_chromatic_adaptation"],
        },
        **kwargs,
    )

    illuminant = optional(illuminant, SDS_ILLUMINANTS["D65"])

    shape = MSDS_ACES_RICD.shape
    if sd.shape != MSDS_ACES_RICD.shape:
        sd = reshape_sd(sd, shape, copy=False)

    if illuminant.shape != MSDS_ACES_RICD.shape:
        illuminant = reshape_sd(illuminant, shape, copy=False)

    s_v = sd.values
    i_v = illuminant.values

    r_bar, g_bar, b_bar = tsplit(MSDS_ACES_RICD.values)

    def k(x: NDArrayFloat, y: NDArrayFloat) -> DTypeFloat:
        """Compute the :math:`K_r`, :math:`K_g` or :math:`K_b` scale factors."""

        return 1 / np.sum(x * y)

    k_r = k(i_v, r_bar)
    k_g = k(i_v, g_bar)
    k_b = k(i_v, b_bar)

    E_r = k_r * np.sum(i_v * s_v * r_bar)
    E_g = k_g * np.sum(i_v * s_v * g_bar)
    E_b = k_b * np.sum(i_v * s_v * b_bar)

    E_rgb = np.array([E_r, E_g, E_b])

    # Accounting for flare.
    E_rgb += FLARE_PERCENTAGE
    E_rgb *= S_FLARE_FACTOR

    if chromatic_adaptation_transform is not None:
        XYZ = RGB_to_XYZ(
            E_rgb,
            RGB_Colourspace(
                "~ACES2065-1",
                RGB_COLOURSPACE_ACES2065_1.primaries,
                XYZ_to_xy(sd_to_XYZ(illuminant) / 100),
                illuminant.name,
            ),
            RGB_COLOURSPACE_ACES2065_1.whitepoint,
            chromatic_adaptation_transform,
        )
        E_rgb = XYZ_to_RGB(XYZ, RGB_COLOURSPACE_ACES2065_1)

    return from_range_1(E_rgb)


sd_to_ACES2065_1 = sd_to_aces_relative_exposure_values

SPECTRAL_SHAPE_RAWTOACES: SpectralShape = SpectralShape(380, 780, 5)
"""Default spectral shape according to *RAW to ACES* v1."""

ROOT_RESOURCES_RAWTOACES: str = os.path.join(
    os.path.dirname(__file__), "datasets", "rawtoaces"
)
"""
*RAW to ACES* resources directory.

Notes
-----
-   *Colour* only ships a minimal dataset from *RAW to ACES*, please see
    `Colour - Datasets <https://github.com/colour-science/colour-datasets>`_
    for the complete *RAW to ACES* v1 dataset, i.e. *3372171*.
"""

_TRAINING_DATA_RAWTOACES_V1: MultiSpectralDistributions | None = None


def read_training_data_rawtoaces_v1() -> MultiSpectralDistributions:
    """
    Read the *RAW to ACES* v1 190 patches.

    Returns
    -------
    :class:`colour.MultiSpectralDistributions`
        *RAW to ACES* v1 190 patches.

    References
    ----------
    :cite:`Dyer2017`

    Examples
    --------
    >>> len(read_training_data_rawtoaces_v1().labels)
    190
    """

    global _TRAINING_DATA_RAWTOACES_V1  # noqa: PLW0603

    if _TRAINING_DATA_RAWTOACES_V1 is not None:
        training_data = _TRAINING_DATA_RAWTOACES_V1
    else:
        path = os.path.join(ROOT_RESOURCES_RAWTOACES, "190_Patches.csv")
        training_data = sds_and_msds_to_msds(
            list(read_sds_from_csv_file(path).values())
        )

        _TRAINING_DATA_RAWTOACES_V1 = training_data

    return training_data


_ILLUMINANTS_RAWTOACES_V1: CanonicalMapping | None = None


def generate_illuminants_rawtoaces_v1() -> CanonicalMapping:
    """
    Generate a series of illuminants according to *RAW to ACES* v1:

    -   *CIE Illuminant D Series* in range [4000, 25000] kelvin degrees.
    -   *Blackbodies* in range [1000, 3500] kelvin degrees.
    -   A.M.P.A.S. variant of *ISO 7589 Studio Tungsten*.

    Returns
    -------
    :class:`colour.utilities.CanonicalMapping`
        Series of illuminants.

    Notes
    -----
    -   This definition introduces a few differences compared to
        *RAW to ACES* v1: *CIE Illuminant D Series* are computed in range
        [4002.15, 7003.77] kelvin degrees and the :math:`C_2` change is not
        used in *RAW to ACES* v1.

    References
    ----------
    :cite:`Dyer2017`

    Examples
    --------
    >>> list(sorted(generate_illuminants_rawtoaces_v1().keys()))
    ['1000K Blackbody', '1500K Blackbody', '2000K Blackbody', \
'2500K Blackbody', '3000K Blackbody', '3500K Blackbody', 'D100', 'D105', \
'D110', 'D115', 'D120', 'D125', 'D130', 'D135', 'D140', 'D145', 'D150', \
'D155', 'D160', 'D165', 'D170', 'D175', 'D180', 'D185', 'D190', 'D195', \
'D200', 'D205', 'D210', 'D215', 'D220', 'D225', 'D230', 'D235', 'D240', \
'D245', 'D250', 'D40', 'D45', 'D50', 'D55', 'D60', 'D65', 'D70', 'D75', \
'D80', 'D85', 'D90', 'D95', 'iso7589']
    """

    global _ILLUMINANTS_RAWTOACES_V1  # noqa: PLW0603

    if _ILLUMINANTS_RAWTOACES_V1 is not None:
        illuminants = _ILLUMINANTS_RAWTOACES_V1
    else:
        illuminants = CanonicalMapping()

        # CIE Illuminants D Series from 4000K to 25000K.
        for i in np.arange(4000, 25000 + 500, 500):
            CCT = i * 1.4388 / 1.4380
            xy = CCT_to_xy_CIE_D(CCT)
            sd = sd_CIE_illuminant_D_series(xy)
            sd.name = f"D{int(CCT / 100):d}"
            illuminants[sd.name] = sd.align(SPECTRAL_SHAPE_RAWTOACES)

        # TODO: Remove when removing the "colour.sd_blackbody" definition
        # warning.
        with suppress_warnings(colour_usage_warnings=True):
            # Blackbody from 1000K to 4000K.
            for i in np.arange(1000, 4000, 500):
                sd = sd_blackbody(i, SPECTRAL_SHAPE_RAWTOACES)
                illuminants[sd.name] = sd

        # A.M.P.A.S. variant of ISO 7589 Studio Tungsten.
        sd = read_sds_from_csv_file(
            os.path.join(
                ROOT_RESOURCES_RAWTOACES, "AMPAS_ISO_7589_Tungsten.csv"
            )
        )["iso7589"]
        illuminants.update({sd.name: sd})

        _ILLUMINANTS_RAWTOACES_V1 = illuminants

    return illuminants


def white_balance_multipliers(
    sensitivities: RGB_CameraSensitivities, illuminant: SpectralDistribution
) -> NDArrayFloat:
    """
    Compute the *RGB* white balance multipliers for given camera *RGB*
    spectral sensitivities and illuminant.

    Parameters
    ----------
    sensitivities
         Camera *RGB* spectral sensitivities.
    illuminant
        Illuminant spectral distribution.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* white balance multipliers.

    References
    ----------
    :cite:`Dyer2017`

    Examples
    --------
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_RAWTOACES,
    ...     "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
    ... )
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values()
    ... )
    >>> illuminant = SDS_ILLUMINANTS["D55"]
    >>> white_balance_multipliers(sensitivities, illuminant)
    ... # doctest: +ELLIPSIS
    array([ 2.3414154...,  1.        ,  1.5163375...])
    """

    shape = sensitivities.shape
    if illuminant.shape != shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{shape}".'
        )
        illuminant = reshape_sd(illuminant, shape, copy=False)

    RGB_w = 1 / np.sum(
        sensitivities.values * illuminant.values[..., None], axis=0
    )
    RGB_w *= 1 / np.min(RGB_w)

    return RGB_w


def best_illuminant(
    RGB_w: ArrayLike,
    sensitivities: RGB_CameraSensitivities,
    illuminants: Mapping,
) -> SpectralDistribution:
    """
    Select the best illuminant for given *RGB* white balance multipliers, and
    sensitivities in given series of illuminants.

    Parameters
    ----------
    RGB_w
        *RGB* white balance multipliers.
    sensitivities
         Camera *RGB* spectral sensitivities.
    illuminants
        Illuminant spectral distributions to choose the best illuminant from.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Best illuminant.

    Examples
    --------
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_RAWTOACES,
    ...     "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
    ... )
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values()
    ... )
    >>> illuminants = generate_illuminants_rawtoaces_v1()
    >>> RGB_w = white_balance_multipliers(
    ...     sensitivities, SDS_ILLUMINANTS["FL2"]
    ... )
    >>> best_illuminant(RGB_w, sensitivities, illuminants).name
    'D40'
    """

    RGB_w = as_float_array(RGB_w)

    sse = np.inf
    illuminant_b = None
    for illuminant in illuminants.values():
        RGB_wi = white_balance_multipliers(sensitivities, illuminant)
        sse_c = np.sum((RGB_wi / RGB_w - 1) ** 2)
        if sse_c < sse:
            sse = sse_c
            illuminant_b = illuminant

    return cast(SpectralDistribution, illuminant_b)


def normalise_illuminant(
    illuminant: SpectralDistribution, sensitivities: RGB_CameraSensitivities
) -> SpectralDistribution:
    """
    Normalise given illuminant with given camera *RGB* spectral sensitivities.

    The multiplicative inverse scaling factor :math:`k` is computed by
    multiplying the illuminant by the sensitivities channel with the maximum
    value.

    Parameters
    ----------
    illuminant
        Illuminant spectral distribution.
    sensitivities
         Camera *RGB* spectral sensitivities.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Normalised illuminant.

    Examples
    --------
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_RAWTOACES,
    ...     "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
    ... )
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values()
    ... )
    >>> illuminant = SDS_ILLUMINANTS["D55"]
    >>> np.sum(illuminant.values)  # doctest: +ELLIPSIS
    7276.1490000...
    >>> np.sum(normalise_illuminant(illuminant, sensitivities).values)
    ... # doctest: +ELLIPSIS
    3.4390373...
    """

    shape = sensitivities.shape
    if illuminant.shape != shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{shape}".'
        )
        illuminant = reshape_sd(illuminant, shape)

    c_i = np.argmax(np.max(sensitivities.values, axis=0))
    k = 1 / np.sum(illuminant.values * sensitivities.values[..., c_i])

    return illuminant * k


def training_data_sds_to_RGB(
    training_data: MultiSpectralDistributions,
    sensitivities: RGB_CameraSensitivities,
    illuminant: SpectralDistribution,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Convert given training data to *RGB* tristimulus values using given
    illuminant and given camera *RGB* spectral sensitivities.

    Parameters
    ----------
    training_data
        Training data multi-spectral distributions.
    sensitivities
         Camera *RGB* spectral sensitivities.
    illuminant
        Illuminant spectral distribution.

    Returns
    -------
    :class:`tuple`
        Tuple of training data *RGB* tristimulus values and white balance
        multipliers.

    Examples
    --------
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_RAWTOACES,
    ...     "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
    ... )
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values()
    ... )
    >>> illuminant = normalise_illuminant(
    ...     SDS_ILLUMINANTS["D55"], sensitivities
    ... )
    >>> training_data = read_training_data_rawtoaces_v1()
    >>> RGB, RGB_w = training_data_sds_to_RGB(
    ...     training_data, sensitivities, illuminant
    ... )
    >>> RGB[:5]  # doctest: +ELLIPSIS
    array([[ 0.0207582...,  0.0196857...,  0.0213935...],
           [ 0.0895775...,  0.0891922...,  0.0891091...],
           [ 0.7810230...,  0.7801938...,  0.7764302...],
           [ 0.1995   ...,  0.1995   ...,  0.1995   ...],
           [ 0.5898478...,  0.5904015...,  0.5851076...]])
    >>> RGB_w  # doctest: +ELLIPSIS
    array([ 2.3414154...,  1.        ,  1.5163375...])
    """

    shape = sensitivities.shape
    if illuminant.shape != shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{shape}".'
        )
        illuminant = reshape_sd(illuminant, shape, copy=False)

    if training_data.shape != shape:
        runtime_warning(
            f'Aligning "{training_data.name}" training data shape to "{shape}".'
        )
        training_data = reshape_msds(training_data, shape, copy=False)

    RGB_w = white_balance_multipliers(sensitivities, illuminant)

    RGB = np.dot(
        np.transpose(illuminant.values[..., None] * training_data.values),
        sensitivities.values,
    )

    RGB *= RGB_w

    return RGB, RGB_w


def training_data_sds_to_XYZ(
    training_data: MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions,
    illuminant: SpectralDistribution,
    chromatic_adaptation_transform: LiteralChromaticAdaptationTransform
    | str
    | None = "CAT02",
) -> NDArrayFloat:
    """
    Convert given training data to *CIE XYZ* tristimulus values using given
    illuminant and given standard observer colour matching functions.

    Parameters
    ----------
    training_data
        Training data multi-spectral distributions.
    cmfs
        Standard observer colour matching functions.
    illuminant
        Illuminant spectral distribution.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.

    Returns
    -------
    :class:`numpy.ndarray`
        Training data *CIE XYZ* tristimulus values.

    Examples
    --------
    >>> from colour import MSDS_CMFS
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_RAWTOACES,
    ...     "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
    ... )
    >>> cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values()
    ... )
    >>> illuminant = normalise_illuminant(
    ...     SDS_ILLUMINANTS["D55"], sensitivities
    ... )
    >>> training_data = read_training_data_rawtoaces_v1()
    >>> training_data_sds_to_XYZ(training_data, cmfs, illuminant)[:5]
    ... # doctest: +ELLIPSIS
    array([[ 0.0174353...,  0.0179504...,  0.0196109...],
           [ 0.0855607...,  0.0895735...,  0.0901703...],
           [ 0.7455880...,  0.7817549...,  0.7834356...],
           [ 0.1900528...,  0.1995   ...,  0.2012606...],
           [ 0.5626319...,  0.5914544...,  0.5894500...]])
    """

    shape = cmfs.shape
    if illuminant.shape != shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{shape}".'
        )
        illuminant = reshape_sd(illuminant, shape, copy=False)

    if training_data.shape != shape:
        runtime_warning(
            f'Aligning "{training_data.name}" training data shape to "{shape}".'
        )
        training_data = reshape_msds(training_data, shape, copy=False)

    XYZ = np.dot(
        np.transpose(illuminant.values[..., None] * training_data.values),
        cmfs.values,
    )

    XYZ *= 1 / np.sum(cmfs.values[..., 1] * illuminant.values)

    XYZ_w = np.dot(np.transpose(cmfs.values), illuminant.values)
    XYZ_w *= 1 / XYZ_w[1]

    if chromatic_adaptation_transform is not None:
        M_CAT = matrix_chromatic_adaptation_VonKries(
            XYZ_w,
            xy_to_XYZ(RGB_COLOURSPACE_ACES2065_1.whitepoint),
            chromatic_adaptation_transform,
        )

        XYZ = vector_dot(M_CAT, XYZ)

    return XYZ


def whitepoint_preserving_matrix(
    M: ArrayLike, RGB_w: ArrayLike = ones(3)
) -> NDArrayFloat:
    """
    Normalise given matrix :math:`M` to preserve given white point
    :math:`RGB_w`.

    Parameters
    ----------
    M
        Matrix :math:`M` to normalise.
    RGB_w
        White point :math:`RGB_w` to normalise the matrix :math:`M` with.

    Returns
    -------
    :class:`numpy.ndarray`
        Normalised matrix :math:`M`.

    Examples
    --------
    >>> M = np.arange(9).reshape([3, 3])
    >>> whitepoint_preserving_matrix(M)
    array([[  0.,   1.,   0.],
           [  3.,   4.,  -6.],
           [  6.,   7., -12.]])
    """

    M = as_float_array(M)
    RGB_w = as_float_array(RGB_w)

    M[..., -1] = RGB_w - np.sum(M[..., :-1], axis=-1)

    return M


def optimisation_factory_rawtoaces_v1() -> (
    Tuple[NDArrayFloat, Callable, Callable, Callable]
):
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function according to *RAW to ACES* v1.

    The objective function returns the Euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in *CIE L\\*a\\*b\\** colourspace.

    It implements whitepoint preservation as an optimisation constraint.

    Returns
    -------
    :class:`tuple`
        :math:`x_0` initial values, objective function, *CIE XYZ* colourspace
        to *CIE L\\*a\\*b\\** colourspace function and finaliser function.

    Examples
    --------
    >>> optimisation_factory_rawtoaces_v1()  # doctest: +SKIP
    (array([ 1.,  0.,  0.,  1.,  0.,  0.]), \
<function optimisation_factory_rawtoaces_v1.<locals> \
.objective_function at 0x...>, \
<function optimisation_factory_rawtoaces_v1.<locals>\
.XYZ_to_optimization_colour_model at 0x...>, \
<function optimisation_factory_rawtoaces_v1.<locals>\
.finaliser_function at 0x...>)
    """

    x_0 = as_float_array([1, 0, 0, 1, 0, 0])

    def objective_function(
        M: NDArrayFloat, RGB: NDArrayFloat, Lab: NDArrayFloat
    ) -> NDArrayFloat:
        """Objective function according to *RAW to ACES* v1."""

        M = finaliser_function(M)

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Lab_t = XYZ_to_optimization_colour_model(XYZ_t)

        return as_float(np.linalg.norm(Lab_t - Lab))

    def XYZ_to_optimization_colour_model(XYZ: ArrayLike) -> NDArrayFloat:
        """*CIE XYZ* colourspace to *CIE L\\*a\\*b\\** colourspace function."""

        return XYZ_to_Lab(XYZ, RGB_COLOURSPACE_ACES2065_1.whitepoint)

    def finaliser_function(M: ArrayLike) -> NDArrayFloat:
        """Finaliser function."""

        return whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

    return (
        x_0,
        objective_function,
        XYZ_to_optimization_colour_model,
        finaliser_function,
    )


def optimisation_factory_Jzazbz() -> (
    Tuple[NDArrayFloat, Callable, Callable, Callable]
):
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the :math:`J_za_zb_z`
    colourspace.

    The objective function returns the Euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in the :math:`J_za_zb_z` colourspace.

    It implements whitepoint preservation as a post-optimisation step.

    Returns
    -------
    :class:`tuple`
        :math:`x_0` initial values, objective function, *CIE XYZ* colourspace
        to :math:`J_za_zb_z` colourspace function and finaliser function.

    Examples
    --------
    >>> optimisation_factory_Jzazbz()  # doctest: +SKIP
    (array([ 1.,  0.,  0.,  1.,  0.,  0.]), \
<function optimisation_factory_Jzazbz.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_Jzazbz.<locals>\
.XYZ_to_optimization_colour_model at 0x...>, \
<function optimisation_factory_Jzazbz.<locals>.\
finaliser_function at 0x...>)
    """

    x_0 = as_float_array([1, 0, 0, 1, 0, 0])

    def objective_function(
        M: ArrayLike, RGB: ArrayLike, Jab: ArrayLike
    ) -> NDArrayFloat:
        """:math:`J_za_zb_z` colourspace based objective function."""

        M = finaliser_function(M)

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Jab_t = XYZ_to_optimization_colour_model(XYZ_t)

        return as_float(np.sum(euclidean_distance(Jab, Jab_t)))

    def XYZ_to_optimization_colour_model(XYZ: ArrayLike) -> NDArrayFloat:
        """*CIE XYZ* colourspace to :math:`J_za_zb_z` colourspace function."""

        return XYZ_to_Jzazbz(XYZ)

    def finaliser_function(M: ArrayLike) -> NDArrayFloat:
        """Finaliser function."""

        return whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 2)), zeros((3, 1))])
        )

    return (
        x_0,
        objective_function,
        XYZ_to_optimization_colour_model,
        finaliser_function,
    )


def optimisation_factory_Oklab_15() -> (
    Tuple[NDArrayFloat, Callable, Callable, Callable]
):
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the *Oklab* colourspace.

    The objective function returns the Euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in the *Oklab* colourspace.

    It implements support for *Finlayson et al. (2015)* root-polynomials of
    degree 2 and produces 15 terms.

    Returns
    -------
    :class:`tuple`
        :math:`x_0` initial values, objective function, *CIE XYZ* colourspace
        to *Oklab* colourspace function and finaliser function.

    References
    ----------
    :cite:`Finlayson2015`

    Examples
    --------
    >>> optimisation_factory_Oklab_15()  # doctest: +SKIP
    (array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., \
0.,  1.]), \
<function optimisation_factory_Oklab_15.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_Oklab_15.<locals>\
.XYZ_to_optimization_colour_model at 0x...>, \
<function optimisation_factory_Oklab_15.<locals>.\
finaliser_function at 0x...>)
    """

    x_0 = as_float_array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

    def objective_function(
        M: ArrayLike, RGB: ArrayLike, Jab: ArrayLike
    ) -> NDArrayFloat:
        """*Oklab* colourspace based objective function."""

        M = finaliser_function(M)

        XYZ_t = np.transpose(
            np.dot(
                RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ,
                np.dot(
                    M,
                    np.transpose(
                        polynomial_expansion_Finlayson2015(RGB, 2, True)
                    ),
                ),
            )
        )

        Jab_t = XYZ_to_optimization_colour_model(XYZ_t)

        return as_float(np.sum(euclidean_distance(Jab, Jab_t)))

    def XYZ_to_optimization_colour_model(XYZ: ArrayLike) -> NDArrayFloat:
        """*CIE XYZ* colourspace to *Oklab* colourspace function."""

        return XYZ_to_Oklab(XYZ)

    def finaliser_function(M: ArrayLike) -> NDArrayFloat:
        """Finaliser function."""

        return whitepoint_preserving_matrix(
            np.hstack([np.reshape(M, (3, 5)), zeros((3, 1))])
        )

    return (
        x_0,
        objective_function,
        XYZ_to_optimization_colour_model,
        finaliser_function,
    )


def matrix_idt(
    sensitivities: RGB_CameraSensitivities,
    illuminant: SpectralDistribution,
    training_data: MultiSpectralDistributions | None = None,
    cmfs: MultiSpectralDistributions | None = None,
    optimisation_factory: Callable = optimisation_factory_rawtoaces_v1,
    optimisation_kwargs: dict | None = None,
    chromatic_adaptation_transform: LiteralChromaticAdaptationTransform
    | str
    | None = "CAT02",
    additional_data: bool = False,
) -> (
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]
    | Tuple[NDArrayFloat, NDArrayFloat]
):
    """
    Compute an *Input Device Transform* (IDT) matrix for given camera *RGB*
    spectral sensitivities, illuminant, training data, standard observer colour
    matching functions and optimisation settings according to *RAW to ACES* v1
    and *P-2013-001* procedures.

    Parameters
    ----------
    sensitivities
         Camera *RGB* spectral sensitivities.
    illuminant
        Illuminant spectral distribution.
    training_data
        Training data multi-spectral distributions, defaults to using the
        *RAW to ACES* v1 190 patches.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    optimisation_factory
        Callable producing the objective function and the *CIE XYZ* to
        optimisation colour model function.
    optimisation_kwargs
        Parameters for :func:`scipy.optimize.minimize` definition.
    chromatic_adaptation_transform
        *Chromatic adaptation* transform, if *None* no chromatic adaptation is
        performed.
    additional_data
        If *True*, the *XYZ* and *RGB* tristimulus values are also returned.

    Returns
    -------
    :class:`tuple`
        Tuple of IDT matrix and white balance multipliers or tuple of IDT
        matrix, white balance multipliers, *XYZ* and *RGB* tristimulus values.

    References
    ----------
    :cite:`Dyer2017`, :cite:`TheAcademyofMotionPictureArtsandSciences2015c`

    Examples
    --------
    Computing the IDT matrix for a *CANON EOS 5DMark II* and
    *CIE Illuminant D Series* *D55* using the method given in *RAW to ACES* v1:

    >>> path = os.path.join(
    ...     ROOT_RESOURCES_RAWTOACES,
    ...     "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
    ... )
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values()
    ... )
    >>> illuminant = SDS_ILLUMINANTS["D55"]
    >>> M, RGB_w = matrix_idt(sensitivities, illuminant)
    >>> np.around(M, 3)
    array([[ 0.865, -0.026,  0.161],
           [ 0.057,  1.123, -0.18 ],
           [ 0.024, -0.203,  1.179]])
    >>> RGB_w  # doctest: +ELLIPSIS
    array([ 2.3414154...,  1.        ,  1.5163375...])

    The *RAW to ACES* v1 matrix for the same camera and optimized by
    `Ceres Solver <http://ceres-solver.org>`__ is as follows::

        0.864994 -0.026302 0.161308
        0.056527 1.122997 -0.179524
        0.023683 -0.202547 1.178864

    >>> M, RGB_w = matrix_idt(
    ...     sensitivities,
    ...     illuminant,
    ...     optimisation_factory=optimisation_factory_Jzazbz,
    ... )
    >>> np.around(M, 3)
    array([[ 0.852, -0.009,  0.158],
           [ 0.054,  1.122, -0.176],
           [ 0.023, -0.224,  1.2  ]])
    >>> RGB_w  # doctest: +ELLIPSIS
    array([ 2.3414154...,  1.        ,  1.5163375...])

    >>> M, RGB_w = matrix_idt(
    ...     sensitivities,
    ...     illuminant,
    ...     optimisation_factory=optimisation_factory_Oklab_15,
    ... )
    >>> np.around(M, 3)
    array([[ 0.645, -0.611,  0.107,  0.736,  0.398, -0.275],
           [-0.159,  0.728, -0.091,  0.651,  0.01 , -0.139],
           [-0.172, -0.403,  1.394,  0.51 , -0.295, -0.034]])
    >>> RGB_w  # doctest: +ELLIPSIS
    array([ 2.3414154...,  1.        ,  1.5163375...])
    """

    training_data = optional(training_data, read_training_data_rawtoaces_v1())

    cmfs, illuminant = handle_spectral_arguments(
        cmfs, illuminant, shape_default=SPECTRAL_SHAPE_RAWTOACES
    )

    shape = cmfs.shape
    if sensitivities.shape != shape:
        runtime_warning(
            f'Aligning "{sensitivities.name}" sensitivities shape to "{shape}".'
        )
        sensitivities = reshape_msds(sensitivities, shape, copy=False)

    if training_data.shape != shape:
        runtime_warning(
            f'Aligning "{training_data.name}" training data shape to "{shape}".'
        )
        training_data = reshape_msds(training_data, shape, copy=False)

    illuminant = normalise_illuminant(illuminant, sensitivities)

    RGB, RGB_w = training_data_sds_to_RGB(
        training_data, sensitivities, illuminant
    )

    XYZ = training_data_sds_to_XYZ(
        training_data, cmfs, illuminant, chromatic_adaptation_transform
    )

    (
        x_0,
        objective_function,
        XYZ_to_optimization_colour_model,
        finaliser_function,
    ) = optimisation_factory()
    optimisation_settings = {
        "method": "BFGS",
        "jac": "2-point",
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    M = minimize(
        objective_function,
        x_0,
        (RGB, XYZ_to_optimization_colour_model(XYZ)),
        **optimisation_settings,
    ).x

    M = finaliser_function(M)

    if additional_data:
        return M, RGB_w, XYZ, RGB
    else:
        return M, RGB_w


def camera_RGB_to_ACES2065_1(
    RGB: ArrayLike,
    B: ArrayLike,
    b: ArrayLike,
    k: ArrayLike = np.ones(3),
    clip: bool = False,
) -> NDArrayFloat:
    """
    Convert given camera *RGB* colourspace array to *ACES2065-1* colourspace
    using the *Input Device Transform* (IDT) matrix :math:`B`, the white
    balance multipliers :math:`b` and the exposure factor :math:`k` according
    to *P-2013-001* procedure.

    Parameters
    ----------
    RGB
        Camera *RGB* colourspace array.
    B
         *Input Device Transform* (IDT) matrix :math:`B`.
    b
         White balance multipliers :math:`b`.
    k
        Exposure factor :math:`k` that results in a nominally "18% gray" object
        in the scene producing ACES values [0.18, 0.18, 0.18].
    clip
        Whether to clip the white balanced camera *RGB* colourspace array
        between :math:`-\\infty` and 1. The intent is to keep sensor saturated
        values achromatic after white balancing.

    Returns
    -------
    :class:`numpy.ndarray`
        *ACES2065-1* colourspace relative exposure values array.

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2015c`

    Examples
    --------
    >>> path = os.path.join(
    ...     ROOT_RESOURCES_RAWTOACES,
    ...     "CANON_EOS_5DMark_II_RGB_Sensitivities.csv",
    ... )
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values()
    ... )
    >>> illuminant = SDS_ILLUMINANTS["D55"]
    >>> B, b = matrix_idt(sensitivities, illuminant)
    >>> camera_RGB_to_ACES2065_1(np.array([0.1, 0.2, 0.3]), B, b)
    ... # doctest: +ELLIPSIS
    array([ 0.270644 ...,  0.1561487...,  0.5012965...])
    """

    RGB = as_float_array(RGB)
    B = as_float_array(B)
    b = as_float_array(b)
    k = as_float_array(k)

    RGB_r = b * RGB / np.min(b)

    RGB_r = np.clip(RGB_r, -np.inf, 1) if clip else RGB_r

    return k * vector_dot(B, RGB_r)
