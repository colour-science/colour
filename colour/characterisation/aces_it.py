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
-   :func:`colour.matrix_idt`
-   :func:`colour.camera_RGB_to_ACES2065_1`

References
----------
-   :cite:`Dyer2017` : Dyer, S., Forsythe, A., Irons, J., Mansencal, T., & Zhu,
    M. (2017). RAW to ACES (Version 1.0) [Computer software].
-   :cite:`Forsythe2018` : Borer, T. (2017). Private Discussion with Mansencal,
    T. and Shaw, N.
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

import numpy as np
import os
from scipy.optimize import minimize

from colour.adaptation import matrix_chromatic_adaptation_VonKries
from colour.algebra import euclidean_distance, vector_dot
from colour.colorimetry import (
    MultiSpectralDistributions,
    SDS_ILLUMINANTS,
    SpectralDistribution,
    SpectralShape,
    handle_spectral_arguments,
    reshape_msds,
    reshape_sd,
    sds_and_msds_to_msds,
    sd_CIE_illuminant_D_series,
    sd_blackbody,
    sd_to_XYZ,
)
from colour.characterisation import MSDS_ACES_RICD, RGB_CameraSensitivities
from colour.hints import (
    ArrayLike,
    Boolean,
    Callable,
    Dict,
    Floating,
    FloatingOrNDArray,
    Literal,
    Mapping,
    NDArray,
    Optional,
    Tuple,
    Union,
    cast,
)
from colour.io import read_sds_from_csv_file
from colour.models import XYZ_to_Jzazbz, XYZ_to_Lab, XYZ_to_xy, xy_to_XYZ
from colour.models.rgb import (
    RGB_COLOURSPACE_ACES2065_1,
    RGB_to_XYZ,
    XYZ_to_RGB,
    normalised_primary_matrix,
)
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float,
    as_float_array,
    from_range_1,
    optional,
    runtime_warning,
    suppress_warnings,
    tsplit,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "FLARE_PERCENTAGE",
    "S_FLARE_FACTOR",
    "sd_to_aces_relative_exposure_values",
    "sd_to_ACES2065_1",
    "SPECTRAL_SHAPE_RAWTOACES",
    "RESOURCES_DIRECTORY_RAWTOACES",
    "read_training_data_rawtoaces_v1",
    "generate_illuminants_rawtoaces_v1",
    "white_balance_multipliers",
    "best_illuminant",
    "normalise_illuminant",
    "training_data_sds_to_RGB",
    "training_data_sds_to_XYZ",
    "optimisation_factory_rawtoaces_v1",
    "optimisation_factory_Jzazbz",
    "matrix_idt",
    "camera_RGB_to_ACES2065_1",
]

FLARE_PERCENTAGE: Floating = 0.00500
"""Flare percentage in the *ACES* system."""

S_FLARE_FACTOR: Floating = 0.18000 / (0.18000 + FLARE_PERCENTAGE)
"""Flare modulation factor in the *ACES* system."""


def sd_to_aces_relative_exposure_values(
    sd: SpectralDistribution,
    illuminant: Optional[SpectralDistribution] = None,
    apply_chromatic_adaptation: Boolean = False,
    chromatic_adaptation_transform: Union[
        Literal[
            "Bianco 2010",
            "Bianco PC 2010",
            "Bradford",
            "CAT02 Brill 2008",
            "CAT02",
            "CAT16",
            "CMCCAT2000",
            "CMCCAT97",
            "Fairchild",
            "Sharp",
            "Von Kries",
            "XYZ Scaling",
        ],
        str,
    ] = "CAT02",
) -> NDArray:
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
    apply_chromatic_adaptation
        Whether to apply chromatic adaptation using given transform.
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
    >>> sd = SDS_COLOURCHECKERS['ColorChecker N Ohta']['dark skin']
    >>> sd_to_aces_relative_exposure_values(sd)  # doctest: +ELLIPSIS
    array([ 0.1171814...,  0.0866360...,  0.0589726...])
    >>> sd_to_aces_relative_exposure_values(sd,
    ...     apply_chromatic_adaptation=True)  # doctest: +ELLIPSIS
    array([ 0.1180779...,  0.0869031...,  0.0589125...])
    """

    illuminant = cast(
        SpectralDistribution, optional(illuminant, SDS_ILLUMINANTS["D65"])
    )

    shape = MSDS_ACES_RICD.shape
    if sd.shape != MSDS_ACES_RICD.shape:
        sd = reshape_sd(sd, shape)

    if illuminant.shape != MSDS_ACES_RICD.shape:
        illuminant = reshape_sd(illuminant, shape)

    s_v = sd.values
    i_v = illuminant.values

    r_bar, g_bar, b_bar = tsplit(MSDS_ACES_RICD.values)

    def k(x: NDArray, y: NDArray) -> NDArray:
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

    if apply_chromatic_adaptation:
        xy = XYZ_to_xy(sd_to_XYZ(illuminant) / 100)
        NPM = normalised_primary_matrix(
            RGB_COLOURSPACE_ACES2065_1.primaries, xy
        )
        XYZ = RGB_to_XYZ(
            E_rgb,
            xy,
            RGB_COLOURSPACE_ACES2065_1.whitepoint,
            NPM,
            chromatic_adaptation_transform,
        )
        E_rgb = XYZ_to_RGB(
            XYZ,
            RGB_COLOURSPACE_ACES2065_1.whitepoint,
            RGB_COLOURSPACE_ACES2065_1.whitepoint,
            RGB_COLOURSPACE_ACES2065_1.matrix_XYZ_to_RGB,
        )

    return from_range_1(E_rgb)


sd_to_ACES2065_1 = sd_to_aces_relative_exposure_values

SPECTRAL_SHAPE_RAWTOACES: SpectralShape = SpectralShape(380, 780, 5)
"""Default spectral shape according to *RAW to ACES* v1."""

RESOURCES_DIRECTORY_RAWTOACES: str = os.path.join(
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

_TRAINING_DATA_RAWTOACES_V1: Optional[MultiSpectralDistributions] = None


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

    global _TRAINING_DATA_RAWTOACES_V1

    if _TRAINING_DATA_RAWTOACES_V1 is not None:
        training_data = _TRAINING_DATA_RAWTOACES_V1
    else:
        path = os.path.join(RESOURCES_DIRECTORY_RAWTOACES, "190_Patches.csv")
        training_data = sds_and_msds_to_msds(
            list(read_sds_from_csv_file(path).values())
        )

        _TRAINING_DATA_RAWTOACES_V1 = training_data

    return training_data


_ILLUMINANTS_RAWTOACES_V1: Optional[CaseInsensitiveMapping] = None


def generate_illuminants_rawtoaces_v1() -> CaseInsensitiveMapping:
    """
    Generate a series of illuminants according to *RAW to ACES* v1:

    -   *CIE Illuminant D Series* in range [4000, 25000] kelvin degrees.
    -   *Blackbodies* in range [1000, 3500] kelvin degrees.
    -   A.M.P.A.S. variant of *ISO 7589 Studio Tungsten*.

    Returns
    -------
    :class:`colour.utilities.CaseInsensitiveMapping`
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

    global _ILLUMINANTS_RAWTOACES_V1

    if _ILLUMINANTS_RAWTOACES_V1 is not None:
        illuminants = _ILLUMINANTS_RAWTOACES_V1
    else:
        illuminants = CaseInsensitiveMapping()

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
                RESOURCES_DIRECTORY_RAWTOACES, "AMPAS_ISO_7589_Tungsten.csv"
            )
        )["iso7589"]
        illuminants.update({sd.name: sd})

        _ILLUMINANTS_RAWTOACES_V1 = illuminants

    return illuminants


def white_balance_multipliers(
    sensitivities: RGB_CameraSensitivities, illuminant: SpectralDistribution
) -> NDArray:
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
    ...     RESOURCES_DIRECTORY_RAWTOACES,
    ...     'CANON_EOS_5DMark_II_RGB_Sensitivities.csv')
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values())
    >>> illuminant = SDS_ILLUMINANTS['D55']
    >>> white_balance_multipliers(sensitivities, illuminant)
    ... # doctest: +ELLIPSIS
    array([ 2.3414154...,  1.        ,  1.5163375...])
    """

    shape = sensitivities.shape
    if illuminant.shape != shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{shape}".'
        )
        illuminant = reshape_sd(illuminant, shape)

    RGB_w = 1 / np.sum(
        sensitivities.values * illuminant.values[..., np.newaxis], axis=0
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
    ...     RESOURCES_DIRECTORY_RAWTOACES,
    ...     'CANON_EOS_5DMark_II_RGB_Sensitivities.csv')
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values())
    >>> illuminants = generate_illuminants_rawtoaces_v1()
    >>> RGB_w = white_balance_multipliers(
    ...     sensitivities, SDS_ILLUMINANTS['FL2'])
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

    return illuminant_b  # type: ignore[return-value]


def normalise_illuminant(
    illuminant: SpectralDistribution, sensitivities: RGB_CameraSensitivities
) -> SpectralDistribution:
    """
    Normalise given illuminant with given camera *RGB* spectral sensitivities.

    The multiplicative inverse scaling factor :math:`k` is computed by
    multiplying the illuminant by the sensitivies channel with the maximum
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
    ...     RESOURCES_DIRECTORY_RAWTOACES,
    ...     'CANON_EOS_5DMark_II_RGB_Sensitivities.csv')
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values())
    >>> illuminant = SDS_ILLUMINANTS['D55']
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
) -> Tuple[NDArray, NDArray]:
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
    ...     RESOURCES_DIRECTORY_RAWTOACES,
    ...     'CANON_EOS_5DMark_II_RGB_Sensitivities.csv')
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values())
    >>> illuminant = normalise_illuminant(
    ...     SDS_ILLUMINANTS['D55'], sensitivities)
    >>> training_data = read_training_data_rawtoaces_v1()
    >>> RGB, RGB_w = training_data_sds_to_RGB(
    ...     training_data, sensitivities, illuminant)
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
        illuminant = reshape_sd(illuminant, shape)

    if training_data.shape != shape:
        runtime_warning(
            f'Aligning "{training_data.name}" training data shape to "{shape}".'
        )
        # pylint: disable=E1102
        training_data = reshape_msds(training_data, shape)

    RGB_w = white_balance_multipliers(sensitivities, illuminant)

    RGB = np.dot(
        np.transpose(
            illuminant.values[..., np.newaxis] * training_data.values
        ),
        sensitivities.values,
    )

    RGB *= RGB_w

    return RGB, RGB_w


def training_data_sds_to_XYZ(
    training_data: MultiSpectralDistributions,
    cmfs: MultiSpectralDistributions,
    illuminant: SpectralDistribution,
    chromatic_adaptation_transform: Union[
        Literal[
            "Bianco 2010",
            "Bianco PC 2010",
            "Bradford",
            "CAT02 Brill 2008",
            "CAT02",
            "CAT16",
            "CMCCAT2000",
            "CMCCAT97",
            "Fairchild",
            "Sharp",
            "Von Kries",
            "XYZ Scaling",
        ],
        str,
    ] = "CAT02",
) -> NDArray:
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
    ...     RESOURCES_DIRECTORY_RAWTOACES,
    ...     'CANON_EOS_5DMark_II_RGB_Sensitivities.csv')
    >>> cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values())
    >>> illuminant = normalise_illuminant(
    ...     SDS_ILLUMINANTS['D55'], sensitivities)
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
        illuminant = reshape_sd(illuminant, shape)

    if training_data.shape != shape:
        runtime_warning(
            f'Aligning "{training_data.name}" training data shape to "{shape}".'
        )
        # pylint: disable=E1102
        training_data = reshape_msds(training_data, shape)

    XYZ = np.dot(
        np.transpose(
            illuminant.values[..., np.newaxis] * training_data.values
        ),
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


def optimisation_factory_rawtoaces_v1() -> Tuple[Callable, Callable]:
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function according to *RAW to ACES* v1.

    The objective function returns the euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in *CIE L\\*a\\*b\\** colourspace.

    Returns
    -------
    :class:`tuple`
        Objective function and *CIE XYZ* colourspace to *CIE L\\*a\\*b\\**
        colourspace function.

    Examples
    --------
    >>> optimisation_factory_rawtoaces_v1()  # doctest: +SKIP
    (<function optimisation_factory_rawtoaces_v1.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_rawtoaces_v1.<locals>\
.XYZ_to_optimization_colour_model at 0x...>)
    """

    def objective_function(
        M: ArrayLike, RGB: ArrayLike, Lab: ArrayLike
    ) -> FloatingOrNDArray:
        """Objective function according to *RAW to ACES* v1."""

        M = np.reshape(M, [3, 3])

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Lab_t = XYZ_to_Lab(XYZ_t, RGB_COLOURSPACE_ACES2065_1.whitepoint)

        return as_float(np.linalg.norm(Lab_t - Lab))

    def XYZ_to_optimization_colour_model(XYZ: ArrayLike) -> NDArray:
        """*CIE XYZ* colourspace to *CIE L\\*a\\*b\\** colourspace function."""

        return XYZ_to_Lab(XYZ, RGB_COLOURSPACE_ACES2065_1.whitepoint)

    return objective_function, XYZ_to_optimization_colour_model


def optimisation_factory_Jzazbz() -> Tuple[Callable, Callable]:
    """
    Produce the objective function and *CIE XYZ* colourspace to optimisation
    colourspace/colour model function based on the :math:`J_za_zb_z`
    colourspace.

    The objective function returns the euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in the :math:`J_za_zb_z` colourspace.

    Returns
    -------
    :class:`tuple`
        Objective function and *CIE XYZ* colourspace to :math:`J_za_zb_z`
        colourspace function.

    Examples
    --------
    >>> optimisation_factory_Jzazbz()  # doctest: +SKIP
    (<function optimisation_factory_Jzazbz.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_Jzazbz.<locals>\
.XYZ_to_optimization_colour_model at 0x...>)
    """

    def objective_function(
        M: ArrayLike, RGB: ArrayLike, Jab: ArrayLike
    ) -> FloatingOrNDArray:
        """:math:`J_za_zb_z` colourspace based objective function."""

        M = np.reshape(M, [3, 3])

        XYZ_t = vector_dot(
            RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ, vector_dot(M, RGB)
        )
        Jab_t = XYZ_to_Jzazbz(XYZ_t)

        return np.sum(euclidean_distance(Jab, Jab_t))

    def XYZ_to_optimization_colour_model(XYZ: ArrayLike) -> NDArray:
        """*CIE XYZ* colourspace to :math:`J_za_zb_z` colourspace function."""

        return XYZ_to_Jzazbz(XYZ)

    return objective_function, XYZ_to_optimization_colour_model


def matrix_idt(
    sensitivities: RGB_CameraSensitivities,
    illuminant: SpectralDistribution,
    training_data: Optional[MultiSpectralDistributions] = None,
    cmfs: Optional[MultiSpectralDistributions] = None,
    optimisation_factory: Callable = optimisation_factory_rawtoaces_v1,
    optimisation_kwargs: Optional[Dict] = None,
    chromatic_adaptation_transform: Union[
        Literal[
            "Bianco 2010",
            "Bianco PC 2010",
            "Bradford",
            "CAT02 Brill 2008",
            "CAT02",
            "CAT16",
            "CMCCAT2000",
            "CMCCAT97",
            "Fairchild",
            "Sharp",
            "Von Kries",
            "XYZ Scaling",
        ],
        str,
    ] = "CAT02",
    additional_data: Boolean = False,
) -> Union[Tuple[NDArray, NDArray, NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """
    Compute an *Input Device Transform* (IDT) matrix for given camera *RGB*
    spectral sensitivities, illuminant, training data, standard observer colour
    matching functions and optimization settings according to *RAW to ACES* v1
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
        Tuple of *Input Device Transform* (IDT) matrix and white balance
        multipliers or tuple of *Input Device Transform* (IDT) matrix, white
        balance multipliers, *XYZ* and *RGB* tristimulus values.

    References
    ----------
    :cite:`Dyer2017`, :cite:`TheAcademyofMotionPictureArtsandSciences2015c`

    Examples
    --------
    Computing the *Input Device Transform* (IDT) matrix for a
    *CANON EOS 5DMark II* and *CIE Illuminant D Series* *D55* using
    the method given in *RAW to ACES* v1:

    >>> path = os.path.join(
    ...     RESOURCES_DIRECTORY_RAWTOACES,
    ...     'CANON_EOS_5DMark_II_RGB_Sensitivities.csv')
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values())
    >>> illuminant = SDS_ILLUMINANTS['D55']
    >>> M, RGB_w = matrix_idt(sensitivities, illuminant)
    >>> np.around(M, 3)
    array([[ 0.85 , -0.016,  0.151],
           [ 0.051,  1.126, -0.185],
           [ 0.02 , -0.194,  1.162]])
    >>> RGB_w  # doctest: +ELLIPSIS
    array([ 2.3414154...,  1.        ,  1.5163375...])

    The *RAW to ACES* v1 matrix for the same camera and optimized by
    `Ceres Solver <http://ceres-solver.org/>`__ is as follows::

        0.864994 -0.026302 0.161308
        0.056527 1.122997 -0.179524
        0.023683 -0.202547 1.178864

    >>> M, RGB_w = matrix_idt(
    ...     sensitivities, illuminant,
    ...     optimisation_factory=optimisation_factory_Jzazbz)
    >>> np.around(M, 3)
    array([[ 0.848, -0.016,  0.158],
           [ 0.053,  1.114, -0.175],
           [ 0.023, -0.225,  1.196]])
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
        # pylint: disable=E1102
        sensitivities = reshape_msds(sensitivities, shape)  # type: ignore[assignment]

    if training_data.shape != shape:
        runtime_warning(
            f'Aligning "{training_data.name}" training data shape to "{shape}".'
        )
        # pylint: disable=E1102
        training_data = reshape_msds(training_data, shape)

    illuminant = normalise_illuminant(illuminant, sensitivities)

    RGB, RGB_w = training_data_sds_to_RGB(
        training_data, sensitivities, illuminant
    )

    XYZ = training_data_sds_to_XYZ(
        training_data, cmfs, illuminant, chromatic_adaptation_transform
    )

    (
        objective_function,
        XYZ_to_optimization_colour_model,
    ) = optimisation_factory()
    optimisation_settings = {
        "method": "BFGS",
        "jac": "2-point",
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    M = minimize(
        objective_function,
        np.ravel(np.identity(3)),
        (RGB, XYZ_to_optimization_colour_model(XYZ)),
        **optimisation_settings,
    ).x.reshape([3, 3])

    if additional_data:
        return M, RGB_w, XYZ, RGB
    else:
        return M, RGB_w


def camera_RGB_to_ACES2065_1(
    RGB: ArrayLike,
    B: ArrayLike,
    b: ArrayLike,
    k: ArrayLike = np.ones(3),
    clip: Boolean = False,
) -> NDArray:
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
    ...     RESOURCES_DIRECTORY_RAWTOACES,
    ...     'CANON_EOS_5DMark_II_RGB_Sensitivities.csv')
    >>> sensitivities = sds_and_msds_to_msds(
    ...     read_sds_from_csv_file(path).values())
    >>> illuminant = SDS_ILLUMINANTS['D55']
    >>> B, b = matrix_idt(sensitivities, illuminant)
    >>> camera_RGB_to_ACES2065_1(np.array([0.1, 0.2, 0.3]), B, b)
    ... # doctest: +ELLIPSIS
    array([ 0.2646811...,  0.1528898...,  0.4944335...])
    """

    RGB = as_float_array(RGB)
    B = as_float_array(B)
    b = as_float_array(b)
    k = as_float_array(k)

    RGB_r = b * RGB / np.min(b)

    RGB_r = np.clip(RGB_r, -np.inf, 1) if clip else RGB_r

    return k * vector_dot(B, RGB_r)
