# -*- coding: utf-8 -*-
"""
Academy Color Encoding System - Input Transform
===============================================

Defines the *Academy Color Encoding System* (ACES) *Input Transform* utilities:

-   :func:`colour.sd_to_aces_relative_exposure_values`
-   :func:`colour.characterisation.read_training_data_rawtoaces_v1`
-   :func:`colour.characterisation.generate_illuminants_rawtoaces_v1`
-   :func:`colour.characterisation.white_balance_multipliers`
-   :func:`colour.characterisation.best_illuminant`
-   :func:`colour.characterisation.normalise_illuminant`
-   :func:`colour.characterisation.training_data_sds_to_RGB`
-   :func:`colour.characterisation.training_data_sds_to_XYZ`
-   :func:`colour.characterisation.optimisation_factory_rawtoaces_v1`
-   :func:`colour.characterisation.optimisation_factory_JzAzBz`
-   :func:`colour.matrix_idt`

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

from __future__ import division, unicode_literals

import numpy as np
import os
from scipy.optimize import minimize

from colour.adaptation import matrix_chromatic_adaptation_VonKries
from colour.algebra import euclidean_distance
from colour.colorimetry import (
    MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape, sds_and_msds_to_msds,
    sd_CIE_illuminant_D_series, sd_blackbody, sd_to_XYZ)
from colour.constants import DEFAULT_INT_DTYPE
from colour.characterisation import MSDS_ACES_RICD
from colour.io import read_sds_from_csv_file
from colour.models import XYZ_to_JzAzBz, XYZ_to_Lab, XYZ_to_xy, xy_to_XYZ
from colour.models.rgb import (RGB_COLOURSPACE_ACES2065_1, RGB_to_XYZ,
                               XYZ_to_RGB, normalised_primary_matrix)
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import (CaseInsensitiveMapping, as_float_array,
                              vector_dot, from_range_1, runtime_warning,
                              tsplit, suppress_warnings)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'FLARE_PERCENTAGE', 'S_FLARE_FACTOR',
    'sd_to_aces_relative_exposure_values', 'SPECTRAL_SHAPE_RAWTOACES',
    'RESOURCES_DIRECTORY_RAWTOACES', 'read_training_data_rawtoaces_v1',
    'generate_illuminants_rawtoaces_v1', 'white_balance_multipliers',
    'best_illuminant', 'normalise_illuminant', 'training_data_sds_to_RGB',
    'training_data_sds_to_XYZ', 'optimisation_factory_rawtoaces_v1',
    'optimisation_factory_JzAzBz', 'matrix_idt'
]

FLARE_PERCENTAGE = 0.00500
"""
Flare percentage in the *ACES* system.

FLARE_PERCENTAGE : float
"""

S_FLARE_FACTOR = 0.18000 / (0.18000 + FLARE_PERCENTAGE)
"""
Flare modulation factor in the *ACES* system.

S_FLARE_FACTOR : float
"""


def sd_to_aces_relative_exposure_values(
        sd,
        illuminant=SDS_ILLUMINANTS['D65'],
        apply_chromatic_adaptation=False,
        chromatic_adaptation_transform='CAT02'):
    """
    Converts given spectral distribution to *ACES2065-1* colourspace relative
    exposure values.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution.
    illuminant : SpectralDistribution, optional
        *Illuminant* spectral distribution.
    apply_chromatic_adaptation : bool, optional
        Whether to apply chromatic adaptation using given transform.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02 Brill 2008',
        'Bianco 2010', 'Bianco PC 2010'}**,
        *Chromatic adaptation* transform.

    Returns
    -------
    ndarray, (3,)
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

    shape = MSDS_ACES_RICD.shape
    if sd.shape != MSDS_ACES_RICD.shape:
        sd = sd.copy().align(shape)

    if illuminant.shape != MSDS_ACES_RICD.shape:
        illuminant = illuminant.copy().align(shape)

    s_v = sd.values
    i_v = illuminant.values

    r_bar, g_bar, b_bar = tsplit(MSDS_ACES_RICD.values)

    def k(x, y):
        """
        Computes the :math:`K_r`, :math:`K_g` or :math:`K_b` scale factors.
        """

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
        NPM = normalised_primary_matrix(RGB_COLOURSPACE_ACES2065_1.primaries,
                                        xy)
        XYZ = RGB_to_XYZ(E_rgb, xy, RGB_COLOURSPACE_ACES2065_1.whitepoint, NPM,
                         chromatic_adaptation_transform)
        E_rgb = XYZ_to_RGB(XYZ, RGB_COLOURSPACE_ACES2065_1.whitepoint,
                           RGB_COLOURSPACE_ACES2065_1.whitepoint,
                           RGB_COLOURSPACE_ACES2065_1.matrix_XYZ_to_RGB)

    return from_range_1(E_rgb)


SPECTRAL_SHAPE_RAWTOACES = SpectralShape(380, 780, 5)
"""
Default spectral shape according to *RAW to ACES* v1.

SPECTRAL_SHAPE_RAWTOACES : SpectralShape
"""

RESOURCES_DIRECTORY_RAWTOACES = os.path.join(
    os.path.dirname(__file__), 'datasets', 'rawtoaces')
"""
*RAW to ACES* resources directory.

Notes
-----
-   *Colour* only ships a minimal dataset from *RAW to ACES*, please see
    `Colour - Datasets <https://github.com/colour-science/colour-datasets>`_
    for the complete *RAW to ACES* v1 dataset, i.e. *3372171*.

RESOURCES_DIRECTORY_RAWTOACES : unicode
"""

_TRAINING_DATA_RAWTOACES_V1 = None


def read_training_data_rawtoaces_v1():
    """
    Reads the *RAW to ACES* v1 190 patches.

    Returns
    -------
    MultiSpectralDistributions
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
        path = os.path.join(RESOURCES_DIRECTORY_RAWTOACES, '190_Patches.csv')
        training_data = sds_and_msds_to_msds(
            read_sds_from_csv_file(path).values())

        _TRAINING_DATA_RAWTOACES_V1 = training_data

    return training_data


_ILLUMINANTS_RAWTOACES_V1 = None


def generate_illuminants_rawtoaces_v1():
    """
    Generates a series of illuminants according to *RAW to ACES* v1:

    -   *CIE Illuminant D Series* in range [4000, 25000] kelvin degrees.
    -   *Blackbodies* in range [1000, 3500] kelvin degrees.
    -   A.M.P.A.S. variant of *ISO 7589 Studio Tungsten*.

    Returns
    -------
    CaseInsensitiveMapping
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
    >>> # Doctests skip for Python 2.x compatibility.
    >>> list(sorted(generate_illuminants_rawtoaces_v1().keys()))
    ... # doctest: +SKIP
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
            sd.name = 'D{0:d}'.format(DEFAULT_INT_DTYPE(CCT / 100))
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
            os.path.join(RESOURCES_DIRECTORY_RAWTOACES,
                         'AMPAS_ISO_7589_Tungsten.csv'))['iso7589']
        illuminants.update({sd.name: sd})

        _ILLUMINANTS_RAWTOACES_V1 = illuminants

    return illuminants


def white_balance_multipliers(sensitivities, illuminant):
    """
    Computes the *RGB* white balance multipliers for given camera *RGB*
    spectral sensitivities and illuminant.

    Parameters
    ----------
    sensitivities : RGB_CameraSensitivities
         Camera *RGB* spectral sensitivities.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.

    Returns
    -------
    ndarray
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
        runtime_warning('Aligning "{0}" illuminant shape to "{1}".'.format(
            illuminant.name, shape))
        illuminant = illuminant.copy().align(shape)

    RGB_w = 1 / np.sum(
        sensitivities.values * illuminant.values[..., np.newaxis], axis=0)
    RGB_w *= 1 / np.min(RGB_w)

    return RGB_w


def best_illuminant(RGB_w, sensitivities, illuminants):
    """
    Select the best illuminant for given *RGB* white balance multipliers,
    and sensitivities in given series of illuminants.

    Parameters
    ----------
    RGB_w : array_like
        *RGB* white balance multipliers.
    sensitivities : RGB_CameraSensitivities
         Camera *RGB* spectral sensitivities.
    illuminants : SpectralDistribution
        Illuminant spectral distributions to choose the best illuminant from.

    Returns
    -------
    SpectralDistribution
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
    >>> # Doctests skip for Python 2.x compatibility.
    >>> best_illuminant(RGB_w, sensitivities, illuminants).name
    ... # doctest: +SKIP
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

    return illuminant_b


def normalise_illuminant(illuminant, sensitivities):
    """
    Normalises given illuminant with given camera *RGB* spectral sensitivities.

    The multiplicative inverse scaling factor :math:`k` is computed by
    multiplying the illuminant by the sensitivies channel with the maximum
    value.

    Parameters
    ----------
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    sensitivities : RGB_CameraSensitivities
         Camera *RGB* spectral sensitivities.

    Returns
    -------
    SpectralDistribution
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
        runtime_warning('Aligning "{0}" illuminant shape to "{1}".'.format(
            illuminant.name, shape))
        illuminant = illuminant.copy().align(shape)

    c_i = np.argmax(np.max(sensitivities.values, axis=0))
    k = 1 / np.sum(illuminant.values * sensitivities.values[..., c_i])

    return illuminant * k


def training_data_sds_to_RGB(training_data, sensitivities, illuminant):
    """
    Converts given training data to *RGB* tristimulus values using given
    illuminant and given camera *RGB* spectral sensitivities.

    Parameters
    ----------
    training_data : MultiSpectralDistributions
        Training data multi-spectral distributions.
    sensitivities : RGB_CameraSensitivities
         Camera *RGB* spectral sensitivities.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.

    Returns
    -------
    ndarray
        Training data *RGB* tristimulus values.

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
    >>> training_data_sds_to_RGB(training_data, sensitivities, illuminant)[:5]
    ... # doctest: +ELLIPSIS
    array([[ 0.0207582...,  0.0196857...,  0.0213935...],
           [ 0.0895775...,  0.0891922...,  0.0891091...],
           [ 0.7810230...,  0.7801938...,  0.7764302...],
           [ 0.1995   ...,  0.1995   ...,  0.1995   ...],
           [ 0.5898478...,  0.5904015...,  0.5851076...]])
    """

    shape = sensitivities.shape
    if illuminant.shape != shape:
        runtime_warning('Aligning "{0}" illuminant shape to "{1}".'.format(
            illuminant.name, shape))
        illuminant = illuminant.copy().align(shape)

    if training_data.shape != shape:
        runtime_warning('Aligning "{0}" training data shape to "{1}".'.format(
            training_data.name, shape))
        training_data = training_data.copy().align(shape)

    RGB_w = white_balance_multipliers(sensitivities, illuminant)

    RGB = np.dot(
        np.transpose(
            illuminant.values[..., np.newaxis] * training_data.values),
        sensitivities.values)
    RGB *= RGB_w

    return RGB


def training_data_sds_to_XYZ(training_data, cmfs, illuminant):
    """
    Converts given training data to *CIE XYZ* tristimulus values using given
    illuminant and given standard observer colour matching functions.

    Parameters
    ----------
    training_data : MultiSpectralDistributions
        Training data multi-spectral distributions.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.

    Returns
    -------
    ndarray
        Training data *CIE XYZ* tristimulus values.

    Examples
    --------
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
        runtime_warning('Aligning "{0}" illuminant shape to "{1}".'.format(
            illuminant.name, shape))
        illuminant = illuminant.copy().align(shape)

    if training_data.shape != shape:
        runtime_warning('Aligning "{0}" training data shape to "{1}".'.format(
            training_data.name, shape))
        training_data = training_data.copy().align(shape)

    XYZ = np.dot(
        np.transpose(
            illuminant.values[..., np.newaxis] * training_data.values),
        cmfs.values)

    XYZ *= 1 / np.sum(cmfs.values[..., 1] * illuminant.values)

    XYZ_w = np.dot(np.transpose(cmfs.values), illuminant.values)
    XYZ_w *= 1 / XYZ_w[1]

    M_CAT = matrix_chromatic_adaptation_VonKries(
        XYZ_w, xy_to_XYZ(RGB_COLOURSPACE_ACES2065_1.whitepoint))

    XYZ = vector_dot(M_CAT, XYZ)

    return XYZ


def optimisation_factory_rawtoaces_v1():
    """
    Factory that returns the objective function and *CIE XYZ* colourspace to
    optimisation colourspace/colour model function according to *RAW to ACES*
    v1.

    The objective function returns the euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in *CIE L\\*a\\*b\\** colourspace.

    Returns
    -------
    tuple
        Objective function and *CIE XYZ* colourspace to *CIE L\\*a\\*b\\**
        colourspace function.

    Examples
    --------
    >>> # Doctests skip for Python 2.x compatibility.
    >>> optimisation_factory_rawtoaces_v1()  # doctest: +SKIP
    (<function optimisation_factory_rawtoaces_v1.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_rawtoaces_v1.<locals>\
.XYZ_to_optimization_colour_model at 0x...>)
    """

    def objective_function(M, RGB, Lab):
        """
        Objective function according to *RAW to ACES* v1.
        """

        M = np.reshape(M, [3, 3])

        XYZ_t = vector_dot(RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ,
                           vector_dot(M, RGB))
        Lab_t = XYZ_to_Lab(XYZ_t, RGB_COLOURSPACE_ACES2065_1.whitepoint)

        return np.linalg.norm(Lab_t - Lab)

    def XYZ_to_optimization_colour_model(XYZ):
        """
        *CIE XYZ* colourspace to *CIE L\\*a\\*b\\** colourspace function.
        """

        return XYZ_to_Lab(XYZ, RGB_COLOURSPACE_ACES2065_1.whitepoint)

    return objective_function, XYZ_to_optimization_colour_model


def optimisation_factory_JzAzBz():
    """
    Factory that returns the objective function and *CIE XYZ* colourspace to
    optimisation colourspace/colour model function based on the
    :math:`J_zA_zB_z` colourspace.

    The objective function returns the euclidean distance between the training
    data *RGB* tristimulus values and the training data *CIE XYZ* tristimulus
    values** in the :math:`J_zA_zB_z` colourspace.

    Returns
    -------
    tuple
        Objective function and *CIE XYZ* colourspace to :math:`J_zA_zB_z`
        colourspace function.

    Examples
    --------
    >>> # Doctests skip for Python 2.x compatibility.
    >>> optimisation_factory_JzAzBz()  # doctest: +SKIP
    (<function optimisation_factory_JzAzBz.<locals>\
.objective_function at 0x...>, \
<function optimisation_factory_JzAzBz.<locals>\
.XYZ_to_optimization_colour_model at 0x...>)
    """

    def objective_function(M, RGB, Jab):
        """
        :math:`J_zA_zB_z` colourspace based objective function.
        """

        M = np.reshape(M, [3, 3])

        XYZ_t = vector_dot(RGB_COLOURSPACE_ACES2065_1.matrix_RGB_to_XYZ,
                           vector_dot(M, RGB))
        Jab_t = XYZ_to_JzAzBz(XYZ_t)

        return np.sum(euclidean_distance(Jab, Jab_t))

    def XYZ_to_optimization_colour_model(XYZ):
        """
        *CIE XYZ* colourspace to :math:`J_zA_zB_z` colourspace function.
        """

        return XYZ_to_JzAzBz(XYZ)

    return objective_function, XYZ_to_optimization_colour_model


def matrix_idt(sensitivities,
               illuminant,
               training_data=None,
               cmfs=MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy()
               .align(SPECTRAL_SHAPE_RAWTOACES),
               optimisation_factory=optimisation_factory_rawtoaces_v1,
               optimisation_kwargs=None):
    """
    Computes an *Input Device Transform* (IDT) matrix for given camera *RGB*
    spectral sensitivities, illuminant, training data, standard observer colour
    matching functions and optimization settings according to *RAW to ACES* v1
    and *P-2013-001* procedures.

    Parameters
    ----------
    sensitivities : RGB_CameraSensitivities
         Camera *RGB* spectral sensitivities.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    training_data : MultiSpectralDistributions, optional
        Training data multi-spectral distributions, defaults to using the
        *RAW to ACES* v1 190 patches.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    optimisation_factory : callable, optional
        Callable producing the objective function and the *CIE XYZ* to
        optimisation colour model function.
    optimisation_kwargs : dict_like, optional
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    ndarray
        *Input Device Transform* (IDT) matrix.

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
    >>> np.around(
    ...     matrix_idt(sensitivities, illuminant), 3)
    array([[ 0.85 , -0.016,  0.151],
           [ 0.051,  1.126, -0.185],
           [ 0.02 , -0.194,  1.162]])

    The *RAW to ACES* v1 matrix for the same camera and optimized by
    `Ceres Solver <http://ceres-solver.org/>`__ is as follows:

    0.864994 -0.026302 0.161308
    0.056527 1.122997 -0.179524
    0.023683 -0.202547 1.178864

    >>> np.around(matrix_idt(
    ...     sensitivities, illuminant,
    ...     optimisation_factory=optimisation_factory_JzAzBz), 3)
    array([[ 0.848, -0.016,  0.158],
           [ 0.053,  1.114, -0.175],
           [ 0.023, -0.225,  1.196]])
    """

    if training_data is None:
        training_data = read_training_data_rawtoaces_v1()

    shape = cmfs.shape
    if sensitivities.shape != shape:
        runtime_warning('Aligning "{0}" sensitivities shape to "{1}".'.format(
            sensitivities.name, shape))
        sensitivities = sensitivities.copy().align(shape)

    if illuminant.shape != shape:
        runtime_warning('Aligning "{0}" illuminant shape to "{1}".'.format(
            illuminant.name, shape))
        illuminant = illuminant.copy().align(shape)

    if training_data.shape != shape:
        runtime_warning('Aligning "{0}" training data shape to "{1}".'.format(
            training_data.name, shape))
        training_data = training_data.copy().align(shape)

    illuminant = normalise_illuminant(illuminant, sensitivities)

    RGB = training_data_sds_to_RGB(training_data, sensitivities, illuminant)
    XYZ = training_data_sds_to_XYZ(training_data, cmfs, illuminant)

    objective_function, XYZ_to_optimization_colour_model = (
        optimisation_factory())
    optimisation_settings = {
        'method': 'BFGS',
        'jac': '2-point',
    }
    if optimisation_kwargs is not None:
        optimisation_settings.update(optimisation_kwargs)

    M = minimize(objective_function, np.ravel(np.identity(3)),
                 (RGB, XYZ_to_optimization_colour_model(XYZ)),
                 **optimisation_settings).x.reshape([3, 3])

    return M
