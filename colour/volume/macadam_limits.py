"""
Optimal Colour Stimuli - MacAdam Limits
=======================================
Defines the objects related to *Optimal Colour Stimuli* computations.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay
from colour.hints import (
    ArrayLike,
    Dict,
    Floating,
    Literal,
    NDArray,
    Optional,
    Union,
)
from colour.colorimetry import (
    MSDS_CMFS,
    reshape_sd,
    SpectralShape,
    SDS_ILLUMINANTS,
)
from colour.models import xyY_to_XYZ
from colour.volume import OPTIMAL_COLOUR_STIMULI_ILLUMINANTS
from colour.utilities import CACHE_REGISTRY, tsplit, zeros, validate_method

__author__ = "Colour Developers", "Christian Greim"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "is_within_macadam_limits",
]

_CACHE_OPTIMAL_COLOUR_STIMULI_XYZ: Dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_OPTIMAL_COLOUR_STIMULI_XYZ"
)

_CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS: Dict = (
    CACHE_REGISTRY.register_cache(
        f"{__name__}._CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS"
    )
)


def _XYZ_optimal_colour_stimuli(
    illuminant: Union[Literal["A", "C", "D65"], str] = "D65"
) -> NDArray:
    """
    Return given illuminant *Optimal Colour Stimuli* in *CIE XYZ* tristimulus
    values and caches it if not existing.

    Parameters
    ----------
    illuminant
        Illuminant name.

    Returns
    -------
    :class:`numpy.ndarray`
        Illuminant *Optimal Colour Stimuli*.
    """

    illuminant = validate_method(
        illuminant,
        list(OPTIMAL_COLOUR_STIMULI_ILLUMINANTS.keys()),
        '"{0}" illuminant is invalid, it must be one of {1}!',
    )

    optimal_colour_stimuli = OPTIMAL_COLOUR_STIMULI_ILLUMINANTS[illuminant]

    vertices = _CACHE_OPTIMAL_COLOUR_STIMULI_XYZ.get(illuminant)

    if vertices is None:
        _CACHE_OPTIMAL_COLOUR_STIMULI_XYZ[illuminant] = vertices = (
            xyY_to_XYZ(optimal_colour_stimuli) / 100
        )

    return vertices


def is_within_macadam_limits(
    xyY: ArrayLike,
    illuminant: Union[Literal["A", "C", "D65"], str] = "D65",
    tolerance: Optional[Floating] = None,
) -> NDArray:
    """
    Return whether given *CIE xyY* colourspace array is within MacAdam limits
    of given illuminant.

    Parameters
    ----------
    xyY
        *CIE xyY* colourspace array.
    illuminant
        Illuminant name.
    tolerance
        Tolerance allowed in the inside-triangle check.

    Returns
    -------
    :class:`numpy.ndarray`
        Whether given *CIE xyY* colourspace array is within MacAdam limits.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> is_within_macadam_limits(np.array([0.3205, 0.4131, 0.51]), "A")
    array(True, dtype=bool)
    >>> a = np.array([[0.3205, 0.4131, 0.51], [0.0005, 0.0031, 0.001]])
    >>> is_within_macadam_limits(a, "A")
    array([ True, False], dtype=bool)
    """

    optimal_colour_stimuli = _XYZ_optimal_colour_stimuli(illuminant)
    triangulation = _CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS.get(
        illuminant
    )

    if triangulation is None:
        _CACHE_OPTIMAL_COLOUR_STIMULI_XYZ_TRIANGULATIONS[
            illuminant
        ] = triangulation = Delaunay(optimal_colour_stimuli)

    simplex = triangulation.find_simplex(xyY_to_XYZ(xyY), tol=tolerance)
    simplex = np.where(simplex >= 0, True, False)

    return simplex


def macadam_limits(
    luminance: Floating = 0.5,
    illuminant=SDS_ILLUMINANTS["E"],
    spectral_range=SpectralShape(360, 830, 1),
    cmfs=MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
) -> NDArray:
    """
    Return an array of CIE -X,Y,Z - Triples containing colour-coordinates
    of the MacAdam-limit for the defined luminance for every
    whavelength defined in spectral_range.
    Target ist a fast running codey, by
    not simply testing the possible optimums step by step but
    more effectively targeted by steps of power of two. The wavelengths
    left and right of a rough optimum are fitted by a rule of proportion,
    so that the wished brightness will be reached exactly.

    Parameters
    ----------
    luminance
        set the wanted luminance
        has to be between 0 and 1
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*
    spectral_range
        SpectralShape according to colour.SpectralShape
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.

    Returns
    -------
    :class:`numpy.ndarray`
        an array of CIE -X,Y,Z - Triples containing colour-coordinates
        of the MacAdam-limit for the definde luminance for every
        whavelength defined in spectral_range
    array([[  3.83917134e-01,   5.00000000e-01,   3.55171511e-01],
       [  3.56913361e-01,   5.00000000e-01,   3.55215349e-01],
       [  3.32781985e-01,   5.00000000e-01,   3.55249953e-01],
       ...
       [  4.44310989e-01,   5.00000000e-01,   3.55056751e-01],
       [  4.13165551e-01,   5.00000000e-01,   3.55118668e-01]])

    References
    ----------
    -   cite: Wyszecki, G., & Stiles, W. S. (2000).
        In Color Science: Concepts and Methods,
        Quantitative Data and Formulae (pp. 181–184). Wiley.
        ISBN:978-0-471-39918-6
    -   cite: Francisco Martínez-Verdú, Esther Perales,
        Elisabet Chorro, Dolores de Fez,
        Valentín Viqueira, and Eduardo Gilabert, "Computation and
        visualization of the MacAdam limits
        for any lightness, hue angle, and light source," J.
        Opt. Soc. Am. A 24, 1501-1515 (2007)
    -   cite: Kenichiro Masaoka. In OPTICS LETTERS, June 15, 2010
        / Vol. 35, No. 1 (pp. 2031 - 2033)

    Examples
    --------
    from matplotlib import pyplot as plt
    import numpy as np
    import math
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_axes([0,0,1,1])
    illuminant = colour.SDS_ILLUMINANTS['D65']
    def plot_Narrowband_Spectra (Yxy_Narrowband_Spectra):
        FirstColumn = 0
        SecondColumn = 1
        x = Yxy_Narrowband_Spectra[...,FirstColumn]
        y = Yxy_Narrowband_Spectra[...,SecondColumn]
        ax.plot(x,y,'orange',label='Spectrum Loci')
        x = [Yxy_Narrowband_Spectra[-1][FirstColumn],
        Yxy_Narrowband_Spectra[0][FirstColumn]]
        y = [Yxy_Narrowband_Spectra[-1][SecondColumn],
        Yxy_Narrowband_Spectra[0][SecondColumn]]
        ax.plot(x,y,'purple',label='Purple Boundary')
        return()
    for n in range(1, 20):
        Yxy_Narrowband_Spectra = colour.XYZ_to_xy(
        colour.macadam_limits(n/20, illuminant))
        plot_Narrowband_Spectra (Yxy_Narrowband_Spectra)
    plt.show()
    """

    target_bright = luminance
    if target_bright > 1 or target_bright < 0:
        raise TypeError(
            "brightness of function macadam_limits( )"
            "has to be between 0 and 1"
        )
    # workarround because illuminant and cmfs are rounded
    # in a different way.
    illuminant = reshape_sd(illuminant, cmfs.shape)
    cmfs = reshape_sd(cmfs, spectral_range)
    illuminant = reshape_sd(illuminant, spectral_range)

    # The cmfs are convolved with the given illuminant
    X_illuminated, Y_illuminated, Z_illuminated = (
        tsplit(cmfs.values) * illuminant.values
    )
    # Generate empty output-array
    out_limits = np.zeros_like(cmfs.values)
    # For examle a SpectralShape(360, 830, 1) has 471 entries
    opti_colour = np.zeros_like(Y_illuminated)
    # The array of optimal colours has the same dimensions
    # like Y_illuminated, in our example: 471
    colour_range = illuminant.values.shape[0]
    a = np.arange(12)
    a = np.ceil(colour_range / 2 ** (a + 1)).astype(int)
    step_sizes = np.append(np.flip(np.unique(a)), 1)
    middle_opti_colour = step_sizes[0] - 1
    # in our example: 235
    width = step_sizes[1] - 1
    # in our example: 117
    step_sizes = np.delete(step_sizes, [0, 1])
    # in our example: np.array([59 30 15  8  4  2  1])
    # The first optimum color has its center initially at zero
    maximum_brightness = np.sum(Y_illuminated)

    def optimum_colour(width, center):
        opti_colour = zeros(colour_range)
        # creates array of 471 zeros and ones which represents optimum-colours
        # All values of the opti_colour-array are initially set to zero
        half_width = width
        center_opti_colour = center
        opti_colour[
            middle_opti_colour
            - half_width : middle_opti_colour
            + 1
            + half_width
        ] = 1
        # we start the construction of the optimum color
        # at the center of the opti_colour-array
        opti_colour = np.roll(
            opti_colour, center_opti_colour - middle_opti_colour
        )
        # the optimum colour is rolled to the right wavelength
        return opti_colour

    def bright_opti_colour(width, center, lightsource):
        brightness = (
            np.sum(optimum_colour(width, center) * lightsource)
            / maximum_brightness
        )
        return brightness

    # here we do some kind of Newton's Method to aproximate the
    # wandted illuminance at the whavelengt.
    # therefore the numbers 127, 64, 32 and so on
    # step_size is in this case np.array([59 30 15  8  4  2  1])
    for wavelength in range(0, colour_range):
        for n in step_sizes:
            brightness = bright_opti_colour(width, wavelength, Y_illuminated)
            if brightness > target_bright or width >= middle_opti_colour:
                width -= n
            else:
                width += n

        brightness = bright_opti_colour(width, wavelength, Y_illuminated)
        if brightness < target_bright:
            width += 1

        rough_optimum = optimum_colour(width, wavelength)
        brightness = np.sum(rough_optimum * Y_illuminated) / maximum_brightness

        # in the following, the both borders of the found rough_optimum
        # are reduced to get more exact results
        bright_difference = (brightness - target_bright) * maximum_brightness
        # discrimination for single-wavelength-spectra
        if width > 0:
            opti_colour = zeros(colour_range)
            opti_colour[
                middle_opti_colour - width : middle_opti_colour + width + 1
            ] = 1
            # instead rolling forward opti_colour, light is rolled backward
            rolled_light = np.roll(
                Y_illuminated, middle_opti_colour - wavelength
            )
            opti_colour_light = opti_colour * rolled_light
            left_opti = opti_colour_light[middle_opti_colour - width]
            right_opti = opti_colour_light[middle_opti_colour + width]
            interpolation = 1 - (bright_difference / (left_opti + right_opti))
            opti_colour[middle_opti_colour - width] = interpolation
            opti_colour[middle_opti_colour + width] = interpolation
            # opti_colour is rolled to right position
            final_optimum = np.roll(
                opti_colour, wavelength - middle_opti_colour
            )
        else:
            final_optimum = rough_optimum / brightness * target_bright

        out_X = np.sum(final_optimum * X_illuminated) / maximum_brightness
        out_Y = target_bright
        out_Z = np.sum(final_optimum * Z_illuminated) / maximum_brightness
        triple = np.array([out_X, out_Y, out_Z])
        out_limits[wavelength] = triple
    return out_limits
