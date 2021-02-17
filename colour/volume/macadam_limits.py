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
from colour.models import xyY_to_XYZ
from colour.volume import OPTIMAL_COLOUR_STIMULI_ILLUMINANTS
from colour.utilities import CACHE_REGISTRY, validate_method
from colour.colorimetry import MSDS_CMFS, SpectralShape, SDS_ILLUMINANTS

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

    optimal_colour_stimuli = OPTIMAL_COLOUR_STIMULI_ILLUMINANTS.get(illuminant)

    if optimal_colour_stimuli is None:
        raise KeyError(
            f'"{illuminant}" not found in factory "Optimal Colour Stimuli": '
            f'"{sorted(OPTIMAL_COLOUR_STIMULI_ILLUMINANTS.keys())}".'
        )

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
    >>> is_within_macadam_limits(np.array([0.3205, 0.4131, 0.51]), 'A')
    array(True, dtype=bool)
    >>> a = np.array([[0.3205, 0.4131, 0.51],
    ...               [0.0005, 0.0031, 0.001]])
    >>> is_within_macadam_limits(a, 'A')
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


def macadam_limits(target_brightness, illuminant=()):
    """
    whavelenght reaches from 360 to 830 nm, in within the programm it is
    handled as 0 to 470. Beyond the references this programm is very fast,
    because the possible optimums are not simply tested step by step but
    more effectively targeted by steps of power of two. The whavelenghts
    left and right of a rough optimum are fited by a rule of proportion,
    so that the wished brightness will be reached exactly.

    Parameters
    ----------
    target_brightness : floating point
        brightness has to be between 0 and 1

    illuminant: object
        illuminant must be out of colorimetry.MSDS_CMFS['XXX']
        If there is no illuminant or it has the wrong form,
        the illuminant SDS_ILLUMINANTS['E']
        is choosen wich has no influence to the calculations,
        because it is an equal-energy-spectrum

    if necessary a third parameter for the
    colour-matching funciton could easily be implemented

    Returns
    -------
    an array of CIE -X,Y,Z - Triples for every single whavelength
    in single nm - Steps in the range from 360 to 830 nm

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

    Example
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
        colour.macadam_limits(n/20, illuminant) / 100)
        plot_Narrowband_Spectra (Yxy_Narrowband_Spectra)

    plt.show()
    """
    target_bright = target_brightness
    if target_bright > 1 or target_bright < 0:
        raise TypeError('brightness of function macadam_limits( )'
                        'has to be between 0 and 1')
    standard_cfms = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    X_cie31 = standard_cfms.values[..., 0]
    Y_cie31 = standard_cfms.values[..., 1]
    Z_cie31 = standard_cfms.values[..., 2]
    try:
        illuminant.interpolator
    except AttributeError:
        illuminant = SDS_ILLUMINANTS['E']


# If there is no illuminant or it has the wrong form,
# an illuminant choosen with no influence
# If the illuminanats do not match the format of the Standard Observer,
# they have to be adaptet
    illuminant.extrapolate(SpectralShape(360, 830))
    illuminant.interpolate(SpectralShape(360, 830, 1))
    # The cie31 cmfs are convolved with the given illuminant
    X_illuminated = X_cie31 * illuminant.values
    Y_illuminated = Y_cie31 * illuminant.values
    Z_illuminated = Z_cie31 * illuminant.values
    # Generate empty output-array
    out_limits = np.zeros_like(standard_cfms.values)
    # This Array has 471 entries for whavelenghts from 360 nm to 830 nm
    opti_colour = np.zeros_like(Y_illuminated)
    # The array of optimal colours has the same dimensions like Y_illuminated
    # and all entries are initialy set to zero
    middle_opti_colour = 235
    # is a constant and not be changed. At 595nm (360 + 235)
    # in the middle of the center_opti_colour-array
    # be aware that counting in array-positions starts at zero
    # The first optimum color has its center initialy at zero
    maximum_brightness = np.sum(Y_illuminated)

    # "integral" over Y_illuminated

    def optimum_colour(width, center):
        opti_colour = np.zeros(471)
        # creates array of 471 zeros and ones which represents optimum-colours
        # All values of the opti_colour-array are intialy set to zero
        half_width = width
        center_opti_colour = center
        middle_opti_colour = 235
        opti_colour[middle_opti_colour - half_width:middle_opti_colour +
                    half_width + 1] = 1
        # we start the construction of the optimum color
        # at the center of the opti_colour-array
        opti_colour = np.roll(opti_colour,
                              center_opti_colour - middle_opti_colour)
        # the optimum colour is rolled to the right whavelenght
        return opti_colour

    def bright_opti_colour(width, center, lightsource):
        brightness = np.sum(
            optimum_colour(width, center) * lightsource) / maximum_brightness
        return brightness

    step_size = np.array([64, 32, 16, 8, 4, 2, 1])
    for whavelength in range(0, 471):
        width = 127
        for n in step_size:
            brightness = bright_opti_colour(width, whavelength, Y_illuminated)
            if brightness > target_bright or width > 234:
                width -= n
            else:
                width += n

        brightness = bright_opti_colour(width, whavelength, Y_illuminated)
        if brightness < target_bright:
            width += 1
            brightness = bright_opti_colour(width, whavelength, Y_illuminated)

        rough_optimum = optimum_colour(width, whavelength)
        brightness = np.sum(rough_optimum * Y_illuminated) / maximum_brightness

        # in the following, the both borders of the found rough_optimum
        # are reduced to get more exact results
        bright_difference = (brightness - target_bright) * maximum_brightness
        # discrimination for single-whavelenght-spectra
        if width > 0:
            opti_colour = np.zeros(471)
            opti_colour[middle_opti_colour - width:middle_opti_colour + width +
                        1] = 1
            # instead rolling foreward opti_colour, light is rolled backward
            rolled_light = np.roll(Y_illuminated,
                                   middle_opti_colour - whavelength)
            opti_colour_light = opti_colour * rolled_light
            left_opti = opti_colour_light[middle_opti_colour - width]
            right_opti = opti_colour_light[middle_opti_colour + width]
            interpolation = 1 - (bright_difference / (left_opti + right_opti))
            opti_colour[middle_opti_colour - width] = interpolation
            opti_colour[middle_opti_colour + width] = interpolation
            # opti_colour is rolled to right possition
            final_optimum = np.roll(opti_colour,
                                    whavelength - middle_opti_colour)
        else:
            final_optimum = rough_optimum / brightness * target_bright

        out_X = np.sum(final_optimum * X_illuminated)
        out_Y = target_bright * maximum_brightness
        out_Z = np.sum(final_optimum * Z_illuminated)
        triple = np.array([out_X, out_Y, out_Z])
        out_limits[whavelength] = triple
    return (out_limits)
