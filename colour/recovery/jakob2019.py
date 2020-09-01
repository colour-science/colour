# -*- coding: utf-8 -*-
"""
Jakob and Hanika (2019) - Reflectance Recovery
==============================================

Defines objects for reflectance recovery, i.e. spectral upsampling, using
*Jakob and Hanika (2019)* method:

-   :func:`colour.recovery.sd_Jakob2019`
-   :func:`colour.recovery.find_coefficients_Jakob2019`
-   :func:`colour.recovery.XYZ_to_sd_Jakob2019`
-   :class:`colour.recovery.Jakob2019Interpolator`

References
----------
-   :cite:`Jakob2019` : Jakob, W., & Hanika, J. (2019). A Low‐Dimensional
    Function Space for Efficient Spectral Upsampling. Computer Graphics Forum,
    38(2), 147–155. doi:10.1111/cgf.13626
"""

from __future__ import division, print_function, unicode_literals

import colour.ndarray as np
import struct
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

from colour import SDS_ILLUMINANTS
from colour.algebra import spow, smoothstep_function
from colour.colorimetry import (
    MSDS_CMFS_STANDARD_OBSERVER, SpectralDistribution, SpectralShape,
    intermediate_lightness_function_CIE1976, sd_ones, sd_to_XYZ)
from colour.difference import JND_CIE1976
from colour.models import XYZ_to_xy, XYZ_to_Lab, RGB_to_XYZ
from colour.utilities import (as_float_array, domain_range_scale, full,
                              index_along_last_axis, to_domain_1,
                              runtime_warning, zeros)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SPECTRAL_SHAPE_JAKOB2019', 'StopMinimizationEarly', 'sd_Jakob2019',
    'error_function', 'dimensionalise_coefficients', 'lightness_scale',
    'find_coefficients_Jakob2019', 'XYZ_to_sd_Jakob2019',
    'Jakob2019Interpolator'
]

SPECTRAL_SHAPE_JAKOB2019 = SpectralShape(360, 780, 5)
"""
Spectral shape for *Jakob and Hanika (2019)* method.

SPECTRAL_SHAPE_JAKOB2019 : SpectralShape
"""


class StopMinimizationEarly(Exception):
    """
    The exception used to stop :func:`scipy.optimize.minimize` once the
    value of the minimized function is small enough. *SciPy* doesn't currently
    offer a better way of doing it.

    Attributes
    ----------
    coefficients : ndarray, (3,)
        Coefficients (function arguments) when this exception was raised.
    error : float
        Error (function value) when this exception was raised.
    """

    def __init__(self, coefficients, error):
        self._coefficients = coefficients
        self._error = error

    @property
    def coefficients(self):
        """
        Getter property for the *Jakob and Hanika (2019)* exception
        coefficients.

        Returns
        -------
        ndarray
            *Jakob and Hanika (2019)* exception coefficients.
        """

        return self._coefficients

    @property
    def error(self):
        """
        Getter property for the *Jakob and Hanika (2019)* exception error
        value.

        Returns
        -------
        ndarray
            *Jakob and Hanika (2019)* exception coefficients.
        """

        return self._error


def sd_Jakob2019(coefficients, shape=SPECTRAL_SHAPE_JAKOB2019):
    """
    Returns a spectral distribution following the spectral model given by
    *Jakob and Hanika (2019)*.

    Parameters
    ----------
    coefficients : array_like
        Dimensionless coefficients for *Jakob and Hanika (2019)* reflectance
        spectral model.
    shape : SpectralShape, optional
        Shape used by the spectral distribution.

    Returns
    -------
    SpectralDistribution
        *Jakob and Hanika (2019)* spectral distribution.

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     sd_Jakob2019([-9e-05, 8.5e-02, -20], SpectralShape(400, 700, 20))
    ...     # doctest: +ELLIPSIS
    SpectralDistribution([[ 400.        ,    0.3143046...],
                          [ 420.        ,    0.4133320...],
                          [ 440.        ,    0.4880034...],
                          [ 460.        ,    0.5279562...],
                          [ 480.        ,    0.5319346...],
                          [ 500.        ,    0.5      ...],
                          [ 520.        ,    0.4326202...],
                          [ 540.        ,    0.3373544...],
                          [ 560.        ,    0.2353056...],
                          [ 580.        ,    0.1507665...],
                          [ 600.        ,    0.0931332...],
                          [ 620.        ,    0.0577434...],
                          [ 640.        ,    0.0367011...],
                          [ 660.        ,    0.0240879...],
                          [ 680.        ,    0.0163316...],
                          [ 700.        ,    0.0114118...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    """

    c_0, c_1, c_2 = as_float_array(coefficients)
    wl = shape.range()
    U = c_0 * wl ** 2 + c_1 * wl + c_2
    R = 1 / 2 + U / (2 * np.sqrt(1 + U ** 2))

    name = 'Jakob (2019) - {0} (COEFF)'.format(coefficients)

    return SpectralDistribution(R, wl, name=name)


def error_function(coefficients,
                   target,
                   cmfs,
                   illuminant,
                   max_error=None,
                   additional_data=False):
    """
    Computes :math:`\\Delta E_{76}` between the target colour and the colour
    defined by given spectral model, along with its gradient.

    Parameters
    ----------
    coefficients : array_like
        Dimensionless coefficients for *Jakob and Hanika (2019)* reflectance
        spectral model.
    target : array_like, (3,)
        *CIE L\\*a\\*b\\** colourspace array of the target colour.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    max_error : float, optional
        Raise ``StopMinimizationEarly`` if the error is smaller than this.
        The default is *None* and the function doesn't raise anything.
    additional_data : bool, optional
        If *True*, some intermediate calculations are returned, for use in
        correctness tests: R, XYZ and Lab.

    Returns
    -------
    error : float
        The computed :math:`\\Delta E_{76}` error.
    derror : ndarray, (3,)
        The gradient of error, i.e. the first derivatives of error with respect
        to the input coefficients.
    R : ndarray
        Computed spectral reflectance.
    XYZ : ndarray, (3,)
        *CIE XYZ* tristimulus values corresponding to ``R``.
    Lab : ndarray, (3,)
        *CIE L\\*a\\*b\\** colourspace array corresponding to ``XYZ``.

    Raises
    ------
    StopMinimizationEarly
        Raised when the error is below ``max_error``.
    """

    c_0, c_1, c_2 = as_float_array(coefficients)
    wv = np.linspace(0, 1, len(cmfs.shape))

    U = c_0 * wv ** 2 + c_1 * wv + c_2
    t1 = np.sqrt(1 + U ** 2)
    R = 1 / 2 + U / (2 * t1)

    t2 = 1 / (2 * t1) - U ** 2 / (2 * t1 ** 3)
    dR = np.array([wv ** 2 * t2, wv * t2, t2])

    E = illuminant.values * R
    dE = illuminant.values * dR

    dw = cmfs.wavelengths[1] - cmfs.wavelengths[0]
    k = 1 / (np.sum(cmfs.values[:, 1] * illuminant.values) * dw)

    XYZ = k * np.dot(E, cmfs.values) * dw
    dXYZ = np.transpose(k * np.dot(dE, cmfs.values) * dw)

    XYZ_n = sd_to_XYZ(illuminant, cmfs)
    XYZ_n /= XYZ_n[1]
    XYZ_XYZ_n = XYZ / XYZ_n

    XYZ_f = intermediate_lightness_function_CIE1976(XYZ, XYZ_n)
    dXYZ_f = np.where(
        XYZ_XYZ_n[..., np.newaxis] > (24 / 116) ** 3,
        1 / (3 * spow(XYZ_n[..., np.newaxis], 1 / 3) * spow(
            XYZ[..., np.newaxis], 2 / 3)) * dXYZ,
        (841 / 108) * dXYZ / XYZ_n[..., np.newaxis],
    )

    def intermediate_XYZ_to_Lab(XYZ_i, offset=16):
        """
        Returns the final intermediate value for the *CIE Lab* to *CIE XYZ*
        conversion.
        """

        return np.array([
            116 * XYZ_i[1] - offset, 500 * (XYZ_i[0] - XYZ_i[1]),
            200 * (XYZ_i[1] - XYZ_i[2])
        ])

    Lab_i = intermediate_XYZ_to_Lab(XYZ_f)
    dLab_i = intermediate_XYZ_to_Lab(dXYZ_f, 0)

    error = np.sqrt(np.sum((Lab_i - target) ** 2))
    if max_error is not None and error <= max_error:
        raise StopMinimizationEarly(coefficients, error)

    derror = np.sum(
        dLab_i * (Lab_i[..., np.newaxis] - target[..., np.newaxis]),
        axis=0) / error

    if additional_data:
        return error, derror, R, XYZ, Lab_i
    else:
        return error, derror


def dimensionalise_coefficients(coefficients, shape):
    """
    Rescales the dimensionless coefficients to given spectral shape.

    A dimensionless form of the reflectance spectral model is used in the
    optimisation process. Instead of the usual spectral shape, specified in
    nanometers, it is normalised to the [0, 1] range. A side effect is that
    computed coefficients work only with the normalised range and need to be
    rescaled to regain units and be compatible with standard shapes.

    Parameters
    ----------
    coefficients : array_like, (3,)
        Dimensionless coefficients.
    shape : SpectralShape
        Spectral distribution shape used in calculations.

    Returns
    -------
    ndarray, (3,)
        Dimensionful coefficients, with units of
        :math:`\\frac{1}{\\mathrm{nm}^2}`, :math:`\\frac{1}{\\mathrm{nm}}`
        and 1, respectively.
    """

    cp_0, cp_1, cp_2 = coefficients
    span = shape.end - shape.start

    c_0 = cp_0 / span ** 2
    c_1 = cp_1 / span - 2 * cp_0 * shape.start / span ** 2
    c_2 = (
        cp_0 * shape.start ** 2 / span ** 2 - cp_1 * shape.start / span + cp_2)

    return np.array([c_0, c_1, c_2])


def lightness_scale(steps):
    """
    Creates a non-linear lightness scale, as described in *Jakob and Hanika
    (2019)*. The spacing between very dark and very bright (and saturated)
    colours is made smaller, because in those regions coefficients tend to
    change rapidly and a finer resolution is needed.

    Parameters
    ----------
    steps : int
        Samples/steps count along the non-linear lightness scale.

    Returns
    -------
    ndarray
        Non-linear lightness scale.

    Examples
    --------
    >>> lightness_scale(5)  # doctest: +ELLIPSIS
    array([ 0.        ,  0.0656127...,  0.5       ,  0.9343872...,  \
1.        ])
    """

    linear = np.linspace(0, 1, steps)

    return smoothstep_function(smoothstep_function(linear))


def find_coefficients_Jakob2019(
        XYZ,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().align(SPECTRAL_SHAPE_JAKOB2019),
        illuminant=SDS_ILLUMINANTS['D65'].copy().align(
            SPECTRAL_SHAPE_JAKOB2019),
        coefficients_0=zeros(3),
        max_error=JND_CIE1976 / 100,
        dimensionalise=True):
    """
    Computes the coefficients for *Jakob and Hanika (2019)* reflectance
    spectral model.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values to find the coefficients for.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    coefficients_0 : array_like, (3,), optional
        Starting coefficients for the solver.
    max_error : float, optional
        Maximal acceptable error. Set higher to save computational time.
        If *None*, the solver will keep going until it is very close to the
        minimum. The default is ``ACCEPTABLE_DELTA_E``.
    dimensionalise : bool, optional
        If *True*, returned coefficients are dimensionful and will not work
        correctly if fed back as ``coefficients_0``. The default is *True*.

    Returns
    -------
    coefficients : ndarray, (3,)
        Computed coefficients that best fit the given colour.
    error : float
        :math:`\\Delta E_{76}` between the target colour and the colour
        corresponding to the computed coefficients.
    """

    shape = cmfs.shape

    if illuminant.shape != shape:
        runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}" colour matching '
            'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    def optimize(target_o, coefficients_0_o):
        """
        Minimises the error function using *L-BFGS-B* method.
        """

        try:
            result = minimize(
                error_function,
                coefficients_0_o, (target_o, cmfs, illuminant, max_error),
                method='L-BFGS-B',
                jac=True)

            return result.x, result.fun
        except StopMinimizationEarly as error:
            return error.coefficients, error.error

    xy_n = XYZ_to_xy(sd_to_XYZ(illuminant))

    XYZ_good = full(3, 0.5)
    coefficients_good = zeros(3)

    divisions = 3
    while divisions < 10:
        XYZ_r = XYZ_good
        coefficient_r = coefficients_good
        keep_divisions = False

        coefficients_0 = coefficient_r
        for i in range(1, divisions):
            XYZ_i = (XYZ - XYZ_r) * i / (divisions - 1) + XYZ_r
            Lab_i = XYZ_to_Lab(XYZ_i)

            coefficients_0, error = optimize(Lab_i, coefficients_0)

            if error > max_error:
                break
            else:
                XYZ_good = XYZ_i
                coefficients_good = coefficients_0
                keep_divisions = True
        else:
            break

        if not keep_divisions:
            divisions += 2

    target = XYZ_to_Lab(XYZ, xy_n)
    coefficients, error = optimize(target, coefficients_0)

    if dimensionalise:
        coefficients = dimensionalise_coefficients(coefficients, shape)

    return coefficients, error


def XYZ_to_sd_Jakob2019(
        XYZ,
        cmfs=MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
        .copy().align(SPECTRAL_SHAPE_JAKOB2019),
        illuminant=sd_ones(SPECTRAL_SHAPE_JAKOB2019),
        optimisation_kwargs=None,
        additional_data=False):
    """
    Recovers the spectral distribution of given RGB colourspace array
    using *Jakob and Hanika (2019)* method.

    Parameters
    ----------
    XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values to recover the spectral distribution from.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    optimisation_kwargs : dict_like, optional
        Parameters for :func:`colour.recovery.find_coefficients_Jakob2019`
        definition.
    additional_data : bool, optional
        If *True*, ``error`` will be returned alongside ``sd``.

    Returns
    -------
    sd : SpectralDistribution
        Recovered spectral distribution.
    error : float
        :math:`\\Delta E_{76}` between the target colour and the colour
        corresponding to the computed coefficients.

    Examples
    --------
    >>> from colour.colorimetry import CCS_ILLUMINANTS, sd_to_XYZ_integration
    >>> from colour.models import XYZ_to_sRGB
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> cmfs = (
    ...     MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> sd = XYZ_to_sd_Jakob2019(XYZ, cmfs)
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.3717653...],
                          [ 370.        ,    0.2551674...],
                          [ 380.        ,    0.1796344...],
                          [ 390.        ,    0.1324685...],
                          [ 400.        ,    0.1025098...],
                          [ 410.        ,    0.0828360...],
                          [ 420.        ,    0.0694814...],
                          [ 430.        ,    0.0601729...],
                          [ 440.        ,    0.0535745...],
                          [ 450.        ,    0.0488773...],
                          [ 460.        ,    0.0455801...],
                          [ 470.        ,    0.0433696...],
                          [ 480.        ,    0.0420540...],
                          [ 490.        ,    0.0415260...],
                          [ 500.        ,    0.0417442...],
                          [ 510.        ,    0.0427256...],
                          [ 520.        ,    0.0445487...],
                          [ 530.        ,    0.0473671...],
                          [ 540.        ,    0.0514381...],
                          [ 550.        ,    0.0571745...],
                          [ 560.        ,    0.0652386...],
                          [ 570.        ,    0.0767126...],
                          [ 580.        ,    0.0934152...],
                          [ 590.        ,    0.1184910...],
                          [ 600.        ,    0.1574567...],
                          [ 610.        ,    0.2196538...],
                          [ 620.        ,    0.3181180...],
                          [ 630.        ,    0.4598423...],
                          [ 640.        ,    0.6224910...],
                          [ 650.        ,    0.7601476...],
                          [ 660.        ,    0.8516183...],
                          [ 670.        ,    0.9061111...],
                          [ 680.        ,    0.9381461...],
                          [ 690.        ,    0.9575256...],
                          [ 700.        ,    0.9697334...],
                          [ 710.        ,    0.9777401...],
                          [ 720.        ,    0.9831864...],
                          [ 730.        ,    0.9870109...],
                          [ 740.        ,    0.9897712...],
                          [ 750.        ,    0.9918114...],
                          [ 760.        ,    0.9933506...],
                          [ 770.        ,    0.9945329...],
                          [ 780.        ,    0.9954555...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd) / 100  # doctest: +ELLIPSIS
    array([ 0.2065209...,  0.1220029...,  0.0513715...])
    """

    XYZ = to_domain_1(XYZ)

    if optimisation_kwargs is None:
        optimisation_kwargs = {}

    with domain_range_scale('ignore'):
        coefficients, error = find_coefficients_Jakob2019(
            XYZ, cmfs, illuminant, **optimisation_kwargs)

    sd = sd_Jakob2019(coefficients, cmfs.shape)
    sd.name = 'Jakob (2019) - {0}'.format(XYZ)

    if additional_data:
        return sd, error
    else:
        return sd


class Jakob2019Interpolator(object):
    """
    Class for working with pre-computed lookup tables for the
    *Jakob and Hanika (2019)* spectral upsampling method. It allows significant
    time savings by performing the expensive numerical optimization ahead of
    time and storing the results in a file.

    The file format is compatible with the code and *.coeff* files in
    supplemental material published alongside the article.

    Attributes
    ----------
    table
    size
    lightness_scale
    coefficients

    Methods
    -------
    RGB_to_coefficients
    RGB_to_sd
    generate
    read
    write
    """

    def __init__(self):
        self._table = None
        self._size = None
        self._lightness_scale = None
        self._coefficients = None

    @property
    def table(self):
        """
        Getter property for the *Jakob and Hanika (2019)* interpolator
        3D table, i.e. the cube storing the interpolation data.

        Returns
        -------
        RegularGridInterpolator
            *Jakob and Hanika (2019)* interpolator 3D table.
        """

        return self._table

    @property
    def size(self):
        """
        Getter property for the *Jakob and Hanika (2019)* interpolator
        size, i.e. the samples count on one side of the 3D table.

        Returns
        -------
        ndarray
            *Jakob and Hanika (2019)* interpolator size.
        """

        return self._size

    @property
    def lightness_scale(self):
        """
        Getter property for the *Jakob and Hanika (2019)* interpolator
        lightness scale.

        Returns
        -------
        int
            *Jakob and Hanika (2019)* interpolator lightness scale.
        """

        return self._lightness_scale

    @property
    def coefficients(self):
        """
        Getter property for the *Jakob and Hanika (2019)* interpolator
        coefficients.

        Returns
        -------
        ndarray
            *Jakob and Hanika (2019)* interpolator coefficients.
        """

        return self._coefficients

    def _create_table(self):
        """
        Creates a :class:`scipy.interpolate.RegularGridInterpolator` class
        instance for read or generated coefficients.
        """

        samples = np.linspace(0, 1, self._size)
        axes = ([0, 1, 2], self._lightness_scale, samples, samples)

        self._table = RegularGridInterpolator(
            axes, self._coefficients, bounds_error=False)

    def RGB_to_coefficients(self, RGB):
        """
        Look up a given *RGB* colourspace array and return corresponding
        coefficients. Interpolation is used for colours not on the table grid.

        Parameters
        ----------
        RGB : ndarray, (3,)
            *RGB* colourspace array.

        Returns
        -------
        coefficients : ndarray, (3,)
            Corresponding coefficients that can be passed to
            :func:`colour.recovery.jakob2019.sd_Jakob2019` to obtain a
            spectral distribution.
        """

        RGB = as_float_array(RGB)

        vmax = np.max(RGB, axis=-1)
        chroma = RGB / (np.expand_dims(vmax, -1) + 1e-10)

        imax = np.argmax(RGB, axis=-1)
        v1 = index_along_last_axis(RGB, imax)
        v2 = index_along_last_axis(chroma, (imax + 2) % 3)
        v3 = index_along_last_axis(chroma, (imax + 1) % 3)

        coords = np.stack([imax, v1, v2, v3], axis=-1)

        return self._table(coords).squeeze()

    def RGB_to_sd(self, RGB, shape=SPECTRAL_SHAPE_JAKOB2019):
        """
        Looks up a given *RGB* colourspace array and return the corresponding
        spectral distribution.

        Parameters
        ----------
        RGB : ndarray, (3,)
            *RGB* colourspace array.
        shape : SpectralShape, optional
            Shape used by the spectral distribution.

        Returns
        -------
        sd : SpectralDistribution
            Spectral distribution corresponding with the RGB* colourspace
            array.
        """

        sd = sd_Jakob2019(self.RGB_to_coefficients(RGB), shape)
        sd.name = 'Jakob (2019) - {0} (RGB)'.format(RGB)

        return sd

    def generate(self,
                 colourspace,
                 cmfs,
                 illuminant,
                 resolution,
                 verbose=False,
                 print_callable=print):
        """
        Creates a lookup table for a given *RGB* colourspace with given
        resolution.

        Parameters
        ----------
        colourspace: RGB_Colourspace
            The *RGB* colourspace to create a lookup table for.
        cmfs : XYZ_ColourMatchingFunctions
            Standard observer colour matching functions.
        illuminant : SpectralDistribution
            Illuminant spectral distribution.
        resolution : int
            The resolution of the lookup table. Higher values will decrease
            errors but at the cost of a much longer run time. The published
            *.coeff* files have a resolution of 64.
        verbose : bool, optional
            If *True*, the default, information about the progress is printed
            to the standard output.
        print_callable : callable, optional
            Callable used to for verbose.
        """

        illuminant_xy = XYZ_to_xy(sd_to_XYZ(illuminant))

        # It could be interesting to have different resolutions for lightness
        # and chromaticity, but the current file format doesn't allow it.
        lightness_steps = resolution
        chroma_steps = resolution

        self._lightness_scale = lightness_scale(lightness_steps)
        self._coefficients = np.empty(
            [3, chroma_steps, chroma_steps, lightness_steps, 3])

        cube_indexes = np.ndindex(3, chroma_steps, chroma_steps)

        # First, create a list of all the fully bright colours with the order
        # matching cube_indexes.
        samples = np.linspace(0, 1, chroma_steps)
        yx = np.meshgrid(*[[1], samples, samples], indexing='ij')
        yx = np.transpose(yx).reshape(-1, 3)
        chromas = np.concatenate(
            [yx, np.roll(yx, 1, axis=1),
             np.roll(yx, 2, axis=1)])

        # TODO: Replace this with a proper progress bar.
        if verbose:
            print_callable(
                '{0:>6} {1:>6} {2:>6}  {3:>13} {4:>13} {5:>13}  {6}'.format(
                    'R', 'G', 'B', 'c0', 'c1', 'c2', 'Delta E'))

        # TODO: Send the list to a multiprocessing pool; this takes a while.
        for (i, j, k), chroma in zip(cube_indexes, chromas):
            if verbose:
                print_callable(
                    'i={0}, j={1}, k={2}, R={3:.6f}, G={4:.6f}, B={5:.6f}'.
                    format(i, j, k, chroma[0], chroma[1], chroma[2]))

            def optimize(L, coefficients_0):
                """
                Solves for a specific lightness and stores the result in the
                appropriate cell.
                """

                RGB = self._lightness_scale[L] * chroma

                XYZ = RGB_to_XYZ(RGB, colourspace.whitepoint, illuminant_xy,
                                 colourspace.RGB_to_XYZ_matrix)

                coefficients, error = find_coefficients_Jakob2019(
                    XYZ,
                    cmfs,
                    illuminant,
                    coefficients_0,
                    dimensionalise=False)

                if verbose:
                    print_callable(
                        '{0:.4f} {1:.4f} {2:.4f}  '
                        '{3:>13.6f} {4:>13.6f} {5:>13.6f} {6:.6f}'.format(
                            RGB[0], RGB[1], RGB[2], coefficients[0],
                            coefficients[1], coefficients[2], error))

                self._coefficients[i, L, j,
                                   k, :] = dimensionalise_coefficients(
                                       coefficients, cmfs.shape)

                return coefficients

            # Starts from somewhere in the middle, similarly to how feedback
            # works in "colour.recovery.find_coefficients_Jakob2019"
            # definition.
            middle_L = lightness_steps // 3
            middle_coefficients = optimize(middle_L, (0, 0, 0))

            # Goes down the lightness scale.
            coefficients_0 = middle_coefficients
            for L in reversed(range(0, middle_L)):
                coefficients_0 = optimize(L, coefficients_0)

            # Goes up the lightness scale.
            coefficients_0 = middle_coefficients
            for L in range(middle_L + 1, lightness_steps):
                coefficients_0 = optimize(L, coefficients_0)

        self._size = lightness_steps
        self._create_table()

    def read(self, path):
        """
        Loads a lookup table from a *.coeff* file.

        Parameters
        ----------
        path : unicode
            Path to the file.
        """

        with open(path, 'rb') as coeff_file:
            if coeff_file.read(4).decode('ISO-8859-1') != 'SPEC':
                raise ValueError(
                    'Bad magic number, this is likely not the right file type!'
                )

            self._size = struct.unpack('i', coeff_file.read(4))[0]
            self._lightness_scale = np.fromfile(
                coeff_file, count=self._size, dtype=np.float32)
            self._coefficients = np.fromfile(
                coeff_file, count=3 * (self._size ** 3) * 3, dtype=np.float32)
            self._coefficients = self._coefficients.reshape(
                3, self._size, self._size, self._size, 3)

        self._create_table()

    def write(self, path):
        """
        Writes the lookup table to a *.coeff* file.

        Parameters
        ----------
        path : unicode
            Path to the file.
        """

        with open(path, 'wb') as coeff_file:
            coeff_file.write(b'SPEC')
            coeff_file.write(struct.pack('i', self._coefficients.shape[1]))
            np.float32(self._lightness_scale).tofile(coeff_file)
            np.float32(self._coefficients).tofile(coeff_file)
