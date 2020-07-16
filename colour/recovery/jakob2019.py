# -*- coding: utf-8 -*-
"""
Jakob and Hanika (2019) - Reflectance Recovery
==============================================

Defines objects for reflectance recovery, i.e. spectral upsampling, using
*Jakob and Hanika (2019)* method:

-   :func:`colour.recovery.RGB_to_sd_Jakob2019`
-   :class:`colour.recovery.Jakob2019Interpolator`

References
----------
-   :cite:`Jakob2019` : Jakob, W., & Hanika, J. (2019). A Low‐Dimensional
    Function Space for Efficient Spectral Upsampling. Computer Graphics Forum,
    38(2), 147–155. doi:10.1111/cgf.13626
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import struct
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

from colour import ILLUMINANT_SDS
from colour.algebra import spow, smoothstep_function
from colour.colorimetry import (
    STANDARD_OBSERVER_CMFS, SpectralDistribution, SpectralShape,
    intermediate_lightness_function_CIE1976, sd_to_XYZ)
from colour.models import XYZ_to_xy, XYZ_to_Lab, RGB_to_XYZ
from colour.utilities import (as_float_array, runtime_warning,
                              index_along_last_axis)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'DEFAULT_SPECTRAL_SHAPE_JAKOB_2019', 'ACCEPTABLE_DELTA_E', 'sd_Jakob2019',
    'StopMinimizationEarly', 'error_function', 'dimensionalise_coefficients',
    'lightness_scale', 'find_coefficients', 'RGB_to_sd_Jakob2019',
    'Jakob2019Interpolator'
]

DEFAULT_SPECTRAL_SHAPE_JAKOB_2019 = SpectralShape(360, 780, 5)
"""
Default spectral shape for *Jakob and Hanika (2019)* method.

DEFAULT_SPECTRAL_SHAPE_JAKOB_2019 : SpectralShape
"""

ACCEPTABLE_DELTA_E = 2.4 / 100  # 1% of JND
"""
Acceptable *perceptual* distance in the *CIE L\\*a\\*b\\** colourspace.

Notes
-----
*Jakob and Hanika (2019)* uses :math:`\\Delta E_{76}` in the
*CIE L\\*a\\*b\\** colourspace as an error metric during the optimization
process. While the *CIE L\\*a\\*b\\** colourspace features decent perceptual
uniformity, it was deemed unsatisfactory when comparing some pair of colors,
compelling the CIE into improving the metric with the CIE 1994
(:math:`\\Delta E_{94}`) quasimetric whose perceptual uniformity was
subsequently corrected with the CIE 2000 (:math:`\\Delta E_{00}`) quasimetric.
Thus, the error metric could be improved by adopting CIE 2000 or even a more
perceptually uniform colourspace such as :math:`IC_TC_P` or :math:`J_zA_zB_z`.

ACCEPTABLE_DELTA_E = 2.4 / 100 : float
"""


def sd_Jakob2019(coefficients, shape=DEFAULT_SPECTRAL_SHAPE_JAKOB_2019):
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

    c_0, c_1, c_2 = coefficients
    wl = shape.range()
    U = c_0 * wl ** 2 + c_1 * wl + c_2
    R = 1 / 2 + U / (2 * np.sqrt(1 + U ** 2))

    name = 'Jakob (2019) - {0} (COEFF)'.format(coefficients)

    return SpectralDistribution(R, wl, name=name)


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
        self.coefficients = coefficients
        self.error = error


def error_function(coefficients,
                   target,
                   shape,
                   cmfs,
                   illuminant,
                   illuminant_XYZ,
                   max_error=None,
                   return_intermediates=False):
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
    shape : SpectralShape
        Spectral shape used in calculations.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    illuminant_XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values of the illuminant.
    max_error : float, optional
        Raise ``StopMinimizationEarly`` if the error is smaller than this.
        The default is ``None`` and the function doesn't raise anything.
    return_intermediates : bool, optional
        If true, some intermediate calculations are returned, for use in
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

    c_0, c_1, c_2 = coefficients
    wv = np.linspace(0, 1, len(shape))

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

    XYZ_n = XYZ / illuminant_XYZ

    f = intermediate_lightness_function_CIE1976(XYZ, illuminant_XYZ)
    df = np.where(
        XYZ_n[..., np.newaxis] > (24 / 116) ** 3,
        1 / (3 * spow(illuminant_XYZ[..., np.newaxis], 1 / 3) * spow(
            XYZ[..., np.newaxis], 2 / 3)) * dXYZ,
        (841 / 108) * dXYZ / illuminant_XYZ[..., np.newaxis],
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

    Lab = intermediate_XYZ_to_Lab(f)
    dLab = intermediate_XYZ_to_Lab(df, 0)

    error = np.sqrt(np.sum((Lab - target) ** 2))
    if max_error is not None and error <= max_error:
        raise StopMinimizationEarly(coefficients, error)

    derror = np.sum(
        dLab * (Lab[..., np.newaxis] - target[..., np.newaxis]),
        axis=0) / error

    if return_intermediates:
        return error, derror, R, XYZ, Lab
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


def find_coefficients(
        RGB,
        colourspace,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        illuminant=ILLUMINANT_SDS['D65'].copy().align(
            DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        coefficients_0=np.array([0, 0, 0]),
        max_error=ACCEPTABLE_DELTA_E,
        dimensionalise=True,
        try_directly_first=False,
        use_feedback='adaptive-from-grey',
        lightness_steps=64):
    """
    Computes the coefficients for *Jakob and Hanika (2019)* reflectance
    spectral model.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace array of the target colour. Values must be linear;
        do not apply cctf encoding.
    colourspace : RGB_Colourspace
        *RGB* colourspace of the target colour.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    coefficients_0 : array_like, (3,), optional
        Starting coefficients for the solver.
    max_error : float, optional
        See the documentation for :func:`colour.recovery.RGB_to_sd_Jakob2019`.
    dimensionalise : bool, optional
        If *True*, returned coefficients are dimensionful and will not work
        correctly if fed back as ``coefficients_0``. The default is *True*.
    try_directly_first: bool, optional
        See the documentation for :func:`colour.recovery.RGB_to_sd_Jakob2019`.
    use_feedback : unicode, optional
        See the documentation for :func:`colour.recovery.RGB_to_sd_Jakob2019`.
    lightness_steps : int, optional
        See the documentation for :func:`colour.recovery.RGB_to_sd_Jakob2019`.

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

    illuminant_XYZ = sd_to_XYZ(illuminant)
    illuminant_XYZ /= illuminant_XYZ[1]
    illuminant_xy = XYZ_to_xy(illuminant_XYZ)

    def RGB_to_Lab(RGB):
        """
        A shorthand for converting given *RGB* colourspace array to
        *CIE L\\*a\\*b\\** colourspace.
        """

        XYZ = RGB_to_XYZ(
            RGB,
            colourspace.whitepoint,
            illuminant_xy,
            colourspace.RGB_to_XYZ_matrix,
        )

        return XYZ_to_Lab(XYZ, illuminant_xy)

    def optimize(target, coefficients_0):
        """
        Minimises the error function using *L-BFGS-B* method.
        """

        try:
            result = minimize(
                error_function,
                coefficients_0,
                (target, shape, cmfs, illuminant, illuminant_XYZ, max_error),
                method='L-BFGS-B',
                jac=True)

            return result.x, result.fun
        except StopMinimizationEarly as error:
            return error.coefficients, error.error

    if try_directly_first:
        target = RGB_to_Lab(RGB)
        coefficients, error = optimize(target, coefficients_0)

        if error < max_error:
            if dimensionalise:
                coefficients = dimensionalise_coefficients(coefficients, shape)

            return coefficients, error

    if use_feedback == 'from-unsaturated':
        target_max = np.max(RGB) + 1e-10
        scale = lightness_scale(lightness_steps)

        i = lightness_steps // 3
        ascending = scale[i] < np.max(RGB)

        while True:
            if ascending:
                if i + 1 == lightness_steps or scale[i + 1] >= target_max:
                    break
            else:
                if i == 0 or scale[i - 1] <= target_max:
                    break

            intermediate_RGB = scale[i] * (RGB + 1e-10) / target_max
            intermediate = RGB_to_Lab(intermediate_RGB)

            coefficients_0, _error = optimize(intermediate, coefficients_0)
            i += 1 if ascending else -1
    elif use_feedback == 'adaptive-from-grey':
        good_RGB = np.array([0.5, 0.5, 0.5])
        good_coefficients = np.array([0, 0, 0])

        divisions = 3
        while divisions < 10:
            keep_divisions = False
            reference_RGB = good_RGB
            reference_coefficients = good_coefficients

            coefficients_0 = reference_coefficients
            for i in range(1, divisions):
                intermediate_RGB = (RGB - reference_RGB) * i / (
                    divisions - 1) + reference_RGB
                intermediate = RGB_to_Lab(intermediate_RGB)

                coefficients_0, error = optimize(intermediate, coefficients_0)

                if error > max_error:
                    break
                else:
                    good_RGB = intermediate_RGB
                    good_coefficients = coefficients_0
                    keep_divisions = True
            else:
                break

            if not keep_divisions:
                divisions += 2
    elif use_feedback is not None:
        raise ValueError('"use_feedback" must be one of: "{0}"'.format(
            ['from-unsaturated', 'adaptive-from-grey']))

    target = RGB_to_Lab(RGB)
    coefficients, error = optimize(target, coefficients_0)

    if dimensionalise:
        coefficients = dimensionalise_coefficients(coefficients, shape)

    return coefficients, error


def RGB_to_sd_Jakob2019(
        RGB,
        colourspace,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        illuminant=ILLUMINANT_SDS['D65'].copy().align(
            DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        coefficients_0=np.array([0, 0, 0]),
        max_error=ACCEPTABLE_DELTA_E,
        try_directly_first=False,
        use_feedback='adaptive-from-grey',
        lightness_steps=64,
        return_error=False):
    """
    Recovers the spectral distribution of given RGB colourspace array
    using *Jakob and Hanika (2019)* method.

    Parameters
    ----------
    RGB : array_like, (3,)
        *RGB* colourspace array of the target colour. Values must be linear and
        without CCTF encoding.
    colourspace : RGB_Colourspace
        *RGB* colourspace of the target colour.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    coefficients_0 : array_like, (3,), optional
        Starting coefficients for the solver.
    max_error : float, optional
        Maximal acceptable error. Set higher to save computational time.
        If ``None``, the solver will keep going until it's very close to the
        minimum. The default is ``ACCEPTABLE_DELTA_E``.
    try_directly_first: bool, optional
        If true and use_feedback is not ``None``, an attempt to solve for
        the target colour will be made, and feedback will be used only if that
        fails to produce an error below ``max_error``.
    use_feedback : unicode, optional
        If set, a less saturated version of the target colour is solved for
        first. Then, colours closer and closer to the target are computed,
        feeding the result of every iteration to the next (as starting
        coefficients). This improves stability of results and greatly improves
        convergence. The possible values are:

            - 'from-unsaturated'   Start from an unsaturated colour with the
                                   same chromaticity and go up or down a
                                   lightness scale toward the target colour.
            - 'adaptive-from-gray' Start from gray and move toward the target
                                   in a straight line (in the *RGB*
                                   colourspace) with an adaptive step size.
            - None                 Don't use feedback.
    lightness_steps : int, optional
        The numbers of steps in the lightness scale used for computing
        intermediate colours when ``use_feedback`` is enabled. The default
        value of 64 is what is used in *Jakob and Hanika (2019)*.
    return_error : bool, optional
        If *True*, ``error`` will be returned alongside ``sd``.

    Returns
    -------
    sd : SpectralDistribution
        Recovered spectral distribution.
    error : float
        :math:`\\Delta E_{76}` between the target colour and the colour
        corresponding to the computed coefficients.
    """

    RGB = as_float_array(RGB)

    coefficients, error = find_coefficients(
        RGB,
        colourspace,
        cmfs,
        illuminant,
        coefficients_0=coefficients_0,
        max_error=max_error,
        try_directly_first=try_directly_first,
        use_feedback=use_feedback,
        lightness_steps=lightness_steps,
    )

    sd = sd_Jakob2019(coefficients, cmfs.shape)
    sd.name = 'Jakob (2019) - {0} {1}'.format(colourspace.name, RGB)

    if return_error:
        return sd, error
    else:
        return sd


class Jakob2019Interpolator:
    """
    Class for working with pre-computed lookup tables for the
    *Jakob and Hanika (2019)* spectral upsampling method. It allows significant
    time savings by performing the expensive numerical optimization ahead of
    time and storing the results in a file.

    The file format is compatible with the code and *.coeff* files in
    supplemental material published alongside the article.

    Attributes
    ----------
    scale
    coefficients
    resolution

    Methods
    -------
    RGB_to_coefficients
    RGB_to_sd
    generate
    read
    write
    """

    def __init__(self):
        self.scale = None
        self.coefficients = None
        self.resolution = None

    def _create_cube(self):
        """
        Creates a :class:`scipy.interpolate.RegularGridInterpolator` class
        instance for read or generated coefficients.
        """

        samples = np.linspace(0, 1, self.resolution)
        axes = ([0, 1, 2], self.scale, samples, samples)

        self.cube = RegularGridInterpolator(
            axes, self.coefficients, bounds_error=False)

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

        return self.cube(coords).squeeze()

    def RGB_to_sd(self, RGB, shape=DEFAULT_SPECTRAL_SHAPE_JAKOB_2019):
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
                 verbose=True,
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

        # It could be interesting to have different resolutions for lightness
        # and chromaticity, but the current file format doesn't allow it.
        lightness_steps = resolution
        chroma_steps = resolution

        self.scale = lightness_scale(lightness_steps)
        self.coefficients = np.empty((3, chroma_steps, chroma_steps,
                                      lightness_steps, 3))

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

                RGB = self.scale[L] * chroma

                coefficients, error = find_coefficients(
                    RGB,
                    colourspace,
                    cmfs,
                    illuminant,
                    coefficients_0,
                    dimensionalise=False,
                    try_directly_first=True,
                    use_feedback='adaptive-from-grey')

                if verbose:
                    print_callable(
                        '{0:.4f} {1:.4f} {2:.4f}  '
                        '{3:>13.6f} {4:>13.6f} {5:>13.6f} {6:.6f}'.format(
                            RGB[0], RGB[1], RGB[2], coefficients[0],
                            coefficients[1], coefficients[2], error))

                self.coefficients[i, L, j, k, :] = dimensionalise_coefficients(
                    coefficients, cmfs.shape)

                return coefficients

            # Starts from somewhere in the middle, similarly to how feedback
            # works in find_coefficients.
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

        self.resolution = lightness_steps
        self._create_cube()

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

            self.resolution = struct.unpack('i', coeff_file.read(4))[0]
            self.scale = np.fromfile(
                coeff_file, count=self.resolution, dtype=np.float32)
            self.coefficients = np.fromfile(
                coeff_file,
                count=3 * self.resolution ** 3 * 3,
                dtype=np.float32)
            self.coefficients = self.coefficients.reshape(
                3, self.resolution, self.resolution, self.resolution, 3)

        self._create_cube()

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
            coeff_file.write(struct.pack('i', self.coefficients.shape[1]))
            np.float32(self.scale).tofile(coeff_file)
            np.float32(self.coefficients).tofile(coeff_file)
