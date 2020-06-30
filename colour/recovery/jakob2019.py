# -*- coding: utf-8 -*-
"""
Jakob and Hanika (2019) - Reflectance Recovery
==============================================

Defines objects for reflectance recovery using *Jakob and Hanika (2019)*
method:

-   :func:`colour.recovery.RGB_to_sd_Jakob2019`

References
----------
-   :cite:`Jakob2019` : Jakob, W., & Hanika, J. (2019). A Low‐Dimensional
    Function Space for Efficient Spectral Upsampling. Computer Graphics Forum,
    38(2), 147–155. doi:10.1111/cgf.13626
"""

from __future__ import division, unicode_literals

import numpy as np
import struct
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

from colour import ILLUMINANT_SDS
from colour.algebra import spow
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
    'DEFAULT_SPECTRAL_SHAPE_JAKOB_2019', 'RGB_to_sd_Jakob2019',
    'Jakob2019Interpolator'
]

DEFAULT_SPECTRAL_SHAPE_JAKOB_2019 = SpectralShape(360, 780, 5)
"""
DEFAULT_SPECTRAL_SHAPE_JAKOB_2019 : SpectralShape
"""
ACCEPTABLE_DELTA_E = 2.4 / 100  # 1% of JND
"""
ACCEPTABLE_DELTA_E = 2.4 / 100 : float
"""


def spectral_model(coefficients,
                   shape=DEFAULT_SPECTRAL_SHAPE_JAKOB_2019,
                   name=None):
    """
    Spectral model given by *Jakob and Hanika (2019)*.
    """

    c_0, c_1, c_2 = coefficients
    wl = shape.range()
    U = c_0 * wl ** 2 + c_1 * wl + c_2
    R = 1 / 2 + U / (2 * np.sqrt(1 + U ** 2))

    if name is None:
        name = "Jakob (2019) - {0} (coeffs.)".format(coefficients)

    return SpectralDistribution(R, wl, name=name)


class StopMinimizationEarly(Exception):
    """
    The exception used to stop :func:`scipy.optimize.minimize` once the
    value of the minimized function is small enough. Currently *SciPy* doesn't
    provide a better way of doing it.

    Attributes
    ----------
    coefficients : ndarray, (3,)
        Coefficients (function arguments) when this exception was thrown.
    error : float
        Error (function value) when this exception was thrown.
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
    Computes :math:`\\Delta E_{76}` between the target colour and the
    colour defined by given spectral model, along with its gradient.

    Parameters
    ----------
    coefficients : array_like
        Dimensionless coefficients for *Jakob and Hanika (2019)* reflectance
        spectral model.
    target : array_like, (3,)
        *CIE L\\*a\\*b\\** colourspace array of the target colour.
    shape : SpectralShape
        Spectral distribution shape used in calculations.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    illuminant_XYZ : array_like, (3,)
        *CIE XYZ* tristimulus values of the illuminant.
    max_error : float, optional
        Raise ``StopMinimizationEarly`` if the error is smaller than this.
        The default is ``None`` and the function doesn't raise anything.

    Other parameters
    ----------------
    return_intermediates : bool, optional
        If true, some intermediate calculations are returned, for use in
        correctness tests: R, XYZ and Lab

    Returns
    -------
    error : float
        The computed :math:`\\Delta E_{76}` error.
    derror : ndarray, (3,)
        The gradient of error, ie. the first derivatives of error with respect
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
    wv = np.linspace(0, 1, len(shape.range()))

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

    XYZ_norm = XYZ / illuminant_XYZ

    f = intermediate_lightness_function_CIE1976(XYZ, illuminant_XYZ)
    df = np.where(
        XYZ_norm[..., np.newaxis] > (24 / 116) ** 3,
        1 / (3 * spow(illuminant_XYZ[..., np.newaxis], 1 / 3) * spow(
            XYZ[..., np.newaxis], 2 / 3)) * dXYZ,
        (841 / 108) * dXYZ / illuminant_XYZ[..., np.newaxis],
    )

    def intermediate_XYZ_to_Lab(XYZ, offset=16):
        return np.array([
            116 * XYZ[1] - offset, 500 * (XYZ[0] - XYZ[1]),
            200 * (XYZ[1] - XYZ[2])
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

    return error, derror


def dimensionalise_coefficients(coefficients, shape):
    """
    Rescale dimensionless coefficients.

    A nondimensionalised form of the reflectance spectral model is used in
    optimisation. Instead of the usual spectral shape, specified in nanometers,
    it's normalised to the [0, 1] range. A side effect is that computed
    coefficients work only with the normalised range and need to be
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


def create_lightness_scale(steps):
    """
    Create a non-linear lightness scale, as described in *Jakob and Hanika
    (2019)*. The spacing between very dark and very bright (and saturated)
    colors is made smaller, because in those regions coefficients tend to
    change rapidly and a finer resolution is needed.
    """

    def smoothstep(x):
        return x ** 2 * (3 - 2 * x)

    linear = np.linspace(0, 1, steps)
    return smoothstep(smoothstep(linear))


def find_coefficients(
        target_RGB,
        colourspace,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        illuminant=ILLUMINANT_SDS['D65'].copy().align(
            DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        coefficients_0=(0, 0, 0),
        dimensionalise=True,
        use_feedback="adaptive-from-grey",
        lightness_steps=64,
        max_error=ACCEPTABLE_DELTA_E,
        try_directly_first=False):
    """
    Computes coefficients for *Jakob and Hanika (2019)* reflectance spectral
    model.

    Parameters
    ----------
    target_RGB : array_like, (3,)
        *RGB* colourspace array of the target colour. Values must be linear;
        do not apply cctf encoding.
    colourspace : RGB_Colourspace
        The RGB colourspace.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    coefficients_0 : array_like, (3,), optional
        Starting coefficients for the solver.
    dimensionalise : bool, optional
        If true, returned coefficients are dimensionful and will not work
        correctly if fed back as ``coefficients_0``. The default is true.
    use_feedback : string, optional
        See the documentation for :func:`RGB_to_sd_Jakob2019`.

    Other parameters
    ----------------
    return_intermediates : bool, optional
        If true, some intermediate calculations are returned, for use in
        correctness tests: R, XYZ and Lab
    lightness_steps : int, optional
        See the documentation for :func:`RGB_to_sd_Jakob2019`.
    max_error : float, optional
        See the documentation for :func:`RGB_to_sd_Jakob2019`.
    try_directly_first: bool, optional
        See the documentation for :func:`RGB_to_sd_Jakob2019`.

    Returns
    -------
    coefficients : ndarray, (3,)
        Computed coefficients that best fit the given colour.
    error : float
        :math:`\\Delta E_{76}` between the target colour and the colour
        corresponding to the computed coefficients.
    """

    if illuminant.shape != cmfs.shape:
        runtime_warning(
            'Aligning "{0}" illuminant shape to "{1}" colour matching '
            'functions shape.'.format(illuminant.name, cmfs.name))
        illuminant = illuminant.copy().align(cmfs.shape)

    shape = illuminant.shape
    illuminant_XYZ = sd_to_XYZ(illuminant)
    illuminant_XYZ /= illuminant_XYZ[1]
    illuminant_xy = XYZ_to_xy(illuminant_XYZ)

    def RGB_to_Lab(RGB):
        """
        A shorthand for converting from *RGB* to *CIE Lab*.
        """
        XYZ = RGB_to_XYZ(
            RGB,
            colourspace.whitepoint,
            illuminant_xy,
            colourspace.RGB_to_XYZ_matrix,
        )
        return XYZ_to_Lab(XYZ, illuminant_xy)

    def _minimize(target, coefficients_0):
        try:
            opt = minimize(
                error_function,
                coefficients_0,
                (target, shape, cmfs, illuminant, illuminant_XYZ, max_error),
                method="L-BFGS-B",
                jac=True)
            return opt.x, opt.fun
        except StopMinimizationEarly as e:
            return e.coefficients, e.error

    if try_directly_first:
        target = RGB_to_Lab(target_RGB)
        coefficients, error = _minimize(target, coefficients_0)

        if error < max_error:
            if dimensionalise:
                coefficients = dimensionalise_coefficients(coefficients, shape)
            return coefficients, error

    if use_feedback == "from-unsaturated":
        target_max = np.max(target_RGB) + 1e-10
        scale = create_lightness_scale(lightness_steps)

        i = lightness_steps // 3
        going_up = scale[i] < np.max(target_RGB)
        while True:
            if going_up:
                if i + 1 == lightness_steps or scale[i + 1] >= target_max:
                    break
            else:
                if i == 0 or scale[i - 1] <= target_max:
                    break

            intermediate_RGB = scale[i] * (target_RGB + 1e-10) / target_max
            intermediate = RGB_to_Lab(intermediate_RGB)

            coefficients_0, _ = _minimize(intermediate, coefficients_0)
            i += 1 if going_up else -1
    elif use_feedback == "adaptive-from-grey":
        good_RGB = np.array([0.5, 0.5, 0.5])
        good_coefficients = np.array([0., 0., 0.])

        divisions = 3
        while divisions < 10:
            keep_divisions = False
            reference_RGB = good_RGB
            reference_coefficients = good_coefficients

            coefficients_0 = reference_coefficients
            for i in range(1, divisions):
                intermediate_RGB = (target_RGB - reference_RGB) * i / (
                                    divisions - 1) + reference_RGB
                intermediate = RGB_to_Lab(intermediate_RGB)

                coefficients_0, error = _minimize(intermediate, coefficients_0)
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
        raise ValueError('Invalid value for use_feedback: "{0}"'.format(
                         use_feedback))

    target = RGB_to_Lab(target_RGB)
    coefficients, error = _minimize(target, coefficients_0)

    if dimensionalise:
        coefficients = dimensionalise_coefficients(coefficients, shape)

    return coefficients, error


def RGB_to_sd_Jakob2019(
        target_RGB,
        colourspace,
        cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        .copy().align(DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        illuminant=ILLUMINANT_SDS['D65'].copy().align(
            DEFAULT_SPECTRAL_SHAPE_JAKOB_2019),
        return_error=False,
        use_feedback="adaptive-from-grey",
        coefficients_0=(0, 0, 0),
        lightness_steps=64,
        max_error=ACCEPTABLE_DELTA_E,
        try_directly_first=False):
    """
    Recovers the spectral distribution of given RGB colourspace array
    using *Jakob and Hanika (2019)* method.

    Parameters
    ----------
    target_RGB : array_like, (3,)
        *RGB* colourspace array of the target colour. Values must be linear;
        do not apply cctf encoding.
    colourspace : RGB_Colourspace
        The *RGB* colourspace.
    cmfs : XYZ_ColourMatchingFunctions
        Standard observer colour matching functions.
    illuminant : SpectralDistribution
        Illuminant spectral distribution.
    return_error : bool, optional
        If true, ``error`` will be returned alongside ``sd``.
    use_feedback : string, optional
        If set, a less saturated version of the target color is solved for
        first. Then, colors closer and closer to the target are computed,
        feeding the result of every iteration to the next (as starting
        coefficients). This improves stability of results and greaty improves
        convergence. The possible values are:

            - 'from-unsaturated'   Start from an unsaturated colour with the
                                   same chromaticity and go up or down a
                                   lightness scale toward the target color.
            - 'adaptive-from-gray' Start from gray and move toward the target
                                   in a straight line (in the RGB colourspace)
                                   with an adaptive step size.
            - None                 Don't use feedback.

    Other parameters
    ----------------
    coefficients_0 : array_like, (3,), optional
        Starting coefficients for the solver.
    lightness_steps : int, optional
        The numbers of steps in the lightness scale used for computing
        intermediate colours when ``use_feedback`` is enabled. The default
        value of 64 is what is used in *Jakob and Hanika (2019)*.
    max_error : float, optional
        Maximal acceptable error. Set higher to save computational time.
        If ``None``, the solver will keep going until it's very close to the
        minimum. The default is ``ACCEPTABLE_DELTA_E``.
    try_directly_first: bool, optional
        If true and use_feedback is not ``None``, an attempt to solve for
        the target colour will be made, and feedback will be used only if that
        fails to produce an error below ``max_error``.

    Returns
    -------
    sd : SpectralDistribution
        Recovered spectral distribution.
    error : float
        :math:`\\Delta E_{76}` between the target colour and the colour
        corresponding to the computed coefficients.
    """

    target_RGB = as_float_array(target_RGB)

    coefficients, error = find_coefficients(
        target_RGB,
        colourspace,
        cmfs,
        illuminant,
        coefficients_0=coefficients_0,
        use_feedback=use_feedback,
        lightness_steps=lightness_steps,
        max_error=max_error)

    sd = spectral_model(
        coefficients,
        cmfs.shape,
        name='Jakob (2019) - {0} {1}'.format(colourspace.name, target_RGB))

    if return_error:
        return sd, error
    return sd


class Jakob2019Interpolator:
    """
    Class for working with pre-computed lookup tables for the
    *Jakob and Hanika (2019)* spectral upsampling method. It allows significant
    time savings by performing the expensive numerical optimization ahead of
    time and storing the results in a file.

    The file format is compatible with the code and .coeff files in
    supplemental material published alongside the article.
    """

    def __init__(self):
        pass

    def __setup_cubes(self):
        """
        Create a RegularGridInterpolator for loaded or generated coefficients.
        """

        samples = np.linspace(0, 1, self.res)
        axes = ([0, 1, 2], self.scale, samples, samples)
        self.cubes = RegularGridInterpolator(
            axes, self.coefficients, bounds_error=False)

    def from_file(self, path):
        """
        Load a lookup table from a file.

        Parameters
        ==========
        path : string
            Path to the file.
        """

        with open(path, 'rb') as fd:
            if fd.read(4).decode('ISO-8859-1') != 'SPEC':
                raise ValueError(
                    'Bad magic number, this likely is not the right file type!'
                )

            self.res = struct.unpack('i', fd.read(4))[0]
            self.scale = np.fromfile(fd, count=self.res, dtype=np.float32)
            self.coefficients = np.fromfile(
                fd, count=3 * self.res ** 3 * 3, dtype=np.float32)
            self.coefficients = self.coefficients.reshape(
                3, self.res, self.res, self.res, 3)

        self.__setup_cubes()

    def RGB_to_coefficients(self, RGB):
        """
        Look up a given *RGB* colourspace array and return corresponding
        coefficients. Interpolation is used for colours not on the table grid.

        Parameters
        ==========
        RGB : ndarray, (3,)
            *RGB* colourspace array.

        Returns
        =======
        coefficients : ndarray, (3,)
            Corresponding coefficients that can be passed to
            :func:`colour.recovery.jakob2019.spectral_model` to obtain a
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
        return self.cubes(coords).squeeze()

    def RGB_to_sd(self, RGB, shape=DEFAULT_SPECTRAL_SHAPE_JAKOB_2019):
        """
        Look up a given *RGB* colourspace array and return the corresponding
        spectral distribution.

        Parameters
        ==========
        RGB : ndarray, (3,)
            *RGB* colourspace array.

        Returns
        =======
        sd : SpectralDistribution
            Corresponding spectral distribution.
        """

        return spectral_model(
            self.RGB_to_coefficients(RGB),
            shape,
            name='Jakob (2019) - {0} (RGB)'.format(RGB))

    def generate(self,
                 colourspace,
                 cmfs,
                 illuminant,
                 resolution,
                 verbose=True):
        """
        Create a lookup table for a given *RGB* colourspace and of a given
        resolution. 

        Parameters
        ==========
        colourspace: RGB_Colourspace
            The *RGB* colourspace to create a lookup table for.
        cmfs : XYZ_ColourMatchingFunctions
            Standard observer colour matching functions.
        illuminant : SpectralDistribution
            Illuminant spectral distribution.
        resolution : int
            The resolution of the lookup table. Higher values will decrease
            errors but at the cost of a much longer run time. The published
            .coeff files have a resolution of 64.
        verbose : bool, optional
            If true (the default), information about the progress is printed
            to the standard output.
        """


        # It could be interesting to have different resolutions for lightness
        # and chromaticity, but the current file format doesn't allow it.
        lightness_steps = resolution
        chroma_steps = resolution

        self.scale = create_lightness_scale(lightness_steps)
        self.coefficients = np.empty((3, chroma_steps, chroma_steps,
                                      lightness_steps, 3))

        # First, create a list of all fully bright colours we want.
        chromas = []
        target = np.empty((3, chroma_steps, chroma_steps, 3))
        for j, x in enumerate(np.linspace(0, 1, chroma_steps)):
            for k, y in enumerate(np.linspace(0, 1, chroma_steps)):
                for i, RGB in enumerate([
                        np.array([1, y, x]),
                        np.array([x, 1, y]),
                        np.array([y, x, 1])
                ]):
                    chromas.append((i, j, k, RGB))
                    target[i, j, k, :] = RGB

        # TODO: replace this with a proper progress bar or something.
        if verbose:
            print('%6s %6s %6s  %13s %13s %13s  %s' % ('R', 'G', 'B', 'c0',
                                                       'c1', 'c2', 'Delta E'))

        # TODO: Send the list to a multiprocessing pool; this takes a while.
        for i, j, k, chroma in chromas:
            if verbose:
                print('i=%d, j=%d, k=%d, R=%g, G=%g, B=%g' %
                      (i, j, k, chroma[0], chroma[1], chroma[2]))

            def optimize(L, coefficients_0):
                """
                Solve for a specific lightness and store the result in the
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
                    use_feedback="adaptive-from-grey",
                    try_directly_first=True)

                if verbose:
                    print('%.4f %.4f %.4f  %13.6g %13.6g %13.6g  %g' %
                          (RGB[0], RGB[1], RGB[2], coefficients[0],
                           coefficients[1], coefficients[2], error))

                self.coefficients[i, L, j, k, :] = dimensionalise_coefficients(
                    coefficients, cmfs.shape)

                return coefficients

            # Start from somewhere in the middle, similarly to how feedback
            # works in find_coefficients.
            middle_L = lightness_steps // 3
            middle_coefficients = optimize(middle_L, (0, 0, 0))

            # Go down the lightness scale
            coefficients_0 = middle_coefficients
            for L in reversed(range(0, middle_L)):
                coefficients_0 = optimize(L, coefficients_0)

            # Go up the lightness scale
            coefficients_0 = middle_coefficients
            for L in range(middle_L + 1, lightness_steps):
                coefficients_0 = optimize(L, coefficients_0)

        self.res = lightness_steps
        self.__setup_cubes()

    def to_file(self, path):
        """
        Write the lookup table to a file.

        Parameters
        ==========
        path : string
            Path to the file.
        """

        with open(path, 'wb') as fd:
            fd.write(b'SPEC')
            fd.write(struct.pack('i', self.coefficients.shape[1]))
            np.float32(self.scale).tofile(fd)
            np.float32(self.coefficients).tofile(fd)
