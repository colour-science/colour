# -*- coding: utf-8 -*-
"""
Jakob and Hanika (2019) - Reflectance Recovery
==============================================

Defines objects for reflectance recovery, i.e. spectral upsampling, using
*Jakob and Hanika (2019)* method:

-   :func:`colour.recovery.sd_Jakob2019`
-   :func:`colour.recovery.find_coefficients_Jakob2019`
-   :func:`colour.recovery.XYZ_to_sd_Jakob2019`
-   :class:`colour.recovery.LUT3D_Jakob2019`

References
----------
-   :cite:`Jakob2019` : Jakob, W., & Hanika, J. (2019). A Low‐Dimensional
    Function Space for Efficient Spectral Upsampling. Computer Graphics Forum,
    38(2), 147-155. doi:10.1111/cgf.13626
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import struct
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator

from colour import SDS_ILLUMINANTS
from colour.algebra import spow, smoothstep_function
from colour.colorimetry import (
    MSDS_CMFS_STANDARD_OBSERVER, SpectralDistribution, SpectralShape,
    intermediate_lightness_function_CIE1976, sd_to_XYZ)
from colour.difference import JND_CIE1976
from colour.models import XYZ_to_xy, XYZ_to_Lab, RGB_to_XYZ
from colour.utilities import (as_float_array, domain_range_scale, full,
                              index_along_last_axis, is_tqdm_installed,
                              message_box, to_domain_1, runtime_warning, zeros)
try:
    from unittest import mock
except ImportError:  # pragma: no cover
    import mock
if is_tqdm_installed():
    from tqdm import tqdm
else:
    tqdm = mock.MagicMock()

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'SPECTRAL_SHAPE_JAKOB2019', 'StopMinimizationEarly', 'sd_Jakob2019',
    'error_function', 'dimensionalise_coefficients', 'lightness_scale',
    'find_coefficients_Jakob2019', 'XYZ_to_sd_Jakob2019', 'LUT3D_Jakob2019'
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
    -   :attr:`~colour.recovery.jakob2019.StopMinimizationEarly.coefficients`
    -   :attr:`~colour.recovery.jakob2019.StopMinimizationEarly.error`
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

    References
    ----------
    :cite:`Jakob2019`

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

    name = '{0} (COEFF) - Jakob (2019)'.format(coefficients)

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

    References
    ----------
    :cite:`Jakob2019`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> find_coefficients_Jakob2019(XYZ)  # doctest: +ELLIPSIS
    (array([  1.3723791...e-04,  -1.3514399...e-01,   3.0838973...e+01]), \
0.0141941...)
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

    xy_n = XYZ_to_xy(sd_to_XYZ(illuminant, cmfs))

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
        illuminant=SDS_ILLUMINANTS['D65'].copy().align(
            SPECTRAL_SHAPE_JAKOB2019),
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

    References
    ----------
    :cite:`Jakob2019`

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
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd_Jakob2019(XYZ, cmfs, illuminant)
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     sd  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.4884502...],
                          [ 370.        ,    0.3251871...],
                          [ 380.        ,    0.2144337...],
                          [ 390.        ,    0.1480663...],
                          [ 400.        ,    0.1085298...],
                          [ 410.        ,    0.0840835...],
                          [ 420.        ,    0.0682934...],
                          [ 430.        ,    0.0577098...],
                          [ 440.        ,    0.0504300...],
                          [ 450.        ,    0.0453634...],
                          [ 460.        ,    0.0418635...],
                          [ 470.        ,    0.0395397...],
                          [ 480.        ,    0.0381585...],
                          [ 490.        ,    0.0375912...],
                          [ 500.        ,    0.0377870...],
                          [ 510.        ,    0.0387631...],
                          [ 520.        ,    0.0406086...],
                          [ 530.        ,    0.0435015...],
                          [ 540.        ,    0.0477476...],
                          [ 550.        ,    0.0538528...],
                          [ 560.        ,    0.0626607...],
                          [ 570.        ,    0.0756177...],
                          [ 580.        ,    0.0952978...],
                          [ 590.        ,    0.1264501...],
                          [ 600.        ,    0.1779277...],
                          [ 610.        ,    0.2648782...],
                          [ 620.        ,    0.4037993...],
                          [ 630.        ,    0.5829234...],
                          [ 640.        ,    0.7442651...],
                          [ 650.        ,    0.8497961...],
                          [ 660.        ,    0.9093483...],
                          [ 670.        ,    0.9424527...],
                          [ 680.        ,    0.9615805...],
                          [ 690.        ,    0.9732085...],
                          [ 700.        ,    0.9806277...],
                          [ 710.        ,    0.9855663...],
                          [ 720.        ,    0.9889743...],
                          [ 730.        ,    0.9913993...],
                          [ 740.        ,    0.9931703...],
                          [ 750.        ,    0.9944931...],
                          [ 760.        ,    0.9955002...],
                          [ 770.        ,    0.9962802...],
                          [ 780.        ,    0.9968932...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([ 0.2065841...,  0.1220125...,  0.0514023...])
    """

    XYZ = to_domain_1(XYZ)

    if optimisation_kwargs is None:
        optimisation_kwargs = {}

    with domain_range_scale('ignore'):
        coefficients, error = find_coefficients_Jakob2019(
            XYZ, cmfs, illuminant, **optimisation_kwargs)

    sd = sd_Jakob2019(coefficients, cmfs.shape)
    sd.name = '{0} (XYZ) - Jakob (2019)'.format(XYZ)

    if additional_data:
        return sd, error
    else:
        return sd


class LUT3D_Jakob2019(object):
    """
    Class for working with pre-computed lookup tables for the
    *Jakob and Hanika (2019)* spectral upsampling method. It allows significant
    time savings by performing the expensive numerical optimization ahead of
    time and storing the results in a file.

    The file format is compatible with the code and *\\*.coeff* files in the
    supplemental material published alongside the article. They are directly
    available from
    `Colour - Datasets <https://github.com/colour-science/colour-datasets>`__
    under the record *4050598*.

    Attributes
    ----------
    -   :attr:`~colour.recovery.LUT3D_Jakob2019.size`
    -   :attr:`~colour.recovery.LUT3D_Jakob2019.lightness_scale`
    -   :attr:`~colour.recovery.LUT3D_Jakob2019.coefficients`
    -   :attr:`~colour.recovery.LUT3D_Jakob2019.interpolator`

    Methods
    -------
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.__init__`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.generate`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.RGB_to_coefficients`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.RGB_to_sd`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.read`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.write`

    References
    ----------
    :cite:`Jakob2019`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> from colour.models import RGB_COLOURSPACE_sRGB
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = (
    ...     MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer'].
    ...     copy().align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    >>> LUT = LUT3D_Jakob2019()
    >>> LUT.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
    >>> path = os.path.join(colour.__path__[0], 'recovery', 'tests',
    ...                     'resources', 'sRGB_Jakob2019.coeff')
    >>> LUT.write(path)  # doctest: +SKIP
    >>> LUT.read(path)  # doctest: +SKIP
    >>> RGB = np.array([0.70573936, 0.19248266, 0.22354169])
    >>> with numpy_print_options(suppress=True):
    ...     # Doctests skip for Python 2.x compatibility.
    ...     LUT.RGB_to_sd(RGB, cmfs.shape)  # doctest: +SKIP
    SpectralDistribution([[ 360.        ,    0.7663248...],
                          [ 370.        ,    0.6248040...],
                          [ 380.        ,    0.4582328...],
                          [ 390.        ,    0.3161403...],
                          [ 400.        ,    0.2196885...],
                          [ 410.        ,    0.1597642...],
                          [ 420.        ,    0.1226653...],
                          [ 430.        ,    0.0990878...],
                          [ 440.        ,    0.0836822...],
                          [ 450.        ,    0.0734525...],
                          [ 460.        ,    0.0667002...],
                          [ 470.        ,    0.0624502...],
                          [ 480.        ,    0.0601529...],
                          [ 490.        ,    0.0595328...],
                          [ 500.        ,    0.0605182...],
                          [ 510.        ,    0.0632235...],
                          [ 520.        ,    0.0679778...],
                          [ 530.        ,    0.0754093...],
                          [ 540.        ,    0.0866232...],
                          [ 550.        ,    0.1035471...],
                          [ 560.        ,    0.1295933...],
                          [ 570.        ,    0.1708525...],
                          [ 580.        ,    0.2377171...],
                          [ 590.        ,    0.3442627...],
                          [ 600.        ,    0.4952907...],
                          [ 610.        ,    0.6605014...],
                          [ 620.        ,    0.7914286...],
                          [ 630.        ,    0.8738002...],
                          [ 640.        ,    0.9212534...],
                          [ 650.        ,    0.9486329...],
                          [ 660.        ,    0.9650124...],
                          [ 670.        ,    0.9752510...],
                          [ 680.        ,    0.9819246...],
                          [ 690.        ,    0.9864387...],
                          [ 700.        ,    0.9895916...],
                          [ 710.        ,    0.9918554...],
                          [ 720.        ,    0.9935199...],
                          [ 730.        ,    0.9947694...],
                          [ 740.        ,    0.9957242...],
                          [ 750.        ,    0.9964656...],
                          [ 760.        ,    0.9970494...],
                          [ 770.        ,    0.9975148...],
                          [ 780.        ,    0.9978900...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    """

    def __init__(self):
        self._interpolator = None
        self._size = None
        self._lightness_scale = None
        self._coefficients = None

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

    @property
    def interpolator(self):
        """
        Getter property for the *Jakob and Hanika (2019)* interpolator.

        Returns
        -------
        RegularGridInterpolator
            *Jakob and Hanika (2019)* interpolator.
        """

        return self._interpolator

    def _create_interpolator(self):
        """
        Creates a :class:`scipy.interpolate.RegularGridInterpolator` class
        instance for read or generated coefficients.
        """

        samples = np.linspace(0, 1, self._size)
        axes = ([0, 1, 2], self._lightness_scale, samples, samples)

        self._interpolator = RegularGridInterpolator(
            axes, self._coefficients, bounds_error=False)

    def generate(self,
                 colourspace,
                 cmfs=MSDS_CMFS_STANDARD_OBSERVER[
                     'CIE 1931 2 Degree Standard Observer']
                 .copy().align(SPECTRAL_SHAPE_JAKOB2019),
                 illuminant=SDS_ILLUMINANTS['D65'].copy().align(
                     SPECTRAL_SHAPE_JAKOB2019),
                 size=64,
                 print_callable=print):
        """
        Generates the lookup table data for given *RGB* colourspace, colour
        matching functions, illuminant and given size.

        Parameters
        ----------
        colourspace: RGB_Colourspace
            The *RGB* colourspace to create a lookup table for.
        cmfs : XYZ_ColourMatchingFunctions, optional
            Standard observer colour matching functions.
        illuminant : SpectralDistribution, optional
            Illuminant spectral distribution.
        size : int, optional
            The resolution of the lookup table. Higher values will decrease
            errors but at the cost of a much longer run time. The published
            *\\*.coeff* files have a resolution of 64.
        print_callable : callable, optional
            Callable used to print progress and diagnostic information.

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> cmfs = MSDS_CMFS_STANDARD_OBSERVER[
        ...         'CIE 1931 2 Degree Standard Observer'].copy().align(
        ...             SpectralShape(360, 780, 10))
        >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> print(LUT.interpolator)
        None
        >>> LUT.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3)
        ======================================================================\
=========
        *                                                                     \
        *
        *   "Jakob et al. (2018)" LUT Optimisation                            \
        *
        *                                                                     \
        *
        ======================================================================\
=========
        <BLANKLINE>
        Optimising 27 coefficients...
        <BLANKLINE>
        >>> print(LUT.interpolator)
        ... # doctest: +ELLIPSIS
        <scipy.interpolate.interpolate.RegularGridInterpolator object at 0x...>
        """

        shape = cmfs.shape

        if illuminant.shape != shape:
            runtime_warning(
                'Aligning "{0}" illuminant shape to "{1}" colour matching '
                'functions shape.'.format(illuminant.name, cmfs.name))
            illuminant = illuminant.copy().align(cmfs.shape)

        xy_n = XYZ_to_xy(sd_to_XYZ(illuminant, cmfs))

        # It could be interesting to have different resolutions for lightness
        # and chromaticity, but the current file format doesn't allow it.
        lightness_steps = size
        chroma_steps = size

        self._lightness_scale = lightness_scale(lightness_steps)
        self._coefficients = np.empty(
            [3, chroma_steps, chroma_steps, lightness_steps, 3])

        cube_indexes = np.ndindex(3, chroma_steps, chroma_steps)
        total_coefficients = chroma_steps ** 2 * 3

        # First, create a list of all the fully bright colours with the order
        # matching cube_indexes.
        samples = np.linspace(0, 1, chroma_steps)
        ij = np.meshgrid(*[[1], samples, samples], indexing='ij')
        ij = np.transpose(ij).reshape(-1, 3)
        chromas = np.concatenate(
            [ij, np.roll(ij, 1, axis=1),
             np.roll(ij, 2, axis=1)])

        message_box(
            '"Jakob et al. (2018)" LUT Optimisation',
            print_callable=print_callable)

        print_callable(
            '\nOptimising {0} coefficients...\n'.format(total_coefficients))

        def optimize(ijkL, coefficients_0):
            """
            Solves for a specific lightness and stores the result in the
            appropriate cell.
            """

            i, j, k, L = ijkL

            RGB = self._lightness_scale[L] * chroma

            XYZ = RGB_to_XYZ(RGB, colourspace.whitepoint, xy_n,
                             colourspace.matrix_RGB_to_XYZ)

            coefficients, _error = find_coefficients_Jakob2019(
                XYZ, cmfs, illuminant, coefficients_0, dimensionalise=False)

            self._coefficients[i, L, j, k, :] = dimensionalise_coefficients(
                coefficients, shape)

            return coefficients

        with tqdm(total=total_coefficients) as progress:
            for ijk, chroma in zip(cube_indexes, chromas):
                progress.update()

                # Starts from somewhere in the middle, similarly to how
                # feedback works in "colour.recovery.\
                # find_coefficients_Jakob2019" definition.
                L_middle = lightness_steps // 3
                coefficients_middle = optimize(
                    np.hstack([ijk, L_middle]), zeros(3))

                # Goes down the lightness scale.
                coefficients_0 = coefficients_middle
                for L in reversed(range(0, L_middle)):
                    coefficients_0 = optimize(
                        np.hstack([ijk, L]), coefficients_0)

                # Goes up the lightness scale.
                coefficients_0 = coefficients_middle
                for L in range(L_middle + 1, lightness_steps):
                    coefficients_0 = optimize(
                        np.hstack([ijk, L]), coefficients_0)

        self._size = size
        self._create_interpolator()

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
            :func:`colour.recovery.jakob2019.sd_Jakob2019` to obtain a spectral
            distribution.

        Examples
        --------
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> cmfs = MSDS_CMFS_STANDARD_OBSERVER[
        ...         'CIE 1931 2 Degree Standard Observer'].copy().align(
        ...             SpectralShape(360, 780, 10))
        >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> LUT.generate(
        ...     RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
        >>> RGB = np.array([0.70573936, 0.19248266, 0.22354169])
        >>> LUT.RGB_to_coefficients(RGB)  # doctest: +ELLIPSIS
        array([  1.5012557...e-04,  -1.4678661...e-01,   3.4017293...e+01])
        """

        RGB = as_float_array(RGB)

        value_max = np.max(RGB, axis=-1)
        chroma = RGB / (np.expand_dims(value_max, -1) + 1e-10)

        i_m = np.argmax(RGB, axis=-1)
        i_1 = index_along_last_axis(RGB, i_m)
        i_2 = index_along_last_axis(chroma, (i_m + 2) % 3)
        i_3 = index_along_last_axis(chroma, (i_m + 1) % 3)

        indexes = np.stack([i_m, i_1, i_2, i_3], axis=-1)

        return self._interpolator(indexes).squeeze()

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

        Examples
        --------
        >>> from colour.utilities import numpy_print_options
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> cmfs = MSDS_CMFS_STANDARD_OBSERVER[
        ...         'CIE 1931 2 Degree Standard Observer'].copy().align(
        ...             SpectralShape(360, 780, 10))
        >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> LUT.generate(
        ...     RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
        >>> RGB = np.array([0.70573936, 0.19248266, 0.22354169])
        >>> with numpy_print_options(suppress=True):
        ...     # Doctests skip for Python 2.x compatibility.
        ...     LUT.RGB_to_sd(RGB, cmfs.shape)  # doctest: +SKIP
        SpectralDistribution([[ 360.        ,    0.7663250...],
                              [ 370.        ,    0.6248043...],
                              [ 380.        ,    0.4582331...],
                              [ 390.        ,    0.3161405...],
                              [ 400.        ,    0.2196887...],
                              [ 410.        ,    0.1597643...],
                              [ 420.        ,    0.1226654...],
                              [ 430.        ,    0.0990879...],
                              [ 440.        ,    0.0836822...],
                              [ 450.        ,    0.0734526...],
                              [ 460.        ,    0.0667003...],
                              [ 470.        ,    0.0624502...],
                              [ 480.        ,    0.060153 ...],
                              [ 490.        ,    0.0595329...],
                              [ 500.        ,    0.0605182...],
                              [ 510.        ,    0.0632236...],
                              [ 520.        ,    0.0679778...],
                              [ 530.        ,    0.0754094...],
                              [ 540.        ,    0.0866233...],
                              [ 550.        ,    0.1035472...],
                              [ 560.        ,    0.1295934...],
                              [ 570.        ,    0.1708527...],
                              [ 580.        ,    0.2377174...],
                              [ 590.        ,    0.3442632...],
                              [ 600.        ,    0.4952913...],
                              [ 610.        ,    0.6605019...],
                              [ 620.        ,    0.7914290...],
                              [ 630.        ,    0.8738004...],
                              [ 640.        ,    0.9212535...],
                              [ 650.        ,    0.9486330...],
                              [ 660.        ,    0.9650125...],
                              [ 670.        ,    0.9752511...],
                              [ 680.        ,    0.9819246...],
                              [ 690.        ,    0.9864387...],
                              [ 700.        ,    0.9895916...],
                              [ 710.        ,    0.9918554...],
                              [ 720.        ,    0.9935199...],
                              [ 730.        ,    0.9947694...],
                              [ 740.        ,    0.9957242...],
                              [ 750.        ,    0.9964656...],
                              [ 760.        ,    0.9970494...],
                              [ 770.        ,    0.9975148...],
                              [ 780.        ,    0.9978900...]],
                             interpolator=SpragueInterpolator,
                             interpolator_kwargs={},
                             extrapolator=Extrapolator,
                             extrapolator_kwargs={...})
        """

        sd = sd_Jakob2019(self.RGB_to_coefficients(RGB), shape)
        sd.name = '{0} (RGB) - Jakob (2019)'.format(RGB)

        return sd

    def read(self, path):
        """
        Loads a lookup table from a *\\*.coeff* file.

        Parameters
        ----------
        path : unicode
            Path to the file.

        Examples
        --------
        >>> import os
        >>> import colour
        >>> from colour.utilities import numpy_print_options
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> cmfs = MSDS_CMFS_STANDARD_OBSERVER[
        ...         'CIE 1931 2 Degree Standard Observer'].copy().align(
        ...             SpectralShape(360, 780, 10))
        >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> LUT.generate(
        ...     RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
        >>> path = os.path.join(colour.__path__[0], 'recovery', 'tests',
        ...                     'resources', 'sRGB_Jakob2019.coeff')
        >>> LUT.write(path)  # doctest: +SKIP
        >>> LUT.read(path)  # doctest: +SKIP
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

        self._create_interpolator()

    def write(self, path):
        """
        Writes the lookup table to a *\\*.coeff* file.

        Parameters
        ----------
        path : unicode
            Path to the file.

        Examples
        --------
        >>> import os
        >>> import colour
        >>> from colour.utilities import numpy_print_options
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> cmfs = MSDS_CMFS_STANDARD_OBSERVER[
        ...         'CIE 1931 2 Degree Standard Observer'].copy().align(
        ...             SpectralShape(360, 780, 10))
        >>> illuminant = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> LUT.generate(
        ...     RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
        >>> path = os.path.join(colour.__path__[0], 'recovery', 'tests',
        ...                     'resources', 'sRGB_Jakob2019.coeff')
        >>> LUT.write(path)  # doctest: +SKIP
        >>> LUT.read(path)  # doctest: +SKIP
        """

        with open(path, 'wb') as coeff_file:
            coeff_file.write(b'SPEC')
            coeff_file.write(struct.pack('i', self._coefficients.shape[1]))
            np.float32(self._lightness_scale).tofile(coeff_file)
            np.float32(self._coefficients).tofile(coeff_file)
