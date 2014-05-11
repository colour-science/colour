#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**transformations.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package color *transformations* objects.

**Others:**

"""

from __future__ import unicode_literals

import bisect
import math
import numpy

import color.chromatic_adaptation
import color.illuminants
import color.exceptions
import color.lightness
import color.matrix
import color.spectral
import color.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "wavelength_to_XYZ",
           "spectral_to_XYZ",
           "XYZ_to_xyY",
           "xyY_to_XYZ",
           "xy_to_XYZ",
           "XYZ_to_xy",
           "XYZ_to_RGB",
           "RGB_to_XYZ",
           "xyY_to_RGB",
           "RGB_to_xyY",
           "XYZ_to_UVW",
           "UVW_to_XYZ",
           "UVW_to_uv",
           "UVW_uv_to_xy",
           "XYZ_to_Luv",
           "Luv_to_XYZ",
           "Luv_to_uv",
           "Luv_uv_to_xy",
           "Luv_to_LCHuv",
           "LCHuv_to_Luv",
           "XYZ_to_Lab",
           "Lab_to_XYZ",
           "Lab_to_LCHab",
           "LCHab_to_Lab"]

LOGGER = color.verbose.install_logger()

def wavelength_to_XYZ(wavelength, cmfs):
    """
    Converts given wavelength to *CIE XYZ* colorspace using given color matching functions, if the retrieved
    wavelength is not available in the color matching function, its value will be calculated using linear interpolation
    between the two closest wavelengths.

    Usage::

        >>> wavelength_to_XYZ(480, color.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer"))
        matrix([[ 0.09564],
            [ 0.13902],
            [ 0.81295]])

    :param wavelength: Wavelength in nm.
    :type wavelength: float
    :param cmfs: Standard observer color matching functions.
    :type cmfs: dict
    :return: *CIE XYZ* matrix.
    :rtype: Matrix
    """

    start, end, steps = cmfs.shape
    if wavelength < start or wavelength > end:
        raise color.exceptions.ProgrammingError(
            "'{0}' nm wavelength not in '{1} - {2}' nm supported wavelengths range!".format(wavelength, start, end))

    wavelengths = numpy.arange(start, end, steps)
    index = bisect.bisect(wavelengths, wavelength)
    if index < len(wavelengths):
        left = wavelengths[index - 1]
        right = wavelengths[index]
    else:
        left = right = wavelengths[-1]

    leftXYZ = numpy.matrix(cmfs.get(left)).reshape((3, 1))
    rightXYZ = numpy.matrix(cmfs.get(right)).reshape((3, 1))

    return color.matrix.linear_interpolate_matrices(left, right, leftXYZ, rightXYZ, wavelength)

def spectral_to_XYZ(spd,
                    cmfs,
                    illuminant=None):
    """
    Converts given relative spectral power distribution to *CIE XYZ* colorspace using given color
    matching functions and illuminant.

    Reference: http://brucelindbloom.com/Eqn_Spect_to_XYZ.html

    Usage::

        >>> cmfs = color.STANDARD_OBSERVERS_COLOR_MATCHING_FUNCTIONS.get("Standard CIE 1931 2 Degree Observer")
        >>> spd = color.SpectralPowerDistribution("Custom", {380: 0.0600, 390: 0.0600}).resparse(*cmfs.shape)
        >>> illuminant = color.ILLUMINANTS_RELATIVE_SPD.get("D50").resparse(*cmfs.shape)
        >>> spectral_to_XYZ(spd, cmfs, illuminant)
        matrix([[  4.57648522e-06],
            [  1.29648668e-07],
            [  2.16158075e-05]])

    :param spd: Spectral power distribution.
    :type spd: SpectralPowerDistribution
    :param cmfs: Standard observer color matching functions.
    :type cmfs: XYZ_ColorMatchingFunctions
    :param illuminant: *Illuminant* spectral power distribution.
    :type illuminant: SpectralPowerDistribution
    :return: *CIE XYZ* matrix.
    :rtype: Matrix

    :note: Spectral power distribution, standard observer color matching functions and illuminant shapes must be aligned.
    """

    if spd.shape != cmfs.shape:
        raise color.exceptions.ProgrammingError(
            "Spectral power distribution and standard observer color matching functions shapes are not aligned: '{0}', '{1}'.".format(
                spd.shape, cmfs.shape))

    if illuminant is None:
        start, end, steps = cmfs.shape
        range = numpy.arange(start, end + steps, steps)
        illuminant = color.spectral.SpectralPowerDistribution(name="1.0",
                                                              spd=dict(zip(*(list(range),
                                                                             [1.] * len(range)))))
    else:
        if illuminant.shape != cmfs.shape:
            raise color.exceptions.ProgrammingError(
                "Illuminant and standard observer color matching functions shapes are not aligned: '{0}', '{1}'.".format(
                    illuminant.shape, cmfs.shape))

    illuminant = illuminant.values
    spd = spd.values

    x_bar, y_bar, z_bar = zip(*cmfs.values)

    denominator = y_bar * illuminant
    spd = spd * illuminant
    xNumerator = spd * x_bar
    yNumerator = spd * y_bar
    zNumerator = spd * z_bar

    XYZ = numpy.matrix([xNumerator.sum() / denominator.sum(),
                        yNumerator.sum() / denominator.sum(),
                        zNumerator.sum() / denominator.sum()])

    return XYZ.reshape((3, 1))

def XYZ_to_xyY(XYZ, illuminant=color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")):
    """
    Converts from *CIE XYZ* colorspace to *CIE xyY* colorspace using given matrix and reference *illuminant*.

    Reference: http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html

    Usage::

        >>> XYZ_to_xyY(numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1)))
        matrix([[  0.4325],
            [  0.3788],
            [ 10.34  ]])

    :param XYZ: *CIE XYZ* matrix.
    :type XYZ: Matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *CIE xyY* matrix.
    :rtype: Matrix (3x1)
    """

    X, Y, Z = numpy.ravel(XYZ)

    if X == 0 and Y == 0 and Z == 0:
        return numpy.matrix([illuminant[0], illuminant[1], Y]).reshape((3, 1))
    else:
        return numpy.matrix([X / (X + Y + Z), Y / (X + Y + Z), Y]).reshape((3, 1))

def xyY_to_XYZ(xyY):
    """
    Converts from *CIE xyY* colorspace to *CIE XYZ* colorspace using given matrix.

    Reference: http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html

    Usage::

        >>> xyY_to_XYZ(numpy.matrix([0.4325, 0.3788, 10.34]).reshape((3, 1)))
        matrix([[ 11.80583421],
            [ 10.34      ],
            [  5.15089229]])

    :param xyY: *CIE xyY* matrix.
    :type xyY: Matrix (3x1)
    :return: *CIE XYZ* matrix.
    :rtype: Matrix (3x1)
    """

    x, y, Y = numpy.ravel(xyY)

    if y == 0:
        return numpy.matrix([0., 0., 0.]).reshape((3, 1))
    else:
        return numpy.matrix([x * Y / y, Y, (1. - x - y) * Y / y]).reshape((3, 1))

def xy_to_XYZ(xy):
    """
    Returns the *CIE XYZ* matrix from given *xy* chromaticity coordinates.

    Usage::

        >>> xy_to_XYZ((0.25, 0.25))
        matrix([[ 1.],
            [ 1.],
            [ 2.]])

    :param xy: *xy* chromaticity coordinate.
    :type xy: tuple
    :return: *CIE XYZ* matrix.
    :rtype: Matrix (3x1)
    """

    return xyY_to_XYZ(numpy.matrix([xy[0], xy[1], 1.]).reshape((3, 1)))

def XYZ_to_xy(XYZ, illuminant=color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")):
    """
    Returns the *xy* chromaticity coordinates from given *CIE XYZ* matrix.

    Usage::

        >>> XYZ_to_xy(numpy.matrix([0.97137399, 1., 1.04462134]).reshape((3, 1)))
        (0.32207410281368043, 0.33156550013623531)
        >>> XYZ_to_xy((0.97137399, 1., 1.04462134))
        (0.32207410281368043, 0.33156550013623531)

    :param XYZ: *CIE XYZ* matrix.
    :type XYZ: Matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *xy* chromaticity coordinates.
    :rtype: tuple
    """

    xyY = numpy.ravel(XYZ_to_xyY(XYZ, illuminant))
    return xyY[0], xyY[1]

def XYZ_to_RGB(XYZ,
               illuminant_XYZ,
               illuminant_RGB,
               chromatic_adaptation_method,
               from_XYZ,
               transfer_function=None):
    """
    Converts from *CIE XYZ* colorspace to *RGB* colorspace using given *CIE XYZ* matrix, *illuminants*,
    *chromatic adaptation* method, *normalized primary matrix* and *transfer function*.

    Usage::

        >>> XYZ = numpy.matrix([11.51847498, 10.08, 5.08937252]).reshape((3, 1))
        >>> illuminant_XYZ =  (0.34567, 0.35850)
        >>> illuminant_RGB =  (0.31271, 0.32902)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> from_XYZ =  numpy.matrix([3.24100326, -1.53739899, -0.49861587, -0.96922426,  1.87592999,  0.04155422, 0.05563942, -0.2040112 ,  1.05714897]).reshape((3, 3))
        >>> XYZ_to_RGB(XYZ, illuminant_XYZ, illuminant_RGB, chromatic_adaptation_method, from_XYZ)
        matrix([[ 17.303501],
            [ 8.8211033],
            [ 5.5672498]])

    :param XYZ: *CIE XYZ* colorspace matrix.
    :type XYZ: Matrix (3x1)
    :param illuminant_XYZ: *CIE XYZ* colorspace *illuminant* chromaticity coordinates.
    :type illuminant_XYZ: tuple
    :param illuminant_RGB: *RGB* colorspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode
    :param from_XYZ: *Normalized primary matrix*.
    :type from_XYZ: Matrix (3x3)
    :param transfer_function: *Transfer function*.
    :type transfer_function: object
    :return: *RGB* colorspace matrix.
    :rtype: Matrix (3x1)
    """

    cat = color.chromatic_adaptation.get_chromatic_adaptation_matrix(
        xy_to_XYZ(illuminant_XYZ),
        xy_to_XYZ(illuminant_RGB),
        method=chromatic_adaptation_method)

    adaptedXYZ = cat * XYZ

    RGB = from_XYZ * adaptedXYZ

    if transfer_function is not None:
        RGB = transfer_function(RGB)

    LOGGER.debug("'Chromatic adaptation' matrix:\n{0}".format(repr(cat)))
    LOGGER.debug("Adapted 'CIE XYZ' matrix:\n{0}".format(repr(adaptedXYZ)))
    LOGGER.debug("'RGB' matrix:\n{0}".format(repr(RGB)))

    return RGB

def RGB_to_XYZ(RGB,
               illuminant_RGB,
               illuminant_XYZ,
               chromatic_adaptation_method,
               to_XYZ,
               inverse_transfer_function=None):
    """
    Converts from *RGB* colorspace to *CIE XYZ* colorspace using given *RGB* matrix, *illuminants*,
    *chromatic adaptation* method, *normalized primary matrix* and *transfer function*.

    Usage::

        >>> RGB = numpy.matrix([17.303501, 8.211033, 5.672498]).reshape((3, 1))
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> illuminant_XYZ = (0.34567, 0.35850)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> to_XYZ = numpy.matrix([0.41238656, 0.35759149, 0.18045049, 0.21263682, 0.71518298, 0.0721802, 0.01933062, 0.11919716, 0.95037259]).reshape((3, 3)))
        >>> RGB_to_XYZ(RGB, illuminant_RGB, illuminant_XYZ, chromatic_adaptation_method, to_XYZ)
        matrix([[ 11.51847498],
            [ 10.0799999 ],
            [  5.08937278]])

    :param RGB: *RGB* colorspace matrix.
    :type RGB: Matrix (3x1)
    :param illuminant_RGB: *RGB* colorspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: tuple
    :param illuminant_XYZ: *CIE XYZ* colorspace *illuminant* chromaticity coordinates.
    :type illuminant_XYZ: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode
    :param to_XYZ: *Normalized primary matrix*.
    :type to_XYZ: Matrix (3x3)
    :param inverse_transfer_function: *Inverse transfer function*.
    :type inverse_transfer_function: object
    :return: *CIE XYZ* colorspace matrix.
    :rtype: Matrix (3x1)
    """

    if inverse_transfer_function is not None:
        RGB = inverse_transfer_function(RGB)

    XYZ = to_XYZ * RGB

    cat = color.chromatic_adaptation.get_chromatic_adaptation_matrix(
        xy_to_XYZ(illuminant_RGB),
        xy_to_XYZ(illuminant_XYZ),
        method=chromatic_adaptation_method)

    adaptedXYZ = cat * XYZ

    LOGGER.debug("'CIE XYZ' matrix:\n{0}".format(repr(XYZ)))
    LOGGER.debug("'Chromatic adaptation' matrix:\n{0}".format(repr(cat)))
    LOGGER.debug("Adapted 'CIE XYZ' matrix:\n{0}".format(repr(adaptedXYZ)))

    return adaptedXYZ

def xyY_to_RGB(xyY,
               illuminant_xyY,
               illuminant_RGB,
               chromatic_adaptation_method,
               from_XYZ,
               transfer_function=None):
    """
    Converts from *CIE xyY* colorspace to *RGB* colorspace using given *CIE xyY* matrix, *illuminants*,
    *chromatic adaptation* method, *normalized primary matrix* and *transfer function*.

    Usage::

        >>> xyY = numpy.matrix([0.4316, 0.3777, 10.08]).reshape((3, 1))
        >>> illuminant_xyY = (0.34567, 0.35850)
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> chromatic_adaptation_method =  "Bradford"
        >>> from_XYZ = numpy.matrix([ 3.24100326, -1.53739899, -0.49861587, -0.96922426,  1.87592999,  0.04155422, 0.05563942, -0.2040112 ,  1.05714897]).reshape((3, 3)))
        >>> xyY_to_RGB(xyY, illuminant_xyY, illuminant_RGB, chromatic_adaptation_method, from_XYZ)
        matrix([[ 17.30350095],
            [  8.21103314],
            [  5.67249761]])

    :param xyY: *CIE xyY* matrix.
    :type xyY: Matrix (3x1)
    :param illuminant_xyY: *CIE xyY* colorspace *illuminant* chromaticity coordinates.
    :type illuminant_xyY: tuple
    :param illuminant_RGB: *RGB* colorspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode
    :param from_XYZ: *Normalized primary matrix*.
    :type from_XYZ: Matrix (3x3)
    :param transfer_function: *Transfer function*.
    :type transfer_function: object
    :return: *RGB* colorspace matrix.
    :rtype: Matrix (3x1)
    """

    return XYZ_to_RGB(xyY_to_XYZ(xyY),
                      illuminant_xyY,
                      illuminant_RGB,
                      chromatic_adaptation_method,
                      from_XYZ,
                      transfer_function)

def RGB_to_xyY(RGB,
               illuminant_RGB,
               illuminant_xyY,
               chromatic_adaptation_method,
               to_XYZ,
               inverse_transfer_function=None):
    """
    Converts from *RGB* colorspace to *CIE xyY* colorspace using given *RGB* matrix, *illuminants*,
    *chromatic adaptation* method, *normalized primary matrix* and *transfer function*.

    Usage::

        >>> RGB = numpy.matrix([17.303501, 8.211033, 5.672498]).reshape((3, 1))
        >>> illuminant_RGB = (0.31271, 0.32902)
        >>> illuminant_xyY = (0.34567, 0.35850)
        >>> chromatic_adaptation_method = "Bradford"
        >>> to_XYZ = numpy.matrix([0.41238656, 0.35759149, 0.18045049, 0.21263682, 0.71518298, 0.0721802, 0.01933062, 0.11919716, 0.95037259]).reshape((3, 3)))
        >>> RGB_to_xyY(RGB, illuminant_RGB, illuminant_xyY, chromatic_adaptation_method, to_XYZ)
        matrix([[  0.4316    ],
            [  0.37769999],
            [ 10.0799999 ]])

    :param RGB: *RGB* colorspace matrix.
    :type RGB: Matrix (3x1)
    :param illuminant_RGB: *RGB* colorspace *illuminant* chromaticity coordinates.
    :type illuminant_RGB: tuple
    :param illuminant_xyY: *CIE xyY* colorspace *illuminant* chromaticity coordinates.
    :type illuminant_xyY: tuple
    :param chromatic_adaptation_method: *Chromatic adaptation* method.
    :type chromatic_adaptation_method: unicode
    :param to_XYZ: *Normalized primary* matrix.
    :type to_XYZ: Matrix (3x3)
    :param inverse_transfer_function: *Inverse transfer* function.
    :type inverse_transfer_function: object
    :return: *CIE XYZ* matrix.
    :rtype: Matrix (3x1)
    """

    return XYZ_to_xyY(RGB_to_XYZ(RGB,
                                 illuminant_RGB,
                                 illuminant_xyY,
                                 chromatic_adaptation_method,
                                 to_XYZ,
                                 inverse_transfer_function))

def XYZ_to_UVW(XYZ):
    """
    Converts from *CIE XYZ* colorspace to *CIE UVW* colorspace using given matrix.

    Reference: http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ

    Usage::

        >>> XYZ_to_UVW(numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1)))
        matrix([[  7.87055614]
            [ 10.34      ]
            [ 12.18252904]])

    :param XYZ: *CIE XYZ* matrix.
    :type XYZ: Matrix (3x1)
    :return: *CIE UVW* matrix.
    :rtype: Matrix (3x1)
    """

    X, Y, Z = numpy.ravel(XYZ)

    return numpy.matrix([2. / 3. * X, Y, 1. / 2. * (-X + 3. * Y + Z)]).reshape((3, 1))

def UVW_to_XYZ(UVW):
    """
    Converts from *CIE UVW* colorspace to *CIE XYZ* colorspace using given matrix.

    Reference: http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ

    Usage::

        >>> UVW_to_XYZ(numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1)))
        matrix([[  7.87055614]
            [ 10.34      ]
            [ 12.18252904]])

    :param UVW: *CIE UVW* matrix.
    :type UVW: Matrix (3x1)
    :return: *CIE XYZ* matrix.
    :rtype: Matrix (3x1)
    """

    U, V, W = numpy.ravel(UVW)

    return numpy.matrix([3. / 2. * U, V, 3. / 2. * U - (3. * V) + (2. * W)]).reshape((3, 1))

def UVW_to_uv(UVW):
    """
    Returns the *uv* chromaticity coordinates from given *CIE UVW* matrix.

    Reference: http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ

    Usage::

        >>> UVW_to_uv(numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1)))
        (0.43249999995420702, 0.378800000065942)

    :param UVW: *CIE UVW* matrix.
    :type UVW: Matrix (3x1)
    :return: *uv* chromaticity coordinates.
    :rtype: tuple
    """

    U, V, W = numpy.ravel(UVW)

    return U / (U + V + W), V / (U + V + W)

def UVW_uv_to_xy(uv):
    """
    Returns the *xy* chromaticity coordinates from given *CIE UVW* colorspace *uv* chromaticity coordinates.

    Reference: http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ

    Usage::

        >>> UVW_uv_to_xy((0.2033733344733139, 0.3140500001549052))
        (0.32207410281368043, 0.33156550013623537)

    :param uv: *CIE UVW uv* chromaticity coordinate.
    :type uv: tuple
    :return: *xy* chromaticity coordinates.
    :rtype: tuple
    """

    return 3. * uv[0] / (2. * uv[0] - 8. * uv[1] + 4.), 2. * uv[1] / (2. * uv[0] - 8. * uv[1] + 4.)

def XYZ_to_Luv(XYZ, illuminant=color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")):
    """
    Converts from *CIE XYZ* colorspace to *CIE Luv* colorspace using given matrix.

    Reference: http://brucelindbloom.com/Eqn_XYZ_to_Luv.html

    Usage::

        >>> XYZ_to_Luv(numpy.matrix([0.92193107, 1., 1.03744246]).reshape((3, 1)))
        matrix([[ 100.        ]
            [ -20.04304247]
            [ -45.09684555]])

    :param XYZ: *CIE XYZ* matrix.
    :type XYZ: Matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *CIE Luv* matrix.
    :rtype: Matrix (3x1)
    """

    X, Y, Z = numpy.ravel(XYZ)
    Xr, Yr, Zr = numpy.ravel(xy_to_XYZ(illuminant))

    yr = Y / Yr

    L = 116. * yr ** (1. / 3.) - 16. if yr > color.lightness.CIE_E else color.lightness.CIE_K * yr
    u = 13. * L * ((4. * X / (X + 15. * Y + 3. * Z)) - (4. * Xr / (Xr + 15. * Yr + 3. * Zr)))
    v = 13. * L * ((9. * Y / (X + 15. * Y + 3. * Z)) - (9. * Yr / (Xr + 15. * Yr + 3. * Zr)))

    return numpy.matrix([L, u, v]).reshape((3, 1))

def Luv_to_XYZ(Luv, illuminant=color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")):
    """
    Converts from *CIE Luv* colorspace to *CIE XYZ* colorspace using given matrix.

    Reference: http://brucelindbloom.com/Eqn_Luv_to_XYZ.html

    Usage::

        >>> Luv_to_XYZ(numpy.matrix([100., -20.04304247, -19.81676035]).reshape((3, 1)))
        matrix([[ 0.92193107]
            [ 1.        ]
            [ 1.03744246]])

    :param Luv: *CIE Luv* matrix.
    :type Luv: Matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *CIE XYZ* matrix.
    :rtype: Matrix (3x1)
    """

    L, u, v = numpy.ravel(Luv)
    Xr, Yr, Zr = numpy.ravel(xy_to_XYZ(illuminant))

    Y = ((L + 16.) / 116.) ** 3. if L > color.lightness.CIE_E * color.lightness.CIE_K else L / color.lightness.CIE_K

    a = 1. / 3. * ((52. * L / (u + 13. * L * (4. * Xr / (Xr + 15. * Yr + 3. * Zr)))) - 1.)
    b = -5. * Y
    c = -1. / 3.0
    d = Y * (39. * L / (v + 13. * L * (9. * Yr / (Xr + 15. * Yr + 3. * Zr))) - 5.)

    X = (d - b) / (a - c)
    Z = X * a + b

    return numpy.matrix([X, Y, Z]).reshape((3, 1))

def Luv_to_uv(Luv, illuminant=color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")):
    """
    Returns the *u'v'* chromaticity coordinates from given *CIE Luv* matrix.

    Reference: http://en.wikipedia.org/wiki/CIELUV#The_forward_transformation

    Usage::

        >>> Luv_to_uv(numpy.matrix([100., -20.04304247, -19.81676035]).reshape((3, 1)))
        (0.19374142100850045, 0.47283165896209456)

    :param Luv: *CIE Luv* matrix.
    :type Luv: Matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *u'v'* chromaticity coordinates.
    :rtype: tuple
    """

    X, Y, Z = numpy.ravel(Luv_to_XYZ(Luv, illuminant))

    return 4. * X / (X + 15. * Y + 3. * Z), 9. * Y / (X + 15. * Y + 3. * Z)

def Luv_uv_to_xy(uv):
    """
    Returns the *xy* chromaticity coordinates from given *CIE Luv* colorspace *u'v'* chromaticity coordinates.

    Reference: http://en.wikipedia.org/wiki/CIELUV#The_reverse_transformation'.

    Usage::

        >>> Luv_uv_to_xy((0.2033733344733139, 0.3140500001549052))
        (0.32207410281368043, 0.33156550013623537)

    :param uv: *CIE Luv u'v'* chromaticity coordinate.
    :type uv: tuple
    :return: *xy* chromaticity coordinates.
    :rtype: tuple
    """

    return 9. * uv[0] / (6. * uv[0] - 16. * uv[1] + 12.), 4. * uv[1] / (6. * uv[0] - 16. * uv[1] + 12.)

def Luv_to_LCHuv(Luv):
    """
    Converts from *CIE Luv* colorspace to *CIE LCHuv* colorspace using given matrix.

    Reference: http://www.brucelindbloom.com/Eqn_Luv_to_LCH.html

    Usage::

        >>> Luv_to_LCHuv(numpy.matrix([100., -20.04304247, -19.81676035]).reshape((3, 1)))
        matrix([[ 100.        ]
            [  28.18559104]
            [ 224.6747382 ]])

    :param Luv: *CIE Luv* matrix.
    :type Luv: Matrix (3x1)
    :return: *CIE LCHuv* matrix.
    :rtype: Matrix (3x1)
    """

    L, u, v = numpy.ravel(Luv)

    H = 180. * math.atan2(v, u) / math.pi
    if H < 0.:
        H += 360.

    return numpy.matrix([L, math.sqrt(u ** 2 + v ** 2), H]).reshape((3, 1))

def LCHuv_to_Luv(LCHuv):
    """
    Converts from *CIE LCHuv* colorspace to *CIE Luv* colorspace using given matrix.

    Reference: http://www.brucelindbloom.com/Eqn_LCH_to_Luv.html

    Usage::

        >>> LCHuv_to_Luv(numpy.matrix([100., 28.18559104, 224.6747382]).reshape((3, 1)))
        matrix([[ 100.        ]
            [ -20.04304247]
            [ -19.81676035]])

    :param LCHuv: *CIE LCHuv* matrix.
    :type LCHuv: Matrix (3x1)
    :return: *CIE Luv* matrix.
    :rtype: Matrix (3x1)
    """

    L, C, H = numpy.ravel(LCHuv)

    return numpy.matrix([L, C * math.cos(math.radians(H)), C * math.sin(math.radians(H))]).reshape((3, 1))

def XYZ_to_Lab(XYZ, illuminant=color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")):
    """
    Converts from *CIE XYZ* colorspace to *CIE Lab* colorspace using given matrix.

    Reference: http://www.brucelindbloom.com/Eqn_XYZ_to_Lab.html

    Usage::

        >>> XYZ_to_Lab(numpy.matrix([0.92193107, 1., 1.03744246]).reshape((3, 1)))
        matrix([[ 100.        ]
            [  -7.41787844]
            [ -15.85742105]])

    :param XYZ: *CIE XYZ* matrix.
    :type XYZ: Matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *CIE Lab* matrix.
    :rtype: Matrix (3x1)
    """

    X, Y, Z = numpy.ravel(XYZ)
    Xr, Yr, Zr = numpy.ravel(xy_to_XYZ(illuminant))

    xr = X / Xr
    yr = Y / Yr
    zr = Z / Zr

    fx = xr ** (1. / 3.) if xr > color.lightness.CIE_E else (color.lightness.CIE_K * xr + 16.) / 116.
    fy = yr ** (1. / 3.) if yr > color.lightness.CIE_E else (color.lightness.CIE_K * yr + 16.) / 116.
    fz = zr ** (1. / 3.) if zr > color.lightness.CIE_E else (color.lightness.CIE_K * zr + 16.) / 116.

    L = 116. * fy - 16.
    a = 500. * (fx - fy)
    b = 200. * (fy - fz)

    return numpy.matrix([L, a, b]).reshape((3, 1))

def Lab_to_XYZ(Lab, illuminant=color.illuminants.ILLUMINANTS.get("Standard CIE 1931 2 Degree Observer").get("D50")):
    """
    Converts from *CIE Lab* colorspace to *CIE XYZ* colorspace using given matrix.

    Reference: http://www.brucelindbloom.com/Eqn_Lab_to_XYZ.html'.

    Usage::

        >>> Lab_to_XYZ(numpy.matrix([100., -7.41787844, -15.85742105]).reshape((3, 1)))
        matrix([[ 0.92193107]
            [ 0.11070565]
            [ 1.03744246]])

    :param Lab: *CIE Lab* matrix.
    :type Lab: Matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *CIE Lab* matrix.
    :rtype: Matrix (3x1)
    """

    L, a, b = numpy.ravel(Lab)
    Xr, Yr, Zr = numpy.ravel(xy_to_XYZ(illuminant))

    fy = (L + 16.) / 116.
    fx = a / 500. + fy
    fz = fy - b / 200.

    xr = fx ** 3. if fx ** 3. > color.lightness.CIE_E else (116. * fx - 16.) / color.lightness.CIE_K
    yr = ((L + 16.) / 116.) ** 3. if L > color.lightness.CIE_K * color.lightness.CIE_E else L / color.lightness.CIE_K
    zr = fz ** 3. if fz ** 3. > color.lightness.CIE_E else (116. * fz - 16.) / color.lightness.CIE_K

    X = xr * Xr
    Y = yr * Yr
    Z = zr * Zr

    return numpy.matrix([X, Y, Z]).reshape((3, 1))

def Lab_to_LCHab(Lab):
    """
    Converts from *CIE Lab* colorspace to *CIE LCHab* colorspace using given matrix.

    Reference: http://www.brucelindbloom.com/Eqn_Lab_to_LCH.html

    Usage::

        >>> Lab_to_LCHab(numpy.matrix([100., -7.41787844, -15.85742105]).reshape((3, 1)))
        matrix([[ 100.        ]
            [  17.50664796]
            [ 244.93046842]])

    :param Lab: *CIE Lab* matrix.
    :type Lab: Matrix (3x1)
    :return: *CIE LCHab* matrix.
    :rtype: Matrix (3x1)
    """

    L, a, b = numpy.ravel(Lab)

    H = 180. * math.atan2(b, a) / math.pi
    if H < 0.:
        H += 360.

    return numpy.matrix([L, math.sqrt(a ** 2 + b ** 2), H]).reshape((3, 1))

def LCHab_to_Lab(LCHab):
    """
    Converts from *CIE LCHab* colorspace to *CIE Lab* colorspace using given matrix.

    Reference: http://www.brucelindbloom.com/Eqn_LCH_to_Lab.html

    Usage::

        >>> LCHab_to_Lab(numpy.matrix([100., 17.50664796, 244.93046842]).reshape((3, 1)))
        matrix([[ 100.        ]
            [  -7.41787844]
            [ -15.85742105]])

    :param LCHab: *CIE LCHab* matrix.
    :type LCHab: Matrix (3x1)
    :return: *CIE Lab* matrix.
    :rtype: Matrix (3x1)
    """

    L, C, H = numpy.ravel(LCHab)

    return numpy.matrix([L, C * math.cos(math.radians(H)), C * math.sin(math.radians(H))]).reshape((3, 1))
