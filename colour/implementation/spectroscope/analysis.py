# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**analysis.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines objects for the homemade spectroscope spectrum images analysis.

    References:

    -  http://thomasmansencal.blogspot.fr/2014/07/a-homemade-spectroscope.html

**Others:**

"""

import matplotlib.image
import matplotlib.pyplot
import numpy as np
import scipy.ndimage

import colour
from colour import Extrapolator1d
from colour import LinearInterpolator
from colour import SpectralPowerDistribution
from colour import TriSpectralPowerDistribution

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["RGB_Spectrum",
           "transfer_function",
           "get_image",
           "get_image_profile",
           "calibrate_RGB_spectrum_profile",
           "get_RGB_spectrum",
           "get_luminance_spd"]


class RGB_Spectrum(TriSpectralPowerDistribution):
    """
    Defines an *RGB* spectrum object implementation.
    """

    def __init__(self, name, data):
        """
        Initialises the class.

        :param name: *RGB* spectrum name.
        :type name: unicode
        :param data: *RGB* spectrum.
        :type data: dict
        """

        TriSpectralPowerDistribution.__init__(self,
                                              name,
                                              data,
                                              mapping={"x": "R",
                                                       "y": "G",
                                                       "z": "B"},
                                              labels={"x": "R",
                                                      "y": "G",
                                                      "z": "B"})

    @property
    def R(self):
        """
        Property for **self.__R** attribute.

        :return: self.__R.
        :rtype: unicode
        """

        return self.x

    @R.setter
    def R(self, value):
        """
        Setter for **self.__R** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError(
            "{0} | '{1}' attribute is read only!".format(
                self.__class__.__name__, "R"))

    @property
    def G(self):
        """
        Property for **self.__G** attribute.

        :return: self.__G.
        :rtype: unicode
        """

        return self.y

    @G.setter
    def G(self, value):
        """
        Setter for **self.__G** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError(
            "{0} | '{1}' attribute is read only!".format(
                self.__class__.__name__, "G"))

    @property
    def B(self):
        """
        Property for **self.__B** attribute.

        :return: self.__B.
        :rtype: unicode
        """

        return self.z

    @B.setter
    def B(self, value):
        """
        Setter for **self.__B** attribute.

        :param value: Attribute value.
        :type value: unicode
        """

        raise AttributeError(
            "{0} | '{1}' attribute is read only!".format(
                self.__class__.__name__, "B"))


def transfer_function(image, colourspace=colour.RGB_COLOURSPACES["sRGB"],
                      to_linear=False):
    """
    Evaluate given colourspace transfer / inverse transfer function on given
    image data.

    :param image: Image to evalute the transfer function.
    :type image: ndarray
    :param colourspace: *RGB* Colourspace.
    :type colourspace: RGB_Colourspace
    :param to_linear: Use colourspace inverse transfer function instead of \
    colourspace transfer function.
    :type to_linear: bool
    :return: Transformed image.
    :rtype: ndarray

    References:

    -  http://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
    """

    vector_linearise = np.vectorize(
        lambda x: colourspace.inverse_transfer_function(
            x) if to_linear else colourspace.transfer_function(x))

    return vector_linearise(image)


def get_image(path, colourspace=colour.RGB_COLOURSPACES["sRGB"],
              to_linear=True):
    """
    Reads image from given path.

    :param path: Path to read the image from.
    :type path: unicode
    :param colourspace: *RGB* Colourspace.
    :type colourspace: RGB_Colourspace
    :param to_linear: Evaluate colourspace inverse transfer function on image \
    data.
    :type to_linear: bool
    :return: Image.
    :rtype: ndarray
    """

    image = matplotlib.image.imread(path)

    if to_linear:
        image = transfer_function(image, colourspace=colourspace,
                                  to_linear=True)

    return image


def get_image_profile(image, line, samples=None):
    """
    Returns the image profile using given line coordinates and given samples
    count.

    :param image: Image to retrieve the profile.
    :type image: ndarray
    :param line: Coordinates as image array indexes to measure the profile.
    :type line: tuple or list or ndarray (x0, y0, x1, y1)
    :param samples: Samples count to retrieve along the line, default to image \
    width.
    :type samples: int
    :return: Profile.
    :rtype: ndarray

    References:

    -  http://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
    """

    height, width, channels = image.shape
    samples = samples if samples else width
    x0, y0, x1, y1 = line

    profile = []
    for i in range(channels):
        x, y = np.linspace(x0, x1, samples), np.linspace(y0, y1, samples)
        z = image[:, :, i]

        profile.append(
            scipy.ndimage.map_coordinates(np.transpose(z), np.vstack((x, y))))

    return np.dstack(profile)


def calibrate_RGB_spectrum_profile(profile, reference, measured, samples=None):
    """
    Calibrates given spectrum profile using given theoretical reference
    wavelength lines in nanometers and measured lines in horizontal axis pixels
    values. If more than 2 lines are provided the profile data will be warped
    to fit the theoretical reference wavelength lines.

    :param profile: Image profile to calibrate.
    :type profile: ndarray
    :param reference: Theoretical reference wavelength lines.
    :type reference: dict
    :param measured: Measured lines in horizontal axis pixels values.
    :type measured: dict
    :param samples: Profile samples count.
    :type samples: int
    :return: Calibrated RGB spectrum.
    :rtype: RGB_Spectrum
    """

    samples = samples if samples else profile.shape[1]
    measured_lines = [line for line, value in
                      sorted(measured.items(), key=lambda x: x[1])]

    # Reference samples.
    r = np.array(map(lambda x: reference.get(x), measured_lines))
    # Measured samples.
    m = np.array(map(lambda x: measured.get(x), measured_lines))

    # Reference range array.
    rr = np.linspace(min(r), max(r))
    # Measured range array.
    mm = np.linspace(min(m), max(m))

    # Interpolator from reference to measured.
    r_to_m_interpolator = Extrapolator1d(LinearInterpolator(r, m))

    # Interpolator from measured range to reference range.
    mm_to_rr_interpolator = Extrapolator1d(LinearInterpolator(mm, rr))

    # Colors interpolator.
    R_interpolator = Extrapolator1d(
        LinearInterpolator(np.arange(0, profile.shape[1]), profile[0, :, 0]))
    G_interpolator = Extrapolator1d(
        LinearInterpolator(np.arange(0, profile.shape[1]), profile[0, :, 1]))
    B_interpolator = Extrapolator1d(
        LinearInterpolator(np.arange(0, profile.shape[1]), profile[0, :, 2]))

    wavelengths = np.linspace(mm_to_rr_interpolator([0]),
                              mm_to_rr_interpolator([profile.shape[1]]),
                              samples)

    R = dict(zip(wavelengths,
                 R_interpolator(r_to_m_interpolator(wavelengths))))
    G = dict(zip(wavelengths,
                 G_interpolator(r_to_m_interpolator(wavelengths))))
    B = dict(zip(wavelengths,
                 B_interpolator(r_to_m_interpolator(wavelengths))))

    return RGB_Spectrum("RGB Spectrum", {"R": R, "G": G, "B": B})


def get_RGB_spectrum(image, reference, measured, samples=None):
    """
    Returns the RGB spectrum of given image.

    :param image: Image to retrieve the RGB spectrum, assuming the spectrum is \
    already properly oriented.
    :type image: ndarray
    :param reference: Theoretical reference wavelength lines.
    :type reference: dict
    :param measured: Measured lines in horizontal axis pixels values.
    :type measured: dict
    :param samples: Spectrum samples count.
    :type samples: int
    :return: RGB spectrum.
    :rtype: RGB_Spectrum
    """

    samples = samples if samples else image.shape[1]
    profile = get_image_profile(image,
                                line=[0, 0, image.shape[1] - 1, 0],
                                samples=samples)

    return calibrate_RGB_spectrum_profile(profile=profile,
                                          reference=reference,
                                          measured=measured,
                                          samples=samples)


def get_luminance_spd(RGB_spectrum,
                      colourspace=colour.RGB_COLOURSPACES["sRGB"]):
    """
    Returns the luminance spectral power distribution of given RGB spectrum.

    :param RGB_spectrum: RGB spectrum to retrieve the luminance from.
    :type RGB_spectrum: RGB_Spectrum
    :param colourspace: *RGB* Colourspace.
    :type colourspace: RGB_Colourspace
    :return: RGB spectrum luminance spectral power distribution, units are \
    arbitrary and normalised to [0, 100] domain.
    :rtype: SpectralPowerDistribution
    """

    RGB_spectrum = RGB_spectrum.clone().normalise(100.)
    get_RGB_luminance = lambda x: colour.get_RGB_luminance(
        x,
        colourspace.primaries,
        colourspace.whitepoint)

    return SpectralPowerDistribution(
        "RGB_spectrum",
        dict([(wavelength, get_RGB_luminance(RGB)) \
              for wavelength, RGB in RGB_spectrum]))
