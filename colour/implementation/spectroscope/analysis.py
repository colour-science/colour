import matplotlib.image
import matplotlib.pyplot
import numpy
import pylab
import scipy.ndimage
from scipy.interpolate import interp1d

import colour
from colour import SpectralPowerDistribution
from colour import TriSpectralPowerDistribution

pylab.rcParams["figure.figsize"] = 14, 7

FRAUNHOFER_PUBLISHED_LINES = {
    "y": 898.765,
    "Z": 822.696,
    "A": 759.370,
    "B": 686.719,
    "C": 656.281,
    "a": 627.661,
    "D1": 589.592,
    "D2": 588.995,
    "D3": 587.5618,
    "e": 546.073,
    "E2": 527.039,
    "b1": 518.362,
    "b2": 517.270,
    "b3": 516.891,
    "b4": 516.733,
    "c": 495.761,
    "F": 486.134,
    "d": 466.814,
    "e": 438.355,
    "G": 430.790,
    "h": 410.175,
    "H": 396.847,
    "K": 393.368,
    "L": 382.044,
    "N": 358.121,
    "P": 336.112,
    "T": 302.108,
    "t": 299.444}

FRAUNHOFER_LINES_ELEMENTS_MAPPING = {
    "y": "O2",
    "Z": "O2",
    "A": "O2",
    "B": "O2",
    "C": "H Alpha",
    "a": "O2",
    "D1": "Na",
    "D2": "Na",
    "D3": "He",
    "e": "Hg",
    "E2": "Fe",
    "b1": "Mg",
    "b2": "Mg",
    "b3": "Fe",
    "b4": "Mg",
    "c": "Fe",
    "F": "H Beta",
    "d": "Fe",
    "e": "Fe",
    "G'": "H Gamma",
    "G": "Fe",
    "G": "Ca",
    "h": "H Delta",
    "H": "Ca+",
    "K": "Ca+",
    "L": "Fe",
    "N": "Fe",
    "P": "Ti+",
    "T": "Fe",
    "t": "Ni"}

FRAUNHOFER_MEASURED_LINES = {
    "G": 134,
    "F": 371,
    "b4": 502,
    "E2": 545,
    "D3": 810,
    "a": 974,
    "C": 1095}


class RGB_Spectrum(TriSpectralPowerDistribution):
    """
    Defines an *RGB* spectrum object implementation.
    """

    def __init__(self, name, data):
        """
        Initialises the class.

        :param name: *RGB* spectrum name.
        :type name: unicode
        :param data:*RGB* spectrum.
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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "R"))

    @R.deleter
    def R(self):
        """
        Deleter for **self.__R** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "R"))

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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "G"))

    @G.deleter
    def G(self):
        """
        Deleter for **self.__G** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "G"))

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

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is read only!".format(self.__class__.__name__, "B"))

    @B.deleter
    def B(self):
        """
        Deleter for **self.__B** attribute.
        """

        raise colour.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "B"))


def extrap1d(interpolator):
    # http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
    xs, ys = interpolator.x, interpolator.y

    def extrapolate(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            return interpolator(x)

    return lambda xs: numpy.array(map(extrapolate, numpy.array(xs)))


def get_image_profile(image, axis, samples=None):
    # http://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
    height, width, channels = image.shape
    samples = samples if samples else width
    x0, y0, x1, y1 = axis
    profile = []

    for i in range(channels):
        x, y = numpy.linspace(x0, x1, samples), numpy.linspace(y0, y1, samples)
        z = image[:, :, i]

        profile.append(scipy.ndimage.map_coordinates(numpy.transpose(z), numpy.vstack((x, y))))

    return numpy.dstack(profile)


def calibrate_RGB_spectrum_profile(reference, measured, profile, samples=100):
    measured_lines = [line for line, value in sorted(measured.items(), key=lambda x: x[1])]

    # Reference samples.
    r = numpy.array(map(lambda x: reference.get(x), measured_lines))
    # Measured samples.
    m = numpy.array(map(lambda x: measured.get(x), measured_lines))

    # Reference range array.
    rr = numpy.linspace(min(r), max(r))
    # Measured range array.
    mm = numpy.linspace(min(m), max(m))

    # Interpolator from reference to measured.
    r_to_m_interpolator = extrap1d(interp1d(r, m))

    # Interpolator from measured range to reference range.
    mm_to_rr_interpolator = extrap1d(interp1d(mm, rr))

    # Colors interpolator.
    R_interpolator = extrap1d(interp1d(numpy.arange(0, profile.shape[1]), profile[0, :, 0]))
    G_interpolator = extrap1d(interp1d(numpy.arange(0, profile.shape[1]), profile[0, :, 1]))
    B_interpolator = extrap1d(interp1d(numpy.arange(0, profile.shape[1]), profile[0, :, 2]))

    wavelengths = numpy.linspace(mm_to_rr_interpolator([0]),
                                 mm_to_rr_interpolator([profile.shape[1]]),
                                 samples)

    R = dict(zip(wavelengths, R_interpolator(r_to_m_interpolator(wavelengths))))
    G = dict(zip(wavelengths, G_interpolator(r_to_m_interpolator(wavelengths))))
    B = dict(zip(wavelengths, B_interpolator(r_to_m_interpolator(wavelengths))))

    return RGB_Spectrum("RGB Spectrum", {"R": R, "G": G, "B": B})


def get_RGB_spectrum(image, reference, measured, samples=None):
    profile = get_image_profile(image,
                                axis=[0, 0, image.shape[1] - 1, 0],
                                samples=samples if samples else image.shape[1])

    return calibrate_RGB_spectrum_profile(reference=reference,
                                          measured=measured,
                                          profile=profile,
                                          samples=samples if samples else profile.shape[1])


def get_spectral_power_distribution(RGB_spectrum, colourspace=colour.COLOURSPACES["sRGB"]):
    RGB_spectrum = RGB_spectrum.clone().normalise(100.)
    get_luminance = lambda x: colour.get_luminance(x, colourspace.primaries, colourspace.whitepoint)
    return SpectralPowerDistribution("RGB_spectrum",
                                     dict([(wavelength, get_luminance(RGB)) for wavelength, RGB in RGB_spectrum]))


def get_image(path, colourspace=colour.COLOURSPACES["sRGB"], linearise=True):
    image = matplotlib.image.imread(path)

    if linearise:
        vector_linearise = numpy.vectorize(lambda x: colourspace.inverse_transfer_function(x))
        image = vector_linearise(image)

    return image

RGB_spectrum = get_RGB_spectrum(
    get_image("/Users/kelsolaar/Documents/Personal/Science/Colour/Spectroscope/Fraunhofer_Lines/Measured.png"),
    {"r": 25,
     "g": 50,
     "y": 55,
     "b": 75},
    {"r": 5,
     "g": 20,
     "y": 87,
     "b": 100})

# pylab.imsave("/Users/kelsolaar/Documents/Personal/Science/Colour/Spectroscope/Fraunhofer_Lines/Adapted.png",
#              numpy.dstack([RGB_spectrum.R.values, RGB_spectrum.G.values, RGB_spectrum.B.values]))

RGB_spectrum = get_RGB_spectrum(
    get_image("/Users/kelsolaar/Documents/Personal/Science/Colour/Spectroscope/Fraunhofer_Lines/Fraunhofer_Lines_002.png"),
    FRAUNHOFER_PUBLISHED_LINES,
    FRAUNHOFER_MEASURED_LINES)

# pylab.imsave(
#     "/Users/kelsolaar/Documents/Personal/Science/Colour/Spectroscope/Fraunhofer_Lines/Fraunhofer_Lines_003.png",
#     numpy.dstack([RGB_spectrum.R.values, RGB_spectrum.G.values, RGB_spectrum.B.values]))

luminance_spd = get_spectral_power_distribution(RGB_spectrum).normalise()

blackbody_spd = colour.blackbody_spectral_power_distribution(5778, *luminance_spd.shape)
blackbody_spd.normalise()

from colour.implementation.matplotlib.plots import *

multi_spd_plot([luminance_spd, blackbody_spd])
multi_spd_plot([RGB_spectrum.R, RGB_spectrum.G, RGB_spectrum.B, luminance_spd])

