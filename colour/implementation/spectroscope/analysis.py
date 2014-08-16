#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectroscope - Analysis
=======================

Defines objects for the homemade spectroscope spectrum images analysis.

References
----------
.. [1]  http://thomasmansencal.blogspot.fr/2014/07/a-homemade-spectroscope.html
"""

from __future__ import division, unicode_literals

import matplotlib.image
import matplotlib.pyplot
import numpy as np
import scipy.ndimage

from colour import Extrapolator1d
from colour import LinearInterpolator1d
from colour import RGB_COLOURSPACES
from colour import SpectralPowerDistribution
from colour import TriSpectralPowerDistribution

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_Spectrum',
           'transfer_function',
           'get_image',
           'get_image_profile',
           'calibrate_RGB_spectrum_profile',
           'get_RGB_spectrum',
           'get_luminance_spd']


class RGB_Spectrum(TriSpectralPowerDistribution):
    """
    Defines an *RGB* spectrum object implementation.

    Parameters
    ----------
    name : unicode
        *RGB* spectrum name.
    data : dict
        *RGB* spectrum.

    Attributes
    ----------
    R
    G
    B
    """

    def __init__(self, name, data):
        TriSpectralPowerDistribution.__init__(self,
                                              name,
                                              data,
                                              mapping={'x': 'R',
                                                       'y': 'G',
                                                       'z': 'B'},
                                              labels={'x': 'R',
                                                      'y': 'G',
                                                      'z': 'B'})

    @property
    def R(self):
        """
        Property for **self.R** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.R

        Warning
        -------
        :attr:`RGB_Spectrum.R` is read only.
        """

        return self.x

    @R.setter
    def R(self, value):
        """
        Setter for **self.R** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('R'))

    @property
    def G(self):
        """
        Property for **self.G** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.G

        Warning
        -------
        :attr:`RGB_Spectrum.G` is read only.
        """

        return self.y

    @G.setter
    def G(self, value):
        """
        Setter for **self.G** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('G'))

    @property
    def B(self):
        """
        Property for **self.B** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.B

        Warning
        -------
        :attr:`RGB_Spectrum.B` is read only.
        """

        return self.z

    @B.setter
    def B(self, value):
        """
        Setter for **self.B** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('B'))


def transfer_function(image,
                      colourspace=RGB_COLOURSPACES['sRGB'],
                      to_linear=False):
    """
    Evaluate given colourspace transfer / inverse transfer function on given
    image data.

    Parameters
    ----------
    image : ndarray
        Image to evaluate the transfer function.
    colourspace : RGB_Colourspace, optional
        *RGB* Colourspace.
    to_linear : bool, optional
        Use colourspace inverse transfer function instead of colourspace
        transfer function.

    Returns
    -------
    ndarray
        Transformed image.

    """

    vector_linearise = np.vectorize(
        lambda x: colourspace.inverse_transfer_function(
            x) if to_linear else colourspace.transfer_function(x))

    return vector_linearise(image)


def get_image(path,
              colourspace=RGB_COLOURSPACES['sRGB'],
              to_linear=True):
    """
    Reads image from given path.

    Parameters
    ----------
    path : unicode
        Path to read the image from.
    colourspace : RGB_Colourspace, optional
        *RGB* Colourspace.
    to_linear : bool, optional
        Evaluate colourspace inverse transfer function on image data.

    Returns
    -------
    ndarray
        Image.
    """

    image = matplotlib.image.imread(path)

    if to_linear:
        image = transfer_function(image,
                                  colourspace=colourspace,
                                  to_linear=True)

    return image


def get_image_profile(image, line, samples=None):
    """
    Returns the image profile using given line coordinates and given samples
    count.

    Parameters
    ----------
    image : ndarray
        image: Image to retrieve the profile.
    line : tuple or list or ndarray, (x0, y0, x1, y1)
        line: Coordinates as image array indexes to measure the profile.
    samples : int, optional
        samples: Samples count to retrieve along the line, default to image
        width.

    Returns
    -------
    ndarray
        Profile.

    References
    ----------
    .. [2]  http://stackoverflow.com/a/7880726/931625
            (Last accessed 8 August 2014)
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

    Parameters
    ----------
    profile : ndarray
        profile: Image profile to calibrate.
    reference : dict
        reference: Theoretical reference wavelength lines.
    measured : dict
        measured: Measured lines in horizontal axis pixels values.
    samples : int, optional
        samples: Profile samples count.

    Returns
    -------
    RGB_Spectrum
        Calibrated RGB spectrum.
    """

    samples = samples if samples else profile.shape[1]
    measured_lines = [line for line, value in
                      sorted(measured.items(), key=lambda x: x[1])]

    # Reference samples.
    r = np.array([reference.get(sample) for sample in measured_lines])
    # Measured samples.
    m = np.array([measured.get(sample) for sample in measured_lines])

    # Reference range array.
    rr = np.linspace(min(r), max(r))
    # Measured range array.
    mm = np.linspace(min(m), max(m))

    # Interpolator from reference to measured.
    r_to_m_interpolator = Extrapolator1d(LinearInterpolator1d(r, m))

    # Interpolator from measured range to reference range.
    mm_to_rr_interpolator = Extrapolator1d(LinearInterpolator1d(mm, rr))

    # Colors interpolator.
    R_interpolator = Extrapolator1d(
        LinearInterpolator1d(np.arange(0, profile.shape[1]), profile[0, :, 0]))
    G_interpolator = Extrapolator1d(
        LinearInterpolator1d(np.arange(0, profile.shape[1]), profile[0, :, 1]))
    B_interpolator = Extrapolator1d(
        LinearInterpolator1d(np.arange(0, profile.shape[1]), profile[0, :, 2]))

    wavelengths = np.linspace(mm_to_rr_interpolator([0]),
                              mm_to_rr_interpolator([profile.shape[1]]),
                              samples)

    R = dict(zip(wavelengths,
                 R_interpolator(r_to_m_interpolator(wavelengths))))
    G = dict(zip(wavelengths,
                 G_interpolator(r_to_m_interpolator(wavelengths))))
    B = dict(zip(wavelengths,
                 B_interpolator(r_to_m_interpolator(wavelengths))))

    return RGB_Spectrum('RGB Spectrum', {'R': R, 'G': G, 'B': B})


def get_RGB_spectrum(image, reference, measured, samples=None):
    """
    Returns the RGB spectrum of given image.

    Parameters
    ----------
    image : ndarray
        image: Image to retrieve the RGB spectrum, assuming the spectrum is
        already properly oriented.
    reference : dict
        reference: Theoretical reference wavelength lines.
    measured : dict
        measured: Measured lines in horizontal axis pixels values.
    samples : int, optional
        samples: Spectrum samples count.

    Returns
    -------
    RGB_Spectrum
        RGB spectrum.
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
                      colourspace=RGB_COLOURSPACES['sRGB']):
    """
    Returns the luminance spectral power distribution of given RGB spectrum.

    Parameters
    ----------
    RGB_spectrum : RGB_Spectrum
        RGB_spectrum: RGB spectrum to retrieve the luminance from.
    colourspace : RGB_Colourspace
        colourspace: *RGB* Colourspace.

    SpectralPowerDistribution
        RGB spectrum luminance spectral power distribution, units are arbitrary
        and normalised to [0, 100] domain.
    """

    RGB_spectrum = RGB_spectrum.clone().normalise(100.)
    get_RGB_luminance = lambda x: colour.get_RGB_luminance(
        x,
        colourspace.primaries,
        colourspace.whitepoint)

    return SpectralPowerDistribution(
        'RGB_spectrum',
        dict([(wavelength, get_RGB_luminance(RGB))
              for wavelength, RGB in RGB_spectrum]))
