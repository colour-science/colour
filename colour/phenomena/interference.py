from __future__ import division, unicode_literals

import math
import colour
import numpy as np
import matplotlib.pyplot as plt
from colour.plotting import *

from colour.colorimetry import DEFAULT_SPECTRAL_SHAPE
from colour.utilities import as_float_array

# There are three different functions here: film_intensity, filmColourSpec, filmColour.
#
# film_intensity takes in the thickness of a film (in nm), the wavelength of some light incident on that film (in nm), a boolean which tells it whether or not the film is made of water because if it is, it can use an empirical algorithm to determine the refractive index for
# that wavelength from the water's absolute temperature (in K) and its density (in kg/m^3). There is also the option to make "water=False" and then "n=2.4" or any other manually inserted refractive index if the liquid in question is, say, oil. The function then outputs
# a "relative intensity" of the light that will come off the film (taking into account the interference with its own partial reflections). This intensity will ALWAYS be in the range [0,1].
#
# filmColour then takes in the same information (in the same units) but not wavelength and assumes a white-light uniform source - you may be able to add in functionality for using illuminants but I haven't read anything about how that stuff works in your library. It then
# outputs to the user's screen a plotted ColourSwatch with the colour you would see reflected from the film in that situation.
#
# filmColourSpec takes in the same information as filmColour but instead of a single thickness, you give it a range of thicknesses in the form of thick_min and thick_max. It then plots a graph for you of how the spectrum of colours reflected varies with different
# thicknesses with a title.
#
# Most of these things have helpful defaults built-in so you can try them straight out of the box to see what I mean! Good luck, thanks again for your help and don't be afraid to ask me for any explanation (or to completely change every part of my code)! I'm honoured
# that anyone would want it!


def light_water_molar_refraction_Schiebener1990(wavelength,
                                                temperature=294,
                                                density=1000):
    wl_s = as_float_array(wavelength) / 589
    T_s = as_float_array(temperature) / 273.15
    p_s = as_float_array(density) / 1000

    a_0 = 0.243905091
    a_1 = 9.53518094 * 10 ** -3
    a_2 = -3.64358110 * 10 ** -3
    a_3 = 2.65666426 * 10 ** -4
    a_4 = 1.59189325 * 10 ** -3
    a_5 = 2.45733798 * 10 ** -3
    a_6 = 0.897478251
    a_7 = -1.63066183 * 10 ** -2
    wl_s_UV = 0.2292020
    wl_s_IR = 5.432937

    wl_s_2 = wl_s ** 2

    LL = (a_0 + a_1 * p_s + a_2 * T_s + a_3 * wl_s_2 * T_s + a_4 / wl_s_2 +
          (a_5 / (wl_s_2 - wl_s_UV ** 2)) + (a_6 / (wl_s_2 - wl_s_IR ** 2)) +
          a_7 * p_s ** 2)

    return LL


def light_water_refractive_index_Schiebener1990(wavelength,
                                                temperature=294,
                                                density=1000):
    p_s = as_float_array(density) / 1000

    LL = light_water_molar_refraction_Schiebener1990(wavelength, temperature,
                                                     density)
    n = np.sqrt((2 * LL + 1 / p_s) / (1 / p_s - LL))

    return n


# https://en.wikipedia.org/wiki/Thin-film_interference
def optical_path_difference(n_1, n_2, theta_i, d):
    # Snell's Law
    theta_t = np.arcsin(n_1 * np.sin(theta_i) / n_2)

    d_cos_theta_t = d / (np.cos(theta_t))

    opd = n_2 * (d_cos_theta_t + d_cos_theta_t) - n_1 * (
        2 * d * np.tan(theta_t) * np.sin(theta_i))

    return opd


def film_intensity(wavelength, thickness, incident=0, n=1.333):
    wl = as_float_array(wavelength)
    thickness = as_float_array(thickness)
    incident = as_float_array(incident)
    n = as_float_array(n)

    return (
        0.5 -
        0.5 * np.cos(4 * n * np.pi * thickness[..., np.newaxis] * np.sqrt(1 - (
            np.sin(incident * 2 * np.pi / 360) / n) ** 2) / wl[np.newaxis]))


def filmColourSpec(
        thickness=np.arange(0, 1000, 1),
        incident=0,
        temperature=294,
        density=1000,
        n=1.333):
    raw = []
    spec = np.arange(300, 800, 1)

    n = light_water_molar_refraction_Schiebener1990(spec, temperature, density)
    f_i = film_intensity(spec, thickness, incident, n)
    for i in f_i:
        raw.append(
            colour.sd_to_XYZ(colour.SpectralDistribution(i, domain=spec)) /
            100)

    raw = colour.XYZ_to_sRGB(raw)
    raw /= np.max(raw)
    raw = np.clip(raw, 0, 1)
    figure = plt.figure()

    axes = figure.add_subplot(211)
    axes.set_xlabel('Thickness (nm)')
    axes.set_title('Spectrum of Expected Colour against Thickness')
    axes.set_yticklabels([])
    axes.set_yticks([])
    colour.plotting.plot_multi_colour_swatches(
        [colour.plotting.ColourSwatch(RGB=c) for c in raw],
        height=500,
        axes=axes,
        standalone=False)
    plt.show()


def filmColour(thickness,
               incident=0,
               water=True,
               temperature=294,
               density=1000,
               n=1.333):
    spec = np.linspace(300, 800, 499)
    raw = np.array([
        colour.XYZ_to_sRGB(
            colour.sd_to_XYZ(
                colour.SpectralDistribution(
                    dict([(int(i) + 1,
                           film_intensity(thickness, i, incident, water,
                                          temperature, density, n))
                          for i in spec]))))
    ])
    raw /= np.max(raw)
    raw = np.clip(raw, 0, 1)
    plot_single_colour_swatch(ColourSwatch(RGB=raw[0]))


filmColourSpec()
