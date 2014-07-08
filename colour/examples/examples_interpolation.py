# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Showcases some **Colour** package *interpolation* related examples.
"""

import pylab

import colour
import colour.implementation.matplotlib.plots


# Comparing *CIE* *Sprague* and *cubic spline* recommended interpolation methods.
uniform_spd_data = {
    340: 0.0000,
    360: 0.0000,
    380: 0.0000,
    400: 0.0641,
    420: 0.0645,
    440: 0.0562,
    460: 0.0537,
    480: 0.0559,
    500: 0.0651,
    520: 0.0705,
    540: 0.0772,
    560: 0.0870,
    580: 0.1128,
    600: 0.1360,
    620: 0.1511,
    640: 0.1688,
    660: 0.1996,
    680: 0.2397,
    700: 0.2852,
    720: 0.0000,
    740: 0.0000,
    760: 0.0000,
    780: 0.0000,
    800: 0.0000,
    820: 0.0000}

non_uniform_spd_data = {
    340.1: 0.0000,
    360: 0.0000,
    380: 0.0000,
    400: 0.0641,
    420: 0.0645,
    440: 0.0562,
    460: 0.0537,
    480: 0.0559,
    500: 0.0651,
    520: 0.0705,
    540: 0.0772,
    560: 0.0870,
    580: 0.1128,
    600: 0.1360,
    620: 0.1511,
    640: 0.1688,
    660: 0.1996,
    680: 0.2397,
    700: 0.2852,
    720: 0.0000,
    740: 0.0000,
    760: 0.0000,
    780: 0.0000,
    800: 0.0000,
    820.9: 0.0000}

base_spd = colour.SpectralPowerDistribution("Reference", uniform_spd_data)
uniform_interpolated_spd = colour.SpectralPowerDistribution("Uniform - Sprague Interpolation",
                                                           uniform_spd_data)
non_uniform_interpolated_spd = colour.SpectralPowerDistribution("Non Uniform - Cubic Spline Interpolation",
                                                               non_uniform_spd_data)

uniform_interpolated_spd.interpolate(steps=1)
non_uniform_interpolated_spd.interpolate(steps=1)

start, end, steps = base_spd.shape
x_limit_min, x_limit_max, y_limit_min, y_limit_max = [], [], [], []

pylab.plot(base_spd.wavelengths,
           base_spd.values,
           "ro-",
           label=base_spd.name,
           linewidth=2.)
pylab.plot(uniform_interpolated_spd.wavelengths,
           uniform_interpolated_spd.values,
           label=uniform_interpolated_spd.name,
           linewidth=2.)
pylab.plot(non_uniform_interpolated_spd.wavelengths,
           non_uniform_interpolated_spd.values,
           label=non_uniform_interpolated_spd.name,
           linewidth=2.)

x_limit_min.append(start)
x_limit_max.append(end)
y_limit_min.append(min(base_spd.values))
y_limit_max.append(max(base_spd.values))

settings = {"x_label": u"Wavelength λ (nm)",
            "y_label": "Spectral Power Distribution",
            "x_tighten": True,
            "legend": True,
            "legend_location": "upper left",
            "x_ticker": True,
            "y_ticker": True,
            "limits": [min(x_limit_min), max(x_limit_max), min(y_limit_min), max(y_limit_max)]}

colour.implementation.matplotlib.plots.bounding_box(**settings)
colour.implementation.matplotlib.plots.aspect(**settings)
colour.implementation.matplotlib.plots.display(**settings)