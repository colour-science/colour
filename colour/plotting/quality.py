#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Quality Plotting
=======================

Defines the colour quality plotting objects:

-   :func:`colour_rendering_index_bars_plot`
"""

from __future__ import division

import matplotlib.pyplot
import numpy as np
import pylab

from colour.models import XYZ_to_sRGB
from colour.quality import (
    colour_quality_scale,
    colour_rendering_index)
from colour.plotting import (
    DEFAULT_FIGURE_WIDTH,
    boundaries,
    canvas,
    decorate,
    display)
from colour.utilities import normalise

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['colour_quality_bars_plot',
           'colour_rendering_index_bars_plot',
           'colour_quality_scale_bars_plot']


def colour_quality_bars_plot(specification, **kwargs):
    """
    Plots the colour quality data of given illuminant or light source colour
    quality specification.

    Parameters
    ----------
    specification : CRI_Specification or VS_ColourQualityScaleData
        Illuminant or light source specification colour quality specification.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS.get('F2')
    >>> colour_quality_bars_plot(illuminant)  # doctest: +SKIP
    True
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    axis = matplotlib.pyplot.gca()

    Q_a, Q_as, colorimetry_data = (specification.Q_a,
                                   specification.Q_as,
                                   specification.colorimetry_data)

    colours = ([[1] * 3] + [normalise(XYZ_to_sRGB(x.XYZ / 100))
                            for x in colorimetry_data[0]])
    x, y = tuple(zip(*[(x[0], x[1].Q_a) for x in sorted(Q_as.items(),
                                                        key=lambda x: x[0])]))
    x, y = np.array([0] + list(x)), np.array([Q_a] + list(y))

    positive = True if np.sign(min(y)) in (0, 1) else False

    width = 0.5
    bars = pylab.bar(x, y, color=colours, width=width)
    y_ticks_steps = 10
    pylab.yticks(range(0 if positive else -100,
                       100 + y_ticks_steps,
                       y_ticks_steps))
    pylab.xticks(x + width / 2,
                 ['Qa'] + ['Q{0}'.format(index) for index in x[1:]])

    def label_bars(bars):
        """
        Add labels above given bars.
        """
        for bar in bars:
            y = bar.get_y()
            height = bar.get_height()
            value = height if np.sign(y) in (0, 1) else -height
            axis.text(bar.get_x() + bar.get_width() / 2,
                      0.025 * height + height + y,
                      '{0:.1f}'.format(value),
                      ha='center', va='bottom')

    label_bars(bars)

    settings.update({
        'title': 'Colour Quality',
        'grid': True,
        'grid_axis': 'y',
        'x_tighten': True,
        'y_tighten': True,
        'limits': (-width,
                   len(Q_as) + width * 2,
                   0 if positive else -110,
                   110),
        'aspect': 1 / ((110 if positive else 220) /
                       (width + len(Q_as) + width * 2))})
    settings.update(kwargs)

    boundaries(**settings)
    decorate(**settings)
    return display(**settings)


def colour_rendering_index_bars_plot(spd, **kwargs):
    """
    Plots the *colour rendering index* of given illuminant or light source.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Illuminant or light source to plot the *colour rendering index*.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS.get('F2')
    >>> colour_rendering_index_bars_plot(illuminant)  # doctest: +SKIP
    True
    """

    if colour_quality_bars_plot(
            colour_rendering_index(
                    spd,
                    additional_data=True),
            standalone=False):
        settings = {
            'title': 'Colour Rendering Index - {0}'.format(spd.title)}

        decorate(**settings)
        return display(**settings)


def colour_quality_scale_bars_plot(spd, **kwargs):
    """
    Plots the *colour quality scale* of given illuminant or light source.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Illuminant or light source to plot the *colour quality scale*.
    \*\*kwargs : \*\*
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> from colour import ILLUMINANTS_RELATIVE_SPDS
    >>> illuminant = ILLUMINANTS_RELATIVE_SPDS.get('F2')
    >>> colour_quality_scale_bars_plot(illuminant)  # doctest: +SKIP
    True
    """

    if colour_quality_bars_plot(
            colour_quality_scale(
                    spd,
                    additional_data=True),
            standalone=False):
        settings = {
            'title': 'Colour Quality Scale - {0}'.format(spd.title)}

        decorate(**settings)
        return display(**settings)
