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

from colour.algebra import normalise
from colour.models import XYZ_to_sRGB
from colour.quality import colour_rendering_index
from colour.plotting import (
    aspect,
    bounding_box,
    display,
    figure_size)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['colour_rendering_index_bars_plot']


@figure_size((8, 8))
def colour_rendering_index_bars_plot(illuminant, **kwargs):
    """
    Plots the *colour rendering index* of given illuminant.

    Parameters
    ----------
    illuminant : SpectralPowerDistribution
        Illuminant to plot the *colour rendering index*.
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

    figure, axis = matplotlib.pyplot.subplots()

    cri, colour_rendering_indexes, additional_data = \
        colour_rendering_index(illuminant, additional_data=True)

    colours = ([[1] * 3] + [normalise(XYZ_to_sRGB(x.XYZ / 100))
                            for x in additional_data[0]])
    x, y = tuple(zip(*sorted(colour_rendering_indexes.items(),
                             key=lambda x: x[0])))
    x, y = np.array([0] + list(x)), np.array(
        [cri] + list(y))

    positive = True if np.sign(min(y)) in (0, 1) else False

    width = 0.5
    bars = pylab.bar(x, y, color=colours, width=width)
    y_ticks_steps = 10
    pylab.yticks(range(0 if positive else -100,
                       100 + y_ticks_steps,
                       y_ticks_steps))
    pylab.xticks(x + width / 2,
                 ['Ra'] + ['R{0}'.format(index) for index in x[1:]])

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

    settings = {
        'title': 'Colour Rendering Index - {0}'.format(illuminant.name),
        'grid': True,
        'x_tighten': True,
        'y_tighten': True,
        'limits': [-width, 14 + width * 2, -10 if positive else -110,
                   110]}
    settings.update(kwargs)

    bounding_box(**settings)
    aspect(**settings)
    return display(**settings)
