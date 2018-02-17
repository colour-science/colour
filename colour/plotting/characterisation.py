# -*- coding: utf-8 -*-
"""
Characterisation Plotting
=========================

Defines the characterisation plotting objects:

-   :func:`colour.plotting.colour_checker_plot`
"""

from __future__ import division

import numpy as np
import pylab

from colour.characterisation import COLOURCHECKERS
from colour.models import RGB_COLOURSPACES
from colour.models import XYZ_to_sRGB, xyY_to_XYZ
from colour.plotting import (ColourSwatch, canvas, multi_colour_swatches_plot,
                             render)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['colour_checker_plot']


def colour_checker_plot(colour_checker='ColorChecker 2005', **kwargs):
    """
    Plots given colour checker.

    Parameters
    ----------
    colour_checker : unicode, optional
        Color checker name.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.
    width : numeric, optional
        {:func:`colour.plotting.multi_colour_swatches_plot`},
        Colour swatch width.
    height : numeric, optional
        {:func:`colour.plotting.multi_colour_swatches_plot`},
        Colour swatch height.
    spacing : numeric, optional
        {:func:`colour.plotting.multi_colour_swatches_plot`},
        Colour swatches spacing.
    columns : int, optional
        {:func:`colour.plotting.multi_colour_swatches_plot`},
        Colour swatches columns count.
    text_display : bool, optional
        {:func:`colour.plotting.multi_colour_swatches_plot`},
        Display colour text.
    text_size : numeric, optional
        {:func:`colour.plotting.multi_colour_swatches_plot`},
        Colour text size.
    text_offset : numeric, optional
        {:func:`colour.plotting.multi_colour_swatches_plot`},
        Colour text offset.

    Returns
    -------
    Figure
        Current figure or None.

    Raises
    ------
    KeyError
        If the given colour rendition chart is not found in the factory colour
        rendition charts.

    Examples
    --------
    >>> colour_checker_plot()  # doctest: +SKIP
    """

    canvas(**kwargs)

    colour_checker, name = COLOURCHECKERS.get(colour_checker), colour_checker
    if colour_checker is None:
        raise KeyError(('Colour checker "{0}" not found in '
                        'factory colour checkers: "{1}".').format(
                            name, sorted(COLOURCHECKERS.keys())))

    _name, data, illuminant = colour_checker
    colour_swatches = []
    for _index, label, xyY in data:
        XYZ = xyY_to_XYZ(xyY)
        RGB = XYZ_to_sRGB(XYZ, illuminant)

        colour_swatches.append(
            ColourSwatch(label.title(), np.clip(np.ravel(RGB), 0, 1)))

    background_colour = '0.1'
    width = height = 1.0
    spacing = 0.25
    columns = 6

    settings = {
        'standalone': False,
        'width': width,
        'height': height,
        'spacing': spacing,
        'columns': columns,
        'text_size': 8,
        'background_colour': background_colour,
        'margins': (-0.125, 0.125, -0.5, 0.125)
    }
    settings.update(kwargs)

    multi_colour_swatches_plot(colour_swatches, **settings)

    text_x = width * (columns / 2) + (columns * (spacing / 2)) - spacing / 2
    text_y = -(len(colour_swatches) / columns + spacing / 2)

    pylab.text(
        text_x,
        text_y,
        '{0} - {1} - Colour Rendition Chart'.format(
            name, RGB_COLOURSPACES['sRGB'].name),
        color='0.95',
        clip_on=True,
        ha='center')

    settings.update({
        'title': name,
        'facecolor': background_colour,
        'edgecolor': None,
        'standalone': True
    })

    return render(**settings)
