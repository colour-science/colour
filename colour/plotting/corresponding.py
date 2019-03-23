# -*- coding: utf-8 -*-
"""
Corresponding Chromaticities Prediction Plotting
================================================

Defines corresponding chromaticities prediction plotting objects:

-   :func:`colour.plotting.plot_corresponding_chromaticities_prediction`
"""

from __future__ import division

from colour.corresponding import corresponding_chromaticities_prediction
from colour.plotting import (COLOUR_STYLE_CONSTANTS, artist,
                             plot_chromaticity_diagram_CIE1976UCS,
                             override_style, render)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['plot_corresponding_chromaticities_prediction']


@override_style()
def plot_corresponding_chromaticities_prediction(experiment=1,
                                                 model='Von Kries',
                                                 transform='CAT02',
                                                 **kwargs):
    """
    Plots given chromatic adaptation model corresponding chromaticities
    prediction.

    Parameters
    ----------
    experiment : int, optional
        Corresponding chromaticities prediction experiment number.
    model : unicode, optional
        Corresponding chromaticities prediction model name.
    transform : unicode, optional
        Transformation to use with *Von Kries* chromatic adaptation model.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.diagrams.plot_chromaticity_diagram`,
        :func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definitions.

    Returns
    -------
    tuple
        Current figure and axes.

    Examples
    --------
    >>> plot_corresponding_chromaticities_prediction(1, 'Von Kries', 'CAT02')
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_\
Plot_Corresponding_Chromaticities_Prediction.png
        :align: center
        :alt: plot_corresponding_chromaticities_prediction
    """

    settings = {'uniform': True}
    settings.update(kwargs)

    _figure, axes = artist(**settings)

    title = (('Corresponding Chromaticities Prediction\n{0} ({1}) - '
              'Experiment {2}\nCIE 1976 UCS Chromaticity Diagram').format(
                  model, transform, experiment)
             if model.lower() in ('von kries', 'vonkries') else
             ('Corresponding Chromaticities Prediction\n{0} - '
              'Experiment {1}\nCIE 1976 UCS Chromaticity Diagram').format(
                  model, experiment))

    settings = {'axes': axes, 'title': title}
    settings.update(kwargs)
    settings['standalone'] = False

    plot_chromaticity_diagram_CIE1976UCS(**settings)

    results = corresponding_chromaticities_prediction(
        experiment, transform=transform)

    for result in results:
        _name, uvp_t, uvp_m, uvp_p = result
        axes.arrow(
            uvp_t[0],
            uvp_t[1],
            uvp_p[0] - uvp_t[0] - 0.1 * (uvp_p[0] - uvp_t[0]),
            uvp_p[1] - uvp_t[1] - 0.1 * (uvp_p[1] - uvp_t[1]),
            color=COLOUR_STYLE_CONSTANTS.colour.dark,
            head_width=0.005,
            head_length=0.005)
        axes.plot(
            uvp_t[0],
            uvp_t[1],
            'o',
            color=COLOUR_STYLE_CONSTANTS.colour.brightest,
            markeredgecolor=COLOUR_STYLE_CONSTANTS.colour.dark,
            markersize=(COLOUR_STYLE_CONSTANTS.geometry.short * 6 +
                        COLOUR_STYLE_CONSTANTS.geometry.short * 0.75),
            markeredgewidth=COLOUR_STYLE_CONSTANTS.geometry.short * 0.75)
        axes.plot(
            uvp_m[0],
            uvp_m[1],
            '^',
            color=COLOUR_STYLE_CONSTANTS.colour.brightest,
            markeredgecolor=COLOUR_STYLE_CONSTANTS.colour.dark,
            markersize=(COLOUR_STYLE_CONSTANTS.geometry.short * 6 +
                        COLOUR_STYLE_CONSTANTS.geometry.short * 0.75),
            markeredgewidth=COLOUR_STYLE_CONSTANTS.geometry.short * 0.75)
        axes.plot(
            uvp_p[0], uvp_p[1], '^', color=COLOUR_STYLE_CONSTANTS.colour.dark)

    settings.update({
        'standalone': True,
        'bounding_box': (-0.1, 0.7, -0.1, 0.7),
    })
    settings.update(kwargs)

    return render(**settings)
