# -*- coding: utf-8 -*-
"""
Corresponding Chromaticities Prediction Plotting
================================================

Defines corresponding chromaticities prediction plotting objects:

-   :func:`colour.plotting.corresponding_chromaticities_prediction_plot`
"""

from __future__ import division
import pylab

from colour.corresponding import corresponding_chromaticities_prediction
from colour.plotting import (chromaticity_diagram_plot_CIE1976UCS,
                             DEFAULT_FIGURE_WIDTH, canvas, render)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['corresponding_chromaticities_prediction_plot']


def corresponding_chromaticities_prediction_plot(experiment=1,
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
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.
    show_diagram_colours : bool, optional
        {:func:`colour.plotting.chromaticity_diagram_plot_CIE1976UCS`}
        Whether to display the chromaticity diagram background colours.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> corresponding_chromaticities_prediction_plot()  # doctest: +SKIP
    """

    settings = {'figure_size': (DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_WIDTH)}
    settings.update(kwargs)

    canvas(**settings)

    settings.update({
        'title': (('Corresponding Chromaticities Prediction\n{0} ({1}) - '
                   'Experiment {2}\nCIE 1976 UCS Chromaticity Diagram').format(
                       model, transform, experiment)
                  if model.lower() in ('von kries', 'vonkries') else
                  ('Corresponding Chromaticities Prediction\n{0} - '
                   'Experiment {1}\nCIE 1976 UCS Chromaticity Diagram').format(
                       model, experiment)),
        'standalone':
            False
    })
    settings.update(kwargs)

    chromaticity_diagram_plot_CIE1976UCS(**settings)

    results = corresponding_chromaticities_prediction(
        experiment, transform=transform)

    for result in results:
        _name, uvp_t, uvp_m, uvp_p = result
        pylab.arrow(
            uvp_t[0],
            uvp_t[1],
            uvp_p[0] - uvp_t[0] - 0.1 * (uvp_p[0] - uvp_t[0]),
            uvp_p[1] - uvp_t[1] - 0.1 * (uvp_p[1] - uvp_t[1]),
            head_width=0.005,
            head_length=0.005,
            linewidth=0.5,
            color='black')
        pylab.plot(uvp_t[0], uvp_t[1], 'o', color='white')
        pylab.plot(uvp_m[0], uvp_m[1], '^', color='white')
        pylab.plot(uvp_p[0], uvp_p[1], '^', color='black')
    settings.update({
        'x_tighten': True,
        'y_tighten': True,
        'limits': (-0.1, 0.7, -0.1, 0.7),
        'standalone': True
    })
    settings.update(kwargs)

    return render(**settings)
