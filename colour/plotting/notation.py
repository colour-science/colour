# -*- coding: utf-8 -*-
"""
Colour Notation Systems Plotting
================================

Defines the colour notation systems plotting objects:

-   :func:`colour.plotting.single_munsell_value_function_plot`
-   :func:`colour.plotting.multi_munsell_value_function_plot`
"""

from __future__ import division

import numpy as np
import pylab

from colour.notation import MUNSELL_VALUE_METHODS
from colour.plotting import DEFAULT_PLOTTING_SETTINGS, canvas, render

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'single_munsell_value_function_plot', 'multi_munsell_value_function_plot'
]


def single_munsell_value_function_plot(function='ASTM D1535-08', **kwargs):
    """
    Plots given *Lightness* function.

    Parameters
    ----------
    function : unicode, optional
        *Munsell* value function to plot.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> single_munsell_value_function_plot('ASTM D1535-08')  # doctest: +SKIP

    .. image:: ../_static/Plotting_Single_Munsell_Value_Function_Plot.png
        :align: center
        :alt: single_munsell_value_function_plot
    """

    settings = {'title': '{0} - Munsell Value Function'.format(function)}
    settings.update(kwargs)

    return multi_munsell_value_function_plot((function, ), **settings)


def multi_munsell_value_function_plot(functions=None, **kwargs):
    """
    Plots given *Munsell* value functions.

    Parameters
    ----------
    functions : array_like, optional
        *Munsell* value functions to plot.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Returns
    -------
    Figure
        Current figure or None.

    Raises
    ------
    KeyError
        If one of the given *Munsell* value function is not found in the
        factory *Munsell* value functions.

    Examples
    --------
    >>> multi_munsell_value_function_plot(['ASTM D1535-08', 'McCamy 1987'])
    ... # doctest: +SKIP

    .. image:: ../_static/Plotting_Multi_Munsell_Value_Function_Plot.png
        :align: center
        :alt: multi_munsell_value_function_plot
    """

    settings = {
        'figure_size': (DEFAULT_PLOTTING_SETTINGS.figure_width,
                        DEFAULT_PLOTTING_SETTINGS.figure_width)
    }
    settings.update(kwargs)

    canvas(**settings)

    if functions is None:
        functions = ('ASTM D1535-08', 'McCamy 1987')

    samples = np.linspace(0, 100, 1000)
    for function in functions:
        function, name = MUNSELL_VALUE_METHODS.get(function), function
        if function is None:
            raise KeyError(
                ('"{0}" "Munsell" value function not found in '
                 'factory "Munsell" value functions: "{1}".').format(
                     name, sorted(MUNSELL_VALUE_METHODS.keys())))

        pylab.plot(
            samples, [function(x) for x in samples],
            label=u'{0}'.format(name))

    settings.update({
        'title': '{0} - Munsell Functions'.format(', '.join(functions)),
        'x_label': 'Luminance Y',
        'y_label': 'Munsell Value V',
        'x_tighten': True,
        'legend': True,
        'legend_location': 'upper left',
        'grid': True,
        'bounding_box': (0, 100, 0, 10),
        'aspect': 10
    })
    settings.update(kwargs)

    return render(**settings)
